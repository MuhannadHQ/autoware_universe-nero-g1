// Copyright 2020 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//
// Author: v1.0 Yukihiro Saito
//
#define EIGEN_MPL2_ONLY

#include "autoware/multi_object_tracker/tracker/model/vehicle_tracker.hpp"

#include "autoware/multi_object_tracker/object_model/shapes.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <autoware/object_recognition_utils/object_recognition_utils.hpp>
#include <autoware_utils/geometry/boost_polygon_utils.hpp>
#include <autoware_utils/math/normalization.hpp>
#include <autoware_utils/math/unit_conversion.hpp>
#include <autoware_utils/ros/msg_covariance.hpp>

#include <bits/stdc++.h>
#include <tf2/utils.h>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif

namespace autoware::multi_object_tracker
{
namespace
{
// NERO_MOT_FLIP_DEFER=<dv_min m/s>,<v_floor m/s>,<v_move m/s>,<t_judge s>,<sigma_stand m/s> (flipspeed, 2026-10-30;
// unset = stock). The bicycle model turns a track by 180 deg as soon as its speed estimate falls below -5 km/h. On a
// new or young track that estimate is the jump of the box centre between two frames (initial speed variance
// (1000 km/h)^2), so the jitter of a standing vehicle's box turns it and gives it a speed. A SUDDEN reverse speed (the
// speed before the update >= v_floor, i.e. not already reverse, and the update took it down by >= dv_min; or a clearly
// forward speed >= -v_floor before an update of this frame or the one before and down by >= dv_min since: two
// channels and duplicate boxes update a track several times per frame) is not turned at once: the track keeps its heading and reports speed 0 (with the speed
// variance of the update, so it gates measurements as widely as the turned one) and the object turned as stock turns
// it runs beside it on the same measurements. Once the measured positions since the jump span t_judge, their
// least-squares speed decides: along the turned heading >= v_move, or after the jump <= -v_move with the turned state
// itself back on the held heading at >= v_move = it moves: the track takes the turned state as is (what stock
// publishes); else it stands: speed 0 with sigma_stand, its heading kept. A track already published when held shows
// the held state while judged; one first published while judged shows what the vehicle track already at its place
// says (the same object seen twice): it moves the turned way = the turn is taken at once; it stands = the held state;
// none = the turned state.
struct NeroFlipDefer
{
  bool enabled{false};
  double dv_min{1.0};
  double v_floor{-0.5};
  double v_move{1.0};
  double t_judge{0.2};
  double sigma_stand{1.0};
  NeroFlipDefer()
  {
    if (const char * env = std::getenv("NERO_MOT_FLIP_DEFER"); env != nullptr && env[0] != '\0') {
      enabled =
        std::sscanf(env, "%lf,%lf,%lf,%lf,%lf", &dv_min, &v_floor, &v_move, &t_judge, &sigma_stand) == 5;
      if (enabled) {
        RCLCPP_INFO(
          rclcpp::get_logger("VehicleTracker"),
          "[NERO_MOT_FLIP_DEFER] on: dv_min %.2f v_floor %.2f v_move %.2f t_judge %.2f sigma_stand %.2f", dv_min,
          v_floor, v_move, t_judge, sigma_stand);
      }
    }
  }
};
const NeroFlipDefer & neroFlipDefer()
{
  static const NeroFlipDefer defer;
  return defer;
}
}  // namespace

bool VehicleTracker::neroFlipDeferEnabled()
{
  return neroFlipDefer().enabled;
}

VehicleTracker::Options VehicleTracker::options{};
VehicleTracker::VehicleTracker(
  const object_model::ObjectModel & object_model, const rclcpp::Time & time,
  const types::DynamicObject & object)
: Tracker(time, object),
  object_model_(object_model),
  logger_(rclcpp::get_logger("VehicleTracker")),
  tracking_offset_(Eigen::Vector2d::Zero())
{
  // velocity deviation threshold
  //   if the predicted velocity is close to the observed velocity,
  //   the observed velocity is used as the measurement.
  velocity_deviation_threshold_ = autoware_utils::kmph2mps(10);  // [m/s]

  if (object.shape.type != autoware_perception_msgs::msg::Shape::BOUNDING_BOX) {
    // set default initial size
    auto & object_extension = object_.shape.dimensions;
    object_extension.x = object_model_.init_size.length;
    object_extension.y = object_model_.init_size.width;
    object_extension.z = object_model_.init_size.height;
  }
  object_.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;

  // set maximum and minimum size
  limitObjectExtension(object_model_);

  // Set motion model parameters
  {
    const double q_stddev_acc_long = object_model_.process_noise.acc_long;
    const double q_stddev_acc_lat = object_model_.process_noise.acc_lat;
    const double q_stddev_yaw_rate_min = object_model_.process_noise.yaw_rate_min;
    const double q_stddev_yaw_rate_max = object_model_.process_noise.yaw_rate_max;
    const double q_stddev_slip_rate_min = object_model_.bicycle_state.slip_rate_stddev_min;
    const double q_stddev_slip_rate_max = object_model_.bicycle_state.slip_rate_stddev_max;
    const double q_max_slip_angle = object_model_.bicycle_state.slip_angle_max;
    const double lf_ratio = object_model_.bicycle_state.wheel_pos_ratio_front;
    const double lf_min = object_model_.bicycle_state.wheel_pos_front_min;
    const double lr_ratio = object_model_.bicycle_state.wheel_pos_ratio_rear;
    const double lr_min = object_model_.bicycle_state.wheel_pos_rear_min;
    motion_model_.setMotionParams(
      q_stddev_acc_long, q_stddev_acc_lat, q_stddev_yaw_rate_min, q_stddev_yaw_rate_max,
      q_stddev_slip_rate_min, q_stddev_slip_rate_max, q_max_slip_angle, lf_ratio, lf_min, lr_ratio,
      lr_min);
  }

  // Set motion limits
  {
    const double max_vel = object_model_.process_limit.vel_long_max;
    const double max_slip = object_model_.bicycle_state.slip_angle_max;
    motion_model_.setMotionLimits(max_vel, max_slip);  // maximum velocity and slip angle
  }

  // Set initial state
  {
    using autoware_utils::xyzrpy_covariance_index::XYZRPY_COV_IDX;
    const double x = object.pose.position.x;
    const double y = object.pose.position.y;
    const double yaw = tf2::getYaw(object.pose.orientation);

    auto pose_cov = object.pose_covariance;
    if (!object.kinematics.has_position_covariance) {
      // initial state covariance
      const auto & p0_cov_x = object_model_.initial_covariance.pos_x;
      const auto & p0_cov_y = object_model_.initial_covariance.pos_y;
      const auto & p0_cov_yaw = object_model_.initial_covariance.yaw;

      const double cos_yaw = std::cos(yaw);
      const double sin_yaw = std::sin(yaw);
      const double sin_2yaw = std::sin(2.0 * yaw);
      pose_cov[XYZRPY_COV_IDX::X_X] = p0_cov_x * cos_yaw * cos_yaw + p0_cov_y * sin_yaw * sin_yaw;
      pose_cov[XYZRPY_COV_IDX::X_Y] = 0.5 * (p0_cov_x - p0_cov_y) * sin_2yaw;
      pose_cov[XYZRPY_COV_IDX::Y_Y] = p0_cov_x * sin_yaw * sin_yaw + p0_cov_y * cos_yaw * cos_yaw;
      pose_cov[XYZRPY_COV_IDX::Y_X] = pose_cov[XYZRPY_COV_IDX::X_Y];
      pose_cov[XYZRPY_COV_IDX::YAW_YAW] = p0_cov_yaw;
    }

    double vel = 0.0;
    double vel_cov = object_model_.initial_covariance.vel_long;
    if (object.kinematics.has_twist) {
      vel = object.twist.linear.x;
    }
    if (object.kinematics.has_twist_covariance) {
      vel_cov = object.twist_covariance[XYZRPY_COV_IDX::X_X];
    }

    const double slip = 0.0;
    const double slip_cov = object_model_.bicycle_state.init_slip_angle_cov;
    const double & length = object_.shape.dimensions.x;

    // initialize motion model
    motion_model_.initialize(time, x, y, yaw, pose_cov, vel, vel_cov, slip, slip_cov, length);
  }
}

bool VehicleTracker::predict(const rclcpp::Time & time)
{
  if (nero_turned_) nero_turned_->predictState(time);
  return motion_model_.predictState(time);
}

bool VehicleTracker::updateMotionModel(
  BicycleMotionModel & motion_model, const types::DynamicObject & object,
  const bool is_yaw_available) const
{
  // velocity capability is checked only when the object has velocity measurement
  // and the predicted velocity is close to the observed velocity
  bool is_velocity_available = false;
  if (object.kinematics.has_twist) {
    const double tracked_vel = motion_model.getStateElement(IDX::VEL);
    const double & observed_vel = object.twist.linear.x;
    if (std::fabs(tracked_vel - observed_vel) < velocity_deviation_threshold_) {
      // Velocity deviation is small
      is_velocity_available = true;
    }
  }

  const double x = object.pose.position.x;
  const double y = object.pose.position.y;
  const double yaw = tf2::getYaw(object.pose.orientation);
  const double vel = object.twist.linear.x;

  if (is_yaw_available && is_velocity_available) {
    // update with yaw angle and velocity
    return motion_model.updateStatePoseHeadVel(
      x, y, yaw, object.pose_covariance, vel, object.twist_covariance);
  } else if (is_yaw_available && !is_velocity_available) {
    // update with yaw angle, but without velocity
    return motion_model.updateStatePoseHead(x, y, yaw, object.pose_covariance);
  } else if (!is_yaw_available && is_velocity_available) {
    // update without yaw angle, but with velocity
    return motion_model.updateStatePoseVel(
      x, y, object.pose_covariance, vel, object.twist_covariance);
  }
  // update without yaw angle and velocity
  return motion_model.updateStatePose(
    x, y, object.pose_covariance);  // update without yaw angle and velocity
}

bool VehicleTracker::measureWithPose(
  const types::DynamicObject & in_object, const types::InputChannel & channel_info)
{
  types::DynamicObject object = in_object;
  // get measurement yaw angle to update
  bool is_yaw_available =
    object.kinematics.orientation_availability != types::OrientationAvailability::UNAVAILABLE &&
    channel_info.trust_orientation;
  if (
    options.use_unavailable_bicycle_yaw && !is_yaw_available && channel_info.trust_orientation &&
    object.kinematics.orientation_availability == types::OrientationAvailability::UNAVAILABLE) {
    using Label = autoware_perception_msgs::msg::ObjectClassification;
    const auto label =
      autoware::object_recognition_utils::getHighestProbLabel(object.classification);
    if (label == Label::BICYCLE || label == Label::MOTORCYCLE) {
      // the uncertainty model inflated the yaw variance x1e3 for UNAVAILABLE; use the model measurement noise instead
      using autoware_utils::xyzrpy_covariance_index::XYZRPY_COV_IDX;
      object.pose_covariance[XYZRPY_COV_IDX::YAW_YAW] = object_model_.measurement_covariance.yaw;
      is_yaw_available = true;  // updateStatePoseHead folds the yaw within +-90 deg (sign unknown)
    }
  }

  // NERO_MOT_FLIP_DEFER: the speed before this update, and the highest speed before any update of this frame and the
  // one before it (two channels and duplicate boxes can take a clearly forward speed below the turn limit in several
  // updates of one frame)
  const auto & defer = neroFlipDefer();
  const bool nero_may_hold = defer.enabled && !nero_turned_;
  double nero_v_before = 0.0;
  double nero_v_frames = 0.0;
  if (defer.enabled) {
    constexpr double nero_ref_window = 0.15;  // [s] this frame and the one before at 10 Hz
    nero_v_before = motion_model_.getStateElement(IDX::VEL);
    nero_v_frames = nero_v_before;
    for (int i = 0; i < static_cast<int>(nero_v_hist_t_.size()); ++i) {
      if (nero_v_hist_n_ > i && nero_now_.seconds() - nero_v_hist_t_[i] <= nero_ref_window) {
        nero_v_frames = std::max(nero_v_frames, nero_v_hist_v_[i]);
      }
    }
    const int k = nero_v_hist_n_ % static_cast<int>(nero_v_hist_t_.size());
    nero_v_hist_t_[k] = nero_now_.seconds();
    nero_v_hist_v_[k] = nero_v_before;
    ++nero_v_hist_n_;
  }

  // update
  bool is_updated = false;
  {
    is_updated = updateMotionModel(motion_model_, object, is_yaw_available);
    if (nero_turned_) {
      updateMotionModel(*nero_turned_, object, is_yaw_available);
      nero_turned_->limitStates();
      // NERO_MOT_FLIP_DEFER: still judged: the held track follows the positions and reports speed 0
      motion_model_.holdStanding(nero_hold_pvv_);
    }
    const double nero_v_after = motion_model_.getStateElement(IDX::VEL);
    const bool nero_sudden =
      (nero_v_before >= defer.v_floor && nero_v_before - nero_v_after >= defer.dv_min) ||
      (nero_v_frames >= -defer.v_floor && nero_v_frames - nero_v_after >= defer.dv_min);
    if (nero_may_hold && nero_sudden && motion_model_.isBeyondReverseLimit()) {
      const double nero_v_ref =
        nero_v_before >= defer.v_floor && nero_v_before - nero_v_after >= defer.dv_min ? nero_v_before
                                                                                      : nero_v_frames;
      // NERO_MOT_FLIP_DEFER: a sudden reverse speed of an unconfirmed track: held as standing, the turn is judged
      nero_turned_.emplace(motion_model_);
      nero_turned_->limitStates();
      nero_turned_count_ = 0;
      nero_hold_published_ = nero_published_;
      // the track's own speeds before this one (the ring, incl. the speed before this update): spanning >= t_judge
      // and never below -v_move = it did not move the turned way before the jump
      {
        const int n = std::min(nero_v_hist_n_, static_cast<int>(nero_v_hist_t_.size()));
        double t_oldest = nero_now_.seconds();
        double v_min = nero_v_before;
        for (int i = 0; i < n; ++i) {
          t_oldest = std::min(t_oldest, nero_v_hist_t_[i]);
          v_min = std::min(v_min, nero_v_hist_v_[i]);
        }
        nero_hold_own_still_ = nero_now_.seconds() - t_oldest >= defer.t_judge && v_min > -defer.v_move;
      }
      nero_show_turned_ = false;
      nero_first_pub_ = "";
      nero_turned_ux_ = std::cos(nero_turned_->getStateElement(IDX::YAW));
      nero_turned_uy_ = std::sin(nero_turned_->getStateElement(IDX::YAW));
      RCLCPP_INFO(
        logger_,
        "[NERO_MOT_FLIP_DEFER] %02x%02x%02x%02x %s hold: t %.2f at (%.2f, %.2f): yaw %.0f v %.2f -> %.2f "
        "(sigma %.2f) would turn to yaw %.0f v %.2f",
        nero_log_uuid_[0], nero_log_uuid_[1], nero_log_uuid_[2], nero_log_uuid_[3], nero_log_model_,
        nero_now_.seconds(),
        motion_model_.getStateElement(IDX::X), motion_model_.getStateElement(IDX::Y),
        motion_model_.getStateElement(IDX::YAW) * 180.0 / M_PI, nero_v_ref,
        motion_model_.getStateElement(IDX::VEL), std::sqrt(motion_model_.getVelocityVariance()),
        nero_turned_->getStateElement(IDX::YAW) * 180.0 / M_PI, nero_turned_->getStateElement(IDX::VEL));
      nero_hold_pvv_ =
        std::max(motion_model_.getVelocityVariance(), defer.sigma_stand * defer.sigma_stand);
      motion_model_.holdStanding(nero_hold_pvv_);
    }
    motion_model_.limitStates();
  }

  // position z
  constexpr double gain = 0.1;
  object_.pose.position.z = (1.0 - gain) * object_.pose.position.z + gain * object.pose.position.z;

  return is_updated;
}

bool VehicleTracker::measureWithShape(const types::DynamicObject & object)
{
  if (object.shape.type != autoware_perception_msgs::msg::Shape::BOUNDING_BOX) {
    // do not update shape if the input is not a bounding box
    return false;
  }

  // check object size abnormality
  constexpr double size_max_multiplier = 1.5;
  constexpr double size_min_multiplier = 0.25;
  if (
    object.shape.dimensions.x > object_model_.size_limit.length_max * size_max_multiplier ||
    object.shape.dimensions.x < object_model_.size_limit.length_min * size_min_multiplier ||
    object.shape.dimensions.y > object_model_.size_limit.width_max * size_max_multiplier ||
    object.shape.dimensions.y < object_model_.size_limit.width_min * size_min_multiplier) {
    return false;
  }

  // update object size
  constexpr double gain = 0.4;
  constexpr double gain_inv = 1.0 - gain;
  auto & object_extension = object_.shape.dimensions;
  object_extension.x = gain_inv * object_extension.x + gain * object.shape.dimensions.x;
  object_extension.y = gain_inv * object_extension.y + gain * object.shape.dimensions.y;
  object_extension.z = gain_inv * object_extension.z + gain * object.shape.dimensions.z;

  // set shape type, which is bounding box
  object_.shape.type = object.shape.type;

  // set maximum and minimum size
  limitObjectExtension(object_model_);

  // update motion model
  motion_model_.updateExtendedState(object_extension.x);
  if (nero_turned_) nero_turned_->updateExtendedState(object_extension.x);

  // update offset into object position
  {
    // rotate back the offset vector from object coordinate to global coordinate
    const double yaw = motion_model_.getStateElement(IDX::YAW);
    const double offset_x_global =
      tracking_offset_.x() * std::cos(yaw) - tracking_offset_.y() * std::sin(yaw);
    const double offset_y_global =
      tracking_offset_.x() * std::sin(yaw) + tracking_offset_.y() * std::cos(yaw);
    motion_model_.adjustPosition(-gain * offset_x_global, -gain * offset_y_global);
    if (nero_turned_) {
      // NERO_MOT_FLIP_DEFER: the turned hypothesis as stock would adjust it (with its own yaw)
      const double yaw_turned = nero_turned_->getStateElement(IDX::YAW);
      nero_turned_->adjustPosition(
        -gain * (tracking_offset_.x() * std::cos(yaw_turned) - tracking_offset_.y() * std::sin(yaw_turned)),
        -gain * (tracking_offset_.x() * std::sin(yaw_turned) + tracking_offset_.y() * std::cos(yaw_turned)));
    }
    // update offset (object coordinate)
    tracking_offset_.x() = gain_inv * tracking_offset_.x();
    tracking_offset_.y() = gain_inv * tracking_offset_.y();
  }

  return true;
}

bool VehicleTracker::measure(
  const types::DynamicObject & in_object, const rclcpp::Time & time,
  const types::InputChannel & channel_info)
{
  // check time gap
  const double dt = motion_model_.getDeltaTime(time);
  if (0.01 /*10msec*/ < dt) {
    RCLCPP_WARN(
      logger_,
      "VehicleTracker::measure There is a large gap between predicted time and measurement "
      "time. (%f)",
      dt);
  }

  // update object
  types::DynamicObject updating_object = in_object;
  shapes::calcAnchorPointOffset(object_, tracking_offset_, updating_object);
  nero_now_ = time;
  measureWithPose(updating_object, channel_info);
  if (channel_info.trust_extension) {
    measureWithShape(updating_object);
  }
  if (nero_turned_) {
    // NERO_MOT_FLIP_DEFER: the measured positions since the sudden reverse speed decide
    neroAddTurnedPosition(updating_object, time);
    neroDecideTurned();
  }

  return true;
}

void VehicleTracker::neroOnPublished(
  const bool has_partner, const double partner_vx, const double partner_vy)
{
  const bool first = !nero_published_;
  nero_published_ = true;
  if (!nero_turned_ || nero_hold_published_ || !first) return;
  // a track first published while judged. A vehicle track already published at its place (its centre in this box) is
  // the same object seen twice, and its own motion tells whether the turn is real: it moves the turned way: the turn
  // is taken now; it stands: the held (standing) state is shown until the decision. With no such track the turned
  // state (what stock publishes) is shown until the decision, unless the track's own speeds before the jump spanned
  // >= t_judge without moving the turned way (a box that jumped back on a tracked vehicle): then the held state.
  if (!has_partner) {
    nero_show_turned_ = !nero_hold_own_still_;
    nero_first_pub_ = nero_hold_own_still_
                        ? " (first published while judged: no vehicle track at its place; its own speeds before the "
                          "jump did not move the turned way)"
                        : " (first published while judged: no vehicle track at its place)";
    return;
  }
  const double yaw_turned = nero_turned_->getStateElement(IDX::YAW);
  const double v_partner = partner_vx * std::cos(yaw_turned) + partner_vy * std::sin(yaw_turned);
  if (v_partner < neroFlipDefer().v_move) {
    nero_show_turned_ = false;
    nero_first_pub_ = " (first published while judged: the vehicle track at its place does not move the turned way)";
    return;
  }
  RCLCPP_INFO(
    logger_,
    "[NERO_MOT_FLIP_DEFER] %02x%02x%02x%02x %s moving: turned: t %.2f at (%.2f, %.2f) first published while judged: "
    "the vehicle track at its place moves %.2f m/s along the turned heading; held yaw %.0f v %.2f, turned yaw %.0f "
    "v %.2f",
    nero_log_uuid_[0], nero_log_uuid_[1], nero_log_uuid_[2], nero_log_uuid_[3], nero_log_model_,
    nero_turned_time_.seconds(), nero_turned_x_, nero_turned_y_, v_partner,
    motion_model_.getStateElement(IDX::YAW) * 180.0 / M_PI, motion_model_.getStateElement(IDX::VEL),
    nero_turned_->getStateElement(IDX::YAW) * 180.0 / M_PI, nero_turned_->getStateElement(IDX::VEL));
  motion_model_.takeStateOf(*nero_turned_);
  nero_turned_.reset();
  nero_show_turned_ = false;
}

void VehicleTracker::neroAddTurnedPosition(
  const types::DynamicObject & object, const rclcpp::Time & time)
{
  const double along =
    object.pose.position.x * nero_turned_ux_ + object.pose.position.y * nero_turned_uy_;
  if (nero_turned_count_ == 0) {
    nero_turned_time_ = time;
    nero_turned_x_ = object.pose.position.x;
    nero_turned_y_ = object.pose.position.y;
  }
  const double t = (time - nero_turned_time_).seconds();
  if (nero_turned_count_ > 0 && t - nero_turned_t_[nero_turned_count_ - 1] < 0.02) {
    // another channel of a frame already counted
    return;
  }
  if (nero_turned_count_ >= static_cast<int>(nero_turned_t_.size())) return;
  nero_turned_t_[nero_turned_count_] = t;
  nero_turned_along_[nero_turned_count_] = along;
  ++nero_turned_count_;
}

void VehicleTracker::neroDecideTurned()
{
  const auto & defer = neroFlipDefer();
  if (!nero_turned_ || nero_turned_count_ < 2) return;
  if (nero_turned_t_[nero_turned_count_ - 1] < defer.t_judge - 0.02) return;
  // least-squares speed of the measured positions along the turned heading: from the jump on, and after it
  const auto slope = [this](const int first) {
    const int n = nero_turned_count_ - first;
    if (n < 2) return 0.0;
    double mt = 0.0, ma = 0.0;
    for (int i = first; i < nero_turned_count_; ++i) {
      mt += nero_turned_t_[i];
      ma += nero_turned_along_[i];
    }
    mt /= n;
    ma /= n;
    double stt = 0.0, sta = 0.0;
    for (int i = first; i < nero_turned_count_; ++i) {
      stt += (nero_turned_t_[i] - mt) * (nero_turned_t_[i] - mt);
      sta += (nero_turned_t_[i] - mt) * (nero_turned_along_[i] - ma);
    }
    return stt < 1e-6 ? 0.0 : sta / stt;
  };
  const double v_turned = slope(0);
  const double v_after = slope(1);
  // it moves the turned way (the turn was right), or after the jump it moves the way it was heading (the reverse
  // speed was wrong, not the motion) and the stock state has turned back to that heading with that motion (a box
  // that jumped back and returns, jitter, leaves the stock state turned or slow): either way the stock state is the
  // object's
  const double yaw_held = motion_model_.getStateElement(IDX::YAW);
  const double yaw_turned = nero_turned_->getStateElement(IDX::YAW);
  const bool turned_back = std::cos(yaw_turned - yaw_held) > 0.0 &&
                           nero_turned_->getStateElement(IDX::VEL) >= defer.v_move;
  const bool moving =
    v_turned >= defer.v_move || (v_after <= -defer.v_move && turned_back);
  RCLCPP_INFO(
    logger_,
    "[NERO_MOT_FLIP_DEFER] %02x%02x%02x%02x %s %s: t %.2f at (%.2f, %.2f) measured %.2f m/s along the turned "
    "heading (%.2f after the jump) over %.2f s%s; held yaw %.0f v %.2f, turned yaw %.0f v %.2f",
    nero_log_uuid_[0], nero_log_uuid_[1], nero_log_uuid_[2], nero_log_uuid_[3], nero_log_model_,
    moving ? "moving: turned" : "standing: kept", nero_turned_time_.seconds(), nero_turned_x_, nero_turned_y_,
    v_turned, v_after, nero_turned_t_[nero_turned_count_ - 1],
    nero_hold_published_ ? " (published when held)" : nero_first_pub_,
    motion_model_.getStateElement(IDX::YAW) * 180.0 / M_PI, motion_model_.getStateElement(IDX::VEL),
    nero_turned_->getStateElement(IDX::YAW) * 180.0 / M_PI, nero_turned_->getStateElement(IDX::VEL));
  if (moving) {
    motion_model_.takeStateOf(*nero_turned_);
  } else {
    motion_model_.holdStanding(defer.sigma_stand * defer.sigma_stand);
  }
  nero_turned_.reset();
  nero_show_turned_ = false;
}

bool VehicleTracker::getTrackedObject(
  const rclcpp::Time & time, types::DynamicObject & object) const
{
  object = object_;

  // predict from motion model
  auto & pose = object.pose;
  auto & pose_cov = object.pose_covariance;
  auto & twist = object.twist;
  auto & twist_cov = object.twist_covariance;
  // NERO_MOT_FLIP_DEFER: a track first published while judged shows the turned state until it is decided
  const auto & model = nero_turned_ && nero_show_turned_ ? *nero_turned_ : motion_model_;
  if (!model.getPredictedState(time, pose, pose_cov, twist, twist_cov)) {
    RCLCPP_WARN(logger_, "VehicleTracker::getTrackedObject: Failed to get predicted state.");
    return false;
  }

  // set shape
  const auto origin_yaw = tf2::getYaw(object_.pose.orientation);
  const auto ekf_pose_yaw = tf2::getYaw(pose.orientation);
  object.shape.footprint =
    autoware_utils::rotate_polygon(object.shape.footprint, origin_yaw - ekf_pose_yaw);

  return true;
}

}  // namespace autoware::multi_object_tracker
