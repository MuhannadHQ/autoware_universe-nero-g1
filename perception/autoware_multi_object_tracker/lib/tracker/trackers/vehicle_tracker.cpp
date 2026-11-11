// Copyright 2020 TIER IV, Inc.
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

#define EIGEN_MPL2_ONLY

#include "autoware/multi_object_tracker/tracker/trackers/vehicle_tracker.hpp"

#include "autoware/multi_object_tracker/object_model/object_model.hpp"
#include "autoware/multi_object_tracker/object_model/shapes_transform.hpp"

#include <autoware_utils_math/normalization.hpp>
#include <autoware_utils_math/unit_conversion.hpp>
#include <tf2/utils.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace autoware::multi_object_tracker
{

namespace
{

void twist_scale_all(types::DynamicObject & object, const double f)
{
  object.twist.linear.x *= f;
  object.twist.linear.y *= f;
  object.twist.angular.z *= f;
}

// NERO_MOT_FLIP_DEFER=<dv_min m/s>,<v_floor m/s>,<v_move m/s>,<t_judge s>,<sigma_stand m/s> (flipspeed, 2026-10-30;
// ported onto upstream by trkup-2, 2026-11-10; unset = upstream). Upstream never turns a track on a reverse speed (its sign
// belief #13050 does), but a young track's backward box jump still becomes a reverse speed spike (s1707: -2.4 / -4.2 m/s on
// a standing truck). A SUDDEN reverse speed (the speed before the update >= v_floor and the update took it down by >= dv_min,
// or a clearly forward speed >= -v_floor before an update of this frame or the one before and down by >= dv_min since; and
// now below -5 km/h) holds the track standing: speed 0 with the update's speed variance (>= sigma_stand^2), re-held after
// every update while judged. Once the measured positions since the jump span t_judge, their least-squares speed along the
// held heading decides: <= -v_move (moving backward) or, after the jump, >= +v_move (the jump was wrong, the motion forward)
// = it moves: that measured speed is taken; else it stands: speed 0 with sigma_stand. Upstream's sign belief keeps the turn.
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

// NERO_MOT_NEWBORN_REV=<frames>,<lo>,<hi> (trkup-2 2026-11-10; unset = upstream). Upstream has no speed prior: a new vehicle track's
// first box jump becomes its speed (sd 7.6 m/s at 0.1 s) and the uncertainty fade lets >= 1.4 m / 0.1 s through; on the 11-08 19:01 trip
// 18 standing vehicles were published at 4-19 m/s within 0.5 s of birth, 14 of them REVERSE (a detector heading-sign flip moving the box
// centre 0.7-1.9 m). While a vehicle track has fewer than <frames> updates from a spawning channel, a reverse speed is exported only if
// the last such update KEPT it (speed before a and after b both reverse, lo <= b / a <= hi, the speed now within [lo, hi] of b) or
// FLIP_DEFER measured the backward motion; a first estimate (a ~ 0) is never exported. Forward speeds are untouched. Output only.
struct NeroNewbornRev
{
  bool enabled{false};
  int frames{8};
  double lo{0.8};
  double hi{2.0};
  NeroNewbornRev()
  {
    if (const char * env = std::getenv("NERO_MOT_NEWBORN_REV"); env != nullptr && env[0] != '\0') {
      enabled = std::sscanf(env, "%d,%lf,%lf", &frames, &lo, &hi) == 3;
      if (enabled) {
        RCLCPP_INFO(
          rclcpp::get_logger("VehicleTracker"), "[NERO_MOT_NEWBORN_REV] on: %d frames, kept ratio %.2f..%.2f", frames, lo,
          hi);
      }
    }
  }
};
const NeroNewbornRev & neroNewbornRev()
{
  static const NeroNewbornRev r;
  return r;
}

// Prediction time over which the axle-covariance blend reaches full decoupling.
constexpr double kAxleBlendTimeConstant = 1.0;  // [s]

types::DynamicObject normalizeYaw(const types::DynamicObject & object, const double reference_yaw)
{
  types::DynamicObject corrected = object;
  const double obs_yaw = tf2::getYaw(corrected.pose.orientation);
  const double yaw_diff = autoware_utils_math::normalize_radian(obs_yaw - reference_yaw);
  if (std::abs(yaw_diff) > M_PI_2) {
    tf2::Quaternion q;
    q.setRPY(0, 0, obs_yaw + M_PI);
    corrected.pose.orientation = tf2::toMsg(q);
    for (auto & pt : corrected.shape.footprint.points) {
      pt.x = -pt.x;
      pt.y = -pt.y;
    }
  }
  return corrected;
}

}  // namespace

VehicleTracker::VehicleTracker(
  const object_model::ObjectModel & object_model, const rclcpp::Time & time,
  const types::DynamicObject & object)
: Tracker(time, object),
  logger_(rclcpp::get_logger("VehicleTracker")),
  object_model_(object_model),
  shape_model_(object_model),
  shape_update_anchor_(BicycleMotionModel::LengthUpdateAnchor::CENTER),
  sign_belief_(object.kinematics.orientation_availability, time)
{
  // set tracker type based on object model
  switch (object_model.type) {
    case object_model::ObjectModelType::GeneralVehicle:
      tracker_type_ = TrackerType::GENERAL_VEHICLE;
      break;
    case object_model::ObjectModelType::NormalVehicle:
      tracker_type_ = TrackerType::NORMAL_VEHICLE;
      break;
    case object_model::ObjectModelType::BigVehicle:
      tracker_type_ = TrackerType::BIG_VEHICLE;
      break;
    case object_model::ObjectModelType::Bicycle:
      tracker_type_ = TrackerType::BICYCLE;
      break;
    default:
      RCLCPP_ERROR(
        logger_, "VehicleTracker: Unsupported object model type: %d",
        static_cast<int>(object_model.type));
      break;
  }

  // Initialize shape manager (forces BBOX, clears footprint, clamps)
  shape_model_.init(object);

  // Seed footprint before the motion model is ready — no tracker pose yet, direct copy
  shape_model_.updateFootprint(object, time);

  // Determine initial vehicle length for motion model initialization
  const double initial_length =
    (object.shape.type == autoware_perception_msgs::msg::Shape::BOUNDING_BOX &&
     object.shape.dimensions.x > 0.0)
      ? std::clamp(
          object.shape.dimensions.x, object_model_.size_limit.length_min,
          object_model_.size_limit.length_max)
      : object_model_.init_size.length;

  // Set motion model parameters
  motion_model_.setMotionParams(
    object_model_.process_noise, object_model_.bicycle_state, object_model_.process_limit);

  // Set initial state
  {
    using autoware_utils_geometry::xyzrpy_covariance_index::XYZRPY_COV_IDX;
    const double x = object.pose.position.x;
    const double y = object.pose.position.y;
    const double yaw = tf2::getYaw(object.pose.orientation);

    auto pose_cov = object.pose_covariance;
    if (!object.kinematics.has_position_covariance) {
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

    double vel_x = 0.0;
    double vel_y = 0.0;
    double vel_x_cov = object_model_.initial_covariance.vel_long;
    double vel_y_cov = object_model_.bicycle_state.init_slip_angle_cov;
    if (object.kinematics.has_twist) {
      vel_x = object.twist.linear.x;
      vel_y = object.twist.linear.y;
    }
    if (object.kinematics.has_twist_covariance) {
      vel_x_cov = object.twist_covariance[XYZRPY_COV_IDX::X_X];
      vel_y_cov = object.twist_covariance[XYZRPY_COV_IDX::Y_Y];
    }

    motion_model_.initialize(
      time, x, y, yaw, pose_cov, vel_x, vel_x_cov, vel_y, vel_y_cov, initial_length);
    motion_model_.setZ(object.pose.position.z);
  }
}

bool VehicleTracker::predict(const rclcpp::Time & time)
{
  // Capture the interval since the last measurement correction.
  time_since_correction_ = getElapsedTimeFromLastUpdate(time);
  return motion_model_.predictState(time);
}

void VehicleTracker::applyAxleCovarianceBlend()
{
  const double blend_ratio = std::clamp(time_since_correction_ / kAxleBlendTimeConstant, 0.0, 1.0);
  if (blend_ratio <= 0.0) return;  // no time elapsed

  motion_model_.blendAxleCovariance(blend_ratio);
}

bool VehicleTracker::updateKinematics(
  const types::DynamicObject & object, const types::InputChannel & channel_info)
{
  // Re-symmetrize the axle covariance so a common-mode lateral bias translates the box to absorb
  // localization error.
  applyAxleCovarianceBlend();

  // Use measurement length only when the channel and shape are trustworthy.
  const bool is_bbox = (object.shape.type == autoware_perception_msgs::msg::Shape::BOUNDING_BOX);
  const bool can_update_shape = channel_info.trust_extension && is_bbox;
  constexpr double min_length = 1.0;
  const double length = can_update_shape ? std::max(object.shape.dimensions.x, min_length)
                                         : std::max(motion_model_.getLength(), min_length);

  const bool is_yaw_available =
    object.kinematics.orientation_availability != types::OrientationAvailability::UNAVAILABLE &&
    channel_info.trust_orientation;
  const bool is_velocity_available = object.kinematics.has_twist;

  bool is_updated = false;
  {
    const double & x = object.pose.position.x;
    const double & y = object.pose.position.y;
    const double & yaw = tf2::getYaw(object.pose.orientation);
    const double & vel_x = object.twist.linear.x;
    const double & vel_y = object.twist.linear.y;

    if (is_yaw_available && is_velocity_available) {
      is_updated = motion_model_.updateStatePoseHeadVel(
        x, y, yaw, object.pose_covariance, vel_x, vel_y, object.twist_covariance, length);
    } else if (is_yaw_available && !is_velocity_available) {
      is_updated = motion_model_.updateStatePoseHead(x, y, yaw, object.pose_covariance, length);
    } else if (!is_yaw_available && is_velocity_available) {
      is_updated = motion_model_.updateStatePoseVel(
        x, y, object.pose_covariance, yaw, vel_x, vel_y, object.twist_covariance, length);
    } else {
      is_updated = motion_model_.updateStatePose(x, y, object.pose_covariance, length);
    }
    motion_model_.limitStates();
  }

  // Low-pass filter on z position (2D motion model does not track z).
  {
    constexpr double gain = 0.1;
    motion_model_.updateZ(object.pose.position.z, gain);
  }

  return is_updated;
}

bool VehicleTracker::updateWheelKinematics(
  const UpdateStrategy & strategy, const types::DynamicObject & measurement,
  const types::DynamicObject & prediction)
{
  // Relax the front/rear covariance asymmetry before the wheel-anchor EKF update.
  applyAxleCovarianceBlend();

  // When polygon and tracked widths disagree, the observed edge center is a biased lateral
  // measurement that the wheel-base lever amplifies into yaw. correctWheelAnchor() nudges the
  // anchor and folds the extra lateral variance into pose_cov.
  std::array<double, 36> pose_cov = measurement.pose_covariance;
  const geometry_msgs::msg::Point anchor_point =
    correctWheelAnchor(prediction, measurement.shape.dimensions.y, strategy.anchor_point, pose_cov);

  const bool measure_front = strategy.type == UpdateStrategyType::FRONT_WHEEL_UPDATE;
  shape_update_anchor_ = measure_front ? BicycleMotionModel::LengthUpdateAnchor::FRONT
                                       : BicycleMotionModel::LengthUpdateAnchor::REAR;
  const bool is_updated =
    motion_model_.updateStatePoseWheel(anchor_point.x, anchor_point.y, pose_cov, measure_front);
  // Wheel-anchor EKF only updates x/y; z position and height are applied here.
  constexpr double z_gain = 0.4;
  motion_model_.updateZ(measurement.pose.position.z, z_gain);
  shape_model_.updateHeight(measurement.shape.dimensions.z);
  return is_updated;
}

bool VehicleTracker::measure(
  const types::DynamicObject & in_object, const rclcpp::Time & time,
  const types::InputChannel & channel_info)
{
  const double dt = motion_model_.getDeltaTime(time);
  if (0.01 /*10msec*/ < dt) {
    RCLCPP_WARN(
      logger_,
      "VehicleTracker::measure There is a large gap between predicted time and measurement "
      "time. (%f)",
      dt);
  }

  // The raw yaw sign votes on the heading-sign belief before normalization discards it.
  const double tracker_yaw = motion_model_.getYawState();
  if (
    in_object.kinematics.orientation_availability != types::OrientationAvailability::UNAVAILABLE &&
    channel_info.trust_orientation) {
    const double raw_yaw_diff =
      autoware_utils_math::normalize_radian(tf2::getYaw(in_object.pose.orientation) - tracker_yaw);
    sign_belief_.vote(
      time, raw_yaw_diff,
      in_object.kinematics.orientation_availability == types::OrientationAvailability::AVAILABLE);
  }

  const types::DynamicObject corrected = normalizeYaw(in_object, tracker_yaw);

  const bool is_bbox = (corrected.shape.type == autoware_perception_msgs::msg::Shape::BOUNDING_BOX);
  neroDeferPre(time);
  updateKinematics(corrected, channel_info);
  neroDeferPost(corrected, time, channel_info);
  if (channel_info.trust_extension && is_bbox) {
    shape_model_.updateShape(corrected);
  }

  // Get current tracker pose for footprint transform
  geometry_msgs::msg::Pose tracker_pose;
  std::array<double, 36> dummy_cov{};
  geometry_msgs::msg::Twist dummy_twist;
  const bool has_pose =
    motion_model_.getPredictedState(time, tracker_pose, dummy_cov, dummy_twist, dummy_cov);
  shape_model_.updateFootprint(
    corrected, time, has_pose ? std::make_optional(tracker_pose) : std::nullopt);

  evaluateSignFlip(time);
  neroStandMeasure(in_object, time, channel_info);

  shape_update_anchor_ = BicycleMotionModel::LengthUpdateAnchor::CENTER;
  removeCache();
  return true;
}

void VehicleTracker::neroDeferPre(const rclcpp::Time & time)
{
  if (neroNewbornRev().enabled) nero_rp_pre_ = motion_model_.getStateElement(IDX::U);
  if (!neroFlipDefer().enabled) return;
  // the speed before this update, and the highest speed before any update of this frame and the one before it (two channels
  // and the merged clusters update a track several times per frame)
  constexpr double ref_window = 0.15;  // [s]
  const double t = time.seconds();
  nero_v_before_ = motion_model_.getStateElement(IDX::U);
  nero_v_frames_ = nero_v_before_;
  const int n = std::min(nero_vh_n_, static_cast<int>(nero_vh_t_.size()));
  for (int i = 0; i < n; ++i) {
    if (t - nero_vh_t_[i] <= ref_window) nero_v_frames_ = std::max(nero_v_frames_, nero_vh_v_[i]);
  }
  const int k = nero_vh_n_ % static_cast<int>(nero_vh_t_.size());
  nero_vh_t_[k] = t;
  nero_vh_v_[k] = nero_v_before_;
  ++nero_vh_n_;
}

void VehicleTracker::neroDeferPost(
  const types::DynamicObject & measurement, const rclcpp::Time & time, const types::InputChannel & channel_info)
{
  if (neroNewbornRev().enabled && channel_info.is_spawn_enabled) {
    ++nero_rp_n_;
    nero_rp_a_ = nero_rp_pre_;  // both from this detection update (a dbt update in between must not count)
    nero_rp_b_ = motion_model_.getStateElement(IDX::U);
  }
  const auto & d = neroFlipDefer();
  if (!d.enabled) return;
  const double t = time.seconds();
  const double v_after = motion_model_.getStateElement(IDX::U);
  if (!nero_held_) {
    constexpr double reverse_limit = -5.0 / 3.6;  // [m/s] 0.44's stock turn limit (isBeyondReverseLimit)
    const bool sudden_now = nero_v_before_ >= d.v_floor && nero_v_before_ - v_after >= d.dv_min;
    const bool sudden_frames = nero_v_frames_ >= -d.v_floor && nero_v_frames_ - v_after >= d.dv_min;
    if (!((sudden_now || sudden_frames) && v_after < reverse_limit)) return;
    nero_held_ = true;
    nero_held_t0_ = t;
    // the car's rule: a track published before the jump, or whose own speeds before it (the last updates, incl. the speed
    // before this one) spanned >= t_judge without moving backward, is held standing while judged; a fresh unpublished
    // track keeps its own state (what upstream publishes) until the decision
    {
      const int n = std::min(nero_vh_n_, static_cast<int>(nero_vh_t_.size()));
      double t_oldest = t, v_min = nero_v_before_;
      for (int i = 0; i < n; ++i) {
        t_oldest = std::min(t_oldest, nero_vh_t_[i]);
        v_min = std::min(v_min, nero_vh_v_[i]);
      }
      const bool own_still = t - t_oldest >= d.t_judge && v_min > -d.v_move;
      nero_hold_state_ = nero_published_ || own_still;
    }
    const double yaw = motion_model_.getYawState();
    nero_held_ux_ = std::cos(yaw);
    nero_held_uy_ = std::sin(yaw);
    nero_hn_ = 0;
    nero_hold_pvv_ =
      std::max(motion_model_.getCovarianceElement(IDX::U, IDX::U), d.sigma_stand * d.sigma_stand);
    RCLCPP_DEBUG(
      logger_, "[NERO_MOT_FLIP_DEFER] hold: t %.2f yaw %.0f v %.2f -> %.2f", t, yaw * 180.0 / M_PI,
      sudden_now ? nero_v_before_ : nero_v_frames_, v_after);
  }
  // still judged: a held track follows the positions and reports speed 0
  if (nero_hold_state_) motion_model_.neroHoldStanding(nero_hold_pvv_);
  const double along =
    measurement.pose.position.x * nero_held_ux_ + measurement.pose.position.y * nero_held_uy_;
  const double tr = t - nero_held_t0_;
  const bool same_frame = nero_hn_ > 0 && tr - nero_ht_[nero_hn_ - 1] < 0.02;  // another channel of a counted frame
  if (!same_frame && nero_hn_ < static_cast<int>(nero_ht_.size())) {
    nero_ht_[nero_hn_] = tr;
    nero_ha_[nero_hn_] = along;
    ++nero_hn_;
  }
  if (nero_hn_ < 2 || nero_ht_[nero_hn_ - 1] < d.t_judge - 0.02) return;
  const auto slope = [this](const int first) {
    const int n = nero_hn_ - first;
    if (n < 2) return 0.0;
    double mt = 0.0, ma = 0.0;
    for (int i = first; i < nero_hn_; ++i) {
      mt += nero_ht_[i];
      ma += nero_ha_[i];
    }
    mt /= n;
    ma /= n;
    double stt = 0.0, sta = 0.0;
    for (int i = first; i < nero_hn_; ++i) {
      stt += (nero_ht_[i] - mt) * (nero_ht_[i] - mt);
      sta += (nero_ht_[i] - mt) * (nero_ha_[i] - ma);
    }
    return stt < 1e-6 ? 0.0 : sta / stt;
  };
  const double v_all = slope(0);
  const double v_after_jump = slope(1);  // logged only
  if (v_all <= -d.v_move) nero_fd_rev_ = true;  // NERO_MOT_NEWBORN_REV: a measured backward motion
  if (v_all <= -d.v_move) {
    // it really moves backward along its heading: a held track takes the measured speed, a fresh one keeps its own
    if (nero_hold_state_) motion_model_.neroSetSpeed(v_all);
  } else {
    // (a forward motion after the jump, from a 2-point slope, is NOT taken: on upstream it published standing cars at 4-12 m/s
    //  (11-08 19:01: 63b90400 12.12, 2d8a0100, 80c00100); the car's version needed its turned copy to agree. It stands.)
    motion_model_.neroHoldStanding(d.sigma_stand * d.sigma_stand);  // it stands
  }
  RCLCPP_DEBUG(
    logger_, "[NERO_MOT_FLIP_DEFER] decided: t %.2f measured %.2f (after the jump %.2f) over %.2f s -> v %.2f", t,
    v_all, v_after_jump, nero_ht_[nero_hn_ - 1], motion_model_.getStateElement(IDX::U));
  nero_held_ = false;
}

void VehicleTracker::neroStandMeasure(
  const types::DynamicObject & box, const rclcpp::Time & time, const types::InputChannel & channel_info)
{
  if (!neroStandSpeedParams().enabled) return;
  // NERO_MOT_STAND_SPEED: 0.44's anchor rule (getNearestCornerOrSurface): the box end the robot is beyond
  const double x = box.pose.position.x, y = box.pose.position.y, yaw = tf2::getYaw(box.pose.orientation);
  const double length = box.shape.dimensions.x;
  double anchor_x = 0.0;
  if (nero_ego_pos_) {
    const double xl = std::cos(yaw) * (nero_ego_pos_->x - x) + std::sin(yaw) * (nero_ego_pos_->y - y);
    anchor_x = xl > 0.5 * length ? 0.5 * length : (xl > -0.5 * length ? 0.0 : -0.5 * length);
  }
  // camghost fields 4-6 (from its candidate source, 7b78575c7): the channel's trust and the robot's position at the measurement
  nero_stand_.measure(
    time.seconds(), channel_info.index,
    {x, y, yaw, length, anchor_x, channel_info.trust_extension && channel_info.trust_orientation,
     nero_ego_pos_.has_value(), nero_ego_pos_ ? nero_ego_pos_->x : 0.0, nero_ego_pos_ ? nero_ego_pos_->y : 0.0},
    motion_model_.getYawState(), motion_model_.getStateElement(IDX::U));
}

void VehicleTracker::evaluateSignFlip(const rclcpp::Time & time)
{
  const double vel_long = motion_model_.getStateElement(IDX::U);
  const double vel_var = motion_model_.getCovarianceElement(IDX::U, IDX::U);
  if (sign_belief_.shouldFlip(time, vel_long, vel_var)) {
    flipOrientationSign();
  }
}

// 180° heading flip of the full tracker state: motion model, stored footprint, and sign belief.
void VehicleTracker::flipOrientationSign()
{
  motion_model_.flipOrientation();
  if (shape_model_.isFootprintValid()) {
    shape_model_.flipFootprintXY();
  }
  sign_belief_.onFlipped();
  // NERO_MOT_STAND_SPEED keeps its frames and start speed along the heading: turn them with it
  if (neroStandSpeedParams().enabled) nero_stand_.onHeadingTurned();
  removeCache();
}

bool VehicleTracker::getMotionState(
  const rclcpp::Time & time, geometry_msgs::msg::Pose & pose, std::array<double, 36> & pose_cov,
  geometry_msgs::msg::Twist & twist, std::array<double, 36> & twist_cov) const
{
  return motion_model_.getPredictedState(time, pose, pose_cov, twist, twist_cov);
}

bool VehicleTracker::getTrackedObject(
  const rclcpp::Time & time, types::DynamicObject & object, const bool to_publish) const
{
  if (!getCachedObject(time, object)) {
    if (!populateKinematicObject(time, time, object)) {
      RCLCPP_WARN(logger_, "VehicleTracker::getTrackedObject: Failed to get predicted state.");
      return false;
    }
    updateCache(object, time);
  }
  // Compose bbox dimensions and stored footprint into the output object.
  assembleShapeTo(object, to_publish);

  if (to_publish) {
    nero_published_ = true;  // NERO_MOT_FLIP_DEFER only
    using autoware_utils_geometry::xyzrpy_covariance_index::XYZRPY_COV_IDX;
    constexpr double vel_cov_buffer = 0.7;
    const double vel_limit =
      std::sqrt(std::max(0.0, object.twist_covariance[XYZRPY_COV_IDX::X_X])) - vel_cov_buffer;
    if (vel_limit > 0.0) {
      // Continuous gain in [0, 1]: 0 when |v| <= vel_limit (too uncertain -> zero), ramps
      // linearly, and reaches 1 when |v| >= 2 * vel_limit (large enough -> export as is).
      auto & twist = object.twist;
      const double gain = std::clamp((std::abs(twist.linear.x) - vel_limit) / vel_limit, 0.0, 1.0);
      twist.linear.x *= gain;
    }
    // vel_limit <= 0: uncertainty within the buffer -> velocity is trustworthy, export as is.

    if (neroNewbornRev().enabled && nero_rp_n_ < neroNewbornRev().frames && object.twist.linear.x < 0.0) {
      // NERO_MOT_NEWBORN_REV: a young track's reverse speed only once an update kept it
      const auto & r = neroNewbornRev();
      const double a = nero_rp_a_, b = nero_rp_b_, now = motion_model_.getStateElement(IDX::U);
      const bool kept = std::fabs(a) >= 0.05 && a * b > 0.0 && a * now > 0.0 && r.lo <= b / a && b / a <= r.hi &&
                        r.lo <= now / b && now / b <= r.hi;
      if (!kept && !nero_fd_rev_) {
        object.twist.linear.x = 0.0;
        object.twist.linear.y = 0.0;
        object.twist.angular.z = 0.0;
      }
    }
    if (neroStandSpeedParams().enabled && nero_stand_.valid()) {
      // NERO_MOT_STAND_SPEED: publish the refined speed (0 while the boxes stand; a measured start speed after a stand),
      // the filter's state untouched; the twist scaled as a whole (a standing vehicle has no yaw rate either)
      const double vel = motion_model_.getStateElement(IDX::U);
      const double v_out = nero_stand_.output();
      if (std::fabs(vel) > 1e-3) {
        const double f = v_out / vel;
        twist_scale_all(object, f);
      } else {
        object.twist.linear.x = v_out;
        object.twist.linear.y = 0.0;
        object.twist.angular.z = 0.0;
      }
    }
  }

  return true;
}

bool VehicleTracker::conditionedUpdate(
  const types::DynamicObject & measurement, const types::DynamicObject & prediction,
  const rclcpp::Time & measurement_time, const types::InputChannel & channel_info)
{
  const auto aligned = shapes::alignClusterToOrientation(measurement, motion_model_.getYawState());
  const types::DynamicObject & meas = aligned ? *aligned : measurement;

  UpdateStrategy strategy = determineUpdateStrategy(meas, prediction);

  if (strategy.type == UpdateStrategyType::WEAK_UPDATE) {
    const types::DynamicObject pseudo_measurement =
      createPseudoMeasurement(measurement, prediction, true);

    const types::DynamicObject pseudo_corrected =
      normalizeYaw(pseudo_measurement, motion_model_.getYawState());
    neroDeferPre(measurement_time);
    updateKinematics(pseudo_corrected, channel_info);
    neroDeferPost(pseudo_corrected, measurement_time, channel_info);

    // Only update height from real measurement (z-span of polygon cluster is reliable)
    shape_model_.updateHeight(measurement.shape.dimensions.z);

    // Store footprint from the real measurement using the post-update tracker pose
    geometry_msgs::msg::Pose tracker_pose;
    std::array<double, 36> dummy_cov{};
    geometry_msgs::msg::Twist dummy_twist;
    const bool has_pose = motion_model_.getPredictedState(
      measurement_time, tracker_pose, dummy_cov, dummy_twist, dummy_cov);
    shape_model_.updateFootprint(
      measurement, measurement_time, has_pose ? std::make_optional(tracker_pose) : std::nullopt);

    evaluateSignFlip(measurement_time);
    neroStandMeasure(meas, measurement_time, channel_info);

    shape_update_anchor_ = BicycleMotionModel::LengthUpdateAnchor::CENTER;
    removeCache();
    return true;
  }

  // Use the aligned measurement so the polygon width/anchor are expressed in the tracker frame
  // (raw cluster dimensions.y is in the cluster's own local frame).
  neroDeferPre(measurement_time);
  const bool is_updated = updateWheelKinematics(strategy, meas, prediction);
  neroDeferPost(meas, measurement_time, channel_info);

  geometry_msgs::msg::Pose tracker_pose;
  std::array<double, 36> dummy_cov{};
  geometry_msgs::msg::Twist dummy_twist;
  const bool has_pose = motion_model_.getPredictedState(
    measurement_time, tracker_pose, dummy_cov, dummy_twist, dummy_cov);
  shape_model_.updateFootprint(
    measurement, measurement_time, has_pose ? std::make_optional(tracker_pose) : std::nullopt);

  evaluateSignFlip(measurement_time);
  neroStandMeasure(meas, measurement_time, channel_info);

  shape_update_anchor_ = BicycleMotionModel::LengthUpdateAnchor::CENTER;
  removeCache();
  return is_updated;
}

void VehicleTracker::assembleShapeTo(types::DynamicObject & output, bool /*to_publish*/) const
{
  // Vehicle length is owned by the motion model and supplied at export time.
  shape_model_.exportTo(output, motion_model_.getLength());
}

void VehicleTracker::setObjectShape(const autoware_perception_msgs::msg::Shape & shape)
{
  const auto new_len = shape_model_.setShape(shape, getLatestMeasurementTime());
  if (new_len) {
    motion_model_.updateStateLength(*new_len, shape_update_anchor_);
  }
  shape_update_anchor_ = BicycleMotionModel::LengthUpdateAnchor::CENTER;
}

}  // namespace autoware::multi_object_tracker
