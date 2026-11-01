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
#define EIGEN_MPL2_ONLY

#include "autoware/multi_object_tracker/tracker/model/unknown_tracker.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <autoware/object_recognition_utils/object_recognition_utils.hpp>
#include <autoware_utils/geometry/boost_polygon_utils.hpp>
#include <autoware_utils/geometry/geometry.hpp>
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
// NERO_MOT_UNKNOWN_MOVER=<v_min m/s>,<rms_max m>,<window s>,<n_min> (oncomingtrack, 2026-10-31; unset = stock). An
// UNKNOWN track starts at speed 0 (sigma 10 km/h) and its CV filter needs ~1 s to follow a fast car; meanwhile its box
// (last measurement + v dt) falls behind the car's front cluster, the IoU / Mahalanobis gates drop it, the shape-jump
// reset puts it back to 0 and the car is re-born at 0 every few frames (10-27 20:47:43: a 12.4 m/s oncoming car
// published as a 0-8 m/s UNKNOWN 1-4 m behind it). Here the track's own measured positions decide: when its last
// <n_min> measurement times within <window> s (spanning >= 0.2 s) lie on a straight line (rms <= rms_max) at
// >= v_min, the track is a mover: its velocity is set to that line's (sigma 1 m/s) whenever the filter is > 1 m/s off,
// a shape-jump reset keeps that velocity instead of 0, a measurement the gates refused that continues the line is
// taken by it (TrackerProcessor::spawn), and it is published heading where it moves (pose yaw = motion direction,
// footprint turned with it, twist along x). Within 1.0 s of its last straight-line fit it stays a mover; a vehicle
// measurement on it continues it (NERO_MOT_REBIRTH). Clutter and standing objects never show such a line (their
// clusters jitter in place); a vehicle's fragments are pruned as before.
struct NeroMover
{
  bool enabled{false};
  double v_min{3.0};
  double rms_max{0.3};
  double window{0.5};
  int n_min{4};
  double d_min{0.0};
  double end_tol{99.0};
  NeroMover()
  {
    if (const char * env = std::getenv("NERO_MOT_UNKNOWN_MOVER"); env != nullptr && env[0] != '\0') {
      enabled = std::sscanf(
                  env, "%lf,%lf,%lf,%d,%lf,%lf", &v_min, &rms_max, &window, &n_min, &d_min, &end_tol) == 6 &&
                n_min >= 3;
      if (enabled) {
        RCLCPP_INFO(
          rclcpp::get_logger("UnknownTracker"),
          "[NERO_MOT_UNKNOWN_MOVER] on: v_min %.2f rms_max %.2f window %.2f n_min %d d_min %.2f end_tol %.2f", v_min,
          rms_max, window, n_min, d_min, end_tol);
      }
    }
  }
};
const NeroMover & neroMoverCfg()
{
  static const NeroMover cfg;
  return cfg;
}
constexpr double kNeroMoverHold = 1.0;     // [s] a mover stays one this long after its last straight-line fit
constexpr double kNeroMoverMinSpan = 0.2;  // [s] the fitted measurements span at least this
constexpr double kNeroMoverSigmaV = 1.0;   // [m/s] the velocity's sigma when set from the line
constexpr double kNeroMoverSigmaVLost = 10.0 / 3.6;  // [m/s] ... when the line is lost (a new track's)
constexpr int kNeroMoverMaxMisses = 2;     // measurement times in a row without a line end a mover
constexpr double kNeroMoverMaxV = 40.0;    // [m/s] no faster object is linked
constexpr double kNeroMoverFragmentBehind = 1.0;  // [m] a mover's measurement this far behind its line is its rear
constexpr double kNeroMoverFragmentAcross = 1.5;  // [m] ... within this of the line
constexpr int kNeroMoverMaxFragments = 3;         // measurements in a row
constexpr double kNeroMoverOncoming = 120.0 * M_PI / 180.0;  // [rad] the line's direction vs the robot's heading
constexpr double kNeroMoverMinArea = 0.4;  // [m2] median footprint area to start a mover
constexpr double kNeroInheritAge = 0.35;  // [s] a track born this recently can hand its positions on
constexpr double kNeroInheritGap = 0.3;   // [s] ... when its last one is at most this old

// least-squares slope of v over t
double neroSlope(const double * t, const double * v, const int n)
{
  double tm = 0.0;
  double vm = 0.0;
  for (int i = 0; i < n; ++i) {
    tm += t[i];
    vm += v[i];
  }
  tm /= static_cast<size_t>(n);
  vm /= static_cast<size_t>(n);
  double stt = 0.0;
  double stv = 0.0;
  for (int i = 0; i < n; ++i) {
    stt += (t[i] - tm) * (t[i] - tm);
    stv += (t[i] - tm) * (v[i] - vm);
  }
  return stt > 1e-9 ? stv / stt : 0.0;
}

// least-squares straight line through the points' (t, x, y): velocity and the rms of the position residuals
bool neroFitLine(const UnknownTracker::NeroView & pts, double & vx, double & vy, double & rms)
{
  const size_t n = static_cast<size_t>(pts.n);
  if (n < 3) return false;
  double tm = 0.0;
  double xm = 0.0;
  double ym = 0.0;
  for (size_t i = 0; i < n; ++i) {
    tm += pts.p[i]->t;
    xm += pts.p[i]->x;
    ym += pts.p[i]->y;
  }
  tm /= n;
  xm /= n;
  ym /= n;
  double stt = 0.0;
  double stx = 0.0;
  double sty = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const auto & p = *pts.p[i];
    stt += (p.t - tm) * (p.t - tm);
    stx += (p.t - tm) * (p.x - xm);
    sty += (p.t - tm) * (p.y - ym);
  }
  if (stt < 1e-6) return false;
  vx = stx / stt;
  vy = sty / stt;
  double ss = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const auto & p = *pts.p[i];
    const double ex = p.x - (xm + vx * (p.t - tm));
    const double ey = p.y - (ym + vy * (p.t - tm));
    ss += ex * ex + ey * ey;
  }
  rms = std::sqrt(ss / n);
  return true;
}
}  // namespace

bool UnknownTracker::neroMoverEnabled()
{
  return neroMoverCfg().enabled;
}

UnknownTracker::NeroPoint UnknownTracker::neroPoint(
  const rclcpp::Time & time, const types::DynamicObject & object)
{
  const double yaw = tf2::getYaw(object.pose.orientation);
  NeroPoint p{time.seconds(), object.pose.position.x, object.pose.position.y, {}, yaw, footprintArea(object)};
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  for (const auto & q : object.shape.footprint.points) {
    p.outline.push_back({p.x + c * q.x - s * q.y, p.y + s * q.x + c * q.y});
  }
  if (p.outline.empty()) p.outline.push_back({p.x, p.y});
  return p;
}

UnknownTracker::NeroLine UnknownTracker::neroLineTest(
  const std::deque<NeroPoint> & points, const int min_n)
{
  NeroView view;
  for (const auto & p : points) view.push(&p);
  return neroLineTest(view, min_n);
}

UnknownTracker::NeroLine UnknownTracker::neroLineTest(const NeroView & points, const int min_n)
{
  // the measured centres on a straight line at >= v_min (rms <= rms_max), moved >= d_min, and the outline's rear-most
  // and front-most points along that line both moving with it (within end_tol x v): a cluster that grows, shrinks or
  // jitters in place moves its centre but not both of its ends. (On the stack; a line that fails the centre / speed /
  // direction tests skips the outline ends and the area.)
  const auto & mover = neroMoverCfg();
  NeroLine line;
  line.n = points.n;
  if (line.n < min_n) return line;
  line.span = points.back().t - points.front().t;
  if (line.span < kNeroMoverMinSpan) return line;
  if (!neroFitLine(points, line.vx, line.vy, line.rms)) return line;
  const double v = std::hypot(line.vx, line.vy);
  line.disp = std::hypot(points.back().x - points.front().x, points.back().y - points.front().y);
  if (v < 1e-3) return line;
  // only against the robot's heading (an oncoming object): a static object wrongly given a speed away from or along
  // with the robot would weaken a stop for it; an approaching one only makes the planners more careful
  const double against =
    std::abs(autoware_utils::normalize_radian(std::atan2(line.vy, line.vx) - points.back().yaw));
  if (!(line.rms <= mover.rms_max && v >= mover.v_min && against >= kNeroMoverOncoming)) return line;
  const double ux = line.vx / v;
  const double uy = line.vy / v;
  std::array<double, kNeroMaxPoints> ts;
  std::array<double, kNeroMaxPoints> near;
  std::array<double, kNeroMaxPoints> far;
  for (int i = 0; i < points.n; ++i) {
    const auto & p = *points.p[i];
    double lo = std::numeric_limits<double>::max();
    double hi = std::numeric_limits<double>::lowest();
    for (const auto & q : p.outline) {
      const double a = q[0] * ux + q[1] * uy;
      lo = std::min(lo, a);
      hi = std::max(hi, a);
    }
    ts[i] = p.t;
    near[i] = lo;
    far[i] = hi;
  }
  line.v_near = neroSlope(ts.data(), near.data(), points.n);
  line.v_far = neroSlope(ts.data(), far.data(), points.n);
  line.ok_cont = line.rms <= mover.rms_max && v >= mover.v_min && std::abs(line.v_far - v) <= mover.end_tol * v &&
                 against >= kNeroMoverOncoming;
  std::array<double, kNeroMaxPoints> areas;
  for (int i = 0; i < points.n; ++i) areas[i] = points.p[i]->area;
  std::nth_element(areas.begin(), areas.begin() + points.n / 2, areas.begin() + points.n);
  line.area = areas[points.n / 2];
  // a mover starts on clusters of a vehicle's face (>= kNeroMoverMinArea): a flickering clutter of a few points (10-27
  // 20:47:06.9 at the kerb, 0.03-0.14 m2) hops its centre along a line too
  line.ok = line.ok_cont && line.disp >= mover.d_min && std::abs(line.v_near - v) <= mover.end_tol * v &&
            line.area >= kNeroMoverMinArea;
  return line;
}

UnknownTracker::Options UnknownTracker::options;

UnknownTracker::UnknownTracker(const rclcpp::Time & time, const types::DynamicObject & object)
: Tracker(time, object), logger_(rclcpp::get_logger("UnknownTracker"))
{
  // Set motion model parameters
  {
    constexpr double q_stddev_x = 0.5;         // [m/s]
    constexpr double q_stddev_y = 0.5;         // [m/s]
    constexpr double q_stddev_vx = 9.8 * 0.3;  // [m/(s*s)]
    constexpr double q_stddev_vy = 9.8 * 0.3;  // [m/(s*s)]
    motion_model_.setMotionParams(q_stddev_x, q_stddev_y, q_stddev_vx, q_stddev_vy);
  }

  // Set motion limits
  motion_model_.setMotionLimits(
    autoware_utils::kmph2mps(60), /* [m/s] maximum velocity, x */
    autoware_utils::kmph2mps(60)  /* [m/s] maximum velocity, y */
  );

  // Set initial state
  initializeState(time, object, false);
  has_meas_ = true;
  last_meas_x_ = object.pose.position.x;
  last_meas_y_ = object.pose.position.y;
  last_meas_area_ = footprintArea(object);
  last_meas_time_ = time;
  if (neroMoverCfg().enabled) {
    nero_points_.push_back(neroPoint(time, object));
    nero_birth_t_ = time.seconds();
  }
}

void UnknownTracker::initializeState(
  const rclcpp::Time & time, const types::DynamicObject & object, const bool zero_velocity)
{
    using autoware_utils::xyzrpy_covariance_index::XYZRPY_COV_IDX;
    const double x = object.pose.position.x;
    const double y = object.pose.position.y;
    auto pose_cov = object.pose_covariance;
    auto twist_cov = object.twist_covariance;
    const double yaw = tf2::getYaw(object.pose.orientation);

    double vx = 0.0;
    double vy = 0.0;
    if (object.kinematics.has_twist && !zero_velocity) {
      const double & vel_x = object.twist.linear.x;
      const double & vel_y = object.twist.linear.y;
      vx = std::cos(yaw) * vel_x - std::sin(yaw) * vel_y;
      vy = std::sin(yaw) * vel_x + std::cos(yaw) * vel_y;
    }

    if (!object.kinematics.has_position_covariance) {
      constexpr double p0_stddev_x = 1.0;  // [m]
      constexpr double p0_stddev_y = 1.0;  // [m]

      const double p0_cov_x = std::pow(p0_stddev_x, 2.0);
      const double p0_cov_y = std::pow(p0_stddev_y, 2.0);

      const double cos_yaw = std::cos(yaw);
      const double sin_yaw = std::sin(yaw);
      const double sin_2yaw = std::sin(2.0 * yaw);
      pose_cov[XYZRPY_COV_IDX::X_X] = p0_cov_x * cos_yaw * cos_yaw + p0_cov_y * sin_yaw * sin_yaw;
      pose_cov[XYZRPY_COV_IDX::X_Y] = 0.5 * (p0_cov_x - p0_cov_y) * sin_2yaw;
      pose_cov[XYZRPY_COV_IDX::Y_Y] = p0_cov_x * sin_yaw * sin_yaw + p0_cov_y * cos_yaw * cos_yaw;
      pose_cov[XYZRPY_COV_IDX::Y_X] = pose_cov[XYZRPY_COV_IDX::X_Y];
    }

    if (!object.kinematics.has_twist_covariance) {
      constexpr double p0_stddev_vx = autoware_utils::kmph2mps(10);  // [m/s]
      constexpr double p0_stddev_vy = autoware_utils::kmph2mps(10);  // [m/s]
      const double p0_cov_vx = std::pow(p0_stddev_vx, 2.0);
      const double p0_cov_vy = std::pow(p0_stddev_vy, 2.0);
      twist_cov[XYZRPY_COV_IDX::X_X] = p0_cov_vx;
      twist_cov[XYZRPY_COV_IDX::X_Y] = 0.0;
      twist_cov[XYZRPY_COV_IDX::Y_X] = 0.0;
      twist_cov[XYZRPY_COV_IDX::Y_Y] = p0_cov_vy;
    }

    // rotate twist covariance matrix, since it is in the vehicle coordinate system
    Eigen::MatrixXd twist_cov_rotate(2, 2);
    twist_cov_rotate(0, 0) = twist_cov[XYZRPY_COV_IDX::X_X];
    twist_cov_rotate(0, 1) = twist_cov[XYZRPY_COV_IDX::X_Y];
    twist_cov_rotate(1, 0) = twist_cov[XYZRPY_COV_IDX::Y_X];
    twist_cov_rotate(1, 1) = twist_cov[XYZRPY_COV_IDX::Y_Y];
    Eigen::MatrixXd R_yaw = Eigen::Rotation2Dd(-yaw).toRotationMatrix();
    Eigen::MatrixXd twist_cov_rotated = R_yaw * twist_cov_rotate * R_yaw.transpose();
    twist_cov[XYZRPY_COV_IDX::X_X] = twist_cov_rotated(0, 0);
    twist_cov[XYZRPY_COV_IDX::X_Y] = twist_cov_rotated(0, 1);
    twist_cov[XYZRPY_COV_IDX::Y_X] = twist_cov_rotated(1, 0);
    twist_cov[XYZRPY_COV_IDX::Y_Y] = twist_cov_rotated(1, 1);

    // initialize motion model
    motion_model_.initialize(time, x, y, pose_cov, vx, vy, twist_cov);
}

double UnknownTracker::footprintArea(const types::DynamicObject & object)
{
  const auto & pts = object.shape.footprint.points;
  if (pts.size() >= 3) {
    double a = 0.0;
    for (size_t i = 0; i < pts.size(); ++i) {
      const auto & p = pts[i];
      const auto & q = pts[(i + 1) % pts.size()];
      a += static_cast<double>(p.x) * q.y - static_cast<double>(q.x) * p.y;
    }
    return std::abs(a) * 0.5;
  }
  return object.shape.dimensions.x * object.shape.dimensions.y;
}

bool UnknownTracker::predict(const rclcpp::Time & time)
{
  return motion_model_.predictState(time);
}

bool UnknownTracker::measureWithPose(const types::DynamicObject & object)
{
  // update motion model
  bool is_updated = false;
  {
    const double x = object.pose.position.x;
    const double y = object.pose.position.y;

    is_updated = motion_model_.updateStatePose(x, y, object.pose_covariance);
    motion_model_.limitStates();
  }

  // position z
  constexpr double gain = 0.1;
  object_.pose.position.z = (1.0 - gain) * object_.pose.position.z + gain * object.pose.position.z;

  return is_updated;
}

bool UnknownTracker::measure(
  const types::DynamicObject & object, const rclcpp::Time & time,
  const types::InputChannel & /*channel_info*/)
{
  // NERO_MOT_UNKNOWN_MOVER
  double frag_vx = 0.0;
  double frag_vy = 0.0;
  const auto & mover = neroMoverCfg();
  if (mover.enabled && has_meas_ && nero_fragments_ < kNeroMoverMaxFragments && neroMover(time, frag_vx, frag_vy)) {
    // a mover's cluster > 1 m behind where it was last measured + its velocity (no car brakes that hard within a
    // frame) is a piece of its rear: the mover coasts on its line (shape, place and history kept) for up to
    // kNeroMoverMaxFragments measurements in a row
    const double dt = std::clamp((time - last_meas_time_).seconds(), 0.0, 0.5);
    const double ex = object.pose.position.x - (last_meas_x_ + frag_vx * dt);
    const double ey = object.pose.position.y - (last_meas_y_ + frag_vy * dt);
    const double v = std::hypot(frag_vx, frag_vy);
    const double along = (ex * frag_vx + ey * frag_vy) / v;
    const double across = (-ex * frag_vy + ey * frag_vx) / v;
    if (along < -kNeroMoverFragmentBehind && std::abs(across) < kNeroMoverFragmentAcross) {
      if (time.seconds() - nero_fragment_t_ > 1e-3) ++nero_fragments_;  // a frame measured twice counts once
      nero_fragment_t_ = time.seconds();
      return true;
    }
  }
  nero_fragments_ = 0;

  // update object shape
  object_.shape = object.shape;
  object_.pose.orientation = object.pose.orientation;

  // check time gap
  const double dt = motion_model_.getDeltaTime(time);
  if (0.01 /*10msec*/ < dt) {
    RCLCPP_WARN(
      logger_,
      "UnknownTracker::measure There is a large gap between predicted time and measurement time. "
      "(%f)",
      dt);
  }

  // update object
  bool reset = false;
  if (options.shape_jump_reset_distance > 0.0 && has_meas_) {
    const double px = motion_model_.getStateElement(IDX::X);
    const double py = motion_model_.getStateElement(IDX::Y);
    const double jump = std::hypot(object.pose.position.x - px, object.pose.position.y - py);
    const double a1 = footprintArea(object);
    const double ratio = std::max(a1, last_meas_area_) / std::max(std::min(a1, last_meas_area_), 1e-3);
    reset = jump > options.shape_jump_reset_distance && ratio > options.shape_jump_area_ratio;
  }
  // NERO_MOT_UNKNOWN_MOVER: the measured positions of the last <window> s on a mover's straight line
  bool line = false;
  double line_vx = 0.0;
  double line_vy = 0.0;
  if (mover.enabled) {
    const double t = time.seconds();
    if (nero_points_.empty() || t - nero_points_.back().t > 1e-3) {
      nero_points_.push_back(neroPoint(time, object));
    }
    while (!nero_points_.empty() && nero_points_.front().t < t - mover.window - 1e-3) {
      nero_points_.pop_front();
    }
    nero_last_line_ = neroLineTest(nero_points_, mover.n_min);
    line = nero_mover_ ? nero_last_line_.ok_cont : nero_last_line_.ok;
    line_vx = nero_last_line_.vx;
    line_vy = nero_last_line_.vy;
    if (line) {
      neroMoverStarts(t, line_vx, line_vy, time);
    } else if (
      nero_mover_ && nero_last_line_.n >= mover.n_min && nero_last_line_.span >= kNeroMoverMinSpan &&
      t - nero_miss_t_ > 1e-3) {
      // the line was tested (enough measurements since the fragments) and failed
      ++nero_misses_;
      nero_miss_t_ = t;
    }
  }
  bool mover_ends = false;
  if (mover.enabled && nero_mover_ && nero_misses_ >= kNeroMoverMaxMisses) {
    // no straight line any more: the velocity is as uncertain as a new track's again; what the last <window> s of
    // positions show when they stand still (else the filter's), so a mover that stopped or a transient line on a
    // cluster leaves no speed behind
    nero_mover_ = false;
    nero_misses_ = 0;
    mover_ends = true;
  }
  double keep_vx = 0.0;
  double keep_vy = 0.0;
  const bool keep = mover.enabled && neroMover(time, keep_vx, keep_vy);
  if (reset && line) {
    // a shape-jump reset in a cycle the line passes restarts with the LINE's velocity, not the filter's (10-27
    // 20:47:41.3: 3.6 m/s for 0b8b's 11 m/s line in the cycle it first passed; 0 while the filter is still slow)
    neroInitWithVelocity(time, object, line_vx, line_vy, kNeroMoverSigmaV);
  } else if (reset && keep) {
    // a mover's cluster changing shape is still the mover: restart there with its velocity, not 0
    neroInitWithVelocity(time, object, keep_vx, keep_vy, kNeroMoverSigmaV);
  } else if (reset) {
    initializeState(time, object, true);
  } else {
    measureWithPose(object);
    if (line) {
      const double fvx = motion_model_.getStateElement(IDX::VX);
      const double fvy = motion_model_.getStateElement(IDX::VY);
      if (std::hypot(fvx - line_vx, fvy - line_vy) > kNeroMoverSigmaV) {
        neroInitWithVelocity(time, object, line_vx, line_vy, kNeroMoverSigmaV);
      }
    } else if (mover_ends) {
      double vx = motion_model_.getStateElement(IDX::VX);
      double vy = motion_model_.getStateElement(IDX::VY);
      if (nero_last_line_.n >= 3 && std::hypot(nero_last_line_.vx, nero_last_line_.vy) < 0.5 * mover.v_min) {
        vx = nero_last_line_.vx;
        vy = nero_last_line_.vy;
      }
      neroInitWithVelocity(time, object, vx, vy, kNeroMoverSigmaVLost);
    }
  }
  has_meas_ = true;
  last_meas_x_ = object.pose.position.x;
  last_meas_y_ = object.pose.position.y;
  last_meas_area_ = footprintArea(object);
  last_meas_time_ = time;

  return true;
}

void UnknownTracker::neroInitWithVelocity(
  const rclcpp::Time & time, const types::DynamicObject & object, const double vx, const double vy,
  const double sigma_v)
{
  // NERO_MOT_UNKNOWN_MOVER: the state restarts at the measurement with the given map-frame velocity
  using autoware_utils::xyzrpy_covariance_index::XYZRPY_COV_IDX;
  types::DynamicObject seeded = object;
  const double yaw = tf2::getYaw(object.pose.orientation);
  seeded.kinematics.has_twist = true;
  seeded.twist.linear.x = std::cos(yaw) * vx + std::sin(yaw) * vy;
  seeded.twist.linear.y = -std::sin(yaw) * vx + std::cos(yaw) * vy;
  seeded.kinematics.has_twist_covariance = true;
  seeded.twist_covariance[XYZRPY_COV_IDX::X_X] = sigma_v * sigma_v;
  seeded.twist_covariance[XYZRPY_COV_IDX::X_Y] = 0.0;
  seeded.twist_covariance[XYZRPY_COV_IDX::Y_X] = 0.0;
  seeded.twist_covariance[XYZRPY_COV_IDX::Y_Y] = sigma_v * sigma_v;
  initializeState(time, seeded, false);
}

bool UnknownTracker::neroMover(const rclcpp::Time & time, double & vx, double & vy) const
{
  if (!neroMoverCfg().enabled || !nero_mover_) return false;
  const double age = (time - nero_mover_time_).seconds();
  if (age > kNeroMoverHold) return false;
  vx = motion_model_.getStateElement(IDX::VX);
  vy = motion_model_.getStateElement(IDX::VY);
  // the filter has followed the line since (a fragment or a stop can take it down: then it is no mover any more)
  return std::hypot(vx, vy) >= 0.5 * neroMoverCfg().v_min;
}

void UnknownTracker::neroMoverStarts(
  const double t, const double vx, const double vy, const rclcpp::Time & time)
{
  if (!nero_mover_ || t - nero_mover_time_.seconds() > kNeroMoverHold) nero_mover_start_ = t;
  nero_mover_ = true;
  nero_mover_vx_ = vx;
  nero_mover_vy_ = vy;
  nero_mover_time_ = time;
  nero_misses_ = 0;
}

double UnknownTracker::neroMoverAge(const rclcpp::Time & time) const
{
  double vx = 0.0;
  double vy = 0.0;
  if (!neroMover(time, vx, vy)) return -1.0;
  return time.seconds() - nero_mover_start_;
}

int UnknownTracker::neroInheritable(const rclcpp::Time & time, const UnknownTracker & newborn) const
{
  // a young track (born within kNeroInheritAge) whose measured positions the new track's first one continues on a
  // straight line at v_min..kNeroMoverMaxV against the robot's heading (3129 at 20:47:44.0-44.4: the det frames
  // -3.1, -2.9, -2.8, -2.7 each started a new track at 0 m/s, the cluster outran each, no track held 4 positions)
  const auto & mover = neroMoverCfg();
  if (!mover.enabled || nero_inherited_ || newborn.nero_points_.empty()) return 0;
  const double t = time.seconds();
  if (nero_birth_t_ < t - kNeroInheritAge - 1e-3) return 0;
  NeroView pts;
  for (const auto & p : nero_points_) {
    if (p.t >= t - mover.window - 1e-3 && p.t < t - 1e-3) pts.push(&p);
  }
  if (pts.n == 0) return 0;
  const auto & m = newborn.nero_points_.back();
  const auto & last = pts.back();
  const double dt = t - last.t;
  const double ex = m.x - last.x;
  const double ey = m.y - last.y;
  const double v1 = std::hypot(ex, ey) / dt;
  if (dt > kNeroInheritGap + 1e-3 || v1 < mover.v_min || v1 > kNeroMoverMaxV) return 0;
  if (std::abs(autoware_utils::normalize_radian(std::atan2(ey, ex) - m.yaw)) < kNeroMoverOncoming) return 0;
  if (pts.n >= 2) {
    // more than one: all on a straight line with the new one (the centre test of the start rule)
    pts.push(&m);
    double vx = 0.0;
    double vy = 0.0;
    double rms = 0.0;
    if (!neroFitLine(pts, vx, vy, rms)) return 0;
    const double v = std::hypot(vx, vy);
    if (rms > mover.rms_max || v < mover.v_min || v > kNeroMoverMaxV) return 0;
    if (std::abs(autoware_utils::normalize_radian(std::atan2(vy, vx) - m.yaw)) < kNeroMoverOncoming) return 0;
    return pts.n - 1;
  }
  return pts.n;
}

void UnknownTracker::neroInheritFrom(
  UnknownTracker & from, const rclcpp::Time & time, const types::DynamicObject & measurement)
{
  const auto & mover = neroMoverCfg();
  const double t = time.seconds();
  std::deque<NeroPoint> older;
  for (const auto & p : from.nero_points_) {
    if (p.t >= t - mover.window - 1e-3 && p.t < t - 1e-3) older.push_back(p);
  }
  from.nero_inherited_ = true;
  for (auto it = older.rbegin(); it != older.rend(); ++it) nero_points_.push_front(*it);
  // the start test at once (4 positions with the inherited ones: a mover from its first frame)
  nero_last_line_ = neroLineTest(nero_points_, mover.n_min);
  if (nero_last_line_.ok) {
    neroMoverStarts(t, nero_last_line_.vx, nero_last_line_.vy, time);
    neroInitWithVelocity(time, measurement, nero_last_line_.vx, nero_last_line_.vy, kNeroMoverSigmaV);
  }
}

bool UnknownTracker::neroMoverContinues(
  const rclcpp::Time & time, const types::DynamicObject & measurement) const
{
  const auto & mover = neroMoverCfg();
  if (!mover.enabled) return false;
  const double t = time.seconds();
  NeroView pts;
  for (const auto & p : nero_points_) {
    if (p.t >= t - mover.window - 1e-3 && p.t < t - 1e-3) pts.push(&p);
  }
  if (pts.n == 0) return false;
  // exact pre-gate (no allocation): on a line with rms <= rms_max over n points every residual is <= sqrt(n) rms_max,
  // so the measurement is within v_max dt + 2 sqrt(n) rms_max of the last point, or the test below fails
  const auto & last = pts.back();
  const double reach = kNeroMoverMaxV * (t - last.t) + 2.0 * std::sqrt(pts.n + 1.0) * mover.rms_max + 1e-6;
  if (
    std::hypot(measurement.pose.position.x - last.x, measurement.pose.position.y - last.y) > reach) {
    return false;
  }
  const NeroPoint m = neroPoint(time, measurement);
  pts.push(&m);
  // two earlier measurements and this one (a new track lost by the gates after its first frames)
  const auto line = neroLineTest(pts, 3);
  return line.ok && std::hypot(line.vx, line.vy) <= kNeroMoverMaxV;
}

bool UnknownTracker::getTrackedObject(
  const rclcpp::Time & time, types::DynamicObject & object) const
{
  object = object_;

  // NERO_MOT_UNKNOWN_MOVER: a mover heads where it moves (the cluster's own yaw is the robot's); its footprint turns
  // with the pose so the polygon stays where it was measured
  double mvx = 0.0;
  double mvy = 0.0;
  if (neroMover(time, mvx, mvy)) {
    const double old_yaw = tf2::getYaw(object.pose.orientation);
    const double new_yaw = std::atan2(mvy, mvx);
    const double c = std::cos(old_yaw - new_yaw);
    const double s = std::sin(old_yaw - new_yaw);
    for (auto & p : object.shape.footprint.points) {
      const double px = p.x;
      const double py = p.y;
      p.x = static_cast<float>(c * px - s * py);
      p.y = static_cast<float>(s * px + c * py);
    }
    object.pose.orientation = autoware_utils::create_quaternion_from_yaw(new_yaw);
  }

  // predict from motion model
  auto & pose = object.pose;
  auto & pose_cov = object.pose_covariance;
  auto & twist = object.twist;
  auto & twist_cov = object.twist_covariance;
  if (!motion_model_.getPredictedState(time, pose, pose_cov, twist, twist_cov)) {
    RCLCPP_WARN(logger_, "UnknownTracker::getTrackedObject: Failed to get predicted state.");
    return false;
  }

  if (options.measured_footprint && has_meas_) {
    const double dt = std::clamp((time - last_meas_time_).seconds(), 0.0, 0.5);
    pose.position.x = last_meas_x_ + motion_model_.getStateElement(IDX::VX) * dt;
    pose.position.y = last_meas_y_ + motion_model_.getStateElement(IDX::VY) * dt;
  }

  return true;
}

}  // namespace autoware::multi_object_tracker
