// [nero trkup] NERO_MOT_UNKNOWN_MOVER on upstream's PolygonTracker: our 0.44.2-line UnknownTracker's mover rule
// (oncomingtrack_1031 + crossmover_1101: /work/ahmed-boxyaw/src/wt/perception/autoware_multi_object_tracker/lib/tracker/
// model/unknown_tracker.cpp), same logic and numbers. Adaptations to upstream's structure (research/um/RESULT.md):
//  - no shape-jump reset upstream: only its non-reset branches are ported (the line steers the filter; a mover that ends
//    restarts its velocity sigma);
//  - the filter is upstream's polygon CV filter (its own process noise), steered by the line as the car steers its own;
//  - publication: an UNKNOWN publishes the mover's velocity (heading where it moves, footprint turned to stay in place,
//    at its last measured position + velocity x time since, clamped 0.5 s = the car's measured_footprint) or 0 m/s;
//    upstream's own UNKNOWN filter speed (enable_motion_output / suppressUncertainVelocity) is never published.
#define EIGEN_MPL2_ONLY

#include "autoware/multi_object_tracker/tracker/nero/nero_unknown_mover.hpp"
#include "autoware/multi_object_tracker/tracker/trackers/polygon_tracker.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <autoware_utils_geometry/msg/covariance.hpp>
#include <autoware_utils_math/normalization.hpp>
#include <autoware_utils_math/unit_conversion.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2/utils.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace autoware::multi_object_tracker
{
namespace nero
{
namespace
{
UnknownMoverParams parseUnknownMover()
{
  UnknownMoverParams p;
  if (const char * env = std::getenv("NERO_MOT_UNKNOWN_MOVER"); env != nullptr && env[0] != '\0') {
    double cv = 0.0;
    double ls = 0.0;
    int ct = 1;
    const int n = std::sscanf(
      env, "%lf,%lf,%lf,%d,%lf,%lf,%lf,%lf,%d", &p.v_min, &p.rms_max, &p.window, &p.n_min, &p.d_min,
      &p.end_tol, &cv, &ls, &ct);
    p.enabled = (n == 6 || n == 8 || n == 9) && p.n_min >= 3;
    p.cross = p.enabled && n >= 8 && cv > 0.0 && ls > 0.0 && ct >= 1;
    if (p.cross) {
      p.cross_v_min = cv;
      p.lead_step = ls;
      p.cross_tests = n == 9 ? ct : 1;
    }
    if (p.enabled) {
      RCLCPP_INFO(
        rclcpp::get_logger("PolygonTracker"),
        "[NERO_MOT_UNKNOWN_MOVER] on: v_min %.2f rms_max %.2f window %.2f n_min %d d_min %.2f end_tol %.2f",
        p.v_min, p.rms_max, p.window, p.n_min, p.d_min, p.end_tol);
    }
    if (p.cross) {
      RCLCPP_INFO(
        rclcpp::get_logger("PolygonTracker"),
        "[NERO_MOT_UNKNOWN_MOVER] crossing movers on: cross_v_min %.2f lead_step %.2f cross_tests %d",
        p.cross_v_min, p.lead_step, p.cross_tests);
    }
  }
  return p;
}

double g_ego_x = 0.0;
double g_ego_y = 0.0;
bool g_ego_ok = false;

// least-squares slope of v over t
double slope(const double * t, const double * v, const int n)
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
}  // namespace

const UnknownMoverParams & unknownMoverParams()
{
  static const UnknownMoverParams p = parseUnknownMover();
  return p;
}

void setEgo(const double x, const double y)
{
  g_ego_x = x;
  g_ego_y = y;
  g_ego_ok = true;
}

bool getEgo(double & x, double & y)
{
  x = g_ego_x;
  y = g_ego_y;
  return g_ego_ok;
}

double moverFloor(const double against)
{
  const auto & mover = unknownMoverParams();
  if (against >= kMoverOncoming) return mover.v_min;
  if (mover.cross && against >= kMoverCrossing) return mover.cross_v_min;
  return -1.0;
}

double footprintArea(const types::DynamicObject & object)
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

MoverPoint moverPoint(const rclcpp::Time & time, const types::DynamicObject & object)
{
  const double yaw = tf2::getYaw(object.pose.orientation);
  MoverPoint p{time.seconds(), object.pose.position.x, object.pose.position.y, {}, yaw, footprintArea(object)};
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  for (const auto & q : object.shape.footprint.points) {
    p.outline.push_back({p.x + c * q.x - s * q.y, p.y + s * q.x + c * q.y});
  }
  if (p.outline.empty()) p.outline.push_back({p.x, p.y});
  return p;
}

bool moverFitLine(const MoverView & pts, double & vx, double & vy, double & rms)
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

MoverLine moverLineTest(const std::deque<MoverPoint> & points, const int min_n)
{
  MoverView view;
  for (const auto & p : points) view.push(&p);
  return moverLineTest(view, min_n);
}

MoverLine moverLineTest(const MoverView & points, const int min_n)
{
  // the measured centres on a straight line at >= v_min (rms <= rms_max), moved >= d_min, and the outline's rear-most
  // and front-most points along that line both moving with it (within end_tol x v): a cluster that grows, shrinks or
  // jitters in place moves its centre but not both of its ends
  const auto & mover = unknownMoverParams();
  MoverLine line;
  line.n = points.n;
  if (line.n < min_n) return line;
  line.span = points.back().t - points.front().t;
  if (line.span < kMoverMinSpan) return line;
  if (!moverFitLine(points, line.vx, line.vy, line.rms)) return line;
  const double v = std::hypot(line.vx, line.vy);
  line.disp = std::hypot(points.back().x - points.front().x, points.back().y - points.front().y);
  if (v < 1e-3) return line;
  // only against the robot's heading (an oncoming object); crossmover: also ACROSS it (60..120 deg), from
  // cross_v_min and only on a steady leading end
  const double against =
    std::abs(autoware_utils_math::normalize_radian(std::atan2(line.vy, line.vx) - points.back().yaw));
  const double floor = moverFloor(against);
  line.crossing = against < kMoverOncoming;
  if (!(line.rms <= mover.rms_max && floor > 0.0 && v >= floor)) return line;
  const double ux = line.vx / v;
  const double uy = line.vy / v;
  std::array<double, kMaxPoints> ts;
  std::array<double, kMaxPoints> near;
  std::array<double, kMaxPoints> far;
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
  line.v_near = slope(ts.data(), near.data(), points.n);
  line.v_far = slope(ts.data(), far.data(), points.n);
  line.ok_cont = line.rms <= mover.rms_max && v >= floor && std::abs(line.v_far - v) <= mover.end_tol * v;
  // crossmover: a crossing line starts a mover only when its leading end advances steadily
  bool steady = true;
  if (line.crossing) {
    double step = 0.0;
    for (int i = 1; i < points.n; ++i) step = std::max(step, far[i] - far[i - 1]);
    const double travel = far[points.n - 1] - far[0];
    line.lead_step = travel > 1e-3 ? step / travel : 99.0;
    steady = line.lead_step <= mover.lead_step;
    // ... and it still advances in its newest step (>= kCrossLastStep x the line speed)
    const double dt = ts[points.n - 1] - ts[points.n - 2];
    line.last_step = dt > 1e-3 ? (far[points.n - 1] - far[points.n - 2]) / dt / v : 0.0;
    steady = steady && line.last_step >= kCrossLastStep;
  }
  std::array<double, kMaxPoints> areas;
  for (int i = 0; i < points.n; ++i) areas[i] = points.p[i]->area;
  std::nth_element(areas.begin(), areas.begin() + points.n / 2, areas.begin() + points.n);
  line.area = areas[points.n / 2];
  // a mover starts on clusters of a vehicle's face (>= kMoverMinArea)
  line.ok = line.ok_cont && line.disp >= mover.d_min && std::abs(line.v_near - v) <= mover.end_tol * v &&
            line.area >= kMoverMinArea && steady;
  return line;
}

}  // namespace nero

using nero::kMoverHold;

int PolygonTracker::neroCrossPasses(const double t, const nero::MoverLine & line)
{
  // consecutive measurement times whose start test passed (one count per measurement time)
  if (t - nero_cross_t_ > 1e-3 || nero_cross_t_ < 0.0) {
    if (line.ok && nero_cross_passes_ == 0) nero_cross_first_t_ = t;
    nero_cross_passes_ = line.ok ? nero_cross_passes_ + 1 : 0;
    nero_cross_t_ = line.ok ? t : -1.0;
  }
  return nero_cross_passes_;
}

bool PolygonTracker::neroMeasure(const types::DynamicObject & object, const rclcpp::Time & time)
{
  const auto & mover = nero::unknownMoverParams();
  // a mover's cluster > 1 m behind where it was last measured + its velocity (no car brakes that hard within a frame)
  // is a piece of its rear: the mover coasts on its line (shape, place and history kept) for up to kMoverMaxFragments
  // measurements in a row
  double frag_vx = 0.0;
  double frag_vy = 0.0;
  if (nero_has_meas_ && nero_fragments_ < nero::kMoverMaxFragments && neroMover(time, frag_vx, frag_vy)) {
    const double dt = std::clamp((time - nero_last_meas_time_).seconds(), 0.0, 0.5);
    const double ex = object.pose.position.x - (nero_last_meas_x_ + frag_vx * dt);
    const double ey = object.pose.position.y - (nero_last_meas_y_ + frag_vy * dt);
    const double v = std::hypot(frag_vx, frag_vy);
    const double along = (ex * frag_vx + ey * frag_vy) / v;
    const double across = (-ex * frag_vy + ey * frag_vx) / v;
    if (along < -nero::kMoverFragmentBehind && std::abs(across) < nero::kMoverFragmentAcross) {
      if (time.seconds() - nero_fragment_t_ > 1e-3) ++nero_fragments_;  // a frame measured twice counts once
      nero_fragment_t_ = time.seconds();
      removeCache();
      return true;
    }
  }
  nero_fragments_ = 0;

  // upstream's measure()
  shape_model_.update(object);
  motion_model_.setOrientation(object.pose.orientation);
  static_motion_model_.setOrientation(object.pose.orientation);
  last_pose_ = object.pose;
  {
    const double dt = motion_model_.getDeltaTime(time);
    if (0.01 /*10msec*/ < dt) {
      RCLCPP_WARN(
        logger_,
        "PolygonTracker::measure There is a large gap between predicted time and measurement time. "
        "(%f)",
        dt);
    }
  }

  // the measured positions of the last <window> s on a mover's straight line
  const double t = time.seconds();
  if (nero_points_.empty() || t - nero_points_.back().t > 1e-3) {
    nero_points_.push_back(nero::moverPoint(time, object));
  }
  while (!nero_points_.empty() && nero_points_.front().t < t - mover.window - 1e-3) {
    nero_points_.pop_front();
  }
  nero_last_line_ = nero::moverLineTest(nero_points_, mover.n_min);
  bool line = nero_mover_ ? nero_last_line_.ok_cont : nero_last_line_.ok;
  if (mover.cross) {
    // crossmover: a crossing start needs <cross_tests> consecutive measurement times whose start test passed
    const int passes = neroCrossPasses(t, nero_last_line_);
    if (!nero_mover_ && nero_last_line_.crossing && passes < mover.cross_tests) line = false;
  }
  const double line_vx = nero_last_line_.vx;
  const double line_vy = nero_last_line_.vy;
  if (line) {
    neroMoverStarts(t, line_vx, line_vy, time);
  } else if (
    nero_mover_ && nero_last_line_.n >= mover.n_min && nero_last_line_.span >= nero::kMoverMinSpan &&
    t - nero_miss_t_ > 1e-3) {
    // the line was tested (enough measurements since the fragments) and failed
    ++nero_misses_;
    nero_miss_t_ = t;
  }
  bool mover_ends = false;
  if (nero_mover_ && nero_misses_ >= nero::kMoverMaxMisses) {
    // no straight line any more: the velocity is as uncertain as a new track's again
    nero_mover_ = false;
    nero_misses_ = 0;
    mover_ends = true;
  }

  updateKinematics(object);
  if (line) {
    const double fvx = motion_model_.getStateElement(CVMotionModel::IDX::VX);
    const double fvy = motion_model_.getStateElement(CVMotionModel::IDX::VY);
    if (std::hypot(fvx - line_vx, fvy - line_vy) > nero::kMoverSigmaV) {
      neroInitWithVelocity(time, object, line_vx, line_vy, nero::kMoverSigmaV);
    }
  } else if (mover_ends) {
    // what the last <window> s of positions show when they stand still (else the filter's), so a mover that stopped or
    // a transient line on a cluster leaves no speed behind
    double vx = motion_model_.getStateElement(CVMotionModel::IDX::VX);
    double vy = motion_model_.getStateElement(CVMotionModel::IDX::VY);
    if (
      nero_last_line_.n >= 3 &&
      std::hypot(nero_last_line_.vx, nero_last_line_.vy) < 0.5 * nero_mover_floor_) {
      vx = nero_last_line_.vx;
      vy = nero_last_line_.vy;
    }
    neroInitWithVelocity(time, object, vx, vy, nero::kMoverSigmaVLost);
  }
  nero_has_meas_ = true;
  nero_last_meas_x_ = object.pose.position.x;
  nero_last_meas_y_ = object.pose.position.y;
  nero_last_meas_time_ = time;

  removeCache();
  return true;
}

void PolygonTracker::neroInitWithVelocity(
  const rclcpp::Time & time, const types::DynamicObject & object, const double vx, const double vy,
  const double sigma_v)
{
  // the state restarts at the measurement with the given map-frame velocity (the car's initializeState with a seeded
  // twist: position covariance from the measurement, else 1 m)
  using autoware_utils_geometry::xyzrpy_covariance_index::XYZRPY_COV_IDX;
  const double yaw = tf2::getYaw(object.pose.orientation);
  auto pose_cov = object.pose_covariance;
  if (!object.kinematics.has_position_covariance) {
    constexpr double p0_cov_x = 1.0;  // [m^2]
    constexpr double p0_cov_y = 1.0;
    const double cos_yaw = std::cos(yaw);
    const double sin_yaw = std::sin(yaw);
    const double sin_2yaw = std::sin(2.0 * yaw);
    pose_cov[XYZRPY_COV_IDX::X_X] = p0_cov_x * cos_yaw * cos_yaw + p0_cov_y * sin_yaw * sin_yaw;
    pose_cov[XYZRPY_COV_IDX::X_Y] = 0.5 * (p0_cov_x - p0_cov_y) * sin_2yaw;
    pose_cov[XYZRPY_COV_IDX::Y_Y] = p0_cov_x * sin_yaw * sin_yaw + p0_cov_y * cos_yaw * cos_yaw;
    pose_cov[XYZRPY_COV_IDX::Y_X] = pose_cov[XYZRPY_COV_IDX::X_Y];
  }
  std::array<double, 36> twist_cov{};
  twist_cov[XYZRPY_COV_IDX::X_X] = sigma_v * sigma_v;
  twist_cov[XYZRPY_COV_IDX::Y_Y] = sigma_v * sigma_v;
  motion_model_.initialize(
    time, object.pose.position.x, object.pose.position.y, pose_cov, vx, vy, twist_cov);
}

bool PolygonTracker::neroMover(const rclcpp::Time & time, double & vx, double & vy) const
{
  if (!nero_on_ || !nero_mover_) return false;
  const double age = (time - nero_mover_time_).seconds();
  if (age > kMoverHold) return false;
  vx = motion_model_.getStateElement(CVMotionModel::IDX::VX);
  vy = motion_model_.getStateElement(CVMotionModel::IDX::VY);
  // the filter has followed the line since (a fragment or a stop can take it down: then it is no mover any more)
  return std::hypot(vx, vy) >= 0.5 * nero_mover_floor_;
}

void PolygonTracker::neroMoverStarts(
  const double t, const double vx, const double vy, const rclcpp::Time & time)
{
  const bool starts = !nero_mover_ || t - nero_mover_time_.seconds() > kMoverHold;
  if (starts) nero_mover_start_ = t;
  // crossmover: a crossing mover's line has passed since the first of its consecutive start tests
  if (starts) {
    nero_mover_since_ =
      nero_last_line_.crossing && nero_cross_passes_ > 0 && nero_cross_first_t_ >= 0.0 ? nero_cross_first_t_ : t;
  }
  nero_mover_cross_ = nero_last_line_.crossing;
  const auto & mover = nero::unknownMoverParams();
  nero_mover_floor_ = nero_last_line_.crossing ? mover.cross_v_min : mover.v_min;
  if (starts && nero_last_line_.crossing) {
    // crossmover: the first-drive log line (track, place, range from the robot, line speed, angle to the heading)
    const auto u = getUUID().uuid;
    const auto & p = nero_points_.back();
    const double angle =
      std::abs(autoware_utils_math::normalize_radian(std::atan2(vy, vx) - p.yaw)) * 180.0 / M_PI;
    double ex = 0.0;
    double ey = 0.0;
    const bool ego = nero::getEgo(ex, ey);
    RCLCPP_INFO(
      logger_,
      "[NERO_MOT_UNKNOWN_MOVER] crossing mover %02x%02x%02x%02x at (%.1f, %.1f), range %.1f m, line %.1f m/s, "
      "%.0f deg to the robot's heading",
      u[0], u[1], u[2], u[3], p.x, p.y, ego ? std::hypot(p.x - ex, p.y - ey) : -1.0, std::hypot(vx, vy), angle);
  }
  nero_mover_ = true;
  nero_mover_vx_ = vx;
  nero_mover_vy_ = vy;
  nero_mover_time_ = time;
  nero_misses_ = 0;
}

double PolygonTracker::neroMoverAge(const rclcpp::Time & time) const
{
  double vx = 0.0;
  double vy = 0.0;
  if (!neroMover(time, vx, vy)) return -1.0;
  return time.seconds() - nero_mover_start_;
}

bool PolygonTracker::neroMoverRecent(const rclcpp::Time & time) const
{
  if (!nero_on_ || !nero::unknownMoverParams().cross || nero_mover_start_ < 0.0 || !nero_mover_cross_) {
    return false;
  }
  return (time - nero_mover_time_).seconds() <= kMoverHold;
}

double PolygonTracker::neroCrossHeld(const rclcpp::Time & time) const
{
  double vx = 0.0;
  double vy = 0.0;
  if (
    !nero_on_ || !nero::unknownMoverParams().cross || !nero_mover_cross_ || nero_mover_since_ < 0.0 ||
    !neroMover(time, vx, vy)) {
    return -1.0;
  }
  return time.seconds() - nero_mover_since_;
}

int PolygonTracker::neroInheritable(const rclcpp::Time & time, const PolygonTracker & newborn) const
{
  // a young track (born within kInheritAge) whose measured positions the new track's first one continues on a
  // straight line at v_min..kMoverMaxV against the robot's heading (crossmover: or across it from cross_v_min)
  const auto & mover = nero::unknownMoverParams();
  if (!nero_on_ || nero_inherited_ || newborn.nero_points_.empty()) return 0;
  const double t = time.seconds();
  if (nero_birth_t_ < t - nero::kInheritAge - 1e-3) return 0;
  nero::MoverView pts;
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
  if (dt > nero::kInheritGap + 1e-3 || v1 > nero::kMoverMaxV) return 0;
  const double floor1 =
    nero::moverFloor(std::abs(autoware_utils_math::normalize_radian(std::atan2(ey, ex) - m.yaw)));
  if (floor1 <= 0.0 || v1 < floor1) return 0;
  if (pts.n >= 2) {
    // more than one: all on a straight line with the new one (the centre test of the start rule)
    pts.push(&m);
    double vx = 0.0;
    double vy = 0.0;
    double rms = 0.0;
    if (!nero::moverFitLine(pts, vx, vy, rms)) return 0;
    const double v = std::hypot(vx, vy);
    const double floor =
      nero::moverFloor(std::abs(autoware_utils_math::normalize_radian(std::atan2(vy, vx) - m.yaw)));
    if (rms > mover.rms_max || floor <= 0.0 || v < floor || v > nero::kMoverMaxV) return 0;
    return pts.n - 1;
  }
  return pts.n;
}

void PolygonTracker::neroInheritFrom(
  PolygonTracker & from, const rclcpp::Time & time, const types::DynamicObject & measurement)
{
  const auto & mover = nero::unknownMoverParams();
  const double t = time.seconds();
  std::deque<nero::MoverPoint> older;
  for (const auto & p : from.nero_points_) {
    if (p.t >= t - mover.window - 1e-3 && p.t < t - 1e-3) older.push_back(p);
  }
  from.nero_inherited_ = true;
  for (auto it = older.rbegin(); it != older.rend(); ++it) nero_points_.push_front(*it);
  // the start test at once (a mover from its first frame); crossmover: a crossing line counts the lost track's
  // consecutive passes up to its last measurement as its own
  nero_last_line_ = nero::moverLineTest(nero_points_, mover.n_min);
  if (mover.cross && !older.empty() && from.nero_cross_t_ >= older.back().t - 1e-3) {
    nero_cross_passes_ = from.nero_cross_passes_;
    nero_cross_t_ = from.nero_cross_t_;
    nero_cross_first_t_ = from.nero_cross_first_t_;
  }
  const bool passes = !mover.cross || neroCrossPasses(t, nero_last_line_) >= mover.cross_tests ||
                      !nero_last_line_.crossing;
  if (nero_last_line_.ok && passes) {
    neroMoverStarts(t, nero_last_line_.vx, nero_last_line_.vy, time);
    neroInitWithVelocity(time, measurement, nero_last_line_.vx, nero_last_line_.vy, nero::kMoverSigmaV);
    removeCache();
  }
}

bool PolygonTracker::neroMoverContinues(
  const rclcpp::Time & time, const types::DynamicObject & measurement) const
{
  if (!nero_on_) return false;
  const auto & mover = nero::unknownMoverParams();
  const double t = time.seconds();
  nero::MoverView pts;
  for (const auto & p : nero_points_) {
    if (p.t >= t - mover.window - 1e-3 && p.t < t - 1e-3) pts.push(&p);
  }
  if (pts.n == 0) return false;
  // exact pre-gate (no allocation): on a line with rms <= rms_max over n points every residual is <= sqrt(n) rms_max
  const auto & last = pts.back();
  const double reach =
    nero::kMoverMaxV * (t - last.t) + 2.0 * std::sqrt(pts.n + 1.0) * mover.rms_max + 1e-6;
  if (
    std::hypot(measurement.pose.position.x - last.x, measurement.pose.position.y - last.y) > reach) {
    return false;
  }
  const nero::MoverPoint m = nero::moverPoint(time, measurement);
  pts.push(&m);
  // two earlier measurements and this one (a new track lost by the gates after its first frames)
  const auto line = nero::moverLineTest(pts, 3);
  return line.ok && std::hypot(line.vx, line.vy) <= nero::kMoverMaxV;
}

void PolygonTracker::neroPublish(const rclcpp::Time & time, types::DynamicObject & object) const
{
  // where it was last measured (last_pose_; also before the 2nd measurement, for the processor's internal reads)
  object.pose.position.x = nero_last_meas_x_;
  object.pose.position.y = nero_last_meas_y_;
  // the rule's state as of the track's last measurement (upstream's publish clamp; the car published its filter state
  // as of then): within the car's mover hold of the last straight-line fit, the line-steered velocity while the filter
  // still follows it (>= 0.5 x the mover's floor, the car's own test). This also spans a lapse where the 2-miss rule
  // ended the mover internally (the car keeps publishing its line-steered filter there).
  const rclcpp::Time t_ref = std::min(time, getLatestMeasurementTime());
  double mvx = motion_model_.getStateElement(CVMotionModel::IDX::VX);
  double mvy = motion_model_.getStateElement(CVMotionModel::IDX::VY);
  const bool moving = nero_mover_start_ >= 0.0 && (t_ref - nero_mover_time_).seconds() <= kMoverHold &&
                      std::hypot(mvx, mvy) >= 0.5 * nero_mover_floor_;
  if (!moving) {
    // no mover: 0 m/s (upstream's own UNKNOWN filter speed is never published)
    object.twist.linear.x = 0.0;
    object.twist.linear.y = 0.0;
    object.twist.angular.z = 0.0;
    return;
  }
  // a mover heads where it moves (the cluster's own yaw is the robot's); its footprint turns with the pose so the
  // polygon stays where it was measured
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
  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, new_yaw);
  object.pose.orientation.x = q.x();
  object.pose.orientation.y = q.y();
  object.pose.orientation.z = q.z();
  object.pose.orientation.w = q.w();
  // the car's measured_footprint: the last measured position + the velocity x the time since (clamped 0.5 s)
  const double dt =
    std::clamp((time - nero_last_meas_time_).seconds(), 0.0, nero::kMeasuredFootprintMaxDt);
  object.pose.position.x = nero_last_meas_x_ + mvx * dt;
  object.pose.position.y = nero_last_meas_y_ + mvy * dt;
  object.twist.linear.x = std::hypot(mvx, mvy);
  object.twist.linear.y = 0.0;
  object.twist.angular.z = 0.0;
}

}  // namespace autoware::multi_object_tracker
