// [nero trkup] NERO_MOT_UNKNOWN_MOVER (oncomingtrack_1031 fields 1-6 + crossmover_1101 fields 7-9), ported from our
// 0.44.2-line UnknownTracker onto upstream's PolygonTracker. Unset = upstream exactly.
//
// NERO_MOT_UNKNOWN_MOVER=<v_min m/s>,<rms_max m>,<window s>,<n_min>,<d_min m>,<end_tol>[,<cross_v_min m/s>,<lead_step>
// [,<cross_tests>]]: an UNKNOWN track becomes a mover when its last >= <n_min> measurement times within <window> s
// (spanning >= 0.2 s) lie on a straight line (rms <= rms_max) at >= v_min against the robot's heading (>= 120 deg; with
// fields 7-9 also across it, 60-120 deg, from cross_v_min on a steadily advancing leading end at <cross_tests>
// consecutive measurement times), travelled >= d_min, with both ends of the measured outline moving with it (within
// end_tol x v) and a vehicle-face-sized footprint (median >= 0.4 m2). The line's velocity steers the track's filter;
// the track is published heading where it moves at the filter's (= line's) speed. On upstream an UNKNOWN that is no
// mover is published at 0 m/s: upstream's own UNKNOWN filter speed is never published while the switch is set.
#ifndef AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__NERO__NERO_UNKNOWN_MOVER_HPP_
#define AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__NERO__NERO_UNKNOWN_MOVER_HPP_

#include "autoware/multi_object_tracker/types.hpp"

#include <rclcpp/rclcpp.hpp>

#include <array>
#include <cmath>
#include <deque>
#include <vector>

namespace autoware::multi_object_tracker::nero
{

struct UnknownMoverParams
{
  bool enabled{false};
  double v_min{3.0};
  double rms_max{0.3};
  double window{0.5};
  int n_min{4};
  double d_min{0.0};
  double end_tol{99.0};
  // crossmover (fields 7-9): crossing lines
  bool cross{false};
  double cross_v_min{0.0};
  double lead_step{0.0};
  int cross_tests{1};
};
const UnknownMoverParams & unknownMoverParams();

constexpr double kMoverHold = 1.0;     // [s] a mover stays one this long after its last straight-line fit
constexpr double kMoverMinSpan = 0.2;  // [s] the fitted measurements span at least this
constexpr double kMoverSigmaV = 1.0;   // [m/s] the velocity's sigma when set from the line
constexpr double kMoverSigmaVLost = 10.0 / 3.6;  // [m/s] ... when the line is lost (a new track's)
constexpr int kMoverMaxMisses = 2;     // measurement times in a row without a line end a mover
constexpr double kMoverMaxV = 40.0;    // [m/s] no faster object is linked
constexpr double kMoverFragmentBehind = 1.0;  // [m] a mover's measurement this far behind its line is its rear
constexpr double kMoverFragmentAcross = 1.5;  // [m] ... within this of the line
constexpr int kMoverMaxFragments = 3;         // measurements in a row
constexpr double kMoverOncoming = 120.0 * M_PI / 180.0;  // [rad] the line's direction vs the robot's heading
constexpr double kMoverCrossing = 60.0 * M_PI / 180.0;   // [rad] crossmover: from here up to kMoverOncoming
constexpr double kCrossLastStep = 0.1;  // crossmover: a crossing start's leading end, newest step / line speed
constexpr double kMoverMinArea = 0.4;   // [m2] median footprint area to start a mover
constexpr double kInheritAge = 0.35;    // [s] a track born this recently can hand its positions on
constexpr double kInheritGap = 0.3;     // [s] ... when its last one is at most this old
constexpr double kMeasuredFootprintMaxDt = 0.5;  // [s] the car's measured_footprint extrapolation clamp

// the track's own measured positions (one per measurement time)
struct MoverPoint
{
  double t;
  double x;
  double y;
  std::vector<std::array<double, 2>> outline;  // the measured footprint in the map frame
  double yaw;   // the measurement's yaw: a cluster carries the robot's heading at its time
  double area;  // the measured footprint's area
};

// a line test's points without copies (the window holds one point per measurement time: <= ~20 in 0.5 s)
constexpr int kMaxPoints = 64;
struct MoverView
{
  std::array<const MoverPoint *, kMaxPoints> p{};
  int n{0};
  void push(const MoverPoint * q)
  {
    if (n == kMaxPoints) {  // keep the newest (never reached at 10-20 Hz in a 0.5 s window)
      for (int i = 1; i < n; ++i) p[i - 1] = p[i];
      --n;
    }
    p[n++] = q;
  }
  const MoverPoint & front() const { return *p[0]; }
  const MoverPoint & back() const { return *p[n - 1]; }
};

// the straight-line test over measured positions
struct MoverLine
{
  int n{0};
  double span{0.0};
  double vx{0.0};
  double vy{0.0};
  double rms{0.0};
  double disp{0.0};    // first -> last measured centre
  double v_near{0.0};  // speed along the line of the outline's rear-most and front-most points
  double v_far{0.0};
  double area{0.0};       // median footprint area of the measurements
  bool crossing{false};   // crossmover: the line runs across the robot's heading (not against it)
  double lead_step{0.0};  // crossmover: the leading end's largest single step / its travel over the window
  double last_step{0.0};  // crossmover: the leading end's newest step as a speed / the line speed
  bool ok{false};       // starts a mover: the centres on the line, travel, both outline ends moving with it
  bool ok_cont{false};  // keeps one: the centres on the line and the leading end moving with it
};

double footprintArea(const types::DynamicObject & object);
MoverPoint moverPoint(const rclcpp::Time & time, const types::DynamicObject & object);
// the speed floor of a line by its direction against the robot's heading (< 0: never a mover)
double moverFloor(const double against);
bool moverFitLine(const MoverView & pts, double & vx, double & vy, double & rms);
MoverLine moverLineTest(const MoverView & points, const int min_n);
MoverLine moverLineTest(const std::deque<MoverPoint> & points, const int min_n);

// crossmover: the robot's position at the current measurement time (the crossing mover's log line only)
void setEgo(const double x, const double y);
bool getEgo(double & x, double & y);

}  // namespace autoware::multi_object_tracker::nero

#endif  // AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__NERO__NERO_UNKNOWN_MOVER_HPP_
