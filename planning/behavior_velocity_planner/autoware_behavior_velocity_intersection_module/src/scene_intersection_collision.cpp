// Copyright 2024 Tier IV, Inc.
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

#include "scene_intersection.hpp"
#include "nero_yield_intent.hpp"
#include "nero_yield_revoke.hpp"  // NERO_IX_YIELD_REVOKE

#include <chrono>
#include <cstdio>
#include <deque>
#include <unordered_map>
#include <utility>
#include <cstdlib>
#include <string>
#include "util.hpp"

#include <autoware/behavior_velocity_planner_common/utilization/boost_geometry_helper.hpp>  // for toGeomPoly
#include <autoware/behavior_velocity_planner_common/utilization/trajectory_utils.hpp>  // for smoothPath
#include <autoware/motion_utils/trajectory/trajectory.hpp>
#include <autoware_lanelet2_extension/utility/query.hpp>  // NERO_IX_PRIORITY_FROM_SIGNS
#include <autoware_lanelet2_extension/utility/utilities.hpp>
#include <autoware_utils/geometry/boost_polygon_utils.hpp>  // for toPolygon2d
#include <autoware_utils/geometry/geometry.hpp>
#include <magic_enum.hpp>

#include <boost/geometry/algorithms/correct.hpp>
#include <boost/geometry/algorithms/intersects.hpp>
#include <boost/geometry/algorithms/within.hpp>
#include <boost/geometry/algorithms/distance.hpp>
#include <boost/geometry/algorithms/convex_hull.hpp>  // NERO_IX_PRIORITY_NO_EXIT

#include <fmt/format.h>
#include <lanelet2_core/geometry/Polygon.h>
#include <tf2/utils.h>
#include <lanelet2_routing/RoutingGraph.h>  // Nero c3: following() of the assigned lanelet

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <list>
#include <memory>
#include <sstream>  // NERO_IX_PRIORITY_EVIDENCE
#include <string>
#include <utility>
#include <vector>

namespace
{
double nero_ego_acc_param()
{
  static const double a = [] {
    const char * v = std::getenv("NERO_IX_EGO_ACC");
    return v != nullptr ? std::max(0.0, std::atof(v)) : 0.0;
  }();
  return a;
}

struct NeroMerge
{
  bool on{false};
  double react{1.0};   // [s] follower reaction time
  double decel{2.0};   // [m/s^2] braking a follower may be asked for
  double v_ref{2.0};   // [m/s] ego speed when it leaves the turn lanelet
  double max{6.0};     // [s] cap of the required headway (and the prediction cut extension)
};

// NERO_IX_MERGE="react,decel,v_ref,max": a car that reaches ego's turn lanelet only after ego has left it
// and then drives on in ego's direction (it will follow ego) needs a headway of
// react + (v_car - v_ref) / (2 decel) behind ego, not upstream's "anything after passing_time is safe".
NeroMerge nero_merge_param()
{
  static const NeroMerge m = [] {
    NeroMerge r;
    const char * v = std::getenv("NERO_IX_MERGE");
    if (v == nullptr) return r;
    double a[4] = {r.react, r.decel, r.v_ref, r.max};
    if (std::sscanf(v, "%lf,%lf,%lf,%lf", &a[0], &a[1], &a[2], &a[3]) >= 2) {
      r.on = true;
      r.react = a[0];
      r.decel = std::max(0.1, a[1]);
      r.v_ref = a[2];
      r.max = a[3];
    }
    return r;
  }();
  return m;
}

double nero_cut_extra()
{
  static const double extra = [] {
    const char * v = std::getenv("NERO_IX_CUT_EXTRA");
    return v != nullptr ? std::max(0.0, std::atof(v)) : 0.0;
  }();
  return extra;
}

// NERO_IX_CREEP="v_creep,t_react,a_stop,acc_th,hold_s,margin" (session 4, Saudi-style pressure): a vehicle that
// CREEPS — at most v_creep for the last hold_s seconds and not accelerating more than acc_th (its own speed history: the
// tracker reports no acceleration) — and that can stop with its whole footprint clear of ego's path (stopping distance
// v*t_react + v^2/(2*a_stop) along its heading; footprint farther than ego's half width + margin from the centreline of
// ego's turn lanelet and the lanelet after it) is treated as yielding: it is left out of the attention objects, i.e. out
// of the TTC collision check and of yield-stuck. Upstream predicts a 1 m/s creeper at constant speed (it "occupies" ego's
// lane for ~8 s: always unsafe) and holds ego for any stopped car within 5 m of its crossing, so impatient drivers rolling
// toward a waiting / turning ego held it at the apex for good. A car that speeds up, or whose stop would still reach ego's
// path, keeps the full upstream check. Unset = upstream.
struct NeroCreepIx
{
  double v_creep{-1.0};
  double t_react{0.5};
  double a_stop{2.0};
  double acc_th{0.3};
  double hold_s{2.0};
  double margin{0.5};
  double t_pass{0.0};  // 7th value: the creeper keeps creeping this long while ego passes (reach v * (t_react + t_pass))
};
const NeroCreepIx & nero_creep_ix()
{
  static const NeroCreepIx p = [] {
    NeroCreepIx r;
    if (const char * e = std::getenv("NERO_IX_CREEP")) {
      std::sscanf(
        e, "%lf,%lf,%lf,%lf,%lf,%lf,%lf", &r.v_creep, &r.t_react, &r.a_stop, &r.acc_th, &r.hold_s, &r.margin,
        &r.t_pass);
    }
    return r;
  }();
  return p;
}
struct NeroCreepHist
{
  std::deque<std::pair<double, double>> s;  // (t, v)
  double last{0.0};
};
std::unordered_map<std::string, NeroCreepHist> & nero_creep_hists()
{
  static std::unordered_map<std::string, NeroCreepHist> h;
  return h;
}
double nero_steady_now()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// records the speed; returns true if the object has been creeping (and is not pulling away)
bool nero_is_creeper(const autoware_perception_msgs::msg::PredictedObject & o, double * v_out, double * acc_out)
{
  const auto & p = nero_creep_ix();
  const double t = nero_steady_now();
  const std::string key(reinterpret_cast<const char *>(o.object_id.uuid.data()), o.object_id.uuid.size());
  auto & h = nero_creep_hists()[key];
  const double v = std::hypot(
    o.kinematics.initial_twist_with_covariance.twist.linear.x,
    o.kinematics.initial_twist_with_covariance.twist.linear.y);
  if (h.s.empty() || t - h.s.back().first > 0.02) {
    h.s.emplace_back(t, v);
  }
  h.last = t;
  while (!h.s.empty() && t - h.s.front().first > p.hold_s + 1.5) h.s.pop_front();
  static double last_prune = 0.0;
  if (t - last_prune > 5.0) {
    last_prune = t;
    for (auto it = nero_creep_hists().begin(); it != nero_creep_hists().end();) {
      if (t - it->second.last > 10.0) it = nero_creep_hists().erase(it); else ++it;
    }
  }
  *v_out = v;
  *acc_out = 0.0;
  if (v > p.v_creep || h.s.size() < 3 || t - h.s.front().first < p.hold_s - 0.2) return false;
  double v_min_1s = v;
  for (const auto & [ti, vi] : h.s) {
    if (t - ti <= p.hold_s && vi > p.v_creep) return false;
    if (t - ti <= 1.0) v_min_1s = std::min(v_min_1s, vi);
  }
  *acc_out = v - v_min_1s;  // speed gained within the last second: a launch shows at once
  return *acc_out <= p.acc_th;
}

// NERO_IX_PULLAWAY_ACC=<a> (pull-away job, 2026-10-24): a vehicle in an attention lane that was standing (or crawling)
// and starts to move is also checked as accelerating at <a> m/s^2 from its current speed along its attention lane (up
// to 13.9 m/s), for up to 4 s after it started while it keeps speeding up (>= 0.25 m/s per 0.5 s), in addition to its
// predicted paths. Upstream checks only its predicted paths, which run at its current speed (map_based_prediction even
// keeps a vehicle below 1.0 m/s where it is), so a car waiting ~12 m before the conflict as if letting ego go stayed
// "safe" until it had sped up (sim: flagged 0.6-0.9 s after it moved; the field tracker's speed lags another 0.4-0.8 s)
// and ego then stopped in its path. Started = the reported speed is >= 0.5 m/s and >= 0.4 m/s above its lowest speed in
// the last 1.5 s, which was <= 0.3 m/s, on a track seen for >= 0.5 s (a track that appears moving is not a start).
// Vehicles and car-sized UNKNOWNs (footprint >= 3 m^2, >= 1 m tall: not the low ghost blobs of 10-20 #12). It only
// adds paths to the check: nothing upstream checks is dropped, a car that stays put or keeps crawling is judged exactly
// as before, and a crawler the car's creep rule treats as yielding stays so until it speeds up. Unset = upstream / the
// car as before.
double nero_pullaway_acc()
{
  static const double a = [] {
    const char * v = std::getenv("NERO_IX_PULLAWAY_ACC");
    return v != nullptr ? std::max(0.0, std::atof(v)) : 0.0;
  }();
  return a;
}
constexpr double kNeroPullawayVGo = 0.5;       // [m/s] started: at least this fast ...
constexpr double kNeroPullawayVStill = 0.3;    // [m/s] ... after being at most this fast ...
constexpr double kNeroPullawayDv = 0.4;        // [m/s] ... and this much faster than then ...
constexpr double kNeroPullawayWindow = 1.5;    // [s] ... within this window
constexpr double kNeroPullawayMinSeen = 0.5;   // [s] track age needed to call it a start
constexpr double kNeroPullawayActive = 4.0;    // [s] the model is checked this long after the start ...
constexpr double kNeroPullawayGainWindow = 0.5;  // [s] ... while its speed keeps rising by at least
constexpr double kNeroPullawayMinGain = 0.25;    // [m/s] this much within this window
constexpr double kNeroPullawayVMax = 13.9;     // [m/s] speed cap of the model
constexpr double kNeroPullawayDt = 0.25;       // [s] model path time step
constexpr double kNeroPullawayHorizon = 10.0;  // [s] model path length (the check cuts it further)
struct NeroPullawayHist
{
  std::deque<std::pair<double, double>> s;  // (t, v)
  double last{0.0};
  double t_start{-1.0};  // steady time of the start, < 0: not started
  double t_end{-1.0};    // steady time the last start window ended (a new start needs a new standstill after it)
  bool logged_unsafe{false};
};
std::unordered_map<std::string, NeroPullawayHist> & nero_pullaway_hists()
{
  static std::unordered_map<std::string, NeroPullawayHist> h;
  return h;
}
std::string nero_uuid_key(const autoware_perception_msgs::msg::PredictedObject & o)
{
  return std::string(reinterpret_cast<const char *>(o.object_id.uuid.data()), o.object_id.uuid.size());
}
bool nero_pullaway_target(const autoware_perception_msgs::msg::PredictedObject & o)
{
  using autoware_perception_msgs::msg::ObjectClassification;
  if (o.classification.empty()) return false;
  const auto label = o.classification.at(0).label;
  if (
    label == ObjectClassification::CAR || label == ObjectClassification::TRUCK ||
    label == ObjectClassification::BUS || label == ObjectClassification::TRAILER ||
    label == ObjectClassification::MOTORCYCLE) {
    return true;
  }
  return label == ObjectClassification::UNKNOWN &&
         o.shape.type == autoware_perception_msgs::msg::Shape::BOUNDING_BOX &&
         o.shape.dimensions.x * o.shape.dimensions.y >= 3.0 && o.shape.dimensions.z >= 1.0;
}
// records the object's speed (once per planning cycle, whichever module sees it first); keeps its start state
void nero_pullaway_update(const autoware_perception_msgs::msg::PredictedObject & o)
{
  const double t = nero_steady_now();
  auto & h = nero_pullaway_hists()[nero_uuid_key(o)];
  const double v = std::hypot(
    o.kinematics.initial_twist_with_covariance.twist.linear.x,
    o.kinematics.initial_twist_with_covariance.twist.linear.y);
  if (!h.s.empty() && t - h.s.back().first <= 0.02) return;  // already recorded this cycle
  h.s.emplace_back(t, v);
  h.last = t;
  while (!h.s.empty() && t - h.s.front().first > 3.0) h.s.pop_front();
  static double last_prune = 0.0;
  if (t - last_prune > 5.0) {
    last_prune = t;
    for (auto it = nero_pullaway_hists().begin(); it != nero_pullaway_hists().end();) {
      if (t - it->second.last > 10.0) it = nero_pullaway_hists().erase(it); else ++it;
    }
  }
  if (h.t_start >= 0.0) {
    // still speeding up? (a car that starts to creep and then keeps its crawl is judged as before again)
    bool speeding_up = true;
    if (t - h.t_start >= kNeroPullawayGainWindow) {
      double v_then = v;
      for (const auto & [ti, vi] : h.s) {
        if (t - ti >= kNeroPullawayGainWindow) v_then = vi;
      }
      speeding_up = v - v_then >= kNeroPullawayMinGain;
    }
    if (t - h.t_start <= kNeroPullawayActive && v >= kNeroPullawayVStill && speeding_up) return;
    h.t_start = -1.0;  // the window is over, it stopped again, or it no longer speeds up
    h.t_end = t;
    h.logged_unsafe = false;
  }
  if (v < kNeroPullawayVGo || h.s.size() < 3 || t - h.s.front().first < kNeroPullawayMinSeen) return;
  double v_min = v;
  for (const auto & [ti, vi] : h.s) {
    if (t - ti <= kNeroPullawayWindow && ti > h.t_end) v_min = std::min(v_min, vi);
  }
  if (v_min <= kNeroPullawayVStill && v - v_min >= kNeroPullawayDv) h.t_start = t;
}
// seconds since the object started from standstill, < 0 if it has not (or the model window is over)
double nero_pullaway_since(const autoware_perception_msgs::msg::PredictedObject & o)
{
  const auto it = nero_pullaway_hists().find(nero_uuid_key(o));
  if (it == nero_pullaway_hists().end() || it->second.t_start < 0.0) return -1.0;
  const double since = nero_steady_now() - it->second.t_start;
  return since <= kNeroPullawayActive ? since : -1.0;
}
// distance / speed of the model after t seconds from v0
double nero_pullaway_dist(const double v0, const double a, const double t)
{
  const double t1 = std::max(0.0, (kNeroPullawayVMax - v0) / a);
  if (t <= t1) return v0 * t + 0.5 * a * t * t;
  return v0 * t1 + 0.5 * a * t1 * t1 + std::max(v0, kNeroPullawayVMax) * (t - t1);
}
double nero_pullaway_speed(const double v0, const double a, const double t)
{
  return std::max(v0, std::min(kNeroPullawayVMax, v0 + a * t));
}
// the model's paths: along the centre line of every attention lane the object is in and heads along (its direction
// of motion within the lane-angle threshold), from where it is to the lane's end (the attention lanes end with the
// conflicting lanelet), timed from its current speed at <a>
std::vector<autoware_perception_msgs::msg::PredictedPath> nero_pullaway_paths(
  const autoware_perception_msgs::msg::PredictedObject & o, const lanelet::ConstLanelets & attention_lanelets,
  const double margin, const double angle_thr)
{
  std::vector<autoware_perception_msgs::msg::PredictedPath> out;
  const auto pose = autoware::behavior_velocity_planner::util::getObjectPoseWithVelocityDirection(o.kinematics);
  const double pose_yaw = tf2::getYaw(pose.orientation);
  const double v0 = std::hypot(
    o.kinematics.initial_twist_with_covariance.twist.linear.x,
    o.kinematics.initial_twist_with_covariance.twist.linear.y);
  const double a = nero_pullaway_acc();
  for (const auto & ll : attention_lanelets) {
    if (!lanelet::utils::isInLanelet(pose, ll, margin)) continue;
    const double ll_yaw = lanelet::utils::getLaneletAngle(ll, pose.position);
    if (std::fabs(autoware_utils::normalize_radian(ll_yaw - pose_yaw)) > angle_thr) continue;
    const auto centerline = ll.centerline2d();
    const double length = lanelet::geometry::length(centerline);
    const double s0 =
      lanelet::geometry::toArcCoordinates(centerline, lanelet::BasicPoint2d(pose.position.x, pose.position.y))
        .length;
    autoware_perception_msgs::msg::PredictedPath path;
    path.confidence = 1.0;
    path.time_step = rclcpp::Duration::from_seconds(kNeroPullawayDt);
    for (double t = 0.0; t <= kNeroPullawayHorizon + 1e-6; t += kNeroPullawayDt) {
      const double s = s0 + nero_pullaway_dist(v0, a, t);
      if (s > length) break;
      const auto p = lanelet::geometry::interpolatedPointAtDistance(centerline, s);
      const auto p_ahead = lanelet::geometry::interpolatedPointAtDistance(centerline, std::min(length, s + 0.5));
      const auto p_back = lanelet::geometry::interpolatedPointAtDistance(centerline, std::max(0.0, s - 0.5));
      geometry_msgs::msg::Pose q;
      q.position.x = p.x();
      q.position.y = p.y();
      q.position.z = pose.position.z;
      q.orientation = autoware_utils::create_quaternion_from_yaw(
        std::atan2(p_ahead.y() - p_back.y(), p_ahead.x() - p_back.x()));
      path.path.push_back(q);
    }
    if (path.path.size() >= 2) out.push_back(path);
  }
  return out;
}

}  // namespace
namespace autoware::behavior_velocity_planner
{
// steady time of the last creeper exemption (read by the pass-judge logic, NERO_IX_CREEP_COMMIT_M)
double & nero_creep_last_exempt_ref()
{
  static double t = -1e9;
  return t;
}
double nero_creep_steady_now()
{
  return nero_steady_now();
}
}  // namespace autoware::behavior_velocity_planner
namespace
{
bool nero_adjacent_fix()
{
  static const bool on = [] {
    const char * v = std::getenv("NERO_IX_ADJACENT_FIX");
    return v != nullptr && std::string(v) == "1";
  }();
  return on;
}

// NERO_IX_PRIORITY_FROM_SIGNS=1 (T junctions, 2026-10-23). The map has no right_of_way elements, so upstream makes
// every conflicting lanelet (+200 m upstream) an attention lane, and ego on the priority road stopped (hard, 0-5 m
// before the junction) for an oncoming car that might turn left into the minor road. The map already encodes the
// priority: minor-road approaches and turns off the priority road across oncoming traffic carry a stop line + stop
// sign, the priority road's straights carry none. For a straight assigned lanelet with no traffic light and no stop
// line / sign of its own, an attention object is treated as yielding (as upstream's safe_under_traffic_control) only
// while it is on the priority road's oncoming lanes that lead into such a signed turn (or straight on) or on a lane
// heading against ego that does not cross ego's centre line (narrow junctions where the opposing straights overlap),
// AND behaves, judged every cycle from its motion (Ahmed 10-23: priority never means ignoring a car): heading its
// lane's way, not turning off it (> 12 deg), before the stop line of its turn or driving on along the straight-on
// lane at >= 2 m/s. A car that could still take the turn (speed <= sqrt(6 m/s2 * radius)) but can no longer stop at
// its line with 3 m/s2 is a potential violator: its turn paths are checked as upstream until ego passes its pass
// judge line. Everything else (minor-road lanes, lanes without stop lines, turning / creeping / wrong-way cars) is
// checked exactly as upstream, and ego does not commit (permanent GO) while a yielding car is near.
// Unset = upstream / the car as before.
bool nero_priority_from_signs()
{
  static const bool on = [] {
    const char * v = std::getenv("NERO_IX_PRIORITY_FROM_SIGNS");
    return v != nullptr && std::string(v) == "1";
  }();
  return on;
}

// NERO_IX_PRIORITY_CAN_STOP=<horizon s> (2026-10-24, on top of the priority rule above). A car that has to yield to ego
// at its stop line is a potential violator (its turn paths are checked) unless it can still stop at that line, judged
// every cycle from its actual motion:
// - a car that keeps its speed (or speeds up): with the module's expected deceleration of a car
//   (ignore_on_amber_traffic_light.object_expected_deceleration.car: upstream's can_stop_before_stopline rule for an
//   amber light), now and, at >= 4 m/s, at the moment ego reaches its 1st pass judge line (at most <horizon> s ahead,
//   its speed carried on), so a car that will no longer be able to stop by the time ego commits is checked before
//   ego commits;
// - a car that is braking for the junction (speed slope over 0.4 s): with up to 3 m/s2, judged by the speed it will
//   have at its line (a fast car that brakes late can still take the turn).
// A car braking harder than that and still unable to stop at its line (it may be slowing down to turn in), or speeding
// up toward its line at >= 2 m/s2 (speed slope, or since it left standstill) so that, carrying on for <horizon> s, it
// could no longer stop at it with 3 m/s2 (a waiting car that pulls away), stays checked after ego's 1st pass judge
// line too.
// Unset = the rule above as before.
double nero_priority_can_stop()
{
  static const double h = [] {
    const char * v = std::getenv("NERO_IX_PRIORITY_CAN_STOP");
    return (v == nullptr || std::string(v).empty()) ? -1.0 : std::max(0.0, std::atof(v));
  }();
  return h;
}

// NERO_IX_PRIORITY_SWING=<deg> (2026-10-25, on top of NERO_IX_PRIORITY_CAN_STOP). Ahmed 10-24: "it should keep driving
// and react when the vehicle slows down and start to swing our way", "if it only started to break without turning it
// shouldnt yield". Braking alone no longer makes a car that has to yield to ego a potential violator: a car seen
// braking is yielding while it keeps heading straight, and a potential violator (its turn paths checked, also after
// ego's 1st pass judge line: a late hard brake for a car that is turning in) once it swings toward its turn by <deg>:
// its heading over the last 0.25 s against its heading 1.0-0.6 s ago (the window-mean pattern of Apollo's lane
// sequence evaluator), on at <deg>, off below half of it. A car not seen braking is judged as above (a car keeping
// turning speed toward its turn is still a potential violator from first sight); a swing makes it one too. Until a
// car has 1 s of heading history only its heading against its lane counts (the turning rule above, > 12 deg: watched).
// Unset = the rules above as before.
double nero_priority_swing()
{
  static const double d = [] {
    const char * v = std::getenv("NERO_IX_PRIORITY_SWING");
    return (v == nullptr || std::string(v).empty()) ? -1.0 : std::max(0.0, std::atof(v));
  }();
  return d;
}

// NERO_IX_PRIORITY_NO_EXIT=1 (2026-10-30, on top of NERO_IX_PRIORITY_CAN_STOP). The rule above takes a car faster than
// a turn (v_turn_max) for one that does not take it: it goes straight on. At a T mouth of a stop-sign road whose every
// way on is a turn (e23: 175 such approaches) a car faster than every one of them cannot go anywhere but straight across
// ego's road (10-29 15:55 #3: 9.4 m/s out of 42755, never slowing, judged yielding until past its line, 0.51 m ahead of
// the robot). Such a car (not braking, not swinging) is judged by the same can-stop test as any car that could still
// take a turn; one that cannot stop at its line is a potential violator, checked also after ego's 1st pass judge line,
// and for 2 s its constant-velocity straight path is checked besides its lane paths. A car with a straight on (the
// priority road's oncoming lane) keeps its verdict. Unset = the rules above as before.
double nero_priority_no_exit()
{
  static const double on = [] {
    const char * v = std::getenv("NERO_IX_PRIORITY_NO_EXIT");
    return (v == nullptr || std::string(v).empty()) ? -1.0 : std::atof(v);
  }();
  return on;
}

// NERO_IX_PRIORITY_EVIDENCE=<pos_s>,<move_m>[,<right_turn>[,<evidence_only>]] (realize-2, 2026-11-03, QUEUE PREDLEFT; on top of
// NERO_IX_PRIORITY_CAN_STOP and NERO_IX_PRIORITY_SWING). Ahmed 11-03 #80: "remove anticipating going left from other cars".
// The priority rule exempts a car that has to yield to ego while it behaves, but CAN_STOP still anticipated a turn from a
// car that merely could not stop at its turn's line (11-03 #97: a car at a steady 4.1 m/s held the robot at its line and
// went straight; #90: two braking cars held the robot's right turn 11 s), and the "turning off its lane" test read a
// fresh track's tracker heading (born 45 deg off its motion) as a turn (#97 first stop). Like Apollo's protected mode
// (prediction/predictor/predictor.cc: with ego's right of way an obstacle not yet in the junction is not followed into
// it), with this switch:
// - the direction of motion of a car (the rule's "heading its lane's way" and "turning off it" tests) comes from its own
//   positions once it has >= <pos_s> s of them over which it moved >= <move_m> m (NO_EXIT's band uses the same), else
//   from the tracker's heading as before;
// - a hand-over (UNKNOWN <-> vehicle) or a position jump restarts a car's speed / heading / position record, a heading flip
//   (> 90 deg) restarts its heading record, a swing of the tracker's heading counts only when its positions also point
//   toward the turn (>= 6 deg against the turn's start), and on a priority approach / its straight on only a turn toward
//   the side of its turns across ego counts as "turning" (none before it has a position record);
// - <evidence_only> = 1 (absent = 1; Ahmed's #80, DECISIONS PREDLEFT-ANT): on an approach whose turn off the priority road
//   has a straight on (< 20 deg, no stop line: the car may legally drive on), a car before its line is a potential violator
//   only when it swings toward the turn (SWING, checked also after ego's pass judge line) or pulls away from a stand
//   toward its line (CAN_STOP's launch clause, the stand seen in its positions); "cannot stop at its line" alone no longer
//   anticipates the turn. 0 = CAN_STOP's anticipation as before. A car with no straight on (a T mouth: it must stop either
//   way) keeps CAN_STOP as before; past its line, turning, creeping or wrong-way it is watched as upstream as before;
// - <right_turn> = 1: the rule also covers ego's right turn without traffic light and without its own stop line / stop
//   sign (a right turn off the priority road; the oncoming left turn into the same exit carries the stop line), the
//   "heading against ego" (PARALLEL) test taken against the right turn's start.
// Unset = the rules above as before.
struct NeroEvidenceParam
{
  bool on{false};
  double pos_s{0.5};
  double move_m{1.0};
  bool right_turn{false};
  bool evidence_only{true};
};
const NeroEvidenceParam & nero_priority_evidence()
{
  static const NeroEvidenceParam p = [] {
    NeroEvidenceParam q;
    const char * v = std::getenv("NERO_IX_PRIORITY_EVIDENCE");
    if (v == nullptr || std::string(v).empty()) return q;
    std::vector<double> f;
    std::stringstream ss(v);
    std::string item;
    while (std::getline(ss, item, ',')) f.push_back(std::atof(item.c_str()));
    if (f.size() < 2 || f.at(0) <= 0.0 || f.at(1) <= 0.0) return q;
    q.on = true;
    q.pos_s = f.at(0);
    q.move_m = f.at(1);
    q.right_turn = f.size() > 2 && f.at(2) > 0.5;
    q.evidence_only = f.size() <= 3 || f.at(3) > 0.5;
    return q;
  }();
  return p;
}

// NERO_IX_PRIORITY_TURN=<win_s>,<dev_deg>,<lat_m>[,<n_cycles>[,<hold_s>[,<gate_tol_deg>[,<gate_s>]]]] (predleft, 2026-11-05; QUEUE PREDLEFT, on top of
// NERO_IX_PRIORITY_EVIDENCE + NERO_IX_PRIORITY_SWING). Ahmed 11-04 (PREDLEFT-ANT (a)): "it should turn and start moving in
// its direction to slow down for it". A car on the priority road's oncoming approach (or already on its turn, before its
// line) SHOWS its turn when its own positions over the last <win_s> s move at least <dev_deg> off its lane's direction
// toward the side of a turn across ego AND at least <lat_m> sideways that way, on <n_cycles> consecutive cycles (the
// tracker's heading lags a real turn by ~0.5 s: SIM.md 19; the positions lead it). From then (held <hold_s> s after the
// last cycle that showed it) it is a potential violator, but only once it is SETTLED: for <gate_s> s before the last <win_s>
// its own heading stayed within <gate_tol_deg> of its lane (a fresh track, a hand-over or a car still turning into the road
// fakes a turn: the census's false triggers; with the gate 0 of 33 straight passes fired): its turn paths are checked, also after ego's 1st pass judge
// line (as a SWING). With NERO_IX_PRIORITY_EVIDENCE's <evidence_only>, a car past the stop line of such a turn whose own
// positions still run along the turn's start (< <dev_deg>) and that has not shown its turn is driving on straight (no
// anticipation there either: it is not "committed" to the turn). Thresholds from the left-turner census of our own
// recordings (predleft README). Unset = off.
struct NeroTurnParam
{
  bool on{false};
  double win_s{0.5};
  double dev_deg{6.0};
  double lat_m{0.25};
  int n_cycles{2};
  double hold_s{1.5};
  double gate_tol_deg{4.0};  // settled: its heading over <win_s> stayed within this of its lane ...
  double gate_s{1.5};        // ... for this long before the last <win_s> (0 = no gate)
};
const NeroTurnParam & nero_priority_turn()
{
  static const NeroTurnParam p = [] {
    NeroTurnParam q;
    const char * v = std::getenv("NERO_IX_PRIORITY_TURN");
    if (v == nullptr || std::string(v).empty()) return q;
    std::vector<double> f;
    std::stringstream ss(v);
    std::string item;
    while (std::getline(ss, item, ',')) f.push_back(std::atof(item.c_str()));
    if (f.size() < 3 || f.at(0) <= 0.0 || f.at(1) <= 0.0 || f.at(2) < 0.0) return q;
    q.on = true;
    q.win_s = f.at(0);
    q.dev_deg = f.at(1);
    q.lat_m = f.at(2);
    if (f.size() > 3 && f.at(3) >= 1.0) q.n_cycles = static_cast<int>(f.at(3) + 0.5);
    if (f.size() > 4 && f.at(4) >= 0.0) q.hold_s = f.at(4);
    if (f.size() > 5 && f.at(5) > 0.0) q.gate_tol_deg = f.at(5);
    if (f.size() > 6 && f.at(6) >= 0.0) q.gate_s = f.at(6);
    return q;
  }();
  return p;
}

// NERO_IX_PRIORITY_TURN_EARLY=<win_s>,<dev_deg>,<move_m>,<near_m>[,<n_cycles>[,<census_gate>]] (ixturnev, 2026-11-06; QUEUE IXTURNEV, on
// top of NERO_IX_PRIORITY_TURN). Ahmed 11-05: "be dynamic if he starts going in our lane then start braking". TURN reads a car's
// turn from the heading of its own positions over 1.0 s; over a shorter window the same positions show it sooner (the census of
// our left turners, replayed through the car's tracker and the turn-onset tracker: 0.10-0.30 s sooner on every at-speed turner),
// but a short window also reads a straight car's lane drift or a slow car's position jitter as a heading. So TURN also shows a
// car's turn when its positions over the last <win_s> s moved at least <move_m> m (a car at speed) and head at least <dev_deg>
// off its lane toward the side of a turn across ego (and at least TURN's <lat_m> sideways), on <n_cycles> consecutive objects
// messages, while it is within <near_m> m of that turn's stop line or past it (a car turning across ego turns there; lane drifts
// further back are not turns), and it is SETTLED by TURN's own gate. Its hold, side, reference direction and effect are TURN's.
// <census_gate> (absent = 1): TURN's settled gate (its 1-s heading within <gate_tol_deg> over <gate_s>) counts the values it
// can compute (>= 5), as the left-turner census measured it (0 of 33 straight passes), instead of failing on the first second
// of a record (no 1-s heading yet): the car's gate needs ~3.5 s of a car's positions, the census's ~2.4 s, so a car first
// published within ~40 m showed its turn up to ~0.7 s later than the census allows; applies to TURN's own 1-s reading too.
// Census (47 straight passes, 7 oncoming, both tracker families): 0 straight passes more than TURN, largest straight-pass value
// 4.8 deg against 6 (TURN's own 5.4). Logs `[NERO_IX] priority turn early: ...` on the cycle it makes a car's turn show while
// TURN alone would not (no such line = the decisions of NERO_IX_PRIORITY_TURN alone). Needs NERO_IX_PRIORITY_TURN. Unset = off.
struct NeroTurnEarlyParam
{
  bool on{false};
  double win_s{0.7};
  double dev_deg{6.0};
  double move_m{2.0};
  double near_m{15.0};
  int n_cycles{2};
  bool census_gate{true};  // TURN's settled gate counts the computable values (>= 5), as the census measured it
};
const NeroTurnEarlyParam & nero_priority_turn_early()
{
  static const NeroTurnEarlyParam p = [] {
    NeroTurnEarlyParam q;
    const char * v = std::getenv("NERO_IX_PRIORITY_TURN_EARLY");
    if (v == nullptr || std::string(v).empty()) return q;
    std::vector<double> f;
    std::stringstream ss(v);
    std::string item;
    while (std::getline(ss, item, ',')) f.push_back(std::atof(item.c_str()));
    if (f.size() < 4 || f.at(0) <= 0.0 || f.at(1) <= 0.0 || f.at(2) < 1.0 || f.at(3) < 0.0) return q;
    q.on = true;
    q.win_s = f.at(0);
    q.dev_deg = f.at(1);
    q.move_m = f.at(2);
    q.near_m = f.at(3);
    if (f.size() > 4 && f.at(4) >= 1.0) q.n_cycles = static_cast<int>(f.at(4) + 0.5);
    if (f.size() > 5) q.census_gate = f.at(5) > 0.5;
    return q;
  }();
  return p;
}

// NERO_IX_PRIORITY_READY=<evidence_s>,<react_s>,<a_stop>,<b_max>[,<horizon_s>[,<away_m>[,<acc_release>[,<gap_m>[,<v_floor>]]]]] (predleft, 2026-11-05; QUEUE PREDLEFT, on
// top of NERO_IX_PRIORITY_EVIDENCE's no-anticipation). Ahmed 10-23: "in saudi you get the occasional crazy driver that
// doesnt respect the right of way ... we are still ready for him". Without anticipation a car that keeps its speed and
// turns across ego exactly as ego arrives shows its turn too late for any stop (the turn shows ~0.5-0.8 s after it starts,
// its body is in ego's path ~1 s after it starts). So while such a car has shown no turn but could still take its turn
// across ego without braking (before its line, not faster than the turn allows), ego keeps itself READY for him: if its
// own plan would put it, at the moment he could start that turn + <evidence_s> (until the turn shows) + <react_s> (until
// ego brakes), too close to the junction to stop at the default stop line with <a_stop> m/s2, and not early enough to
// clear his way before he could reach it, ego eases off with the gentlest deceleration that restores it (at most <b_max>
// m/s2, a speed cap ahead of ego: never a stop, never a hard brake for a car that has not turned). He shows no turn ->
// ego drives on; he turns -> NERO_IX_PRIORITY_TURN / SWING make him a potential violator and the module stops ego in time.
// Cars more than <horizon_s> s from their turn are not considered; a car is let go when it shows it goes on straight:
// past the start of its turn by more than it would need to show the turn, drifting at least <away_m> m away from the
// turn's side over the last ~1.5 s, or speeding up by at least <acc_release> m/s2. <gap_m> > 0: "ready" = able to stop
// <gap_m> short of the stretch of ego's lane his body would sweep (if that is beyond the stop line), 0 = at the default
// stop line. <v_floor>: the cap never asks for less than this speed [m/s] (default 1.0). Unset = off.
struct NeroReadyParam
{
  bool on{false};
  double evidence_s{0.8};
  double react_s{0.5};
  double a_stop{3.0};
  double b_max{1.0};
  double horizon_s{8.0};
  double away_m{0.3};       // released when it drifts this far away from its turn side over the last ~1.5 s (0 = off)
  double acc_release{0.5};  // released when it speeds up at least this much [m/s2] (0 = off)
  double gap_m{0.0};        // ego may plan its stop up to this short of his swept band instead of the stop line (0 = the line)
  double v_floor{1.0};      // [m/s] the cap never asks for less
};
const NeroReadyParam & nero_priority_ready()
{
  static const NeroReadyParam p = [] {
    NeroReadyParam q;
    const char * v = std::getenv("NERO_IX_PRIORITY_READY");
    if (v == nullptr || std::string(v).empty()) return q;
    std::vector<double> f;
    std::stringstream ss(v);
    std::string item;
    while (std::getline(ss, item, ',')) f.push_back(std::atof(item.c_str()));
    if (f.size() < 4 || f.at(0) < 0.0 || f.at(1) < 0.0 || f.at(2) <= 0.0 || f.at(3) <= 0.0) return q;
    q.on = true;
    q.evidence_s = f.at(0);
    q.react_s = f.at(1);
    q.a_stop = f.at(2);
    q.b_max = f.at(3);
    if (f.size() > 4 && f.at(4) > 0.0) q.horizon_s = f.at(4);
    if (f.size() > 5 && f.at(5) >= 0.0) q.away_m = f.at(5);
    if (f.size() > 6 && f.at(6) >= 0.0) q.acc_release = f.at(6);
    if (f.size() > 7 && f.at(7) >= 0.0) q.gap_m = f.at(7);
    if (f.size() > 8 && f.at(8) > 0.0) q.v_floor = f.at(8);
    return q;
  }();
  return p;
}

// the first 4 bytes of an object's uuid in hex (log lines)
std::string nero_hex_id(const autoware_perception_msgs::msg::PredictedObject & o)
{
  return fmt::format(
    "{:02x}{:02x}{:02x}{:02x}", o.object_id.uuid[0], o.object_id.uuid[1], o.object_id.uuid[2], o.object_id.uuid[3]);
}

// NERO_IX_PRIORITY_DEBUG=1: log only, one line per cycle and car with a priority verdict (numbers of the rule above)
bool nero_priority_debug()
{
  static const bool on = [] {
    const char * v = std::getenv("NERO_IX_PRIORITY_DEBUG");
    return v != nullptr && std::string(v) == "1";
  }();
  return on;
}

// stop lines of a lanelet: road_marking -> refers (type stop_line), traffic_sign (stop_sign) -> ref_line
std::vector<lanelet::ConstLineString3d> nero_stop_lines(const lanelet::ConstLanelet & ll)
{
  std::vector<lanelet::ConstLineString3d> out;
  for (const auto & reg : ll.regulatoryElements()) {
    const std::string subtype = reg->attributeOr(lanelet::AttributeName::Subtype, "none");
    if (subtype == "road_marking") {
      for (const auto & ls : reg->getParameters<lanelet::ConstLineString3d>("refers")) {
        if (std::string(ls.attributeOr(lanelet::AttributeName::Type, "none")) == "stop_line") out.push_back(ls);
      }
    } else if (subtype == "traffic_sign") {
      bool stop_sign = false;
      for (const auto & ls : reg->getParameters<lanelet::ConstLineString3d>("refers")) {
        if (std::string(ls.attributeOr(lanelet::AttributeName::Subtype, "none")) == "stop_sign") stop_sign = true;
      }
      if (!stop_sign) continue;
      for (const auto & ls : reg->getParameters<lanelet::ConstLineString3d>("ref_line")) {
        if (std::string(ls.attributeOr(lanelet::AttributeName::Type, "none")) == "stop_line") out.push_back(ls);
      }
    }
  }
  return out;
}

double nero_lanelet_yaw(const lanelet::ConstLanelet & ll)
{
  const auto c = ll.centerline2d();
  return std::atan2(c.back().y() - c.front().y(), c.back().x() - c.front().x());
}
}  // namespace

namespace autoware::behavior_velocity_planner
{
namespace bg = boost::geometry;

// NERO_IX_UNKNOWN_FLAT=<h>[,<v>] (ixghost job, 2026-10-26): an UNKNOWN whose reported height has stayed below <h> m
// since its track appeared, and that reports less than <v> m/s now, is not a collision target of this module
// (obstacle_stop still brakes for it on the path). The field's UNKNOWN polygons carry the robot's own yaw (tracker
// convex hulls), so a flat ground return (a speed bump, broken surface: 0.03-0.07 m) inside a turn lane that merges into
// / diverges from ego's lane passes the attention lane's heading test; standing (or in its track's first cycle) in ego's
// lane, or drifting into it, its passage covered ego's whole window and one cycle became a CollisionStop held 1.5 s on
// the priority road (10-26 #5 / #6 / #14). No vehicle is reported that flat (field: 97k vehicle-class samples up to
// 60 m, none below 0.5 m); a track that was ever >= <h> stays a target for good, and a flat one reporting >= <v> m/s
// counts in that cycle (a fast real car whose first cycles are reported flat: no later reaction; field ghosts report
// 0.1 m/s median, >= 3.5 m/s in 0.3 % of their samples). <v> omitted = any speed. Unset = the car.
struct NeroUnknownFlatParam
{
  double h{0.0};
  double v{std::numeric_limits<double>::infinity()};
};
const NeroUnknownFlatParam & nero_unknown_flat()
{
  static const NeroUnknownFlatParam p = [] {
    NeroUnknownFlatParam r;
    if (const char * e = std::getenv("NERO_IX_UNKNOWN_FLAT")) {
      double h = 0.0, v = 0.0;
      const int n = std::sscanf(e, "%lf,%lf", &h, &v);
      r.h = n >= 1 ? std::max(0.0, h) : 0.0;
      if (n >= 2 && v > 0.0) r.v = v;
    }
    return r;
  }();
  return p;
}
struct NeroUnknownFlatSeen
{
  double max_h{0.0};
  double last{0.0};
  std::vector<int64_t> logged_lanes{};
};
// the object's track (keyed by uuid, forgotten 2 s after it was last seen): its tallest reported height so far
NeroUnknownFlatSeen & nero_unknown_flat_seen(const autoware_perception_msgs::msg::PredictedObject & o)
{
  static std::unordered_map<std::string, NeroUnknownFlatSeen> seen;
  static double last_prune = 0.0;
  const double t =
    std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  if (t - last_prune > 1.0) {
    last_prune = t;
    for (auto it = seen.begin(); it != seen.end();) {
      if (t - it->second.last > 2.0) it = seen.erase(it); else ++it;
    }
  }
  auto & s = seen[uuidKey(o.object_id)];
  s.max_h = std::max(s.max_h, static_cast<double>(o.shape.dimensions.z));
  s.last = t;
  return s;
}

bool IntersectionModule::isTargetCollisionVehicleType(
  const autoware_perception_msgs::msg::PredictedObject & object) const
{
  const auto label = object.classification.at(0).label;
  const auto & p = planner_param_.collision_detection.target_type;

  if (label == autoware_perception_msgs::msg::ObjectClassification::CAR && p.car) {
    return true;
  }
  if (label == autoware_perception_msgs::msg::ObjectClassification::BUS && p.bus) {
    return true;
  }
  if (label == autoware_perception_msgs::msg::ObjectClassification::TRUCK && p.truck) {
    return true;
  }
  if (label == autoware_perception_msgs::msg::ObjectClassification::TRAILER && p.trailer) {
    return true;
  }
  if (label == autoware_perception_msgs::msg::ObjectClassification::MOTORCYCLE && p.motorcycle) {
    return true;
  }
  if (label == autoware_perception_msgs::msg::ObjectClassification::BICYCLE && p.bicycle) {
    return true;
  }
  if (label == autoware_perception_msgs::msg::ObjectClassification::UNKNOWN && p.unknown) {
    const auto & flat = nero_unknown_flat();
    if (flat.h > 0.0) {  // NERO_IX_UNKNOWN_FLAT: a flat, slow UNKNOWN is left to obstacle_stop
      auto & seen = nero_unknown_flat_seen(object);
      const auto & tw = object.kinematics.initial_twist_with_covariance.twist.linear;
      if (seen.max_h < flat.h && std::hypot(tw.x, tw.y) < flat.v) {
        if (
          std::find(seen.logged_lanes.begin(), seen.logged_lanes.end(), lane_id_) ==
          seen.logged_lanes.end()) {
          seen.logged_lanes.push_back(lane_id_);
          const auto & pos = object.kinematics.initial_pose_with_covariance.pose.position;
          RCLCPP_INFO(
            logger_,
            "[NERO_IX] unknown_flat: lane %ld, UNKNOWN %02x%02x at (%.1f, %.1f) %.2f x %.2f x %.2f m, v %.2f: not a "
            "collision target (never >= %.2f m tall, < %.1f m/s); obstacle_stop keeps it",
            lane_id_, object.object_id.uuid[0], object.object_id.uuid[1], pos.x, pos.y,
            object.shape.dimensions.x, object.shape.dimensions.y, object.shape.dimensions.z,
            std::hypot(tw.x, tw.y), flat.h, flat.v);
        }
        return false;
      }
    }
    // Nero c3 (unknown_filter): with the filter on, an UNKNOWN is a target only if it is big or
    // has been moving fast for a while (small/static UNKNOWN blobs such as curbs, bushes or ghost
    // points no longer make ego yield). Filter off = stock behaviour.
    if (planner_param_.collision_detection.unknown_filter.enable) {
      return isUnknownObjectCollisionTarget(object);
    }
    return true;
  }
  return false;
}

// ==========================================================================================
// Nero c3 (unknown_filter)
// ==========================================================================================
namespace
{
double footprintArea(const autoware_perception_msgs::msg::Shape & shape)
{
  using autoware_perception_msgs::msg::Shape;
  if (shape.type == Shape::BOUNDING_BOX) {
    return shape.dimensions.x * shape.dimensions.y;
  }
  if (shape.type == Shape::CYLINDER) {
    const double r = shape.dimensions.x / 2.0;  // dimensions.x is the diameter
    return M_PI * r * r;
  }
  // POLYGON: shoelace formula on the footprint
  const auto & pts = shape.footprint.points;
  double twice_area = 0.0;
  for (size_t i = 0; i < pts.size(); ++i) {
    const auto & a = pts.at(i);
    const auto & b = pts.at((i + 1) % pts.size());
    twice_area += static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
  }
  return std::fabs(twice_area) / 2.0;
}

double objectSpeed(const autoware_perception_msgs::msg::PredictedObject & object)
{
  const auto & v = object.kinematics.initial_twist_with_covariance.twist.linear;
  return std::hypot(v.x, v.y);
}
}  // namespace

void IntersectionModule::updateUnknownSpeedHistory()
{
  const auto & uf = planner_param_.collision_detection.unknown_filter;
  if (!uf.enable || !planner_param_.collision_detection.target_type.unknown) {
    unknown_speed_history_.clear();
    return;
  }
  const auto now = clock_->now();
  for (const auto & object : planner_data_->predicted_objects->objects) {
    if (
      object.classification.empty() || object.classification.at(0).label !=
                                          autoware_perception_msgs::msg::ObjectClassification::UNKNOWN) {
      continue;
    }
    auto & history = unknown_speed_history_[uuidKey(object.object_id)];
    history.last_seen = now;
    if (objectSpeed(object) >= uf.min_speed) {
      if (!history.fast_since) {
        history.fast_since = now;  // start of a new "fast" streak
      }
    } else {
      history.fast_since = std::nullopt;  // streak broken
    }
  }
  // forget objects that have not been seen for a while (track lost / uuid changed)
  static constexpr double forget_after_sec = 1.0;
  for (auto it = unknown_speed_history_.begin(); it != unknown_speed_history_.end();) {
    if ((now - it->second.last_seen).seconds() > forget_after_sec) {
      it = unknown_speed_history_.erase(it);
    } else {
      ++it;
    }
  }
}

bool IntersectionModule::isUnknownObjectCollisionTarget(
  const autoware_perception_msgs::msg::PredictedObject & object) const
{
  const auto & uf = planner_param_.collision_detection.unknown_filter;
  // (a) big enough footprint
  if (footprintArea(object.shape) >= uf.min_area) {
    return true;
  }
  // (b) speed >= min_speed held continuously for >= min_duration
  const auto it = unknown_speed_history_.find(uuidKey(object.object_id));
  if (it == unknown_speed_history_.end() || !it->second.fast_since) {
    return false;
  }
  return (clock_->now() - it->second.fast_since.value()).seconds() >= uf.min_duration;
}

// ==========================================================================================
// Nero c3 (wrong_way_watch)
// ==========================================================================================
bool IntersectionModule::isWrongWayOnLanelets(
  const autoware_perception_msgs::msg::PredictedObject & object,
  const lanelet::ConstLanelets & ego_lanelets) const
{
  const auto & ww = planner_param_.collision_detection.wrong_way_watch;
  // slow objects are excluded: the tracker often flips the heading of slow/stopped vehicles, and
  // a flipped lead car must not look like an oncoming wrong-way car
  if (objectSpeed(object) < ww.min_speed) {  // planar speed (UNKNOWN hulls move in vy)
    return false;
  }
  // heading of motion (flipped if the object reports negative longitudinal velocity)
  const auto pose = util::getObjectPoseWithVelocityDirection(object.kinematics);
  const double pose_angle = tf2::getYaw(pose.orientation);
  for (const auto & ll : ego_lanelets) {
    // object centre must be inside the lanelet itself (no margin): an oncoming car on the
    // neighbouring (opposite) lane is not "on ego's lane"
    if (!lanelet::utils::isInLanelet(pose, ll, 0.0)) {
      continue;
    }
    const double ll_angle = lanelet::utils::getLaneletAngle(ll, pose.position);
    const double angle_diff = autoware_utils::normalize_radian(ll_angle - pose_angle, -M_PI);
    if (std::fabs(angle_diff) > M_PI / 2.0) {
      return true;
    }
  }
  return false;
}

// ==========================================================================================
// NERO_IX_PRIORITY_FROM_SIGNS
// ==========================================================================================
void IntersectionModule::neroBuildPriorityLanes()
{
  using Kind = NeroPriorityLane::Kind;
  nero_priority_lanes_ = std::vector<NeroPriorityLane>{};
  auto & lanes = nero_priority_lanes_.value();
  const auto lanelet_map_ptr = planner_data_->route_handler_->getLaneletMapPtr();
  const auto routing_graph_ptr = planner_data_->route_handler_->getRoutingGraphPtr();
  const auto assigned_lanelet = lanelet_map_ptr->laneletLayer.get(lane_id_);
  std::string why;
  // NERO_IX_PRIORITY_EVIDENCE <right_turn>: also ego's right turn off the priority road
  const bool ev_right_turn = nero_priority_evidence().on && nero_priority_evidence().right_turn &&
                             turn_direction_ == "right";
  if (turn_direction_ != "straight" && !ev_right_turn) {
    why = "turn_direction " + turn_direction_;
  } else if (has_traffic_light_) {
    why = "traffic light";
  } else if (!nero_stop_lines(assigned_lanelet).empty()) {
    why = "own stop line / stop sign";
  }
  if (!why.empty()) {
    RCLCPP_INFO(logger_, "[NERO_IX] priority_from_signs: lane %ld not in scope (%s)", lane_id_, why.c_str());
    return;
  }

  // ego's own lanes as in generateObjectiveLanelets()
  lanelet::ConstLanelets ego_lanelets{};
  for (const auto & previous_lanelet : routing_graph_ptr->previous(assigned_lanelet)) {
    ego_lanelets.push_back(previous_lanelet);
    for (const auto & following_lanelet : routing_graph_ptr->following(previous_lanelet)) {
      if (!lanelet::utils::contains(ego_lanelets, following_lanelet)) {
        ego_lanelets.push_back(following_lanelet);
      }
    }
  }
  const auto conflicting_lanelets =
    lanelet::utils::getConflictingLanelets(routing_graph_ptr, assigned_lanelet);
  const auto & detection_lanelets = intersection_lanelets_.value().attention_non_preceding();

  std::map<lanelet::Id, size_t> index;
  auto add = [&](const lanelet::ConstLanelet & ll, const Kind kind, const double stop_s) {
    if (index.count(ll.id()) > 0) return;
    index[ll.id()] = lanes.size();
    lanes.push_back(NeroPriorityLane{ll, kind, stop_s, 0.0, {}});
  };
  double ego_yaw = nero_lanelet_yaw(assigned_lanelet);
  const auto ego_centerline = assigned_lanelet.centerline2d();
  if (ev_right_turn && ego_centerline.size() >= 2) {
    // NERO_IX_PRIORITY_EVIDENCE: a lane "heading against ego" is judged against the direction ego comes from
    ego_yaw = std::atan2(
      ego_centerline[1].y() - ego_centerline[0].y(), ego_centerline[1].x() - ego_centerline[0].x());
  }
  for (const auto & ll : detection_lanelets) {
    if (!lanelet::utils::contains(conflicting_lanelets, ll)) {
      add(ll, Kind::NORMAL, 0.0);  // wrong-direction neighbours (consider_wrong_direction_vehicle)
      continue;
    }
    const auto stop_lines = nero_stop_lines(ll);
    if (!stop_lines.empty()) {
      double stop_s = std::numeric_limits<double>::max();
      for (const auto & stop_line : stop_lines) {
        geometry_msgs::msg::Pose mid;
        mid.position.x = (stop_line.front().x() + stop_line.back().x()) / 2.0;
        mid.position.y = (stop_line.front().y() + stop_line.back().y()) / 2.0;
        stop_s = std::min(stop_s, lanelet::utils::getArcCoordinates({ll}, mid).length);
      }
      add(ll, Kind::YIELD_MINOR, stop_s);  // YIELD_PRIORITY_ROAD is decided below from its approaches
      continue;
    }
    const double dyaw = std::fabs(autoware_utils::normalize_radian(nero_lanelet_yaw(ll) - ego_yaw));
    const bool crosses_ego =
      bg::intersects(ll.centerline2d().basicLineString(), ego_centerline.basicLineString());
    add(ll, (dyaw > 150.0 / 180.0 * M_PI && !crosses_ego) ? Kind::PARALLEL : Kind::NORMAL, 0.0);
  }
  // upstream lanes exactly as the module extends the attention area
  for (const auto & ll : detection_lanelets) {
    for (const auto & sequence : lanelet::utils::query::getPrecedingLaneletSequences(
           routing_graph_ptr, ll, planner_param_.common.attention_area_length, ego_lanelets)) {
      for (const auto & l : sequence) add(l, Kind::APPROACH_OTHER, 0.0);
    }
  }
  // a turn with a stop line is "off the priority road" if every lane entering it also carries traffic that goes on
  // through the junction without crossing ego (a successor outside the attention area, or a PARALLEL lane): the
  // oncoming lane of the priority road, whose left-turners must stop at the line. Otherwise it is a minor-road turn.
  auto goes_on_at_junction = [&](const lanelet::ConstLanelet & entry) {
    for (const auto & next : routing_graph_ptr->following(entry)) {
      if (next.id() == assigned_lanelet.id() || lanelet::utils::contains(ego_lanelets, next)) continue;
      const auto it = index.find(next.id());
      if (it == index.end() || lanes.at(it->second).kind == Kind::PARALLEL) return true;
    }
    return false;
  };
  for (auto & lane : lanes) {
    if (lane.kind != Kind::YIELD_MINOR) continue;
    const auto entries = routing_graph_ptr->previous(lane.lanelet);
    if (!entries.empty() && std::all_of(entries.begin(), entries.end(), goes_on_at_junction)) {
      lane.kind = Kind::YIELD_PRIORITY_ROAD;
    }
  }
  // an upstream lane is a priority-road approach if no path from it enters a lane that is watched (a lane without
  // stop line, a minor-road turn, ego's own lanes) before it leaves the attention area or reaches the stop line of a
  // turn off the priority road / a PARALLEL lane
  std::map<lanelet::Id, bool> rated;
  std::function<bool(const lanelet::ConstLanelet &, int)> exempt_from = [&](const lanelet::ConstLanelet & ll,
                                                                           const int depth) {
    if (const auto it = rated.find(ll.id()); it != rated.end()) return it->second;
    rated[ll.id()] = false;  // a cycle counts as "watched"
    bool exempt = depth < 64;
    for (const auto & next : routing_graph_ptr->following(ll)) {
      if (next.id() == assigned_lanelet.id() || lanelet::utils::contains(ego_lanelets, next)) {
        exempt = false;  // feeds ego's own lanes
        continue;
      }
      const auto it = index.find(next.id());
      if (it == index.end()) continue;  // leaves the attention area
      const auto kind = lanes.at(it->second).kind;
      if (kind == Kind::APPROACH_OTHER) {
        exempt = exempt && exempt_from(next, depth + 1);
      } else if (kind != Kind::PARALLEL && kind != Kind::YIELD_PRIORITY_ROAD) {
        exempt = false;
      }
    }
    rated[ll.id()] = exempt;
    return exempt;
  };
  for (auto & lane : lanes) {
    if (lane.kind == Kind::APPROACH_OTHER && exempt_from(lane.lanelet, 0)) lane.kind = Kind::APPROACH_PRIORITY;
  }
  // the straight-on lanes next to a turn off the priority road (a car inside one is still going straight)
  const size_t n_classified = lanes.size();
  for (size_t i = 0; i < n_classified; ++i) {
    if (lanes.at(i).kind != Kind::YIELD_PRIORITY_ROAD) continue;
    for (const auto & entry : routing_graph_ptr->previous(lanes.at(i).lanelet)) {
      for (const auto & sibling : routing_graph_ptr->following(entry)) {
        if (sibling.id() != assigned_lanelet.id() && !lanelet::utils::contains(ego_lanelets, sibling)) {
          add(sibling, Kind::SIBLING, 0.0);
          const auto sit = index.find(sibling.id());  // NERO_IX_PRIORITY_EVIDENCE: the turns it is the straight on of
          if (sit != index.end() && lanes.at(sit->second).kind == Kind::SIBLING) {
            nero_ev_sib_turns_[sit->second].push_back(i);
          }
        }
      }
    }
  }
  // a turn off the priority road can be taken at most at sqrt(a_lat * R): 6 m/s2 sideways is beyond aggressive
  // driving; a car faster than that cannot turn without braking first (re-judged every cycle)
  constexpr double kTurnLatAcc = 6.0;
  for (auto & lane : lanes) {
    if (lane.kind != Kind::YIELD_PRIORITY_ROAD) continue;
    const auto c = lane.lanelet.centerline2d();
    if (c.size() < 3) continue;
    const double h0 = std::atan2(c[1].y() - c[0].y(), c[1].x() - c[0].x());
    const double h1 = std::atan2(
      c[c.size() - 1].y() - c[c.size() - 2].y(), c[c.size() - 1].x() - c[c.size() - 2].x());
    const double dtheta = std::max(0.1, std::fabs(autoware_utils::normalize_radian(h1 - h0)));
    lane.v_turn_max = std::sqrt(kTurnLatAcc * lanelet::utils::getLaneletLength2d(lane.lanelet) / dtheta);
    lane.turn_sign = autoware_utils::normalize_radian(h1 - h0) >= 0.0 ? 1.0 : -1.0;  // NERO_IX_PRIORITY_SWING
    lane.h_start = h0;                                                               // NERO_IX_PRIORITY_EVIDENCE
    if (nero_priority_ready().on) {
      // NERO_IX_PRIORITY_READY: the first point of this turn's centre line within reach of ego's lane (a car's half width
      // 1.0 m + ego's half width + 0.2 m from ego's centre line), sampled every 0.25 m
      const double reach = 1.2 + 0.5 * planner_data_->vehicle_info_.vehicle_width_m;
      const auto ego_cl = assigned_lanelet.centerline2d();
      const double len = lanelet::utils::getLaneletLength2d(lane.lanelet);
      // and the stretch of ego's centre line that its body sweeps there (+- its half width 1.0 m along ego's lane)
      double near = std::numeric_limits<double>::infinity(), far = -near;
      for (double s = 0.0; s <= len + 1e-6 && ego_cl.size() >= 2; s += 0.25) {
        const auto pt = lanelet::geometry::interpolatedPointAtDistance(lane.lanelet.centerline2d(), s);
        if (lanelet::geometry::distance2d(ego_cl, pt) <= reach) {
          if (lane.ready_s_cross < 0.0) {
            lane.ready_s_cross = s;
            lane.ready_cross.x = pt.x();
            lane.ready_cross.y = pt.y();
          }
          const double ea = lanelet::geometry::toArcCoordinates(ego_cl, pt).length;
          near = std::min(near, ea);
          far = std::max(far, ea);
        }
      }
      if (lane.ready_s_cross >= 0.0) {
        const double ego_len = lanelet::geometry::length(ego_cl);
        const auto pn = lanelet::geometry::interpolatedPointAtDistance(ego_cl, std::clamp(near - 1.0, 0.0, ego_len));
        const auto pf = lanelet::geometry::interpolatedPointAtDistance(ego_cl, std::clamp(far + 1.0, 0.0, ego_len));
        lane.ready_near.x = pn.x();
        lane.ready_near.y = pn.y();
        lane.ready_far.x = pf.x();
        lane.ready_far.y = pf.y();
      }
    }
  }
  // NERO_IX_PRIORITY_EVIDENCE: a turn off the priority road every entry of which also goes straight on (< 20 deg) into a
  // lane without a stop line / stop sign (the priority road continues there)
  if (nero_priority_evidence().on) {
    constexpr double kStraightOnRad = 20.0 / 180.0 * M_PI;
    auto goes_straight_on = [&](const lanelet::ConstLanelet & entry, const lanelet::ConstLanelet & turn) {
      for (const auto & sib : routing_graph_ptr->following(entry)) {
        if (
          sib.id() == turn.id() || sib.id() == assigned_lanelet.id() ||
          lanelet::utils::contains(ego_lanelets, sib)) {
          continue;
        }
        const auto c = sib.centerline2d();
        if (c.size() < 2) continue;
        const double h0 = std::atan2(c[1].y() - c[0].y(), c[1].x() - c[0].x());
        const double h1 = std::atan2(
          c[c.size() - 1].y() - c[c.size() - 2].y(), c[c.size() - 1].x() - c[c.size() - 2].x());
        if (
          std::fabs(autoware_utils::normalize_radian(h1 - h0)) < kStraightOnRad && nero_stop_lines(sib).empty()) {
          return true;
        }
      }
      return false;
    };
    for (auto & lane : lanes) {
      if (lane.kind != Kind::YIELD_PRIORITY_ROAD) continue;
      const auto entries = routing_graph_ptr->previous(lane.lanelet);
      lane.straight_on = !entries.empty() && std::all_of(entries.begin(), entries.end(), [&](const auto & e) {
        return goes_straight_on(e, lane.lanelet);
      });
    }
  }
  // distance from the end of each priority-road approach lane to the stop line of each turn it leads into
  std::function<std::vector<std::pair<size_t, double>>(const lanelet::ConstLanelet &, int)> to_line =
    [&](const lanelet::ConstLanelet & ll, const int depth) {
      std::vector<std::pair<size_t, double>> out;
      if (depth > 8) return out;
      for (const auto & next : routing_graph_ptr->following(ll)) {
        const auto it = index.find(next.id());
        if (it == index.end()) continue;
        const auto & n = lanes.at(it->second);
        if (n.kind == Kind::YIELD_PRIORITY_ROAD) {
          out.emplace_back(it->second, n.stop_s);
        } else if (n.kind == Kind::APPROACH_PRIORITY) {
          const double len = lanelet::utils::getLaneletLength2d(next);
          for (const auto & [y, d] : to_line(next, depth + 1)) {
            if (len + d < 80.0) out.emplace_back(y, len + d);
          }
        }
      }
      return out;
    };
  for (auto & lane : lanes) {
    if (lane.kind == Kind::APPROACH_PRIORITY) lane.to_line = to_line(lane.lanelet, 0);
  }
  // NERO_IX_PRIORITY_NO_EXIT: the fastest way on from each priority-road approach: any exit it leads into (its turns,
  // a straight on next to them, a lane outside the attention area) at the same kTurnLatAcc sideways
  if (nero_priority_no_exit() > 0.0) {
    constexpr double kStraightOnRad = 20.0 / 180.0 * M_PI;  // an exit turning less than this goes straight on
    auto v_way = [&](const lanelet::ConstLanelet & ll) {
      const auto c = ll.centerline2d();
      if (c.size() < 2) return std::numeric_limits<double>::infinity();
      const double h0 = std::atan2(c[1].y() - c[0].y(), c[1].x() - c[0].x());
      const double h1 = std::atan2(
        c[c.size() - 1].y() - c[c.size() - 2].y(), c[c.size() - 1].x() - c[c.size() - 2].x());
      const double dtheta = std::fabs(autoware_utils::normalize_radian(h1 - h0));
      if (dtheta < kStraightOnRad) return std::numeric_limits<double>::infinity();  // a straight on (any length)
      return std::sqrt(kTurnLatAcc * lanelet::utils::getLaneletLength2d(ll) / dtheta);
    };
    std::function<double(const lanelet::ConstLanelet &, int)> v_exit = [&](
                                                                          const lanelet::ConstLanelet & ll,
                                                                          const int depth) {
      double best = 0.0;
      for (const auto & next : routing_graph_ptr->following(ll)) {
        const auto it = index.find(next.id());
        if (it != index.end() && lanes.at(it->second).kind == Kind::APPROACH_PRIORITY && depth < 8) {
          best = std::max(best, v_exit(next, depth + 1));
        } else {
          best = std::max(best, v_way(next));
        }
      }
      return best > 0.0 ? best : std::numeric_limits<double>::infinity();
    };
    for (auto & lane : lanes) {
      if (lane.kind == Kind::APPROACH_PRIORITY) lane.v_exit_max = v_exit(lane.lanelet, 0);
    }
    for (auto & lane : lanes) {
      if (lane.kind != Kind::YIELD_PRIORITY_ROAD) continue;
      double v = 0.0;
      for (const auto & entry : routing_graph_ptr->previous(lane.lanelet)) {
        const auto it = index.find(entry.id());
        v = std::max(
          v, (it != index.end() && lanes.at(it->second).kind == Kind::APPROACH_PRIORITY)
               ? lanes.at(it->second).v_exit_max
               : std::numeric_limits<double>::infinity());
      }
      lane.v_exit_max = v > 0.0 ? v : std::numeric_limits<double>::infinity();
    }
    // a SIBLING of such a mouth that is itself a turn is the mouth's other turn, not a straight on: a car heading
    // straight inside it near the mouth is not "turning off its straight lane" (a late braker crawling to its line)
    for (auto & lane : lanes) {
      if (lane.kind != Kind::SIBLING || !std::isfinite(v_way(lane.lanelet))) continue;
      for (const auto & entry : routing_graph_ptr->previous(lane.lanelet)) {
        const auto it = index.find(entry.id());
        if (
          it != index.end() && lanes.at(it->second).kind == Kind::APPROACH_PRIORITY &&
          std::isfinite(lanes.at(it->second).v_exit_max)) {
          lane.turn_sibling = true;
        }
      }
    }
  }
  std::string text;
  for (const auto & lane : lanes) {
    if (lane.kind == Kind::APPROACH_OTHER) continue;
    text += " " + std::to_string(lane.lanelet.id()) + ":" + std::string(magic_enum::enum_name(lane.kind));
    if (lane.kind == Kind::YIELD_MINOR || lane.kind == Kind::YIELD_PRIORITY_ROAD) {
      text += fmt::format("(stop {:.1f} m", lane.stop_s);
      if (lane.kind == Kind::YIELD_PRIORITY_ROAD) text += fmt::format(", turn <= {:.1f} m/s", lane.v_turn_max);
      if (lane.kind == Kind::YIELD_PRIORITY_ROAD && std::isfinite(lane.v_exit_max)) {
        text += fmt::format(", ways on <= {:.1f} m/s", lane.v_exit_max);  // NERO_IX_PRIORITY_NO_EXIT
      }
      if (lane.straight_on) text += ", straight on";  // NERO_IX_PRIORITY_EVIDENCE
      text += ")";
    }
    if (lane.kind == Kind::APPROACH_PRIORITY && std::isfinite(lane.v_exit_max) && lane.v_exit_max < 15.0) {
      text += fmt::format("(ways on <= {:.1f} m/s)", lane.v_exit_max);  // NERO_IX_PRIORITY_NO_EXIT
    }
    if (lane.turn_sibling) text += "(a turn)";  // NERO_IX_PRIORITY_NO_EXIT
  }
  RCLCPP_INFO(
    logger_, "[NERO_IX] priority_from_signs: lane %ld in scope%s:%s", lane_id_,
    ev_right_turn ? " (right turn, NERO_IX_PRIORITY_EVIDENCE)" : "", text.c_str());
}

int IntersectionModule::neroPriorityVerdict(
  const autoware_perception_msgs::msg::PredictedObject & object,
  const geometry_msgs::msg::Pose & object_direction, const bool is_parked_vehicle,
  std::vector<size_t> * pv_turns) const
{
  using Kind = NeroPriorityLane::Kind;
  constexpr int kWatched = 0, kYielding = 1, kPotential = 2;
  if (!nero_priority_lanes_ || nero_priority_lanes_->empty()) return kWatched;
  // re-judged every cycle from the car's actual motion (Ahmed 10-23: priority never means ignoring a car):
  // a car that turns off its straight lane (motion direction more than kTurnDeg off the lane), creeps past the stop
  // line of its turn, or is not heading the right way is watched exactly as upstream; a car that could still take a
  // turn across ego but is too fast to stop at its stop line (v^2 / (2 kStopDecel) > distance) is a potential
  // violator
  constexpr double kTurnDeg = 12.0;       // straight driving: tracker heading within ~4 deg of the lane (captures)
  constexpr double kTurnMinSpeed = 1.0;   // [m/s] below this the heading is not reliable
  constexpr double kStraightMinSpeed = 2.0;  // [m/s] past the stop line, slower = creeping (NERO_IX_CREEP's v_creep)
  constexpr double kStopDecel = 3.0;      // [m/s2] the comfortable braking a car that yields uses (2-3, Ahmed 10-23)
  const double angle_thr = planner_param_.common.attention_area_angle_threshold;
  const double margin = planner_param_.common.attention_area_margin;
  const auto & lanes = nero_priority_lanes_.value();
  const auto & twist = object.kinematics.initial_twist_with_covariance.twist.linear;
  const double speed = std::hypot(twist.x, twist.y);
  // direction of motion: velocity direction (heading flipped when reversing, and so the lateral velocity) plus the
  // lateral velocity
  const double lateral = twist.x < 0.0 ? -twist.y : twist.y;
  double motion_angle =
    tf2::getYaw(object_direction.orientation) + std::atan2(lateral, std::fabs(twist.x));
  // NERO_IX_PRIORITY_EVIDENCE: from its own positions once it has >= pos_s s of them over which it moved >= move_m (a
  // fresh track's heading can be tens of degrees off its motion and swing back over 1-2 s: 11-03 #97)
  const auto & ev = nero_priority_evidence();
  bool motion_from_positions = false;
  if (ev.on) {
    const auto xy_it = nero_priority_xy_hist_.find(uuidKey(object.object_id));
    if (xy_it != nero_priority_xy_hist_.end() && !xy_it->second.empty()) {
      // the oldest position within the last second; when that is younger than pos_s (a gap: the car was outside the
      // attention area for a while) the newest one at least pos_s old (the history keeps 1.5 s)
      const auto & h = xy_it->second;
      const auto & last = h.back();
      const std::array<double, 3> * ref = nullptr;
      for (const auto & q : h) {
        if (q[0] < last[0] - 1.0 - 1e-3) continue;
        ref = &q;
        break;
      }
      if (ref == nullptr || last[0] - (*ref)[0] < ev.pos_s - 1e-3) {
        ref = nullptr;
        for (auto it = h.rbegin(); it != h.rend(); ++it) {
          if (last[0] - (*it)[0] >= ev.pos_s - 1e-3) {
            ref = &(*it);
            break;
          }
        }
      }
      if (ref != nullptr && std::hypot(last[1] - (*ref)[1], last[2] - (*ref)[2]) >= ev.move_m) {
        motion_angle = std::atan2(last[2] - (*ref)[2], last[1] - (*ref)[1]);
        motion_from_positions = true;
      }
    }
  }
  // NERO_IX_PRIORITY_EVIDENCE: its own positions show it stood (moved < 0.3 m over >= 0.5 s) within its record (1.5 s): a
  // fresh track born at 0 m/s or a hand-over reports a "launch" while the car was moving all along
  auto stood_by_positions = [&]() {
    const auto xy_it = nero_priority_xy_hist_.find(uuidKey(object.object_id));
    if (xy_it == nero_priority_xy_hist_.end()) return false;
    const auto & h = xy_it->second;
    for (size_t a = 0; a < h.size(); ++a) {
      for (size_t b = a + 1; b < h.size(); ++b) {
        if (h[b][0] - h[a][0] < 0.5 - 1e-3) continue;
        if (std::hypot(h[b][1] - h[a][1], h[b][2] - h[a][2]) < 0.3) return true;
        break;
      }
    }
    return false;
  };
  const double stop_dist = speed * speed / (2.0 * kStopDecel);
  // NERO_IX_PRIORITY_CAN_STOP: can it still stop at its line (speed v, distance d), from its actual motion?
  const double can_stop_horizon = nero_priority_can_stop();
  const double a_exp =
    planner_param_.collision_detection.ignore_on_amber_traffic_light.object_expected_deceleration.car;
  const double a_obs = nero_priority_a_obs_;
  constexpr double kLookAheadMinSpeed = 4.0;  // [m/s] slower, a car stops within 2-4 m (2-3 m/s2): rolling up to its line
  constexpr double kPullAwayAcc = 2.0;  // [m/s2] speeding up at least this much = pulling away (a roll-up is gentler)
  const double ego_v = std::max(1.0, planner_data_->current_velocity->twist.linear.x);
  const double t_ahead = (can_stop_horizon > 0.0 && nero_dist_to_1st_pass_judge_ > 0.0)
                           ? std::min(can_stop_horizon, nero_dist_to_1st_pass_judge_ / ego_v)
                           : 0.0;
  const bool braking = nero_priority_braking_now_;
  nero_priority_hard_ = false;
  // NERO_IX_PRIORITY_NO_EXIT: faster than every way on from where it is (set per lane below) = it can only go straight
  // (judged from first sight; a late braker that is already braking then is let go as soon as its braking shows:
  // the stop it caused ends without the collision hold, scene_intersection.cpp)
  const bool no_exit_rule = nero_priority_no_exit() > 0.0 && can_stop_horizon >= 0.0;
  double v_exit_here = std::numeric_limits<double>::infinity();
  bool no_exit_before_line = false;  // only a car still before its line (past it, it is watched as upstream anyway)
  double no_exit_lane_yaw = 0.0;     // the direction of the lane it is judged on (its band's second direction)
  bool no_exit_gate = false;
  nero_priority_no_exit_now_ = false;
  // NERO_IX_PRIORITY_SWING: braking alone does not make it a potential violator; a swing toward its turn does
  const double swing_deg = nero_priority_swing();
  const bool swing_rule = swing_deg > 0.0 && can_stop_horizon >= 0.0;
  const std::string swing_key = uuidKey(object.object_id);
  auto swinging_toward = [&](const size_t y) {
    if (!nero_priority_swing_known_) return false;  // < 1 s of heading history: the turning rule watches it
    const std::string key = swing_key + (lanes.at(y).turn_sign > 0.0 ? "L" : "R");
    const double toward = lanes.at(y).turn_sign * nero_priority_swing_;
    if (ev.on && motion_from_positions) {
      // NERO_IX_PRIORITY_EVIDENCE: a swing of the tracker's heading counts only if its own positions also point at least
      // half the turning angle toward the turn (against the turn's start = straight on): at a real turn's onset the
      // positions lead the reported heading (SIM.md 19); a converging fresh heading moves the heading alone
      const double dev = lanes.at(y).turn_sign *
                         autoware_utils::normalize_radian(motion_angle - lanes.at(y).h_start) * 180.0 / M_PI;
      if (dev < 0.5 * kTurnDeg) {
        if (toward >= swing_deg && nero_priority_debug()) {
          nero_priority_dbg_ += fmt::format(" | swing {:.0f} not in its motion (dev {:.0f})", toward, dev);
        }
        nero_priority_swinging_.erase(key);
        return false;
      }
    }
    if (toward >= swing_deg) {
      nero_priority_swinging_.insert(key);
    } else if (toward < 0.5 * swing_deg) {
      nero_priority_swinging_.erase(key);
    }
    return nero_priority_swinging_.count(key) > 0;
  };
  const bool dbg = nero_priority_debug();
  if (dbg) {
    nero_priority_dbg_ = fmt::format(
      "v {:.2f} a_obs {:.2f}{}{} t_pj {:.2f}{}", speed, a_obs, braking ? " brk" : "",
      nero_priority_a_launch_ > 0.0 ? fmt::format(" launch {:.2f}", nero_priority_a_launch_) : "", t_ahead,
      nero_priority_swing_known_ ? fmt::format(" swing {:.1f}", nero_priority_swing_) : "");
    if (motion_from_positions) {  // NERO_IX_PRIORITY_EVIDENCE
      nero_priority_dbg_ += fmt::format(" pos {:.0f} (yaw {:.0f})", motion_angle * 180.0 / M_PI,
        (tf2::getYaw(object_direction.orientation) + std::atan2(lateral, std::fabs(twist.x))) * 180.0 / M_PI);
    }
  }
  auto cannot_stop = [&](const size_t y, const double v, const double d) {
    // it could take the turn only at <= v_turn_max: a car braking toward it is judged by its speed at the line
    const double v_line = std::sqrt(std::max(0.0, v * v - 2.0 * (braking ? a_obs : 0.0) * std::max(0.0, d)));
    if (v_line > lanes.at(y).v_turn_max) {
      // NERO_IX_PRIORITY_NO_EXIT: too fast for this turn goes straight on, unless it is too fast for every way on
      if (!no_exit_rule || !no_exit_before_line || v_line <= v_exit_here) return false;
      no_exit_gate = true;
    }
    // a car keeping its speed must be able to stop with the expected deceleration, one that is braking for the
    // junction with up to kStopDecel (the braking a car that yields uses, 2-3 m/s2)
    const double a_req = d > 0.05 ? v * v / (2.0 * d) : (v > 0.5 ? 1e9 : 0.0);
    return a_req > (braking ? kStopDecel : a_exp);
  };
  // NERO_IX_PRIORITY_TURN: the direction of the lane the car is judged on (set in the lane loop below before potential())
  double cur_lane_angle = 0.0;
  // NERO_IX_PRIORITY_TURN: past the stop line of the turn being judged, still moving along the turn's start (set below)
  bool past_straight = false;
  const auto & tp = nero_priority_turn();
  const auto & rp = nero_priority_ready();
  // NERO_IX_PRIORITY_TURN: do its own positions over the last win_s s move toward the side of turn y (vs its lane)?
  auto turn_shows = [&](const size_t y, const double dist_to_line) {
    if (!tp.on || !ev.on) return false;
    const std::string tkey = uuidKey(object.object_id) + "#" + std::to_string(y);
    const double now = nero_steady_now();
    bool fire = false, computed = false;
    double dev = 0.0, lat = 0.0;
    const auto xy_it = nero_priority_xy_hist_.find(uuidKey(object.object_id));
    if (xy_it != nero_priority_xy_hist_.end() && xy_it->second.size() >= 2) {
      const auto & h = xy_it->second;
      const auto & last = h.back();
      const std::array<double, 3> * ref = nullptr;
      for (auto it = h.rbegin(); it != h.rend(); ++it) {
        if (last[0] - (*it)[0] >= tp.win_s - 1e-3) {
          ref = &(*it);
          break;
        }
      }
      if (ref != nullptr) {
        const double dx = last[1] - (*ref)[1], dy = last[2] - (*ref)[2];
        const double lon = dx * std::cos(cur_lane_angle) + dy * std::sin(cur_lane_angle);
        lat = lanes.at(y).turn_sign * (-dx * std::sin(cur_lane_angle) + dy * std::cos(cur_lane_angle));
        if (std::hypot(dx, dy) >= 1.0 && lon > 0.0) {
          dev = std::atan2(lat, lon) * 180.0 / M_PI;
          computed = true;
          fire = dev >= tp.dev_deg && lat >= tp.lat_m;
        }
      }
    }
    const double stamp = rclcpp::Time(planner_data_->predicted_objects->header.stamp).seconds();
    // its heading history (this turn's side, NaN while not computable) for the settled gate, one value per message
    auto & dh = nero_turn_dev_[tkey];
    if (dh.empty() || stamp > dh.back()[0] + 1e-6) {
      dh.push_back({stamp, computed ? dev : std::numeric_limits<double>::quiet_NaN()});
    }
    while (!dh.empty() && dh.front()[0] < stamp - (tp.win_s + tp.gate_s + 0.5)) dh.pop_front();
    // NERO_IX_PRIORITY_TURN_EARLY: the same positions over a shorter window, a car at speed near its turn's line
    const auto & te = nero_priority_turn_early();
    bool efire = false;
    double edev = 0.0, elat = 0.0;
    if (
      te.on && dist_to_line <= te.near_m && xy_it != nero_priority_xy_hist_.end() && xy_it->second.size() >= 2) {
      const auto & h = xy_it->second;
      const auto & last = h.back();
      const std::array<double, 3> * ref = nullptr;
      for (auto it = h.rbegin(); it != h.rend(); ++it) {
        if (last[0] - (*it)[0] >= te.win_s - 0.05) {  // the newest position ~win_s old (10 Hz objects)
          ref = &(*it);
          break;
        }
      }
      if (ref != nullptr) {
        const double dx = last[1] - (*ref)[1], dy = last[2] - (*ref)[2];
        const double lon = dx * std::cos(cur_lane_angle) + dy * std::sin(cur_lane_angle);
        elat = lanes.at(y).turn_sign * (-dx * std::sin(cur_lane_angle) + dy * std::cos(cur_lane_angle));
        if (std::hypot(dx, dy) >= te.move_m && lon > 0.0) {
          edev = std::atan2(elat, lon) * 180.0 / M_PI;
          efire = edev >= te.dev_deg && elat >= tp.lat_m;
        }
      }
    }
    bool tfire = fire;  // NERO_IX_PRIORITY_TURN_EARLY: TURN's own reading, judged by the switch's gate
    if ((fire || efire) && tp.gate_s > 0.0) {
      // TURN's gate as the car runs it: every value in the window computable and within tol (>= 5 of them); with
      // NERO_IX_PRIORITY_TURN_EARLY's <census_gate> the values not yet computable (its first second of positions) are skipped
      // and >= 5 computable ones must be within tol: the gate the left-turner census measured (0 false of 33)
      int n = 0, n_fin = 0;
      bool settled = true, settled_c = true;
      for (const auto & q : dh) {
        if (q[0] < stamp - tp.win_s - tp.gate_s - 1e-6 || q[0] > stamp - tp.win_s + 1e-6) continue;
        if (!std::isfinite(q[1]) || std::fabs(q[1]) > tp.gate_tol_deg) settled = false;
        if (std::isfinite(q[1])) {
          ++n_fin;
          if (std::fabs(q[1]) > tp.gate_tol_deg) settled_c = false;
        }
        ++n;
      }
      settled = settled && n >= 5;
      settled_c = settled_c && n_fin >= 5;
      const bool settled_e = te.census_gate ? settled_c : settled;
      if (!settled) {
        if (dbg && fire) {
          nero_priority_dbg_ += fmt::format(" | turn {} pos dev {:.1f} not settled", lanes.at(y).lanelet.id(), dev);
        }
        fire = false;
      }
      if (!settled_e) {
        if (dbg && efire) {
          nero_priority_dbg_ += fmt::format(" | turn {} early dev {:.1f} not settled", lanes.at(y).lanelet.id(), edev);
        }
        efire = false;
        tfire = false;
      }
    }
    // [0] consecutive objects messages that showed it, [1] steady time it last held, [2] the objects stamp counted last
    // (a car in two lanes is judged twice per cycle: count each message once)
    auto & st = nero_turn_shows_[tkey];
    if (stamp > st[2] + 1e-6) {
      st[2] = stamp;
      st[0] = fire ? st[0] + 1.0 : 0.0;
      if (fire && st[0] >= tp.n_cycles - 1e-6) st[1] = now;
    }
    const bool shows_turn = st[1] > 0.0 && now - st[1] <= tp.hold_s;
    bool shows = shows_turn;
    if (te.on) {
      // NERO_IX_PRIORITY_TURN_EARLY: its own count of consecutive messages and its own last time it held (TURN's hold)
      auto & es = nero_turn_early_[tkey];
      const bool new_msg = stamp > es[2] + 1e-6;
      if (new_msg) {
        es[2] = stamp;
        // TURN's own reading through the switch's gate, or the short window
        const bool any = efire || (te.census_gate && tfire);
        es[0] = any ? es[0] + 1.0 : 0.0;
        if (any && es[0] >= (efire ? te.n_cycles : tp.n_cycles) - 1e-6) es[1] = now;
      }
      const bool eshows = es[1] > 0.0 && now - es[1] <= tp.hold_s;
      if (eshows && !shows_turn && new_msg) {
        const std::string turn_dev = computed ? fmt::format("{:.1f} deg", dev) : std::string("unknown");
        RCLCPP_INFO(
          logger_,
          "[NERO_IX] priority turn early: lane %ld obj %s turn %ld d %.1f v %.2f: positions over %.1f s head %.1f deg "
          "(lat %.2f m) toward the turn%s, TURN's %.1f s heading %s%s",
          lane_id_, nero_hex_id(object).c_str(), lanes.at(y).lanelet.id(), dist_to_line, speed, te.win_s, edev, elat,
          efire ? "" : " (held)", tp.win_s, turn_dev.c_str(), (tfire && !efire) ? " (census gate)" : "");
      }
      if (dbg && (efire || eshows)) {
        nero_priority_dbg_ += fmt::format(" | turn {} early dev {:.1f} lat {:.2f}{}", lanes.at(y).lanelet.id(), edev,
          elat, eshows ? " SHOWS" : "");
      }
      shows = shows_turn || eshows;
    }
    if (dbg && (fire || shows_turn)) {
      nero_priority_dbg_ += fmt::format(" | turn {} pos dev {:.1f} lat {:.2f}{}", lanes.at(y).lanelet.id(), dev, lat,
        shows_turn ? " SHOWS" : "");
    }
    return shows;
  };
  // NERO_IX_PRIORITY_READY: a car judged straight on that could still take turn y without braking (or braking into it)
  auto ready_candidate = [&](const size_t y, const double dist_to_line) {
    if (!rp.on || !ev.on || speed < 1.0) return;
    NeroReadyCand c;
    c.key = uuidKey(object.object_id);
    c.turn = y;
    c.dist_to_line = dist_to_line;
    c.v = speed;
    c.a = braking ? std::max(0.0, a_obs) : 0.0;
    c.acc_up = std::max(0.0, std::max(-a_obs, nero_priority_a_launch_));
    if (braking) nero_ready_slowed_.insert(c.key);
    c.slowed = nero_ready_slowed_.count(c.key) > 0;
    // its sideways drift toward the side of turn y over its record (>= 1 s of positions, the oldest within 1.5 s)
    const auto xy_it = nero_priority_xy_hist_.find(c.key);
    if (xy_it != nero_priority_xy_hist_.end() && xy_it->second.size() >= 2) {
      const auto & h = xy_it->second;
      const auto & a = h.front();
      const auto & b = h.back();
      if (b[0] - a[0] >= 1.0 - 1e-3) {
        const double dx = b[1] - a[1], dy = b[2] - a[2];
        c.toward = lanes.at(y).turn_sign * (-dx * std::sin(cur_lane_angle) + dy * std::cos(cur_lane_angle));
      }
    }
    c.length = object.shape.dimensions.x;
    c.width = object.shape.dimensions.y;
    c.id = nero_hex_id(object);
    nero_ready_cands_.push_back(c);
  };
  auto potential = [&](const size_t y, const double dist_to_line) {
    if (can_stop_horizon < 0.0) {
      if (speed <= lanes.at(y).v_turn_max && stop_dist > dist_to_line) pv_turns->push_back(y);
      if (dbg) {
        nero_priority_dbg_ += fmt::format(" | turn {} d {:.1f} a_req {:.2f}", lanes.at(y).lanelet.id(),
          dist_to_line, dist_to_line > 0.05 ? speed * speed / (2.0 * dist_to_line) : 99.0);
      }
      return;
    }
    if (swing_rule) {
      if (swinging_toward(y)) {
        // turning in toward ego: its turn paths are checked, also after ego's 1st pass judge line
        nero_priority_hard_ = true;
        pv_turns->push_back(y);
        if (dbg) {
          nero_priority_dbg_ += fmt::format(" | turn {} d {:.1f} -> SWING toward it (late check)",
            lanes.at(y).lanelet.id(), dist_to_line);
        }
        return;
      }
      if (turn_shows(y, dist_to_line)) {
        // NERO_IX_PRIORITY_TURN: its positions move toward its turn: its turn paths are checked, also after ego's 1st pass
        // judge line (as a swing)
        nero_priority_hard_ = true;
        pv_turns->push_back(y);
        if (dbg) {
          nero_priority_dbg_ += fmt::format(" | turn {} d {:.1f} -> TURN SHOWS (late check)",
            lanes.at(y).lanelet.id(), dist_to_line);
        }
        return;
      }
      if (braking) {
        // slowing down while heading straight: a signal, not a violation (Ahmed 10-24)
        // NERO_IX_PRIORITY_READY: it may also be slowing into its turn: ego stays ready for it (gentle, never a stop)
        if (lanes.at(y).straight_on) ready_candidate(y, dist_to_line);
        if (dbg) {
          nero_priority_dbg_ += fmt::format(" | turn {} d {:.1f} -> braking, no swing: yielding",
            lanes.at(y).lanelet.id(), dist_to_line);
        }
        return;
      }
      if (ev.on && ev.evidence_only && lanes.at(y).straight_on && (dist_to_line > 0.0 || past_straight)) {
        // NERO_IX_PRIORITY_EVIDENCE: heading straight before its line with a straight on: no turn is anticipated
        // ("cannot stop at its line" alone is no violation, Ahmed 11-03 #80); a launch from a stand toward its line
        // still makes it a potential violator (CAN_STOP's clause below, checked also after ego's pass judge line)
        const double a_up = std::max(-a_obs, nero_priority_a_launch_);
        bool launch = false;
        if (a_up >= kPullAwayAcc && can_stop_horizon > 0.0 && stood_by_positions()) {
          const double v2 = speed + a_up * can_stop_horizon;
          const double d2 = dist_to_line - 0.5 * (speed + v2) * can_stop_horizon;
          launch = v2 <= lanes.at(y).v_turn_max && (d2 > 0.05 ? v2 * v2 / (2.0 * d2) : 1e9) > kStopDecel;
        }
        if (launch) {
          nero_priority_hard_ = true;
          pv_turns->push_back(y);
        } else if (speed <= lanes.at(y).v_turn_max) {
          ready_candidate(y, dist_to_line);  // NERO_IX_PRIORITY_READY: it could still take the turn without braking
        }
        if (dbg) {
          nero_priority_dbg_ += fmt::format(" | turn {} d {:.1f} -> {}", lanes.at(y).lanelet.id(), dist_to_line,
            launch ? "LAUNCH toward it (late check)" : "straight on, no swing: yielding (evidence)");
        }
        return;
      }
    }
    no_exit_gate = false;
    no_exit_before_line = dist_to_line > 0.0;
    bool pv = cannot_stop(y, speed, dist_to_line);
    double v1 = speed, d1 = dist_to_line;
    if (!pv && !braking && t_ahead > 0.0 && speed >= kLookAheadMinSpeed) {
      // a car that keeps its speed toward the junction, carried on until ego reaches its 1st pass judge line
      d1 = dist_to_line - speed * t_ahead;
      pv = cannot_stop(y, v1, d1);
    }
    // braking harder than a car that yields and still unable to stop at its line with that braking: it may be
    // slowing down to take the turn (a fast car that brakes late and turns in)
    if (pv && braking && a_obs >= kStopDecel) nero_priority_hard_ = true;
    // speeding up toward its line (a waiting car that pulls away) so that, carrying on for <horizon> s, it could no
    // longer stop at its line with kStopDecel
    const double a_up = std::max(-a_obs, nero_priority_a_launch_);  // speeding up (slope, or since it stood)
    if (!braking && a_up >= kPullAwayAcc && can_stop_horizon > 0.0) {
      const double v2 = speed + a_up * can_stop_horizon;
      const double d2 = dist_to_line - 0.5 * (speed + v2) * can_stop_horizon;
      if (v2 <= lanes.at(y).v_turn_max && (d2 > 0.05 ? v2 * v2 / (2.0 * d2) : 1e9) > kStopDecel) {
        pv = true;
        nero_priority_hard_ = true;
      }
    }
    if (pv && no_exit_gate) {
      // NERO_IX_PRIORITY_NO_EXIT: it cannot take any way on and cannot stop: checked also after ego's pass judge line
      nero_priority_hard_ = true;
      nero_priority_no_exit_now_ = true;
      nero_priority_no_exit_lane_yaw_ = no_exit_lane_yaw;
    }
    if (pv) pv_turns->push_back(y);
    if (dbg) {
      nero_priority_dbg_ += fmt::format(" | turn {} d {:.1f} a_req {:.2f} at_pj v {:.2f} d {:.1f} -> {}{}",
        lanes.at(y).lanelet.id(), dist_to_line, dist_to_line > 0.05 ? speed * speed / (2.0 * dist_to_line) : 99.0,
        v1, d1, pv ? (nero_priority_hard_ ? "CANNOT stop (late check)" : "CANNOT stop") : "can stop",
        no_exit_gate ? fmt::format(" NO EXIT (ways on <= {:.1f})", v_exit_here) : "");
    }
  };
  bool watched = false;      // on a lane checked as upstream, or misbehaving
  bool committed = false;    // past the stop line of a turn off the priority road
  bool exempt = false;
  bool straight_on = false;  // inside (no margin) the straight-on lane next to that turn, driving along it
  for (size_t i = 0; i < lanes.size(); ++i) {
    const auto & lane = lanes.at(i);
    if (!lanelet::utils::isInLanelet(object_direction, lane.lanelet, margin)) continue;
    past_straight = false;  // NERO_IX_PRIORITY_TURN
    const double lane_angle = lanelet::utils::getLaneletAngle(lane.lanelet, object_direction.position);
    const double angle_diff = autoware_utils::normalize_radian(lane_angle - motion_angle, -M_PI);
    const bool along = std::fabs(angle_diff) < angle_thr;
    const bool reversed_parked =
      is_parked_vehicle && std::fabs(autoware_utils::normalize_radian(angle_diff + M_PI)) < angle_thr;
    const bool straight_lane =
      lane.kind == Kind::APPROACH_PRIORITY || lane.kind == Kind::PARALLEL || lane.kind == Kind::SIBLING;
    switch (lane.kind) {
      case Kind::NORMAL:
      case Kind::YIELD_MINOR:
      case Kind::APPROACH_OTHER:
        if (along || reversed_parked) watched = true;  // the module's own heading rule
        continue;
      default:
        break;
    }
    if (!along) continue;  // exempt only when heading the lane's way (a reversed / wrong-way car stays watched)
    const bool turning = speed >= kTurnMinSpeed && std::fabs(angle_diff) > kTurnDeg / 180.0 * M_PI;
    // NERO_IX_PRIORITY_EVIDENCE: on a priority-road approach only a turn toward the side of one of its turns across ego
    // counts (a car easing back into its lane from over the centre line, or veering to its kerb, is not turning in)
    bool turning_in = turning;
    const double side = angle_diff < 0.0 ? 1.0 : -1.0;  // +1: it moves to the left of its lane's direction
    if (
      ev.on && turning && !motion_from_positions &&
      ((lane.kind == Kind::APPROACH_PRIORITY && !lane.to_line.empty()) ||
       (lane.kind == Kind::SIBLING && nero_ev_sib_turns_.count(i) > 0))) {
      // no record of its own motion yet (< pos_s of positions: a fresh track's heading can be tens of degrees off)
      turning_in = false;
      if (dbg) nero_priority_dbg_ += " | no motion record yet";
    } else if (ev.on && turning && lane.kind == Kind::APPROACH_PRIORITY && !lane.to_line.empty()) {
      turning_in = std::any_of(lane.to_line.begin(), lane.to_line.end(), [&](const auto & yd) {
        return side * lanes.at(yd.first).turn_sign > 0.0;
      });
      if (dbg && !turning_in) nero_priority_dbg_ += " | veers away from its turns";
    } else if (ev.on && turning && lane.kind == Kind::SIBLING && nero_ev_sib_turns_.count(i) > 0) {
      // the straight on next to its turns: the same, toward the side of those turns
      const auto & sib_turns = nero_ev_sib_turns_.at(i);
      turning_in = std::any_of(sib_turns.begin(), sib_turns.end(), [&](const size_t y) {
        return side * lanes.at(y).turn_sign > 0.0;
      });
      if (dbg && !turning_in) nero_priority_dbg_ += " | veers away from the turn beside";
    }
    // leaving its straight lane, toward ego or not (NERO_IX_PRIORITY_NO_EXIT: a T mouth's other turn is no straight lane)
    if (straight_lane && turning_in && !lane.turn_sibling) watched = true;
    if (lane.kind == Kind::YIELD_PRIORITY_ROAD) {
      const double s_obj = lanelet::utils::getArcCoordinates({lane.lanelet}, object_direction).length;
      past_straight = false;
      if (
        s_obj > lane.stop_s && tp.on && ev.on && ev.evidence_only && lane.straight_on && motion_from_positions &&
        speed >= kStraightMinSpeed &&
        std::fabs(autoware_utils::normalize_radian(motion_angle - lane.h_start)) < tp.dev_deg / 180.0 * M_PI) {
        // NERO_IX_PRIORITY_TURN: past its line but still running along the turn's start: driving on straight, judged
        // by its turn evidence (potential() below), not committed to the turn
        past_straight = true;
        if (dbg) nero_priority_dbg_ += " | past its line, straight";
      }
      if (s_obj > lane.stop_s && !past_straight) {
        committed = true;
      } else {
        exempt = true;
      }
      v_exit_here = lane.v_exit_max;  // NERO_IX_PRIORITY_NO_EXIT
      no_exit_lane_yaw = lane_angle;
      cur_lane_angle = lane.h_start;  // NERO_IX_PRIORITY_TURN: against the turn's start (its local direction already turns)
      potential(i, lane.stop_s - s_obj);  // negative past the line: it may still turn if slow enough
    } else if (lane.kind == Kind::APPROACH_PRIORITY) {
      exempt = true;
      const double rest = lanelet::utils::getLaneletLength2d(lane.lanelet) -
                          lanelet::utils::getArcCoordinates({lane.lanelet}, object_direction).length;
      v_exit_here = lane.v_exit_max;  // NERO_IX_PRIORITY_NO_EXIT
      no_exit_lane_yaw = lane_angle;
      cur_lane_angle = lane_angle;  // NERO_IX_PRIORITY_TURN
      for (const auto & [y, d] : lane.to_line) potential(y, rest + d);
    } else {  // PARALLEL, SIBLING
      if (lane.kind == Kind::PARALLEL) exempt = true;
      if (
        !turning_in && speed >= kStraightMinSpeed &&  // turning_in == turning unless NERO_IX_PRIORITY_EVIDENCE
        lanelet::utils::isInLanelet(object_direction, lane.lanelet, 0.0)) {
        straight_on = true;
      }
    }
  }
  if (watched) return kWatched;
  if (committed && !straight_on) return kWatched;
  if (!exempt && !straight_on) return kWatched;
  std::sort(pv_turns->begin(), pv_turns->end());
  pv_turns->erase(std::unique(pv_turns->begin(), pv_turns->end()), pv_turns->end());
  return pv_turns->empty() ? kYielding : kPotential;
}

void IntersectionModule::neroReadyCap(PathWithLaneId * path)
{
  // NERO_IX_PRIORITY_READY (predleft, 2026-11-05): see nero_priority_ready()
  const auto & rp = nero_priority_ready();
  if (!rp.on || nero_ready_cands_.empty() || !nero_priority_lanes_ || !nero_ready_stop_known_) return;
  const double d_sl = nero_ready_dist_to_stop_;  // ego (base_link) to the default stop line
  if ((d_sl <= 0.0 && rp.gap_m <= 0.0) || path->points.size() < 2) return;
  const double v_e = std::max(0.0, planner_data_->current_velocity->twist.linear.x);
  const double kVFloor = rp.v_floor;  // [m/s] the cap never asks for less
  if (v_e < 1.0) return;  // (between 1.0 and the floor the cap holds ego at the floor while it is not ready)
  constexpr double kStopMargin = 0.5;  // [m] ego's stop short of the default stop line
  constexpr double kClearMargin = 0.5;  // [s] ego through his way this long before he could reach it
  // [m/s2] the stop ego keeps ready for a car slowing into its turn, as planned here: a short stop at the line is realized
  // ~1.6x firmer by the controller under field braking (rt90 slow_left_n: planned 1.33-1.60 -> command 2.16-2.73), so 1.3
  // keeps the command near 2.0
  constexpr double kGentleStop = 1.3;
  const auto & lanes = nero_priority_lanes_.value();
  const auto & ego_pos = planner_data_->current_odometry->pose.position;
  const double rear = planner_data_->vehicle_info_.rear_overhang_m;
  constexpr double kInf = std::numeric_limits<double>::infinity();
  // arc length along the path to a point's projection on its nearest segment = motion_utils' findNearestSegmentIndex +
  // calcLongitudinalOffsetToSegment arithmetic without their per-call path copies (nero_lint; trajgap / bppstall)
  const size_t n_pts = path->points.size();
  std::vector<double> cum(n_pts, 0.0);
  for (size_t i = 1; i < n_pts; ++i) {
    const auto & a = path->points.at(i - 1).point.pose.position;
    const auto & q = path->points.at(i).point.pose.position;
    cum.at(i) = cum.at(i - 1) + std::hypot(q.x - a.x, q.y - a.y);
  }
  auto lon_to_seg = [&](const size_t k, const geometry_msgs::msg::Point & p) {
    const auto & a = path->points.at(k).point.pose.position;
    const auto & q = path->points.at(k + 1).point.pose.position;
    const double dx = q.x - a.x, dy = q.y - a.y, len = std::hypot(dx, dy);
    return len > 1e-9 ? ((p.x - a.x) * dx + (p.y - a.y) * dy) / len : 0.0;
  };
  auto seg_of = [&](const geometry_msgs::msg::Point & p) -> size_t {
    const size_t i = autoware::motion_utils::findNearestIndex(path->points, p);  // a scan, no copy
    if (i == 0) return 0;
    if (i >= n_pts - 1) return n_pts - 2;
    return lon_to_seg(i, p) <= 0.0 ? i - 1 : i;
  };
  auto s_of = [&](const geometry_msgs::msg::Point & p) {
    const size_t k = seg_of(p);
    return cum.at(k) + lon_to_seg(k, p);
  };
  const size_t seg = seg_of(ego_pos);
  const double off = lon_to_seg(seg, ego_pos);
  const double s_ego = cum.at(seg) + off;
  // ego's arc length to the near / far end of each turn's swept stretch of its lane (one lookup per turn and cycle)
  std::map<size_t, std::pair<double, double>> d_band;
  const double front = planner_data_->vehicle_info_.max_longitudinal_offset_m;
  double b_need = 0.0;
  const NeroReadyCand * worst = nullptr;
  double worst_ts = 0.0;
  double worst_allow = d_sl;
  bool worst_full = true;
  for (const auto & c : nero_ready_cands_) {
    const auto & turn = lanes.at(c.turn);
    if (turn.ready_s_cross < 0.0) continue;
    // his time to the start of the turn and until his front reaches ego's lane through it, at his speed; a car braking
    // into its turn as slowing at its present rate down to a turning speed (kTurnV) and keeping that
    constexpr double kTurnV = 2.5;  // [m/s] the slowest real turners kept through the turn (census: predleft README)
    auto t_at = [&](const double d) {
      if (d <= 0.0) return 0.0;
      if (c.a > 0.1 && c.v > kTurnV) {
        const double d_slow = (c.v * c.v - kTurnV * kTurnV) / (2.0 * c.a);
        if (d <= d_slow) return (c.v - std::sqrt(std::max(0.0, c.v * c.v - 2.0 * c.a * d))) / c.a;
        return (c.v - kTurnV) / c.a + (d - d_slow) / kTurnV;
      }
      return d / c.v;
    };
    const double d_start = c.dist_to_line - turn.stop_s;
    // let go: past the start of its turn by more than it needs to show the turn, drifting away from its turn's side,
    // or speeding up (a car does not speed up into a turn across ego)
    if (d_start < -(c.v * rp.evidence_s + 1.0)) continue;
    if (rp.away_m > 0.0 && c.toward <= -rp.away_m) continue;
    if (rp.acc_release > 0.0 && c.acc_up >= rp.acc_release) continue;
    const double t_s = t_at(d_start);
    const double t_in = t_at(d_start + turn.ready_s_cross - 0.5 * c.length);
    if (!std::isfinite(t_s) || t_s > rp.horizon_s) continue;
    if (d_band.count(c.turn) == 0) {
      d_band[c.turn] = {s_of(turn.ready_near) - s_ego, s_of(turn.ready_far) - s_ego};
    }
    const double d_near = d_band.at(c.turn).first, d_far = d_band.at(c.turn).second;
    if (d_far + rear < 0.0) continue;  // ego is through his way
    // ego, keeping its speed, is through his way (its rear past the far end) before he could reach ego's lane
    if (std::isfinite(t_in) && d_far + rear <= v_e * (t_in - kClearMargin)) continue;
    // where ego must be able to stop: the default stop line, or (a car keeping its speed) up to gap_m short of his swept
    // stretch; a car slowing into its turn is the likely real turner: ego keeps to the line and to a gentle stop (also
    // once it has slowed to its turning speed and holds it: rt90 slow_left_n, 6 -> 3.1 m/s, then a 3.0 stop was kept)
    const bool slowing = c.a > 0.1 || c.slowed;
    const double a_stop = slowing ? std::min(rp.a_stop, kGentleStop) : rp.a_stop;
    const double d_allow =
      std::max(d_sl - kStopMargin, (rp.gap_m > 0.0 && !slowing) ? d_near - front - rp.gap_m : -kInf);
    // can ego still stop at the default stop line when his turn would show (t_s + evidence) and it reacts (+ react)?
    const double t_e = t_s + rp.evidence_s + rp.react_s;
    auto ready_with = [&](const double b) {
      const double t1 = b > 1e-6 ? std::min(t_e, std::max(0.0, (v_e - kVFloor) / b)) : t_e;
      const double v1 = v_e - b * t1;
      const double x = v_e * t1 - 0.5 * b * t1 * t1 + v1 * (t_e - t1);
      return d_allow - x >= v1 * v1 / (2.0 * a_stop);
    };
    if (d_allow <= 0.0) continue;  // ego cannot stop short of it any more: the module's own checks decide
    if (ready_with(0.0)) continue;
    double b = rp.b_max;
    const bool full = ready_with(rp.b_max);
    if (full) {
      double lo = 0.0, hi = rp.b_max;
      for (int k = 0; k < 14; ++k) {
        const double mid = 0.5 * (lo + hi);
        (ready_with(mid) ? hi : lo) = mid;
      }
      b = hi;
    }
    if (b > b_need) {
      b_need = b;
      worst = &c;
      worst_ts = t_s;
      worst_allow = d_allow;
      worst_full = full;
    }
  }
  if (b_need <= 0.0 || worst == nullptr) return;
  // the cap: v(s) = sqrt(v_e^2 - 2 b s) ahead of ego (not below kVFloor), ego's speed on the points behind it (the
  // trajectory starts behind ego: an uncapped point there leaks into the speed planned at ego)
  double s = -off;
  size_t n_capped = 0;
  for (size_t i = 0; i < path->points.size(); ++i) {
    if (i > seg) {
      const auto & a = path->points.at(i - 1).point.pose.position;
      const auto & q = path->points.at(i).point.pose.position;
      s += std::hypot(q.x - a.x, q.y - a.y);
    }
    const double cap = i <= seg ? v_e : std::sqrt(std::max(kVFloor * kVFloor, v_e * v_e - 2.0 * b_need * std::max(0.0, s)));
    auto & vel = path->points.at(i).point.longitudinal_velocity_mps;
    if (vel > cap) {
      vel = static_cast<float>(cap);
      ++n_capped;
    }
    if (i > seg && s > std::max(d_sl, worst_allow) + 2.0) break;
  }
  char txt[96];
  std::snprintf(txt, sizeof(txt), " | NERO ready -%.2f for %s (%.1f s to its turn)", b_need, worst->id.c_str(), worst_ts);
  internal_debug_data_.decision_type += txt;
  const double now = nero_steady_now();
  if (now - nero_ready_last_log_ > 0.5 || nero_priority_debug()) {
    nero_ready_last_log_ = now;
    RCLCPP_INFO(
      logger_,
      "[NERO_IX] priority ready: lane %ld, car %s v %.1f m/s (a %.1f), %.1f m to its line (turn %ld, %.1f s to its "
      "turn): ego %.1f m/s, %.1f m to its stop line, eases off %.2f m/s2%s%s (%zu pts)",
      lane_id_, worst->id.c_str(), worst->v, worst->a, worst->dist_to_line, lanes.at(worst->turn).lanelet.id(),
      worst_ts, v_e, d_sl, b_need, worst_full ? "" : " (the most it may: not fully ready)",
      worst->slowed ? " [slowed: gentle stop]" : "", n_capped);
  }
}

void IntersectionModule::updateObjectInfoManagerArea()
{
  const auto & intersection_lanelets = intersection_lanelets_.value();
  const auto & attention_lanelets = intersection_lanelets.attention();
  const auto & attention_lanelet_stoplines = intersection_lanelets.attention_stoplines();
  const auto & adjacent_lanelets = intersection_lanelets.adjacent();
  const auto lanelet_map_ptr = planner_data_->route_handler_->getLaneletMapPtr();
  const auto & assigned_lanelet = lanelet_map_ptr->laneletLayer.get(lane_id_);
  const auto intersection_area = util::getIntersectionArea(assigned_lanelet, lanelet_map_ptr);

  // Nero c3 (unknown_filter): keep the per-object speed streaks up to date
  updateUnknownSpeedHistory();
  nero_ready_cands_.clear();  // NERO_IX_PRIORITY_READY: this cycle's candidates
  if (!nero_turn_shows_.empty()) {  // NERO_IX_PRIORITY_TURN: forget cars not judged for a while
    const double now = nero_steady_now();
    const double stamp = rclcpp::Time(planner_data_->predicted_objects->header.stamp).seconds();
    for (auto it = nero_turn_shows_.begin(); it != nero_turn_shows_.end();) {
      if (stamp - it->second[2] > 5.0 && now - it->second[1] > 5.0) {
        nero_turn_dev_.erase(it->first);
        nero_turn_early_.erase(it->first);  // NERO_IX_PRIORITY_TURN_EARLY
        it = nero_turn_shows_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // NERO_IX_PRIORITY_FROM_SIGNS: classify the lanes once, then flag yielding objects every cycle
  nero_priority_yield_uuids_.clear();
  nero_yi_yielders_.clear();  // NERO_IX_YIELD_INTENT
  if (nero_yield_intent::param().on) {
    nero_yi_was_.next(rclcpp::Time(planner_data_->predicted_objects->header.stamp).seconds());
    nero_yi_pos_was_.next(rclcpp::Time(planner_data_->predicted_objects->header.stamp).seconds());
  }
  nero_priority_pv_.clear();
  nero_priority_pv_late_.clear();
  if (nero_priority_from_signs() && !nero_priority_lanes_) {
    neroBuildPriorityLanes();
  }
  nero_priority_dbg_text_.clear();
  for (auto it = nero_priority_no_exit_.begin(); it != nero_priority_no_exit_.end();) {  // NERO_IX_PRIORITY_NO_EXIT
    if (nero_steady_now() - it->second > 10.0) {
      nero_priority_no_exit_lane_yaws_.erase(it->first);
      it = nero_priority_no_exit_.erase(it);
    } else {
      ++it;
    }
  }
  if (!nero_priority_v_hist_.empty()) {  // NERO_IX_PRIORITY_CAN_STOP: forget cars not seen for 2 s
    const double t_now = rclcpp::Time(planner_data_->predicted_objects->header.stamp).seconds();
    for (auto it = nero_priority_v_hist_.begin(); it != nero_priority_v_hist_.end();) {
      if (it->second.empty() || it->second.back().first < t_now - 2.0) {
        nero_priority_braking_.erase(it->first);
        nero_ready_slowed_.erase(it->first);       // NERO_IX_PRIORITY_READY
        nero_priority_yaw_hist_.erase(it->first);  // NERO_IX_PRIORITY_SWING
        nero_priority_xy_hist_.erase(it->first);   // NERO_IX_PRIORITY_NO_EXIT
        nero_ev_vehicle_.erase(it->first);         // NERO_IX_PRIORITY_EVIDENCE
        nero_priority_swinging_.erase(it->first + "L");
        nero_priority_swinging_.erase(it->first + "R");
        it = nero_priority_v_hist_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Nero c3 (wrong_way_watch): a vehicle driving against ego's exit lane (following lanelets of
  // the assigned lanelet) is registered as an attention object even if it is not on an attention
  // lanelet, and is never dropped as an "adjacent" (same-lane) object. Once flagged, it stays
  // flagged while it drives against the assigned lanelet itself (it came out of the exit lane into
  // the junction towards ego). The assigned lanelet alone is NOT enough to flag an object, since
  // crossing/turning traffic inside the junction often lies on it with a heading > pi/2 off.
  const bool wrong_way_watch = planner_param_.collision_detection.wrong_way_watch.enable;
  const auto prev_wrong_way_uuids = wrong_way_ego_lane_object_uuids_;
  wrong_way_ego_lane_object_uuids_.clear();
  lanelet::ConstLanelets ego_exit_lanelets{};
  if (wrong_way_watch) {
    for (const auto & ll :
         planner_data_->route_handler_->getRoutingGraphPtr()->following(assigned_lanelet)) {
      ego_exit_lanelets.push_back(ll);
    }
  }

  // ==========================================================================================
  // entries that are not observed in this iteration need to be cleared
  //
  // NOTE: old_map is not referenced because internal data of object_info_manager is cleared
  // ==========================================================================================
  const auto old_map = object_info_manager_.getObjectsMap();
  object_info_manager_.clearObjects();

  // NERO_IX_CREEP: ego's path = the centreline of its turn lanelet and of the lanelet after it
  autoware_utils::LineString2d nero_ego_path;
  if (nero_creep_ix().v_creep > 0.0 || nero_yield_intent::param().on) {  // (also NERO_IX_YIELD_INTENT's clear check)
    for (const auto & p : assigned_lanelet.centerline2d()) nero_ego_path.emplace_back(p.x(), p.y());
    const auto nexts = planner_data_->route_handler_->getRoutingGraphPtr()->following(assigned_lanelet);
    if (!nexts.empty()) {
      for (const auto & p : nexts.front().centerline2d()) nero_ego_path.emplace_back(p.x(), p.y());
    }
  }
  const double nero_half_width = planner_data_->vehicle_info_.vehicle_width_m / 2.0;

  for (const auto & predicted_object : planner_data_->predicted_objects->objects) {
    if (!isTargetCollisionVehicleType(predicted_object)) {
      continue;
    }
    // NERO_IX_PULLAWAY_ACC: every candidate's speed history (also while the creep rule leaves it out)
    if (nero_pullaway_acc() > 0.0 && nero_pullaway_target(predicted_object)) {
      nero_pullaway_update(predicted_object);
    }
    // NERO_IX_YIELD_REVOKE: the creep rule's speed history of every candidate, also while the yield rule leaves it out
    // (a revoked yielder is judged on its real gain, not on samples from before its yield)
    if (nero_yield_revoke::param().on && nero_creep_ix().v_creep > 0.0) {
      double v_c = 0.0, a_c = 0.0;
      nero_is_creeper(predicted_object, &v_c, &a_c);
    }

    // ==========================================================================================
    // NOTE: is_parked_vehicle is used because sometimes slow vehicle direction is
    // incorrect/reversed/flipped due to tracking. if is_parked_vehicle is true, object direction
    // is not checked
    // ==========================================================================================
    const auto object_direction =
      util::getObjectPoseWithVelocityDirection(predicted_object.kinematics);
    // Nero c3: use the full planar speed (UNKNOWN hull objects move mostly in vy)
    const auto is_parked_vehicle =
      std::hypot(
        predicted_object.kinematics.initial_twist_with_covariance.twist.linear.x,
        predicted_object.kinematics.initial_twist_with_covariance.twist.linear.y) <
      planner_param_.occlusion.ignore_parked_vehicle_speed_threshold;

    const bool is_wrong_way_on_ego_lane =
      wrong_way_watch &&
      (isWrongWayOnLanelets(predicted_object, ego_exit_lanelets) ||
       (prev_wrong_way_uuids.count(uuidKey(predicted_object.object_id)) > 0 &&
        isWrongWayOnLanelets(predicted_object, lanelet::ConstLanelets{assigned_lanelet})));
    if (is_wrong_way_on_ego_lane) {
      wrong_way_ego_lane_object_uuids_.insert(uuidKey(predicted_object.object_id));
    }

    // NERO: with NERO_IX_ADJACENT_FIX=1 an object that is in an attention lanelet (with a matching
    // heading) is kept even when it also lies on ego's own / sibling turn lanelets ("adjacent"). Upstream
    // drops it first, so cars crossing the U-turn apex or merging into its exit vanish from the check
    // exactly while they are in the conflict zone (the U-turn lanelets overlap the junction).
    const auto belong_attention_lanelet_id =
      checkAngleForTargetLanelets(object_direction, attention_lanelets, is_parked_vehicle);
    const auto belong_adjacent_lanelet_id =
      checkAngleForTargetLanelets(object_direction, adjacent_lanelets, false);
    if (
      belong_adjacent_lanelet_id && !is_wrong_way_on_ego_lane &&
      !(nero_adjacent_fix() && belong_attention_lanelet_id)) {
      continue;
    }
    const auto & obj_pos = predicted_object.kinematics.initial_pose_with_covariance.pose.position;
    const bool in_intersection_area = [&]() {
      if (!intersection_area) {
        return false;
      }
      return bg::within(autoware_utils::Point2d{obj_pos.x, obj_pos.y}, intersection_area.value());
    }();
    std::optional<lanelet::ConstLanelet> attention_lanelet{std::nullopt};
    std::optional<lanelet::ConstLineString3d> stopline{std::nullopt};
    if (!belong_attention_lanelet_id && !in_intersection_area && !is_wrong_way_on_ego_lane) {
      continue;
    } else if (belong_attention_lanelet_id) {
      const auto idx = belong_attention_lanelet_id.value();
      attention_lanelet = attention_lanelets.at(idx);
      stopline = attention_lanelet_stoplines.at(idx);
    }
    // Nero c3 (wrong_way_watch): a wrong-way vehicle on ego's own lane ahead is an attention object
    const bool belong_attention_area =
      belong_attention_lanelet_id.has_value() || is_wrong_way_on_ego_lane;

    // NERO_IX_PRIORITY_CAN_STOP: speed history of every car in an attention lane (also a waiting car that the creep
    // rule below leaves out, so its start from standstill is seen)
    if (
      nero_priority_from_signs() && belong_attention_area && !is_wrong_way_on_ego_lane &&
      (nero_priority_can_stop() >= 0.0 || nero_priority_debug())) {
      // observed deceleration = -slope of its speed over the last 0.4 s
      auto & hist = nero_priority_v_hist_[uuidKey(predicted_object.object_id)];
      const double t_obj = rclcpp::Time(planner_data_->predicted_objects->header.stamp).seconds();
      const auto & tw = predicted_object.kinematics.initial_twist_with_covariance.twist.linear;
      if (nero_priority_evidence().on) {
        // NERO_IX_PRIORITY_EVIDENCE: a hand-over (UNKNOWN <-> vehicle) or a jump of its position starts a new motion
        // record: the history before it belongs to another track (11-03 #97: a standing cluster's 0 m/s and ego-yaw,
        // taken over by a vehicle track at 5.5 m/s, read as a launch and a 168 deg swing toward its turn)
        const std::string key = uuidKey(predicted_object.object_id);
        const bool vehicle =
          predicted_object.classification.at(0).label != autoware_perception_msgs::msg::ObjectClassification::UNKNOWN;
        const auto & p = predicted_object.kinematics.initial_pose_with_covariance.pose.position;
        bool reset = false;
        const auto cls_it = nero_ev_vehicle_.find(key);
        if (cls_it != nero_ev_vehicle_.end() && cls_it->second != vehicle) reset = true;
        nero_ev_vehicle_[key] = vehicle;
        const auto xy_it = nero_priority_xy_hist_.find(key);
        if (!reset && xy_it != nero_priority_xy_hist_.end() && !xy_it->second.empty()) {
          const auto & q = xy_it->second.back();
          const double dt = t_obj - q[0];
          if (dt > 1e-3 && std::hypot(p.x - q[1], p.y - q[2]) > 1.0 + (std::hypot(tw.x, tw.y) + 2.0) * dt) {
            reset = true;
          }
        }
        if (reset) {
          hist.clear();
          for (auto it = nero_turn_dev_.lower_bound(key + "#"); it != nero_turn_dev_.end() &&
                                                                it->first.compare(0, key.size() + 1, key + "#") == 0;) {
            it = nero_turn_dev_.erase(it);  // NERO_IX_PRIORITY_TURN: its heading history belongs to the old record
          }
          nero_priority_xy_hist_.erase(key);
          nero_priority_yaw_hist_.erase(key);
          nero_priority_braking_.erase(key);
          nero_ready_slowed_.erase(key);  // NERO_IX_PRIORITY_READY
          nero_priority_swinging_.erase(key + "L");
          nero_priority_swinging_.erase(key + "R");
        }
      }
      if (hist.empty() || t_obj > hist.back().first + 1e-3) hist.emplace_back(t_obj, std::hypot(tw.x, tw.y));
      while (!hist.empty() && hist.front().first < t_obj - 1.5) hist.erase(hist.begin());
      // NERO_IX_PRIORITY_NO_EXIT (and NERO_IX_PRIORITY_EVIDENCE): its direction of motion
      if (nero_priority_no_exit() > 0.0 || nero_priority_evidence().on) {
        auto & xy = nero_priority_xy_hist_[uuidKey(predicted_object.object_id)];
        const auto & p = predicted_object.kinematics.initial_pose_with_covariance.pose.position;
        if (xy.empty() || t_obj > xy.back()[0] + 1e-3) xy.push_back({t_obj, p.x, p.y});
        while (!xy.empty() && xy.front()[0] < t_obj - 1.5) xy.erase(xy.begin());
      }
      double n = 0.0, st = 0.0, sv = 0.0, stt = 0.0, stv = 0.0, t_first = t_obj;
      for (const auto & [t, v] : hist) {
        if (t < t_obj - 0.4 - 1e-3) continue;
        const double dt = t - t_obj;
        n += 1.0, st += dt, sv += v, stt += dt * dt, stv += dt * v;
        t_first = std::min(t_first, t);
      }
      const double den = n * stt - st * st;
      nero_priority_a_obs_ =
        (n >= 3.0 && t_obj - t_first >= 0.2 && den > 1e-9) ? -(n * stv - st * sv) / den : 0.0;
      // a car that stood (<= 0.3 m/s) within the last 1.5 s: its acceleration since it left standstill
      nero_priority_a_launch_ = 0.0;
      for (auto it = hist.rbegin(); it != hist.rend(); ++it) {
        if (it->second > 0.3) continue;
        if (hist.back().second > 0.5 && t_obj - it->first >= 0.15) {
          nero_priority_a_launch_ = (hist.back().second - it->second) / (t_obj - it->first);
        }
        break;
      }
      // braking for the junction: on at >= 1 m/s2, off below 0.5 m/s2 (the estimate wobbles around a gentle braking)
      if (nero_priority_a_obs_ >= 1.0) {
        nero_priority_braking_.insert(uuidKey(predicted_object.object_id));
      } else if (nero_priority_a_obs_ < 0.5) {
        nero_priority_braking_.erase(uuidKey(predicted_object.object_id));
      }
      nero_priority_braking_now_ = nero_priority_braking_.count(uuidKey(predicted_object.object_id)) > 0;
      // NERO_IX_PRIORITY_SWING: heading over the last 0.25 s against its heading 1.0-0.6 s ago
      nero_priority_swing_known_ = false;
      nero_priority_swing_ = 0.0;
      if (nero_priority_swing() >= 0.0) {
        auto & yh = nero_priority_yaw_hist_[uuidKey(predicted_object.object_id)];
        const double yaw =
          tf2::getYaw(predicted_object.kinematics.initial_pose_with_covariance.pose.orientation);
        if (
          nero_priority_evidence().on && !yh.empty() &&
          std::fabs(autoware_utils::normalize_radian(yaw - yh.back().second)) > M_PI / 2.0) {
          yh.clear();  // NERO_IX_PRIORITY_EVIDENCE: a heading flip is not a swing
        }
        if (yh.empty() || t_obj > yh.back().first + 1e-3) yh.emplace_back(t_obj, yaw);
        while (!yh.empty() && yh.front().first < t_obj - 1.5) yh.erase(yh.begin());
        double n_now = 0.0, s_now = 0.0, n_ref = 0.0, s_ref = 0.0;
        for (const auto & [t, y] : yh) {
          const double d = autoware_utils::normalize_radian(y - yaw);  // relative to the latest heading
          if (t >= t_obj - 0.25 - 1e-3) {
            n_now += 1.0, s_now += d;
          } else if (t >= t_obj - 1.0 - 1e-3 && t <= t_obj - 0.6 + 1e-3) {
            n_ref += 1.0, s_ref += d;
          }
        }
        if (n_now >= 2.0 && n_ref >= 2.0) {
          nero_priority_swing_known_ = true;
          nero_priority_swing_ = (s_now / n_now - s_ref / n_ref) * 180.0 / M_PI;
        }
      }
    }

    // NERO_IX_YIELD_INTENT (yieldsignal, 2026-11-01, QUEUE YSIG): a vehicle that braked and now stands / crawls while
    // ego waits, nothing standing right ahead of it, whose stop (the creep rule's reach) stays clear of ego's path, is
    // yielding to ego: left out of the attention objects (so out of the TTC check and of yield-stuck), re-judged every
    // cycle (it is a full target again the cycle it speeds up). The car's creep rule needs 2 s at <= 2 m/s and the
    // stock check keeps a braking car "crossing" at its current speed until it is below 0.5 m/s, then the 1.5 s hold
    if (
      nero_yield_intent::param().on && nero_yield_intent::is_vehicle(predicted_object) && nero_ego_path.size() >= 2 &&
      !is_wrong_way_on_ego_lane) {
      const auto & yp = nero_yield_intent::param();
      const double t_yi = rclcpp::Time(planner_data_->predicted_objects->header.stamp).seconds();
      auto & yh = nero_yield_intent::update(predicted_object, t_yi);
      const auto yv = nero_yield_intent::view(yh, t_yi);
      const double ego_v_yi = std::abs(planner_data_->current_velocity->twist.linear.x);
      const std::string ykey = nero_yield_intent::key(predicted_object);
      const bool was_yielder = nero_yi_was_.was(ykey);  // in this module, last cycle
      bool yielder = false;
      double clear = -1.0;
      bool is_queued = false;
      // (ego's speed gates only the start of a yield: once ego goes on it, only the car's own motion ends it)
      // yieldroll (fields 11-12): or this module's collision stop is bringing ego to its line (nero_yi_roll_d_ >= 0)
      const bool ego_rolls_to_line = nero_yi_roll_d_ >= 0.0 && ego_v_yi >= yp.ego_v && !was_yielder;
      // yieldroll (fields 13-14): its own positions show it stands: may start a yield while ego rolls to its line under
      // this module's stop, and keeps a yield (never starts one at the line: there yieldsignal_1101's evidence decides)
      // (ego still moving: >= 0.3 m/s; a robot already standing at its line keeps yieldsignal_1101's evidence, and a
      // yield the reported speeds started keeps their test)
      const bool pos_was = nero_yi_pos_was_.was(ykey);
      const bool pos_yield = yv.pos_stand && !yv.yielding_motion &&
                             ((nero_yi_roll_d_ >= 0.0 && ego_v_yi >= 0.3 && !was_yielder) || pos_was);
      if ((yv.yielding_motion && (ego_v_yi < yp.ego_v || was_yielder || ego_rolls_to_line)) || pos_yield) {
        const auto & cp = nero_creep_ix();
        const double t_react = cp.v_creep > 0.0 ? cp.t_react : 0.5;
        const double t_pass = cp.v_creep > 0.0 ? cp.t_pass : 1.5;
        const double a_stop = cp.v_creep > 0.0 ? cp.a_stop : 2.0;
        // (a car that stands by its positions: its reach at the speed its positions allow at most)
        const double v = pos_yield ? std::min(yv.v, yp.r_pos / yp.t_pos) : yv.v;
        const double d_stop = v * (t_react + t_pass) + v * v / (2.0 * a_stop);
        auto stop_pose = predicted_object.kinematics.initial_pose_with_covariance.pose;
        const double yaw = tf2::getYaw(stop_pose.orientation);
        const double sgn = predicted_object.kinematics.initial_twist_with_covariance.twist.linear.x < 0.0 ? -1.0 : 1.0;
        stop_pose.position.x += sgn * d_stop * std::cos(yaw);
        stop_pose.position.y += sgn * d_stop * std::sin(yaw);
        const auto stop_poly = autoware_utils::to_polygon2d(stop_pose, predicted_object.shape);
        clear = bg::distance(stop_poly, nero_ego_path) - nero_half_width;
        is_queued = clear > yp.margin &&
                    nero_yield_intent::queued(predicted_object, *planner_data_->predicted_objects, nero_yi_was_);
        yielder = clear > yp.margin && !is_queued;
      }
      // NERO_IX_YIELD_REVOKE: a yield that ends because the car speeds up: no other exemption takes it as yielding
      if (nero_yield_revoke::param().on) {
        if (yielder) {
          nero_yield_revoke::mark(predicted_object, t_yi, true);
        } else if (was_yielder && (yv.gain_1s > yp.acc_th || yv.v > yp.v_crawl)) {
          nero_yield_revoke::mark(predicted_object, t_yi, false);
          RCLCPP_INFO(
            logger_,
            "[NERO_IX] yield_revoke: lane %ld %s drives on (v %.2f, gained %.2f in 1 s): no creep / priority exemption, "
            "pull-away model, a stop only where it ends short of his way",
            lane_id_, nero_hex_id(predicted_object).c_str(), yv.v, yv.gain_1s);
        }
      }
      const std::string yid = nero_hex_id(predicted_object);
      if (yielder) {
        if (!was_yielder) {
          RCLCPP_INFO(
            logger_,
            "[NERO_IX] yield_intent: lane %ld %s braked from %.1f m/s (%.1f s ago), stands %.1f s / crawls %.1f s at "
            "%.2f m/s (gained %.2f in 1 s), its stop %.2f m clear of ego's path (ego v %.2f): yielding, left out",
            lane_id_, yid.c_str(), yv.v_peak, yv.since_fast, yv.stand_s, yv.crawl_s, yv.v, yv.gain_1s, clear,
            ego_v_yi);
          if (ego_rolls_to_line || pos_yield) {
            RCLCPP_INFO(
              logger_,
              "[NERO_IX] yieldroll: lane %ld %s yields while the collision stop brings ego to its line (%.2f m "
              "ahead, ego v %.2f, roll gate %d), by its positions %d (stands %.2f s within %.2f m, reported %.2f)",
              lane_id_, yid.c_str(), nero_yi_roll_d_, ego_v_yi, ego_rolls_to_line ? 1 : 0, pos_yield ? 1 : 0,
              yv.pos_stand_s, yp.r_pos, yv.v);
          }
        }
        nero_yi_was_.now.insert(ykey);
        if (pos_yield || pos_was) nero_yi_pos_was_.now.insert(ykey);  // yieldroll: started by its positions
        nero_yi_yielders_.insert(uuidKey(predicted_object.object_id));
        continue;  // it braked for ego and waits: yielding
      }
      if (was_yielder) {
        RCLCPP_INFO(
          logger_,
          "[NERO_IX] yield_intent: lane %ld %s no longer yields (v %.2f, gained %.2f in 1 s, stands %.1f s / crawls "
          "%.1f s, clear %.2f%s, ego v %.2f): judged as any vehicle",
          lane_id_, yid.c_str(), yv.v, yv.gain_1s, yv.stand_s, yv.crawl_s, clear, is_queued ? ", queued" : "",
          ego_v_yi);
      }
    }
    if (nero_creep_ix().v_creep > 0.0 && nero_ego_path.size() >= 2 && !is_wrong_way_on_ego_lane) {
      double v = 0.0, acc = 0.0;
      if (
        nero_is_creeper(predicted_object, &v, &acc) &&
        !nero_yield_revoke::drives_on(  // NERO_IX_YIELD_REVOKE: a yielder that drives on is no creeper
          predicted_object, rclcpp::Time(planner_data_->predicted_objects->header.stamp).seconds())) {
        const auto & cp = nero_creep_ix();
        const double d_stop = v * (cp.t_react + cp.t_pass) + v * v / (2.0 * cp.a_stop);
        auto stop_pose = predicted_object.kinematics.initial_pose_with_covariance.pose;
        const double yaw = tf2::getYaw(stop_pose.orientation);
        const double sgn = predicted_object.kinematics.initial_twist_with_covariance.twist.linear.x < 0.0 ? -1.0 : 1.0;
        stop_pose.position.x += sgn * d_stop * std::cos(yaw);
        stop_pose.position.y += sgn * d_stop * std::sin(yaw);
        const auto stop_poly = autoware_utils::to_polygon2d(stop_pose, predicted_object.shape);
        const double clear = bg::distance(stop_poly, nero_ego_path) - nero_half_width;
        if (clear > cp.margin) {
          static double last_log = 0.0;
          const double now = nero_steady_now();
          if (now - last_log > 1.0) {
            last_log = now;
            RCLCPP_INFO(logger_, "[NERO_IX] creeper treated as yielding: v %.2f acc %.2f, stops %.2f m clear of ego's path",
              v, acc, clear);
          }
          nero_creep_last_exempt_ref() = nero_steady_now();
          continue;  // it creeps and can stop clear of ego's path: yielding
        }
      }
    }
    if (nero_priority_from_signs() && belong_attention_area && !is_wrong_way_on_ego_lane) {
      std::vector<size_t> pv_turns;
      const int verdict =
        neroPriorityVerdict(predicted_object, object_direction, is_parked_vehicle, &pv_turns);
      if (nero_priority_debug()) {
        nero_priority_dbg_text_[uuidKey(predicted_object.object_id)] =
          fmt::format("{} {}", verdict == 0 ? "WATCHED" : (verdict == 1 ? "YIELDING" : "POTENTIAL"),
            nero_priority_dbg_);
      }
      if (verdict != 0) {
        const auto key = uuidKey(predicted_object.object_id);
        if (verdict == 1) {
          nero_priority_yield_uuids_.insert(key);
        } else {
          nero_priority_pv_[key] = pv_turns;
          if (nero_priority_hard_) nero_priority_pv_late_.insert(key);
          if (nero_priority_no_exit_now_) {  // NERO_IX_PRIORITY_NO_EXIT
            if (nero_priority_no_exit_.count(key) == 0) {
              const auto & tw = predicted_object.kinematics.initial_twist_with_covariance.twist.linear;
              const double d_line = pv_turns.empty() ? -1.0 : [&] {
                const auto & turn = nero_priority_lanes_->at(pv_turns.front());
                double d = turn.stop_s - lanelet::utils::getArcCoordinates({turn.lanelet}, object_direction).length;
                for (const auto & lane : nero_priority_lanes_.value()) {  // on its approach: the rest + to its line
                  if (lane.kind != NeroPriorityLane::Kind::APPROACH_PRIORITY) continue;
                  if (!lanelet::utils::isInLanelet(object_direction, lane.lanelet, 0.0)) continue;
                  for (const auto & [y, dl] : lane.to_line) {
                    if (y != pv_turns.front()) continue;
                    d = lanelet::utils::getLaneletLength2d(lane.lanelet) -
                        lanelet::utils::getArcCoordinates({lane.lanelet}, object_direction).length + dl;
                  }
                }
                return d;
              }();
              RCLCPP_INFO(
                logger_,
                "[NERO_IX] priority no_exit: lane %ld, car %s at (%.1f, %.1f) v %.1f m/s, %.1f m to its line (turn "
                "%ld): faster than every way on from its approach (<= %.1f m/s), not braking, cannot stop at its "
                "line: a violator, its straight path is checked",
                lane_id_, nero_hex_id(predicted_object).c_str(), object_direction.position.x, object_direction.position.y,
                std::hypot(tw.x, tw.y), d_line,
                pv_turns.empty() ? 0L : nero_priority_lanes_->at(pv_turns.front()).lanelet.id(),
                pv_turns.empty() ? 0.0 : nero_priority_lanes_->at(pv_turns.front()).v_exit_max);
            }
            nero_priority_no_exit_[key] = nero_steady_now();
            nero_priority_no_exit_lane_yaws_[key] = nero_priority_no_exit_lane_yaw_;
          }
        }
        nero_priority_last_yield_ = nero_steady_now();
        static double last_log = 0.0;
        if (nero_priority_last_yield_ - last_log > 1.0) {
          last_log = nero_priority_last_yield_;
          RCLCPP_INFO(
            logger_, "[NERO_IX] priority: lane %ld, object at (%.1f, %.1f) v %.1f %s", lane_id_,
            object_direction.position.x, object_direction.position.y,
            std::hypot(
              predicted_object.kinematics.initial_twist_with_covariance.twist.linear.x,
              predicted_object.kinematics.initial_twist_with_covariance.twist.linear.y),
            verdict == 1 ? "must yield to ego (map stop line)"
                         : "could still turn across ego without stopping at its line: its turn is checked");
        }
      }
    }
    const auto object_it = old_map.find(predicted_object.object_id);
    if (object_it != old_map.end()) {
      auto object_info = object_it->second;
      object_info_manager_.registerExistingObject(
        predicted_object.object_id, belong_attention_area, in_intersection_area, is_parked_vehicle,
        object_info);
      object_info->initialize(predicted_object, attention_lanelet, stopline);
    } else {
      auto object_info = object_info_manager_.registerObject(
        predicted_object.object_id, belong_attention_area, in_intersection_area,
        is_parked_vehicle);
      object_info->initialize(predicted_object, attention_lanelet, stopline);
    }
  }
}

void IntersectionModule::updateObjectInfoManagerCollision(
  const PathLanelets & path_lanelets,
  const IntersectionModule::TimeDistanceArray & time_distance_array,
  const IntersectionModule::TrafficPrioritizedLevel & traffic_prioritized_level,
  const bool passed_1st_judge_line_first_time, const bool passed_2nd_judge_line_first_time,
  autoware_internal_debug_msgs::msg::Float64MultiArrayStamped * object_ttc_time_array)
{
  const auto & intersection_lanelets = intersection_lanelets_.value();
  nero_no_exit_bands_.clear();  // NERO_IX_PRIORITY_NO_EXIT

  if (passed_1st_judge_line_first_time) {
    object_info_manager_.setPassed1stPassJudgeLineFirstTime(clock_->now());
  }
  if (passed_2nd_judge_line_first_time) {
    object_info_manager_.setPassed2ndPassJudgeLineFirstTime(clock_->now());
  }

  const double passing_time = time_distance_array.back().first;
  const auto & concat_lanelets = path_lanelets.all;
  const auto closest_arc_coords =
    lanelet::utils::getArcCoordinates(concat_lanelets, planner_data_->current_odometry->pose);
  const auto & ego_lane = path_lanelets.ego_or_entry2exit;
  debug_data_.ego_lane = ego_lane.polygon3d();
  const auto ego_poly = ego_lane.polygon2d().basicPolygon();
  const auto nero_merge = nero_merge_param();
  const auto ego_centerline = ego_lane.centerline2d();
  const bool has_ego_exit_yaw = ego_centerline.size() >= 2;
  const double ego_exit_yaw =
    has_ego_exit_yaw ? std::atan2(
                         ego_centerline[ego_centerline.size() - 1].y() -
                           ego_centerline[ego_centerline.size() - 2].y(),
                         ego_centerline[ego_centerline.size() - 1].x() -
                           ego_centerline[ego_centerline.size() - 2].x())
                     : 0.0;

  // Nero c3 (wrong_way_watch): for wrong-way objects on ego's own lane the collision area is ego's
  // whole remaining path from its current position: ego_or_entry2exit plus path_lanelets.next
  // (the exit lane), because a head-on conflict with such a car happens on the exit lane, which
  // the stock ego_poly (assigned lane part only) does not cover.
  const auto & ww_param = planner_param_.collision_detection.wrong_way_watch;
  lanelet::BasicPolygon2d ego_forward_poly{};
  if (ww_param.enable && !wrong_way_ego_lane_object_uuids_.empty()) {
    const auto forward_poly3d = lanelet::utils::getPolygonFromArcLength(
      concat_lanelets, std::max(0.0, closest_arc_coords.length),
      lanelet::utils::getLaneletLength2d(concat_lanelets));
    for (const auto & p : forward_poly3d) {
      ego_forward_poly.emplace_back(p.x(), p.y());
    }
  }

  // ==========================================================================================
  // dynamically change TTC margin according to traffic light color to gradually relax from green to
  // red
  // ==========================================================================================
  const auto [collision_start_margin_time, collision_end_margin_time] = [&]() {
    if (traffic_prioritized_level == TrafficPrioritizedLevel::FULLY_PRIORITIZED) {
      return std::make_pair(
        planner_param_.collision_detection.fully_prioritized.collision_start_margin_time,
        planner_param_.collision_detection.fully_prioritized.collision_end_margin_time);
    }
    if (traffic_prioritized_level == TrafficPrioritizedLevel::PARTIALLY_PRIORITIZED) {
      return std::make_pair(
        planner_param_.collision_detection.partially_prioritized.collision_start_margin_time,
        planner_param_.collision_detection.partially_prioritized.collision_end_margin_time);
    }
    return std::make_pair(
      planner_param_.collision_detection.not_prioritized.collision_start_margin_time,
      planner_param_.collision_detection.not_prioritized.collision_end_margin_time);
  }();

  constexpr size_t object_debug_size = 57;
  {
    object_ttc_time_array->stamp = clock_->now();
    object_ttc_time_array->layout.dim.resize(3);
    object_ttc_time_array->layout.dim.at(0).label = "objects";
    object_ttc_time_array->layout.dim.at(0).size =
      1;  // incremented in the loop, first row is lane_id
    object_ttc_time_array->layout.dim.at(1).label =
      "[x, y, th, length, width, speed, dangerous, ref_obj_enter_time, ref_obj_exit_time, "
      "start_time, start_dist, "
      "end_time, end_dist, first_collision_x, first_collision_y, last_collision_x, "
      "last_collision_y, "
      "prd_x[0], ... pred_x[19], pred_y[0], ... pred_y[19]]";
    object_ttc_time_array->layout.dim.at(1).size = object_debug_size;
    for (unsigned i = 0; i < object_debug_size; ++i) {
      object_ttc_time_array->data.push_back(lane_id_);
    }
  }

  for (auto & object_info : object_info_manager_.attentionObjects()) {
    const auto & predicted_object = object_info->predicted_object();
    // Nero c3 (wrong_way_watch)
    const bool is_wrong_way_on_ego_lane =
      ww_param.enable && ego_forward_poly.size() >= 3 &&
      wrong_way_ego_lane_object_uuids_.count(uuidKey(predicted_object.object_id)) > 0;
    bool safe_under_traffic_control = false;
    const auto label = predicted_object.classification.at(0).label;
    const auto expected_deceleration =
      (label == autoware_perception_msgs::msg::ObjectClassification::MOTORCYCLE ||
       label == autoware_perception_msgs::msg::ObjectClassification::BICYCLE)
        ? planner_param_.collision_detection.ignore_on_amber_traffic_light
            .object_expected_deceleration.bike
        : planner_param_.collision_detection.ignore_on_amber_traffic_light
            .object_expected_deceleration.car;
    if (
      traffic_prioritized_level == TrafficPrioritizedLevel::PARTIALLY_PRIORITIZED &&
      object_info->can_stop_before_stopline(expected_deceleration)) {
      safe_under_traffic_control = true;
    }
    if (
      traffic_prioritized_level == TrafficPrioritizedLevel::FULLY_PRIORITIZED &&
      object_info->can_stop_before_ego_lane(
        expected_deceleration,
        planner_param_.collision_detection.ignore_on_red_traffic_light.object_margin_to_path,
        ego_lane)) {
      safe_under_traffic_control = true;
    }
    // NERO_IX_PRIORITY_FROM_SIGNS: it has to yield to ego and has not committed to crossing (a potential violator only
    // once ego is over its pass judge line: then a stop for a car that has not started to turn would be a late hard
    // brake; until then its turn paths are checked as upstream)
    const auto nero_pv_it = nero_priority_pv_.find(uuidKey(predicted_object.object_id));
    const bool nero_pv = nero_pv_it != nero_priority_pv_.end();
    if (
      nero_priority_yield_uuids_.count(uuidKey(predicted_object.object_id)) > 0 ||
      (nero_pv && nero_over_1st_pass_judge_ &&
       nero_priority_pv_late_.count(uuidKey(predicted_object.object_id)) == 0)) {
      safe_under_traffic_control = true;
    }
    if (is_wrong_way_on_ego_lane) {
      // Nero c3: a wrong-way driver is not expected to obey the traffic control of its lane
      safe_under_traffic_control = false;
    }
    // NERO_IX_YIELD_REVOKE: nor is a driver who yielded and drives on after all (its positions show the restart)
    if (
      safe_under_traffic_control &&
      nero_yield_revoke::drives_on(
        predicted_object, rclcpp::Time(planner_data_->predicted_objects->header.stamp).seconds())) {
      safe_under_traffic_control = false;
    }

    // ==========================================================================================
    // check the PredictedPath in the ascending order of its confidence to save the safe/unsafe
    // CollisionKnowledge for most probable path
    // ==========================================================================================
    std::list<const autoware_perception_msgs::msg::PredictedPath *> sorted_predicted_paths;
    for (unsigned i = 0; i < predicted_object.kinematics.predicted_paths.size(); ++i) {
      sorted_predicted_paths.push_back(&predicted_object.kinematics.predicted_paths.at(i));
    }
    sorted_predicted_paths.sort(
      [](const auto path1, const auto path2) { return path1->confidence > path2->confidence; });
    // NERO_IX_PULLAWAY_ACC: a vehicle that has just started from standstill is also checked as accelerating along
    // its attention lane (after its own predicted paths, which are all still checked)
    std::vector<autoware_perception_msgs::msg::PredictedPath> nero_pullaway;
    const double nero_pullaway_t =
      (nero_pullaway_acc() > 0.0 && !is_wrong_way_on_ego_lane && nero_pullaway_target(predicted_object))
        ? nero_pullaway_since(predicted_object)
        : -1.0;
    // NERO_IX_YIELD_REVOKE: a yielder that drives on is checked as pulling away from its restart (its positions)
    const bool nero_yr_pullaway =
      nero_pullaway_t < 0.0 && nero_pullaway_acc() > 0.0 && !is_wrong_way_on_ego_lane &&
      nero_pullaway_target(predicted_object) &&
      nero_yield_revoke::drives_on(
        predicted_object, rclcpp::Time(planner_data_->predicted_objects->header.stamp).seconds());
    if (nero_pullaway_t >= 0.0 || nero_yr_pullaway) {
      nero_pullaway = nero_pullaway_paths(
        predicted_object, intersection_lanelets.attention(), planner_param_.common.attention_area_margin,
        planner_param_.common.attention_area_angle_threshold);
      for (const auto & p : nero_pullaway) {
        sorted_predicted_paths.push_back(&p);
      }
    }
    const auto nero_is_pullaway_path = [&](const autoware_perception_msgs::msg::PredictedPath * p) {
      return !nero_pullaway.empty() && p >= nero_pullaway.data() && p < nero_pullaway.data() + nero_pullaway.size();
    };
    // NERO_IX_PRIORITY_NO_EXIT: a car flagged within the last 2 s (it can take no way on) is also checked on its
    // constant-velocity straight path from where it is now, not only on the lane paths of its prediction
    autoware_perception_msgs::msg::PredictedPath nero_no_exit_path;
    if (!nero_priority_no_exit_.empty() && !is_wrong_way_on_ego_lane) {
      const auto it = nero_priority_no_exit_.find(uuidKey(predicted_object.object_id));
      if (it != nero_priority_no_exit_.end() && nero_steady_now() - it->second < 2.0) {
        const auto & pose = predicted_object.kinematics.initial_pose_with_covariance.pose;
        const auto & tw = predicted_object.kinematics.initial_twist_with_covariance.twist.linear;
        double yaw = tf2::getYaw(pose.orientation) + std::atan2(tw.y, tw.x);  // direction of motion
        // ... from its positions over the last second when there are >= 0.5 s of them (the heading is noisier);
        // until then the band also covers the direction of its lane (a first heading can be several degrees off)
        bool yaw_from_positions = false;
        const auto xy_it = nero_priority_xy_hist_.find(it->first);
        if (xy_it != nero_priority_xy_hist_.end() && !xy_it->second.empty()) {
          const auto & h = xy_it->second;
          const auto & last = h.back();
          for (const auto & q : h) {
            if (q[0] < last[0] - 1.0 - 1e-3) continue;
            if (last[0] - q[0] >= 0.5 && std::hypot(last[1] - q[1], last[2] - q[2]) > 1.0) {
              yaw = std::atan2(last[2] - q[2], last[1] - q[1]);
              yaw_from_positions = true;
            }
            break;
          }
        }
        const double v = std::hypot(tw.x, tw.y);
        constexpr double dt = 0.25, horizon = 10.0;
        nero_no_exit_path.time_step = rclcpp::Duration::from_seconds(dt);
        nero_no_exit_path.confidence = 1.0;
        for (double t = 0.0; t <= horizon + 1e-6; t += dt) {
          geometry_msgs::msg::Pose p = pose;
          p.position.x += v * t * std::cos(yaw);
          p.position.y += v * t * std::sin(yaw);
          p.orientation = autoware_utils::create_quaternion_from_yaw(yaw);
          nero_no_exit_path.path.push_back(p);
        }
        sorted_predicted_paths.push_back(&nero_no_exit_path);
        // its straight swept band (the box now and 6 s on): where ego must keep the margin from (neroNoExitStopIndex)
        auto end_pose = pose;
        end_pose.position.x += v * 6.0 * std::cos(yaw);
        end_pose.position.y += v * 6.0 * std::sin(yaw);
        end_pose.orientation = autoware_utils::create_quaternion_from_yaw(yaw);
        auto start_pose = pose;
        start_pose.orientation = end_pose.orientation;
        autoware_utils::MultiPoint2d pts;
        std::vector<geometry_msgs::msg::Pose> band_poses{start_pose, end_pose};
        const auto ly_it = nero_priority_no_exit_lane_yaws_.find(it->first);
        if (!yaw_from_positions && ly_it != nero_priority_no_exit_lane_yaws_.end()) {
          auto lane_end = pose;  // the box 6 s on along its lane's direction
          lane_end.position.x += v * 6.0 * std::cos(ly_it->second);
          lane_end.position.y += v * 6.0 * std::sin(ly_it->second);
          lane_end.orientation = autoware_utils::create_quaternion_from_yaw(ly_it->second);
          band_poses.push_back(lane_end);
        }
        for (const auto & q : band_poses) {
          const auto box = autoware_utils::to_polygon2d(q, predicted_object.shape);  // named: outer() of a temporary dangles
          for (const auto & c : box.outer()) pts.push_back(c);
        }
        autoware_utils::Polygon2d band;
        bg::convex_hull(pts, band);
        nero_no_exit_bands_[uuidKey(predicted_object.object_id)] = band;
      }
    }
    bool nero_pullaway_unsafe = false;

    // ==========================================================================================
    // if all of the predicted path is lower confidence/geometrically does not intersect with ego
    // path, both will be null, which is interpreted as SAFE. if any of the path is "normal", either
    // of them has value, not both
    // ==========================================================================================
    std::optional<CollisionInterval> unsafe_interval{std::nullopt};
    std::optional<CollisionInterval> safe_interval{std::nullopt};
    std::optional<std::vector<double>> object_debug_info{std::nullopt};

    for (const auto & predicted_path_ptr : sorted_predicted_paths) {
      auto predicted_path = *predicted_path_ptr;
      if (
        predicted_path.confidence <
        planner_param_.collision_detection.min_predicted_path_confidence) {
        continue;
      }
      if (nero_pv && !safe_under_traffic_control && predicted_path_ptr != &nero_no_exit_path) {
        // potential violator: only the paths that take one of its turns past the stop line are checked
        const auto & nero_lanes = nero_priority_lanes_.value();
        const bool takes_turn = std::any_of(
          predicted_path.path.begin(), predicted_path.path.end(), [&](const auto & pose) {
            return std::any_of(nero_pv_it->second.begin(), nero_pv_it->second.end(), [&](const size_t y) {
              const auto & turn = nero_lanes.at(y);
              return lanelet::utils::isInLanelet(pose, turn.lanelet, 0.0) &&
                     lanelet::utils::getArcCoordinates({turn.lanelet}, pose).length > turn.stop_s;
            });
          });
        if (!takes_turn) continue;
      }
      if (is_wrong_way_on_ego_lane) {
        // ==========================================================================================
        // Nero c3 (wrong_way_watch): the object drives against ego's lane, so any overlap of its
        // predicted footprint with ego's remaining path (from ego to the end of the exit lane) is a
        // head-on conflict, whatever the timing. ego's time/distance profile only spans the
        // intersection, so the stock TTC window is not used here. The horizon is at least
        // prediction_horizon so that a conflict further down the exit lane is still seen.
        // ==========================================================================================
        cutPredictPathWithinDuration(
          planner_data_->predicted_objects->header.stamp,
          std::max(passing_time, ww_param.prediction_horizon), &predicted_path);
        const auto ww_interval_opt = findPassageInterval(
          predicted_path, predicted_object.shape, ego_forward_poly,
          intersection_lanelets.first_attention_lane(),
          intersection_lanelets.second_attention_lane());
        if (ww_interval_opt) {
          safe_interval = std::nullopt;
          unsafe_interval = ww_interval_opt;
          break;
        }
        continue;
      }
      // junction-yield 10-11 (autoware_universe#11774): with the flag the horizon covers the start margin, so an
      // object entering up to collision_start_margin_time after the ego's (modelled) exit is still checked
      // NERO: NERO_IX_CUT_EXTRA=<s> keeps the predicted path up to passing_time + s (NERO_IX_MERGE: its max).
      // Upstream 0.44.2 cuts at passing_time, so a car that reaches the merge into ego's exit lane just after
      // ego leaves the turn lanelet is never a conflict, although it then closes on the slow ego from behind.
      // With both this and the start-margin flag set, the longer horizon is used.
      const double prediction_horizon =
        passing_time +
        std::max(
          planner_param_.collision_detection.cut_predicted_path_with_start_margin
            ? collision_start_margin_time
            : 0.0,
          nero_merge.on ? nero_merge.max : nero_cut_extra());
      cutPredictPathWithinDuration(
        planner_data_->predicted_objects->header.stamp, prediction_horizon, &predicted_path);
      const auto object_passage_interval_opt = findPassageInterval(
        predicted_path, predicted_object.shape, ego_poly,
        intersection_lanelets.first_attention_lane(),
        intersection_lanelets.second_attention_lane());
      if (!object_passage_interval_opt) {
        // there is no chance of geometric collision for the entire prediction horizon
        continue;
      }
      const auto & object_passage_interval = object_passage_interval_opt.value();
      const auto object_enter_exit_time = object_passage_interval.interval_time;
      const auto object_enter_time = std::get<0>(object_enter_exit_time);
      const auto object_exit_time = std::get<1>(object_enter_exit_time);
      bool nero_force_unsafe = false;
      if (nero_merge.on && object_enter_time > passing_time) {
        // reaches ego's turn lanelet only after ego has left it
        const auto & last_pose =
          object_passage_interval.path.at(object_passage_interval.interval_position.second);
        const double dyaw =
          autoware_utils::normalize_radian(tf2::getYaw(last_pose.orientation) - ego_exit_yaw);
        if (!has_ego_exit_yaw || std::fabs(dyaw) > M_PI / 4.0) {
          continue;  // crosses where ego has already been: no conflict (as upstream)
        }
        // NERO_IX_PULLAWAY_ACC: a model path arrives at the model's speed, not the reported one
        const double v_obj =
          nero_is_pullaway_path(predicted_path_ptr)
            ? nero_pullaway_speed(
                std::hypot(
                  predicted_object.kinematics.initial_twist_with_covariance.twist.linear.x,
                  predicted_object.kinematics.initial_twist_with_covariance.twist.linear.y),
                nero_pullaway_acc(), object_enter_time)
            : std::max(0.0, predicted_object.kinematics.initial_twist_with_covariance.twist.linear.x);
        const double headway = std::min(
          nero_merge.max,
          nero_merge.react + std::max(0.0, v_obj - nero_merge.v_ref) / (2.0 * nero_merge.decel));
        if (object_enter_time >= passing_time + headway) {
          continue;  // far enough behind ego to follow it comfortably
        }
        nero_force_unsafe = true;  // would close on ego from behind harder than `decel`
      }
      auto ego_start_itr = std::lower_bound(
        time_distance_array.begin(), time_distance_array.end(),
        object_enter_time - collision_start_margin_time,
        [](const auto & a, const double b) { return a.first < b; });
      if (ego_start_itr == time_distance_array.end() && nero_force_unsafe) {
        ego_start_itr = time_distance_array.end() - 1;
      }
      if (ego_start_itr == time_distance_array.end()) {
        // ==========================================================================================
        // this is the case where at time "object_enter_time - collision_start_margin_time", ego is
        // arriving at the exit of the intersection, which means even if we assume that the object
        // accelerates and the first collision happens faster by the TTC margin, ego will be already
        // arriving at the exist of the intersection.
        // ==========================================================================================
        continue;
      }
      auto ego_end_itr = std::lower_bound(
        time_distance_array.begin(), time_distance_array.end(),
        object_exit_time + collision_end_margin_time,
        [](const auto & a, const double b) { return a.first < b; });
      if (ego_end_itr == time_distance_array.end()) {
        ego_end_itr = time_distance_array.end() - 1;
      }
      const double ego_start_arc_length = std::max(
        0.0, closest_arc_coords.length + ego_start_itr->second -
               planner_data_->vehicle_info_.rear_overhang_m);
      const double ego_end_arc_length = std::min(
        closest_arc_coords.length + ego_end_itr->second +
          planner_data_->vehicle_info_.max_longitudinal_offset_m,
        lanelet::utils::getLaneletLength2d(concat_lanelets));
      const auto trimmed_ego_polygon = lanelet::utils::getPolygonFromArcLength(
        concat_lanelets, ego_start_arc_length, ego_end_arc_length);
      if (trimmed_ego_polygon.empty() && !nero_force_unsafe) {
        continue;
      }
      Polygon2d polygon{};
      for (const auto & p : trimmed_ego_polygon) {
        polygon.outer().emplace_back(p.x(), p.y());
      }
      bg::correct(polygon);
      debug_data_.candidate_collision_ego_lane_polygon = toGeomPoly(polygon);

      const auto & object_path = object_passage_interval.path;
      const auto [begin, end] = object_passage_interval.interval_position;
      bool collision_detected = nero_force_unsafe;
      for (auto i = begin; i <= end && !collision_detected; ++i) {
        if (bg::intersects(
              polygon, autoware_utils::to_polygon2d(object_path.at(i), predicted_object.shape))) {
          collision_detected = true;
          break;
        }
      }
      auto get_object_info = [&]() {
        // debug info
        const auto & pose = predicted_object.kinematics.initial_pose_with_covariance.pose;
        const auto & shape = predicted_object.shape;
        const auto [object_start_itr, object_end_itr] = object_passage_interval.interval_position;
        const auto & object_start_pos = object_passage_interval.path.at(object_start_itr).position;
        const auto & object_end_pos = object_passage_interval.path.at(object_end_itr).position;
        std::vector<double> debug;
        debug.reserve(object_debug_size);

        debug.insert(
          debug.end(),
          {pose.position.x, pose.position.y, tf2::getYaw(pose.orientation), shape.dimensions.x,
           shape.dimensions.y,
           predicted_object.kinematics.initial_twist_with_covariance.twist.linear.x,
           1.0 * static_cast<int>(collision_detected), object_enter_time, object_exit_time,
           ego_start_itr->first, ego_start_itr->second, ego_end_itr->first, ego_end_itr->second,
           object_start_pos.x, object_start_pos.y, object_end_pos.x, object_end_pos.y});
        for (unsigned i = 0; i < 20; ++i) {
          const auto & pos = object_passage_interval.path
                               .at(std::min<size_t>(i, object_passage_interval.path.size() - 1))
                               .position;
          debug.push_back(pos.x);
          debug.push_back(pos.y);
        }
        return debug;
      };

      if (collision_detected) {
        // if judged as UNSAFE, return
        safe_interval = std::nullopt;
        unsafe_interval = object_passage_interval;
        object_debug_info = get_object_info();
        nero_pullaway_unsafe = nero_is_pullaway_path(predicted_path_ptr);
        break;
      }
      if (!safe_interval) {
        // ==========================================================================================
        // save the safe_decision_knowledge for the most probable path. this value is nullified if
        // judged UNSAFE during the iteration
        // ==========================================================================================
        safe_interval = object_passage_interval;
        object_debug_info = get_object_info();
      }
    }
    object_info->update_safety(unsafe_interval, safe_interval, safe_under_traffic_control);
    if (nero_pullaway_t >= 0.0) {
      auto & h = nero_pullaway_hists()[nero_uuid_key(predicted_object)];
      const auto & pos = predicted_object.kinematics.initial_pose_with_covariance.pose.position;
      const double v = std::hypot(
        predicted_object.kinematics.initial_twist_with_covariance.twist.linear.x,
        predicted_object.kinematics.initial_twist_with_covariance.twist.linear.y);
      if (nero_pullaway_unsafe && !h.logged_unsafe) {
        h.logged_unsafe = true;
        RCLCPP_INFO(
          logger_,
          "[NERO_IX] pullaway: lane %ld, object at (%.1f, %.1f) v %.2f started %.2f s ago: collision as it "
          "accelerates at %.1f m/s2 (enters ego's lane in %.1f s)%s",
          lane_id_, pos.x, pos.y, v, nero_pullaway_t, nero_pullaway_acc(),
          unsafe_interval->interval_time.first, safe_under_traffic_control ? ", but it yields (traffic control)" : "");
      } else {
        static double last_log = 0.0;
        const double now = nero_steady_now();
        if (now - last_log > 1.0) {
          last_log = now;
          RCLCPP_INFO(
            logger_, "[NERO_IX] pullaway: lane %ld, object at (%.1f, %.1f) v %.2f started %.2f s ago: %zu lane path(s), %s",
            lane_id_, pos.x, pos.y, v, nero_pullaway_t, nero_pullaway.size(),
            unsafe_interval ? (nero_pullaway_unsafe ? "unsafe (model)" : "unsafe (its own prediction)") : "safe");
        }
      }
    }
    if (passed_1st_judge_line_first_time) {
      object_info->setDecisionAt1stPassJudgeLinePassage(CollisionKnowledge{
        clock_->now(),  // stamp
        unsafe_interval
          ? CollisionKnowledge::SafeType::UNSAFE
          : (safe_under_traffic_control ? CollisionKnowledge::SafeType::SAFE_UNDER_TRAFFIC_CONTROL
                                        : CollisionKnowledge::SafeType::SAFE),  // safe
        unsafe_interval ? unsafe_interval : safe_interval,                      // interval
        predicted_object.kinematics.initial_twist_with_covariance.twist.linear
          .x  // observed_velocity
      });
    }
    if (passed_2nd_judge_line_first_time) {
      object_info->setDecisionAt2ndPassJudgeLinePassage(CollisionKnowledge{
        clock_->now(),  // stamp
        unsafe_interval
          ? CollisionKnowledge::SafeType::UNSAFE
          : (safe_under_traffic_control ? CollisionKnowledge::SafeType::SAFE_UNDER_TRAFFIC_CONTROL
                                        : CollisionKnowledge::SafeType::SAFE),  // safe
        unsafe_interval ? unsafe_interval : safe_interval,                      // interval
        predicted_object.kinematics.initial_twist_with_covariance.twist.linear
          .x  // observed_velocity
      });
    }

    if (nero_priority_debug()) {  // NERO_IX_PRIORITY_DEBUG (log only)
      const auto dbg_it = nero_priority_dbg_text_.find(uuidKey(predicted_object.object_id));
      if (dbg_it != nero_priority_dbg_text_.end()) {
        const auto & pos = predicted_object.kinematics.initial_pose_with_covariance.pose.position;
        RCLCPP_INFO(
          logger_, "[NERO_IX] priority dbg: lane %ld obj %02x%02x at (%.1f, %.1f) %s | ego v %.2f pj %.1f m over %d | %s %s",
          lane_id_, predicted_object.object_id.uuid[0], predicted_object.object_id.uuid[1], pos.x, pos.y,
          dbg_it->second.c_str(), planner_data_->current_velocity->twist.linear.x,
          nero_dist_to_1st_pass_judge_, nero_over_1st_pass_judge_ ? 1 : 0,
          safe_under_traffic_control ? "safe_tc" : "checked",
          unsafe_interval ? fmt::format("UNSAFE {:.1f}-{:.1f} s", unsafe_interval->interval_time.first,
                                        unsafe_interval->interval_time.second).c_str()
                          : "safe");
      }
    }

    // debug
    if (object_debug_info) {
      const auto & data = object_debug_info.value();
      object_ttc_time_array->layout.dim.at(0).size++;
      std::copy(data.begin(), data.end(), std::back_inserter(object_ttc_time_array->data));
    }
  }
}

void IntersectionModule::cutPredictPathWithinDuration(
  const builtin_interfaces::msg::Time & object_stamp, const double time_thr,
  autoware_perception_msgs::msg::PredictedPath * path) const
{
  const rclcpp::Time current_time = clock_->now();
  const auto original_path = path->path;
  path->path.clear();

  for (size_t k = 0; k < original_path.size(); ++k) {  // each path points
    const auto & predicted_pose = original_path.at(k);
    const auto predicted_time =
      rclcpp::Time(object_stamp) + rclcpp::Duration(path->time_step) * static_cast<double>(k);
    if ((predicted_time - current_time).seconds() < time_thr) {
      path->path.push_back(predicted_pose);
    }
  }
}

std::optional<NonOccludedCollisionStop> IntersectionModule::isGreenPseudoCollisionStatus(
  const size_t closest_idx, const size_t collision_stopline_idx,
  const IntersectionStopLines & intersection_stoplines) const
{
  // ==========================================================================================
  // if there are any vehicles on the attention area when ego entered the intersection on green
  // light, do pseudo collision detection because collision is likely to happen.
  // ==========================================================================================
  if (initial_green_light_observed_time_) {
    const auto now = clock_->now();
    const bool still_wait =
      (rclcpp::Duration((now - initial_green_light_observed_time_.value())).seconds() <
       planner_param_.collision_detection.yield_on_green_traffic_light.duration);
    if (!still_wait) {
      return std::nullopt;
    }
    const auto & attention_objects = object_info_manager_.attentionObjects();
    const bool exist_close_vehicles = std::any_of(
      attention_objects.begin(), attention_objects.end(), [&](const auto & object_info) {
        return object_info->before_stopline_by(
          planner_param_.collision_detection.yield_on_green_traffic_light.object_dist_to_stopline);
      });
    if (exist_close_vehicles) {
      const auto occlusion_stopline_idx = intersection_stoplines.occlusion_peeking_stopline.value();
      return NonOccludedCollisionStop{
        closest_idx, collision_stopline_idx, occlusion_stopline_idx, std::string("")};
    }
  }
  return std::nullopt;
}

std::string IntersectionModule::generateDetectionBlameDiagnosis(
  const std::vector<
    std::pair<IntersectionModule::CollisionStatus::BlameType, std::shared_ptr<ObjectInfo>>> &
    too_late_detect_objects,
  const std::vector<
    std::pair<IntersectionModule::CollisionStatus::BlameType, std::shared_ptr<ObjectInfo>>> &
    misjudge_objects) const
{
  std::string diag;
  if (!safely_passed_1st_judge_line_time_) {
    return diag;
  }
  const auto [passed_1st_judge_line_time, passed_1st_judge_line_pose] =
    safely_passed_1st_judge_line_time_.value();
  const auto passed_1st_judge_line_time_double =
    static_cast<double>(passed_1st_judge_line_time.nanoseconds()) / 1e+9;

  const auto now = clock_->now();
  const auto now_double = static_cast<double>(now.nanoseconds()) / 1e+9;

  // CAVEAT: format library causes runtime fault if the # of placeholders is more than the # of
  // vargs
  for (const auto & [blame_type, object_info] : too_late_detect_objects) {
    if (
      blame_type == CollisionStatus::BLAME_AT_FIRST_PASS_JUDGE && object_info->unsafe_interval()) {
      const auto & unsafe_interval = object_info->unsafe_interval().value();
      const double time_diff = now_double - passed_1st_judge_line_time_double;
      diag += fmt::format(
        "object {0} was not detected when ego passed the 1st pass judge line at {1}, but now at "
        "{2}, collision is detected after {3}~{4} seconds on the first attention lanelet of type "
        "{5}.\n",
        object_info->uuid_str,                                // 0
        passed_1st_judge_line_time_double,                    // 1
        now_double,                                           // 2
        unsafe_interval.interval_time.first,                  // 3
        unsafe_interval.interval_time.second,                 // 4
        magic_enum::enum_name(unsafe_interval.lane_position)  // 5
      );
      const auto past_position_opt = object_info->estimated_past_position(time_diff);
      if (past_position_opt) {
        const auto & past_position = past_position_opt.value();
        diag += fmt::format(
          "this object is estimated to have been at x = {0}, y = {1} when ego passed the 1st pass "
          "judge line({2} seconds before from now) given the estimated current velocity {3}[m/s]. "
          "ego was at x = {4}, y = {5} when it passed the 1st pass judge line so it is the fault "
          "of detection side that failed to detect around {6}[m] range at that time.\n",
          past_position.x,                                                            // 0
          past_position.y,                                                            // 1
          time_diff,                                                                  // 2
          object_info->observed_velocity(),                                           // 3
          passed_1st_judge_line_pose.position.x,                                      // 4
          passed_1st_judge_line_pose.position.y,                                      // 5
          autoware_utils::calc_distance2d(passed_1st_judge_line_pose, past_position)  // 6
        );
      }
    }
    if (
      safely_passed_2nd_judge_line_time_ &&
      blame_type == CollisionStatus::BLAME_AT_SECOND_PASS_JUDGE && object_info->unsafe_interval()) {
      const auto [passed_2nd_judge_line_time, passed_2nd_judge_line_pose] =
        safely_passed_2nd_judge_line_time_.value();
      const auto passed_2nd_judge_line_time_double =
        static_cast<double>(passed_2nd_judge_line_time.nanoseconds()) / 1e+9;

      const auto & unsafe_interval = object_info->unsafe_interval().value();
      const double time_diff = now_double - passed_2nd_judge_line_time_double;
      diag += fmt::format(
        "object {0} was not detected when ego passed the 2nd pass judge line at {1}, but now at "
        "{2}, collision is detected after {3}~{4} seconds on the lanelet of type {5}.\n",
        object_info->uuid_str,                                // 0
        passed_2nd_judge_line_time_double,                    // 1
        now_double,                                           // 2
        unsafe_interval.interval_time.first,                  // 3
        unsafe_interval.interval_time.second,                 // 4
        magic_enum::enum_name(unsafe_interval.lane_position)  // 5
      );
      const auto past_position_opt = object_info->estimated_past_position(time_diff);
      if (past_position_opt) {
        const auto & past_position = past_position_opt.value();
        diag += fmt::format(
          "this object is estimated to have been at x = {0}, y = {1} when ego passed the 2nd pass "
          "judge line({2} seconds before from now) given the estimated current velocity {3}[m/s]. "
          "ego was at x = {4}, y = {5} when it passed the 2nd pass judge line so it is the fault "
          "of detection side that failed to detect around {6}[m] range at that time.\n",
          past_position.x,                                                            // 0
          past_position.y,                                                            // 1
          time_diff,                                                                  // 2
          object_info->observed_velocity(),                                           // 3
          passed_2nd_judge_line_pose.position.x,                                      // 4
          passed_2nd_judge_line_pose.position.y,                                      // 5
          autoware_utils::calc_distance2d(passed_2nd_judge_line_pose, past_position)  // 6
        );
      }
    }
  }
  for (const auto & [blame_type, object_info] : misjudge_objects) {
    if (
      blame_type == CollisionStatus::BLAME_AT_FIRST_PASS_JUDGE && object_info->unsafe_interval() &&
      object_info->decision_at_1st_pass_judge_line_passage()) {
      const auto & decision_at_1st_pass_judge_line =
        object_info->decision_at_1st_pass_judge_line_passage().value();
      const auto decision_at_1st_pass_judge_line_time =
        static_cast<double>(decision_at_1st_pass_judge_line.stamp.nanoseconds()) / 1e+9;
      const auto & unsafe_interval = object_info->unsafe_interval().value();
      diag += fmt::format(
        "object {0} was judged as {1} when ego passed the 1st pass judge line at time {2} "
        "previously with the estimated velocity {3}[m/s], but now at {4} collision is detected "
        "after {5}~{6} seconds on the first attention lanelet of type {7} with the estimated "
        "current velocity {8}[m/s]\n",
        object_info->uuid_str,                                             // 0
        magic_enum::enum_name(decision_at_1st_pass_judge_line.safe_type),  // 1
        decision_at_1st_pass_judge_line_time,                              // 2
        decision_at_1st_pass_judge_line.observed_velocity,                 // 3
        now_double,                                                        // 4
        unsafe_interval.interval_time.first,                               // 5
        unsafe_interval.interval_time.second,                              // 6
        magic_enum::enum_name(unsafe_interval.lane_position),              // 7
        object_info->observed_velocity()                                   // 8
      );
    }
    if (
      blame_type == CollisionStatus::BLAME_AT_SECOND_PASS_JUDGE && object_info->unsafe_interval() &&
      object_info->decision_at_2nd_pass_judge_line_passage()) {
      const auto & decision_at_2nd_pass_judge_line =
        object_info->decision_at_2nd_pass_judge_line_passage().value();
      const auto decision_at_2nd_pass_judge_line_time =
        static_cast<double>(decision_at_2nd_pass_judge_line.stamp.nanoseconds()) / 1e+9;
      const auto & unsafe_interval = object_info->unsafe_interval().value();
      diag += fmt::format(
        "object {0} was judged as {1} when ego passed the 2nd pass judge line at time {2} "
        "previously with the estimated velocity {3}[m/s], but now at {4} collision is detected "
        "after {5}~{6} seconds on the lanelet of type {7} with the estimated current velocity "
        "{8}[m/s]\n",
        object_info->uuid_str,                                             // 0
        magic_enum::enum_name(decision_at_2nd_pass_judge_line.safe_type),  // 1
        decision_at_2nd_pass_judge_line_time,                              // 2
        decision_at_2nd_pass_judge_line.observed_velocity,                 // 3
        now_double,                                                        // 4
        unsafe_interval.interval_time.first,                               // 5
        unsafe_interval.interval_time.second,                              // 6
        magic_enum::enum_name(unsafe_interval.lane_position),              // 7
        object_info->observed_velocity()                                   // 8
      );
    }
  }
  return diag;
}

std::string IntersectionModule::generateEgoRiskEvasiveDiagnosis(
  const autoware_internal_planning_msgs::msg::PathWithLaneId & path, const size_t closest_idx,
  const IntersectionModule::TimeDistanceArray & ego_time_distance_array,
  const std::vector<
    std::pair<IntersectionModule::CollisionStatus::BlameType, std::shared_ptr<ObjectInfo>>> &
    too_late_detect_objects,
  [[maybe_unused]] const std::vector<
    std::pair<IntersectionModule::CollisionStatus::BlameType, std::shared_ptr<ObjectInfo>>> &
    misjudge_objects) const
{
  static constexpr double min_vel = 1e-2;
  std::string diag;
  const double ego_vel = planner_data_->current_velocity->twist.linear.x;
  for (const auto & [blame_type, object_info] : too_late_detect_objects) {
    if (!object_info->unsafe_interval()) {
      continue;
    }
    // object side
    const auto & unsafe_interval = object_info->unsafe_interval().value();
    const auto [begin, end] = unsafe_interval.interval_position;
    const auto &p1 = unsafe_interval.path.at(begin).position,
               p2 = unsafe_interval.path.at(end).position;
    const auto collision_pos =
      autoware_utils::create_point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2);
    const auto object_dist_to_margin_point =
      autoware_utils::calc_distance2d(
        object_info->predicted_object().kinematics.initial_pose_with_covariance.pose.position,
        collision_pos) -
      planner_param_.collision_detection.avoid_collision_by_acceleration
          .object_time_margin_to_collision_point *
        object_info->observed_velocity();
    if (object_dist_to_margin_point <= 0.0) {
      continue;
    }
    const auto object_eta_to_margin_point =
      object_dist_to_margin_point / std::max(min_vel, object_info->observed_velocity());
    // ego side
    const double ego_dist_to_collision_pos =
      autoware::motion_utils::calcSignedArcLength(path.points, closest_idx, collision_pos);
    const auto ego_eta_to_collision_pos_it = std::lower_bound(
      ego_time_distance_array.begin(), ego_time_distance_array.end(), ego_dist_to_collision_pos,
      [](const auto & a, const double b) { return a.second < b; });
    if (ego_eta_to_collision_pos_it == ego_time_distance_array.end()) {
      continue;
    }
    const double ego_eta_to_collision_pos = ego_eta_to_collision_pos_it->first;
    if (ego_eta_to_collision_pos < object_eta_to_margin_point) {
      // this case, ego will pass the collision point before this object get close to the collision
      // point within margin just by keeping current plan, so no need to accelerate
      continue;
    }
    diag += fmt::format(
      "the object is expected to approach the collision point by the TTC margin in {0} seconds, "
      "while ego will arrive there in {1} seconds, so ego needs to accelerate from current "
      "velocity {2}[m/s] to {3}[m/s]\n",
      object_eta_to_margin_point,                             // 0
      ego_eta_to_collision_pos,                               // 1
      ego_vel,                                                // 2
      ego_dist_to_collision_pos / object_eta_to_margin_point  // 3
    );
  }
  return diag;
}

IntersectionModule::CollisionStatus IntersectionModule::detectCollision(
  const bool is_over_1st_pass_judge_line,
  const std::optional<bool> is_over_2nd_pass_judge_line) const
{
  // ==========================================================================================
  // if collision is detected for multiple objects, we prioritize collision on the first
  // attention lanelet
  // ==========================================================================================
  bool collision_at_first_lane = false;
  bool collision_at_non_first_lane = false;

  // ==========================================================================================
  // find the objects which are judged as UNSAFE after ego passed pass judge lines.
  //
  // misjudge_objects are those that were once judged as safe when ego passed the pass judge line
  //
  // too_late_detect_objects are those that (1) were not detected when ego passed the pass judge
  // line (2) were judged as dangerous at the same time when ego passed the pass judge line, which
  // means they were expected to have been detected when ego passed the pass judge lines or in the
  // prior iteration, because ego could have judged them as UNSAFE if their information was
  // available at that time.
  //
  // that case is both "too late to stop" and "too late to go" for the planner. and basically
  // detection side is responsible for this fault
  // ==========================================================================================
  std::vector<std::pair<CollisionStatus::BlameType, std::shared_ptr<ObjectInfo>>> misjudge_objects;
  std::vector<std::pair<CollisionStatus::BlameType, std::shared_ptr<ObjectInfo>>>
    too_late_detect_objects;
  for (const auto & object_info : object_info_manager_.attentionObjects()) {
    if (object_info->is_safe_under_traffic_control()) {
      debug_data_.safe_under_traffic_control_targets.objects.push_back(
        object_info->predicted_object());
      continue;
    }
    const auto unsafe_info_opt = object_info->unsafe_info();
    if (!unsafe_info_opt) {
      continue;
    }
    const auto & unsafe_info = unsafe_info_opt.value();
    // ==========================================================================================
    // if ego is over the pass judge lines, then the visualization as "too_late_objects" or
    // "misjudge_objects" is more important than that for "unsafe"
    //
    // NOTE: consider a vehicle which was not detected at 1st_pass_judge_passage, and now collision
    // detected on the 1st lane, which is "too_late" for 1st lane passage, but once it decelerated
    // or yielded, so it turned safe, and ego passed the 2nd pass judge line, but at the same it
    // accelerated again, which is "misjudge" for 2nd lane passage. In this case this vehicle is
    // visualized as "misjudge"
    // ==========================================================================================
    auto * debug_container = &debug_data_.unsafe_targets.objects;
    if (unsafe_info.lane_position == CollisionInterval::LanePosition::FIRST) {
      collision_at_first_lane = true;
    } else {
      collision_at_non_first_lane = true;
    }
    if (
      is_over_1st_pass_judge_line &&
      unsafe_info.lane_position == CollisionInterval::LanePosition::FIRST) {
      const auto & decision_at_1st_pass_judge_opt =
        object_info->decision_at_1st_pass_judge_line_passage();
      if (!decision_at_1st_pass_judge_opt) {
        too_late_detect_objects.emplace_back(
          CollisionStatus::BlameType::BLAME_AT_FIRST_PASS_JUDGE, object_info);
        debug_container = &debug_data_.too_late_detect_targets.objects;
      } else {
        const auto & decision_at_1st_pass_judge = decision_at_1st_pass_judge_opt.value();
        if (decision_at_1st_pass_judge.safe_type != CollisionKnowledge::SafeType::UNSAFE) {
          misjudge_objects.emplace_back(
            CollisionStatus::BlameType::BLAME_AT_FIRST_PASS_JUDGE, object_info);
          debug_container = &debug_data_.misjudge_targets.objects;
        } else {
          too_late_detect_objects.emplace_back(
            CollisionStatus::BlameType::BLAME_AT_FIRST_PASS_JUDGE, object_info);
          debug_container = &debug_data_.too_late_detect_targets.objects;
        }
      }
    }
    if (is_over_2nd_pass_judge_line && is_over_2nd_pass_judge_line.value()) {
      const auto & decision_at_2nd_pass_judge_opt =
        object_info->decision_at_2nd_pass_judge_line_passage();
      if (!decision_at_2nd_pass_judge_opt) {
        too_late_detect_objects.emplace_back(
          CollisionStatus::BlameType::BLAME_AT_SECOND_PASS_JUDGE, object_info);
        debug_container = &debug_data_.too_late_detect_targets.objects;
      } else {
        const auto & decision_at_2nd_pass_judge = decision_at_2nd_pass_judge_opt.value();
        if (decision_at_2nd_pass_judge.safe_type != CollisionKnowledge::SafeType::UNSAFE) {
          misjudge_objects.emplace_back(
            CollisionStatus::BlameType::BLAME_AT_SECOND_PASS_JUDGE, object_info);
          debug_container = &debug_data_.misjudge_targets.objects;
        } else {
          too_late_detect_objects.emplace_back(
            CollisionStatus::BlameType::BLAME_AT_SECOND_PASS_JUDGE, object_info);
          debug_container = &debug_data_.too_late_detect_targets.objects;
        }
      }
    }
    debug_container->emplace_back(object_info->predicted_object());
  }
  if (collision_at_first_lane) {
    return {true, CollisionInterval::FIRST, too_late_detect_objects, misjudge_objects};
  } else if (collision_at_non_first_lane) {
    return {true, CollisionInterval::ELSE, too_late_detect_objects, misjudge_objects};
  }
  return {false, CollisionInterval::ELSE, too_late_detect_objects, misjudge_objects};
}

std::optional<size_t> IntersectionModule::checkAngleForTargetLanelets(

  const geometry_msgs::msg::Pose & pose, const lanelet::ConstLanelets & target_lanelets,
  const bool is_parked_vehicle) const
{
  const double detection_area_angle_thr = planner_param_.common.attention_area_angle_threshold;
  // Nero c3 fix: this used to read common.attention_area_angle_threshold (a non-zero double),
  // which made it always true regardless of collision_detection.consider_wrong_direction_vehicle
  const bool consider_wrong_direction_vehicle =
    planner_param_.collision_detection.consider_wrong_direction_vehicle;
  const double dist_margin = planner_param_.common.attention_area_margin;

  for (unsigned i = 0; i < target_lanelets.size(); ++i) {
    const auto & ll = target_lanelets.at(i);
    if (!lanelet::utils::isInLanelet(pose, ll, dist_margin)) {
      continue;
    }
    const double ll_angle = lanelet::utils::getLaneletAngle(ll, pose.position);
    const double pose_angle = tf2::getYaw(pose.orientation);
    const double angle_diff = autoware_utils::normalize_radian(ll_angle - pose_angle, -M_PI);
    if (consider_wrong_direction_vehicle) {
      if (std::fabs(angle_diff) > 1.57 || std::fabs(angle_diff) < detection_area_angle_thr) {
        return std::make_optional<size_t>(i);
      }
    } else {
      if (std::fabs(angle_diff) < detection_area_angle_thr) {
        return std::make_optional<size_t>(i);
      }
      // NOTE: sometimes parked vehicle direction is reversed even if its longitudinal velocity is
      // positive
      if (
        is_parked_vehicle && (std::fabs(angle_diff) < detection_area_angle_thr ||
                              (std::fabs(angle_diff + M_PI) < detection_area_angle_thr))) {
        return std::make_optional<size_t>(i);
      }
    }
  }
  return std::nullopt;
}

IntersectionModule::TimeDistanceArray IntersectionModule::calcIntersectionPassingTime(
  const autoware_internal_planning_msgs::msg::PathWithLaneId & path, const bool is_prioritized,
  const IntersectionStopLines & intersection_stoplines,
  autoware_internal_debug_msgs::msg::Float64MultiArrayStamped * ego_ttc_array) const
{
  const double intersection_velocity =
    planner_param_.collision_detection.velocity_profile.default_velocity;
  const double minimum_ego_velocity =
    planner_param_.collision_detection.velocity_profile.minimum_default_velocity;
  const bool use_upstream_velocity =
    planner_param_.collision_detection.velocity_profile.use_upstream;
  const double minimum_upstream_velocity =
    planner_param_.collision_detection.velocity_profile.minimum_upstream_velocity;
  const double current_velocity = planner_data_->current_velocity->twist.linear.x;

  // ==========================================================================================
  // if ego is waiting for collision detection, the entry time into the intersection
  // is a bit delayed for the chattering hold, so we need to "shift" the TimeDistanceArray by
  // this delay
  // ==========================================================================================
  const bool is_go_out = (activated_ && occlusion_activated_);
  const double time_delay = (is_go_out || is_prioritized)
                              ? 0.0
                              : (planner_param_.collision_detection.collision_detection_hold_time -
                                 collision_state_machine_.getDuration());

  // ==========================================================================================
  // to account for the stopline generated by upstream behavior_velocity modules (like walkway,
  // crosswalk), if use_upstream flag is true, the raw velocity of path points after
  // last_intersection_stopline_candidate_idx is used, which maybe almost-zero. at those almost-zero
  // velocity path points, ego future profile is almost "fixed" there.
  //
  // last_intersection_stopline_candidate_idx must be carefully decided especially when ego
  // velocity is almost zero, because if last_intersection_stopline_candidate_idx is at the
  // closest_idx for example, ego is almost "fixed" at current position for the entire
  // spatiotemporal profile, which is judged as SAFE because that profile does not collide
  // with the predicted paths of objects.
  //
  // if second_attention_lane exists, second_attention_stopline_idx is used. if not,
  // max(occlusion_stopline_idx, first_attention_stopline_idx) is used because
  // occlusion_stopline_idx varies depending on the peeking offset parameter
  // ==========================================================================================
  const auto second_attention_stopline_idx = intersection_stoplines.second_attention_stopline;
  const auto occlusion_stopline_idx = intersection_stoplines.occlusion_peeking_stopline.value();
  const auto first_attention_stopline_idx = intersection_stoplines.first_attention_stopline.value();
  const auto closest_idx = intersection_stoplines.closest_idx;
  const auto last_intersection_stopline_candidate_idx =
    second_attention_stopline_idx ? second_attention_stopline_idx.value()
                                  : std::max(occlusion_stopline_idx, first_attention_stopline_idx);

  bool assigned_lane_found = false;
  // crop intersection part of the path, and set the reference velocity to intersection_velocity
  // for ego's ttc
  PathWithLaneId reference_path;
  std::optional<size_t> upstream_stopline{std::nullopt};
  for (size_t i = 0; i + 1 < path.points.size(); ++i) {
    auto reference_point = path.points.at(i);
    // assume backward velocity is current ego velocity
    if (i < closest_idx) {
      reference_point.point.longitudinal_velocity_mps = current_velocity;
    }
    if (
      i > last_intersection_stopline_candidate_idx &&
      std::fabs(reference_point.point.longitudinal_velocity_mps) <
        std::numeric_limits<double>::epsilon() &&
      !upstream_stopline) {
      upstream_stopline = i;
    }
    if (!use_upstream_velocity) {
      reference_point.point.longitudinal_velocity_mps = intersection_velocity;
    }
    reference_path.points.push_back(reference_point);
    bool has_objective_lane_id = util::hasLaneIds(path.points.at(i), associative_ids_);
    if (assigned_lane_found && !has_objective_lane_id) {
      break;
    }
    assigned_lane_found = has_objective_lane_id;
  }
  if (!assigned_lane_found) {
    return {{0.0, 0.0}};  // has already passed the intersection.
  }
  // NERO_IX_STANDSTILL_REJUDGE (overrunyield, 2026-10-26): a standing ego past its first attention stop line (after a
  // stop-line overrun) has the path points behind it set to its own speed, 0, above; those are not a stop line of an
  // upstream module, but they were taken as one, so ego was predicted to crawl through the intersection at
  // minimum_upstream_velocity (0.7 m/s: 5.7 m in 9 s instead of ~5 s) and a car 30 m away held it for 11 s where a
  // robot standing at its line goes at once. With the switch the search starts at ego, as it does for an ego before
  // that line (the only case that changes: standing past it, which upstream never re-judges).
  {
    static const bool nero_sr = [] {
      const char * e = std::getenv("NERO_IX_STANDSTILL_REJUDGE");
      return e != nullptr && std::atof(e) > 0.0;
    }();
    if (nero_sr && upstream_stopline && upstream_stopline.value() < closest_idx) {
      upstream_stopline = std::nullopt;
      for (size_t i = std::max(closest_idx, last_intersection_stopline_candidate_idx + 1);
           i < reference_path.points.size(); ++i) {
        if (
          std::fabs(reference_path.points.at(i).point.longitudinal_velocity_mps) <
          std::numeric_limits<double>::epsilon()) {
          upstream_stopline = i;
          break;
        }
      }
    }
  }

  // apply smoother to reference velocity
  // NERO: with NERO_IX_EGO_ACC=<a> the smoother is skipped and ego is modelled as accelerating from its
  // current speed at <a> m/s^2 up to the path's own speed: v(s) = min(v_path(s), sqrt(v0^2 + 2 a s)).
  // Upstream's smoother output for a standing ego stays below minimum_default_velocity over the whole
  // turn, so the standing ego was assumed to creep at that floor (0.5 m/s: 13 m in 26 s, really ~6 s),
  // and a car crossing the U-turn's second half "would pass before ego gets there" when it would not.
  const double nero_ego_acc = nero_ego_acc_param();
  PathWithLaneId smoothed_reference_path = reference_path;
  if (nero_ego_acc > 0.0) {
    smoothed_reference_path = reference_path;
  } else if (!smoothPath(reference_path, smoothed_reference_path, planner_data_)) {
    smoothed_reference_path = reference_path;
  }

  // calculate when ego is going to reach each (interpolated) points on the path
  TimeDistanceArray time_distance_array{};
  double dist_sum = 0.0;
  double passing_time = time_delay;
  time_distance_array.emplace_back(passing_time, dist_sum);

  // NOTE: `reference_path` is resampled in `reference_smoothed_path`, so
  // `last_intersection_stopline_candidate_idx` makes no sense
  const auto smoothed_path_closest_idx =
    autoware::motion_utils::findFirstNearestIndexWithSoftConstraints(
      smoothed_reference_path.points, path.points.at(closest_idx).point.pose,
      planner_data_->ego_nearest_dist_threshold, planner_data_->ego_nearest_yaw_threshold);

  const std::optional<size_t> upstream_stopline_idx_opt = [&]() -> std::optional<size_t> {
    if (upstream_stopline) {
      const auto upstream_stopline_point =
        reference_path.points.at(upstream_stopline.value()).point.pose;
      return autoware::motion_utils::findFirstNearestIndexWithSoftConstraints(
        smoothed_reference_path.points, upstream_stopline_point,
        planner_data_->ego_nearest_dist_threshold, planner_data_->ego_nearest_yaw_threshold);
    } else {
      return std::nullopt;
    }
  }();

  // junction-yield 10-11 (param-gated, standstill_start_acceleration 0 = stock): while (almost) standing the smoothed
  // reference velocity is ~0 and the profile collapses to minimum_default_velocity, a constant speed the robot never
  // drives (it waits, then accelerates). Model the pull-away instead: v(s) = min(sqrt(v0^2 + 2 a s), raw path velocity),
  // plus standstill_start_delay when standing.
  const double standstill_acc =
    planner_param_.collision_detection.velocity_profile.standstill_start_acceleration;
  const bool standstill_model = standstill_acc > 0.0 && current_velocity < 0.5;
  std::vector<double> raw_s;
  std::vector<double> raw_v;
  if (standstill_model) {
    double s = 0.0;
    for (size_t i = std::min(closest_idx, reference_path.points.size() - 1); i < reference_path.points.size();
         ++i) {
      if (!raw_s.empty()) {
        s += autoware_utils::calc_distance2d(reference_path.points.at(i - 1), reference_path.points.at(i));
      }
      raw_s.push_back(s);
      raw_v.push_back(path.points.at(i).point.longitudinal_velocity_mps);
    }
    if (current_velocity < 0.1) {
      passing_time += planner_param_.collision_detection.velocity_profile.standstill_start_delay;
      time_distance_array.back().first = passing_time;
    }
  }

  for (size_t i = smoothed_path_closest_idx; i + 1 < smoothed_reference_path.points.size(); ++i) {
    const auto & p1 = smoothed_reference_path.points.at(i);
    const auto & p2 = smoothed_reference_path.points.at(i + 1);

    const double dist = autoware_utils::calc_distance2d(p1, p2);
    dist_sum += dist;

    // use average velocity between p1 and p2
    double average_velocity =
      (p1.point.longitudinal_velocity_mps + p2.point.longitudinal_velocity_mps) / 2.0;
    if (nero_ego_acc > 0.0) {
      const double v0 = std::max(current_velocity, 0.0);
      const double v_reach = std::sqrt(v0 * v0 + 2.0 * nero_ego_acc * (dist_sum - 0.5 * dist));
      average_velocity = std::min(std::max(average_velocity, 0.0), v_reach);
    }
    const double passing_velocity = [&]() {
      if (use_upstream_velocity) {
        if (upstream_stopline_idx_opt && i > upstream_stopline_idx_opt.value()) {
          return minimum_upstream_velocity;
        }
      }
      if (standstill_model && !raw_s.empty()) {
        const double s_mid = std::max(dist_sum - dist / 2.0, 0.0);
        const auto it = std::upper_bound(raw_s.begin(), raw_s.end(), s_mid);
        const size_t k = (it == raw_s.begin()) ? 0 : static_cast<size_t>(it - raw_s.begin() - 1);
        // a stop point in the raw path (the stop sign the ego stands at) is not a speed: use the intersection velocity
        const double cap = raw_v.at(k) < 0.1 ? intersection_velocity : raw_v.at(k);
        const double v_kin =
          std::sqrt(current_velocity * current_velocity + 2.0 * standstill_acc * s_mid);
        return std::max(std::min(v_kin, cap), 0.1);
      }
      return std::max<double>(average_velocity, minimum_ego_velocity);
    }();
    passing_time += (dist / passing_velocity);

    time_distance_array.emplace_back(passing_time, dist_sum);
  }
  ego_ttc_array->stamp = clock_->now();
  ego_ttc_array->layout.dim.resize(3);
  ego_ttc_array->layout.dim.at(0).label = "lane_id_@[0][0], ttc_time, ttc_dist, path_x, path_y";
  constexpr size_t ego_debug_size = 5;
  ego_ttc_array->layout.dim.at(0).size = ego_debug_size;
  ego_ttc_array->layout.dim.at(1).label = "values";
  ego_ttc_array->layout.dim.at(1).size = time_distance_array.size();
  ego_ttc_array->data.reserve(time_distance_array.size() * ego_debug_size);
  for (unsigned i = 0; i < time_distance_array.size(); ++i) {
    ego_ttc_array->data.push_back(lane_id_);
  }
  for (const auto & [t, d] : time_distance_array) {
    ego_ttc_array->data.push_back(t);
  }
  for (const auto & [t, d] : time_distance_array) {
    ego_ttc_array->data.push_back(d);
  }
  for (size_t i = smoothed_path_closest_idx; i < smoothed_reference_path.points.size(); ++i) {
    const auto & p = smoothed_reference_path.points.at(i).point.pose.position;
    ego_ttc_array->data.push_back(p.x);
  }
  for (size_t i = smoothed_path_closest_idx; i < smoothed_reference_path.points.size(); ++i) {
    const auto & p = smoothed_reference_path.points.at(i).point.pose.position;
    ego_ttc_array->data.push_back(p.y);
  }
  return time_distance_array;
}

}  // namespace autoware::behavior_velocity_planner
