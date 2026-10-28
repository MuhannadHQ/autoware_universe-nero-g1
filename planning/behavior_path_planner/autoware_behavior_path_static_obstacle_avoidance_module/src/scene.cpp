// Copyright 2023 TIER IV, Inc.
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

#include "autoware/behavior_path_static_obstacle_avoidance_module/scene.hpp"

#include "autoware/behavior_path_planner_common/interface/scene_module_visitor.hpp"
#include "autoware/behavior_path_planner_common/utils/drivable_area_expansion/static_drivable_area.hpp"
#include "autoware/behavior_path_planner_common/utils/path_safety_checker/objects_filtering.hpp"
#include "autoware/behavior_path_planner_common/utils/path_safety_checker/safety_check.hpp"
#include "autoware/behavior_path_planner_common/utils/path_utils.hpp"
#include "autoware/behavior_path_planner_common/utils/traffic_light_utils.hpp"
#include "autoware/behavior_path_planner_common/utils/utils.hpp"
#include "autoware/behavior_path_static_obstacle_avoidance_module/debug.hpp"
#include "autoware/behavior_path_static_obstacle_avoidance_module/utils.hpp"

#include <autoware/motion_utils/trajectory/interpolation.hpp>
#include <autoware_lanelet2_extension/regulatory_elements/road_marking.hpp>
#include <autoware_lanelet2_extension/regulatory_elements/speed_bump.hpp>
#include <autoware_lanelet2_extension/utility/message_conversion.hpp>
#include <autoware_lanelet2_extension/utility/query.hpp>
#include <autoware_lanelet2_extension/utility/utilities.hpp>
#include <autoware_utils/geometry/geometry.hpp>
#include <autoware_utils/math/normalization.hpp>
#include <magic_enum.hpp>
#include <tf2/utils.h>
#include <lanelet2_core/geometry/LineString.h>
#include <lanelet2_core/primitives/BasicRegulatoryElements.h>
#include <autoware_utils/system/time_keeper.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace autoware::behavior_path_planner
{
namespace
{
bool isDrivingSameLane(
  const lanelet::ConstLanelets & previous_lanes, const lanelet::ConstLanelets & current_lanes)
{
  std::multiset<lanelet::Id> prev_ids{};
  std::multiset<lanelet::Id> curr_ids{};
  std::multiset<lanelet::Id> same_ids{};

  std::for_each(
    previous_lanes.begin(), previous_lanes.end(), [&](const auto & l) { prev_ids.insert(l.id()); });
  std::for_each(
    current_lanes.begin(), current_lanes.end(), [&](const auto & l) { curr_ids.insert(l.id()); });

  std::set_intersection(
    prev_ids.begin(), prev_ids.end(), curr_ids.begin(), curr_ids.end(),
    std::inserter(same_ids, same_ids.end()));

  return !same_ids.empty();
}

bool isBestEffort(const std::string & policy)
{
  return policy == "best_effort";
}

lanelet::BasicLineString3d toLineString3d(const std::vector<Point> & bound)
{
  lanelet::BasicLineString3d ret{};
  std::for_each(
    bound.begin(), bound.end(), [&](const auto & p) { ret.emplace_back(p.x, p.y, p.z); });
  return ret;
}
}  // namespace

// ---- nero (session 5, 2026-10-17): moving around objects with traffic in the lane next to ego. Every switch is read
// once from the environment; unset = stock 0.44.2.
//  NERO_SOA_FIX_10794=1   upstream #10794: insertPrepareVelocity's "ego too fast for the nominal shift" test was
//                         inverted, so a decel point to velocity_map[0] went in whenever ego was NOT much slower
//  NERO_SOA_EGO_PRED=1    the safety check predicts ego along its planned speed, capped at max_vel (stock: +0.5 m/s²
//                         without a cap, so ego "outruns" cars behind it and "catches" cars ahead of it)
//  NERO_SOA_APPROVAL_KEEPS_SAFETY=1  the operator's RTC approval of an ambiguous car authorises the move but does not
//                         override the traffic check (stock: an approval while unsafe = "force", no yield)
//  NERO_SOA_HOLD_IN_LANE=1  committed (shifting) but unsafe while the body is still in its lane: hold before the line
//                         (stock: the unsafe verdict is overwritten to safe once ego moved th_avoid_execution sideways)
//  NERO_SOA_START_SAFE_CYCLES=N  from (nearly) standstill the move needs N safe cycles in a row (stock: 3, and one
//                         flicker of "safe" between two cars started it)
//  NERO_SOA_STILL_SHARP_S=s  standing still behind the object for s seconds: a sharper move than max_jerk[0] is allowed
//  NERO_SOA_BUMP_V=v       ego predicted at v (m/s) on mapped speed bumps (and braking at 0.8 m/s² toward them)
//  NERO_SOA_GAP="a_dec,v_min,dv,up_cycles,safe_cycles"  gap timing (needs NERO_SOA_EGO_PRED): when the move around is
//                         unsafe at the current speed, search for the highest approach speed (steps of dv down to
//                         v_min) at which it is safe — a car in the next lane passes before ego reaches the shift —
//                         approve the move at that speed and ramp ego down to it at a_dec (no stop); raise it again one
//                         step per up_cycles safe cycles; claim safe only after safe_cycles consecutive safe cycles
namespace nero_soa
{
struct Env
{
  bool fix_10794{false};
  bool ego_pred{false};
  bool gap{false};
  double gap_a_dec{0.4};
  double gap_v_min{1.0};
  double gap_dv{0.25};
  int gap_up_cycles{10};
  int gap_safe_cycles{3};
  bool approval_keeps_safety{false};
  bool hold_in_lane{false};
  int start_safe_cycles{0};
  double still_sharp_s{0.0};
  double bump_v{0.0};
  bool gap_fast{false};
  bool return_away{false};
};

const Env & env()
{
  static const Env e = [] {
    Env r;
    auto on = [](const char * k) {
      const char * v = std::getenv(k);
      return v && std::atoi(v) > 0;
    };
    r.fix_10794 = on("NERO_SOA_FIX_10794");
    r.ego_pred = on("NERO_SOA_EGO_PRED");
    r.approval_keeps_safety = on("NERO_SOA_APPROVAL_KEEPS_SAFETY");
    r.hold_in_lane = on("NERO_SOA_HOLD_IN_LANE");
    if (const char * v = std::getenv("NERO_SOA_START_SAFE_CYCLES"); v) {
      r.start_safe_cycles = std::max(0, std::atoi(v));
    }
    if (const char * v = std::getenv("NERO_SOA_STILL_SHARP_S"); v) {
      r.still_sharp_s = std::max(0.0, std::atof(v));
    }
    if (const char * v = std::getenv("NERO_SOA_BUMP_V"); v) {
      r.bump_v = std::max(0.0, std::atof(v));
    }
    if (const char * v = std::getenv("NERO_SOA_GAP"); v && *v && std::string(v) != "0") {
      double a = 0.4, vmin = 1.0, dv = 0.25, up = 10, sc = 3;
      std::sscanf(v, "%lf,%lf,%lf,%lf,%lf", &a, &vmin, &dv, &up, &sc);
      r.gap = true;
      r.gap_a_dec = std::max(0.05, a);
      r.gap_v_min = std::max(0.0, vmin);
      r.gap_dv = std::max(0.05, dv);
      r.gap_up_cycles = std::max(1, static_cast<int>(up));
      r.gap_safe_cycles = std::max(1, static_cast<int>(sc));
      r.ego_pred = true;
    }
    r.gap_fast = on("NERO_SOA_GAP_FAST");
    if (r.gap_fast) {
      std::fprintf(stderr, "[NERO_SOA] gap_fast=1\n");
    }
    r.return_away = on("NERO_SOA_RETURN_AWAY");
    if (r.return_away) {
      std::fprintf(stderr, "[NERO_SOA] return_away=1\n");
    }
    std::fprintf(
      stderr,
      "[NERO_SOA] fix_10794=%d ego_pred=%d gap=%d (a_dec %.2f v_min %.2f dv %.2f up %d safe %d) "
      "approval_keeps_safety=%d hold_in_lane=%d start_safe_cycles=%d still_sharp_s=%.1f bump_v=%.2f\n",
      r.fix_10794, r.ego_pred, r.gap, r.gap_a_dec, r.gap_v_min, r.gap_dv, r.gap_up_cycles,
      r.gap_safe_cycles, r.approval_keeps_safety, r.hold_in_lane, r.start_safe_cycles, r.still_sharp_s,
      r.bump_v);
    return r;
  }();
  return e;
}
}  // namespace nero_soa

// ---- nero (2026-10-24, oncominglate job): an oncoming car that comes into view once ego is out in the oncoming
// lane passing a parked car. Every switch is read once from the environment; unset = overtake v4 exactly.
//  NERO_SOA_ONCOMING_DEBUG=1   log only: one line per cycle while ego moves sideways or a moving object is in the
//                              safety check: ego's shift, how far its body is over its lane's bound, the module's
//                              verdicts, and each moving safety-check object (speed, distance, safe / reason)
//  An "oncoming car" below: an object ahead on the shift side, heading against ego (> 135 deg), whose box reaches
//  into the strip ego's body uses while it is out, once it has (a) moved >= 0.5 m toward ego within 0.6 s on one
//  track id while reporting >= 1.0 m/s toward ego at both ends (49 field captures 10-18..10-22: never on 986
//  standing UNKNOWN / 376 standing CAR tracks, on all 5 real oncoming cars 0.35-0.41 s after they were first logged),
//  or (b) kept >= 1.0 m/s (vehicles) / >= 2.5 m/s (UNKNOWN: a standing UNKNOWN never read >= 2.0 m/s toward the
//  robot) for 0.3 s. Kept across track ids by place while it slows down.
//  NERO_SOA_ONCOMING_PAUSE=m   an oncoming car while ego's shift is under way and ego can still get its body back
//                              into its lane before the target: hold the shift m metres inside the lane and stop
//                              before the target; go on after 1 s without an oncoming car (port of the RC1 study's
//                              yield.pause_during_shifting, 2026-10-07/08; stock overwrites the unsafe verdict with
//                              safe once the shift is under way)
//  NERO_SOA_ONCOMING_COMMITTED=r,a  past that point (beside the target): judge the oncoming car from its motion every
//                              cycle (Ahmed 10-24: "in real life a car wont boldoze like that it will start slowing
//                              down as a signal for the robot to go through"). Go through while it slows down (>= 1 m/s
//                              below its speed of the last 2 s), once ego is past the targets, or when ego is back in
//                              its lane before it arrives; then return into the lane as soon as the rear has cleared the
//                              targets (sharpest lateral jerk, only over a way back clear of standing objects). Else
//                              (it keeps coming, ego beside the target): stop at once (2.5 m/s2, the planning limit)
//                              until it slows or has passed. r, a = a driver reacting in r s and braking at a m/s2:
//                              only the logged judgement "could it still stop for ego" (can_stop)
//  NERO_SOA_ONCOMING_CORRIDOR=1  ego's body leaves its lane only if the planned pass, where the body is over the line,
//                              keeps every standing object outside ego's lane (other than targets the path avoids)
//                              outside obstacle_stop's lateral margin (vehicles 0.1 m, others 0.3 m): else
//                              obstacle_stop strands ego in the
//                              oncoming lane (sim c08 dense: 29 s in the oncoming lane behind kerb debris, then a car
//                              drove into it); it waits before the line like the stock yield
//  NERO_SOA_ONCOMING_START_YIELD=1  (2026-10-25, passstart job) an object ahead makes the pass unsafe while ego has
//                              not started to move sideways (shift <= lateral_execution_threshold), but the shift line
//                              is already registered and starts within the minimum prepare distance: yield (the stock
//                              yield: the shift is dropped, the path is straight again and obstacle_stop holds ego
//                              before the target, where it was already stopping it), as when the same object is seen a
//                              moment earlier. Stock cannot yield there ("prevent sudden steering"), overwrites the
//                              verdict with safe and starts the shift; obstacle_stop then takes the approaching car as
//                              its stop target 15 m out and ego speeds up toward it (sim c06 oncoming: a car seen
//                              0.2 s before the shift start, ego 5 m before the parked car at 0.7 m/s -> contact).
//                              Ego has not moved sideways, so dropping the shift is no sudden steering. Objects behind
//                              ego are left to NERO_SOA_HOLD_IN_LANE / NERO_SOA_GAP
//  NERO_SOA_REAR_START_YIELD=1  (2026-10-26, yieldmrm job) the same yield for an object behind / beside ego (a car
//                              coming up the lane next to ego, the shift side): stock yields for it when it is seen a
//                              moment earlier (prepare distance enough), but within the prepare distance it refuses,
//                              NERO_SOA_HOLD_IN_LANE then holds nothing when the shift stays inside the lane (the stop
//                              before the line is the path end) and ego moves sideways toward the car (field 10-25
//                              20:47:50, lanelet 41746: "hold before the line" at 15 km/h, b627 9.4 m behind-left).
//                              Ego has not moved sideways: it waits in its lane behind the target, as in the stock yield
//  NERO_SOA_RETURN_AWAY=1      (2026-10-26, yieldmrm job) NERO_SOA_HOLD_IN_LANE's hold lets a return through: while
//                              ego is shifted and the pass is unsafe only for objects on the side ego is shifted to,
//                              new shift lines that only bring it back toward its lane centre go ahead (stock overwrites
//                              the verdict with safe there anyway; the hold kept the return waiting while the car came
//                              alongside: sim k15c, the car passed 0.23 m from a robot still shifted toward it)
//  NERO_SOA_GAP_FAST=1         (2026-10-26, yieldmrm job) NERO_SOA_GAP's speed search without the repeated work:
//                              the safety-check targets and the speed bumps of the cycle once (stock: per speed tried),
//                              and the descending search stops at the first unsafe speed below the lowest speed ego's
//                              predicted braking reaches before the shift (every lower speed gives the same predicted
//                              path, so the same verdict). Same decisions; at 15-20 km/h 4-5 checks instead of up to
//                              16 per cycle (field 10-25 20:29:20 / 20:47:51: the search ran while the pass was unsafe,
//                              the trajectory fell under 5 Hz -> MRM; sim at the car's load: 92-105 ms of 100-126)
//  NERO_SOA_ONCOMING_SIGHT_M=R[,V]  commit check: ego's body leaves its lane only if a car unseen beyond R m, coming
//                              at V m/s (default 5.56), could not reach ego before its body is back in its lane (the
//                              pass's remaining length at ego's speed, accelerating at 0.5 m/s2 up to 4.17 m/s); else it
//                              waits before the line like the stock yield. Independent of any seen traffic: it decides
//                              which passes are made at all. The debug line logs the sight each pass needs (need_sight)
namespace nero_ovl
{
struct Env
{
  bool debug{false};
  double pause_margin{-1.0};  // < 0: off
  bool committed{false};
  double driver_reaction_s{1.0};
  double driver_decel{3.0};
  double sight_m{-1.0};  // < 0: off
  double sight_v{5.56};
  bool corridor{false};
  bool start_yield{false};
  bool rear_start_yield{false};
};

const Env & env()
{
  static const Env e = [] {
    Env r;
    if (const char * v = std::getenv("NERO_SOA_ONCOMING_DEBUG"); v) r.debug = std::atoi(v) > 0;
    if (const char * v = std::getenv("NERO_SOA_ONCOMING_PAUSE"); v && *v) r.pause_margin = std::atof(v);
    if (const char * v = std::getenv("NERO_SOA_ONCOMING_COMMITTED"); v && *v && std::string(v) != "0") {
      double rs = 1.0, a = 3.0;
      std::sscanf(v, "%lf,%lf", &rs, &a);
      r.committed = true;
      r.driver_reaction_s = std::max(0.0, rs);
      r.driver_decel = std::max(0.5, a);
    }
    if (const char * v = std::getenv("NERO_SOA_ONCOMING_CORRIDOR"); v) r.corridor = std::atoi(v) > 0;
    if (const char * v = std::getenv("NERO_SOA_ONCOMING_START_YIELD"); v) r.start_yield = std::atoi(v) > 0;
    if (const char * v = std::getenv("NERO_SOA_REAR_START_YIELD"); v) r.rear_start_yield = std::atoi(v) > 0;
    if (const char * v = std::getenv("NERO_SOA_ONCOMING_SIGHT_M"); v && *v && std::string(v) != "0") {
      double m = -1.0, vv = 5.56;
      std::sscanf(v, "%lf,%lf", &m, &vv);
      r.sight_m = m;
      r.sight_v = std::max(0.5, vv);
    }
    std::fprintf(
      stderr,
      "[NERO_SOA_OVL] debug=%d pause_margin=%.2f committed=%d (driver reaction %.1f s, decel %.1f) corridor=%d "
      "sight_m=%.1f (unseen car %.2f m/s) start_yield=%d\n",
      r.debug, r.pause_margin, r.committed, r.driver_reaction_s, r.driver_decel, r.corridor, r.sight_m, r.sight_v,
      r.start_yield);
    if (r.rear_start_yield) {
      std::fprintf(stderr, "[NERO_SOA_OVL] rear_start_yield=1\n");
    }
    return r;
  }();
  return e;
}
bool active()
{
  return env().pause_margin >= 0.0 || env().committed || env().sight_m > 0.0 || env().corridor;
}
// NERO_SOA_LANE_BLOCKED=1 (2026-10-26, returnshift job, QUEUE OLS, capture 10-25 #6): a pass whose path is still out of
// the lane where obstacle_stop will stop ego for a standing vehicle in ego's own lane that the module does not pass
// (not a target: traffic, a truck unloading 3 m before a junction) ends with ego standing in the oncoming lane (#6: 12 s
// beside the truck's tail lift, then the operator). Such a pass counts as blocked like overtake v6's corridor check: not
// yet moving sideways -> the stock yield (ego waits in its lane); moving sideways -> v6's pause, its stop before that
// vehicle, back inside the lane if the way back fits, else holding the shift ego has (never further out). Needs overtake
// v6's corridor check and pause switched on (the car since 10-25).
}  // namespace nero_ovl

namespace nero_ret
{
// NERO_SOA_RETURN_NO_NEED (shift_line_generator.cpp has the full note): here, a target that needs no shift is not the
// target a pause stops before
bool no_need()
{
  static const bool on = [] {
    const char * v = std::getenv("NERO_SOA_RETURN_NO_NEED");
    return v && std::atoi(v) > 0;
  }();
  return on;
}
}  // namespace nero_ret

namespace nero_rcl
{
// NERO_SOA_RETURN_CLEAR=1 (2026-10-26, narrowweave job, QUEUE RSB, captures 10-26 #7 "nice" / #8 "scenario", lanelet
// 44311): out in the oncoming lane beside a row of cars parked in ego's lane, the module planned a line back toward
// the lane (the return of a target it had passed, once the next car was "wait and see") straight into a car standing
// in the lane that is not a target (parked facing ego, "queued"); obstacle_stop stopped ego in the oncoming lane
// inside the gap between the rows and the line was never re-planned (field: 7 s, then the operator drove it through
// at 11.5 km/h; sim on the car: stands for good). Out of its lane, such a line is replaced by holding ego's shift past
// every standing object it ran into (where holding clears them by obstacle_stop's lateral margin) and a way back that
// clears every standing object, over before the next stop line / junction lanelet; if none fits, the module's plan
// stays. Upstream has no such check (avoid / return lines are checked against targets only).
bool on()
{
  static const bool r = [] {
    const char * v = std::getenv("NERO_SOA_RETURN_CLEAR");
    const bool b = v && std::atoi(v) > 0;
    std::fprintf(stderr, "[NERO_SOA] return_clear=%d\n", b);
    return b;
  }();
  return r;
}
constexpr double LAT_EXTRA = 0.15;    // path smoothing / tracking on top of obstacle_stop's lateral margin
constexpr double LON_MARGIN = 1.0;    // ego's rear past the object before the way back starts
constexpr double HORIZON = 40.0;      // how far ahead the module's plan is checked
constexpr double TOWARD_EPS = 0.05;   // a plan point counts as back toward the lane once this much nearer the centre
// NERO_SOA_RETURN_DEBUG=1 (log only): a line a second while out of the lane with the plan back clear, and the time
bool debug()
{
  static const bool r = [] {
    const char * v = std::getenv("NERO_SOA_RETURN_DEBUG");
    return v && std::atoi(v) > 0;
  }();
  return r;
}
}  // namespace nero_rcl

namespace nero_cgw
{
// NERO_SOA_CORRIDOR_GIVEWAY="gap_m,park_s,side_m,decel" (2026-10-27, corridor job, QUEUE NWC, capture 10-26e #24 "what
// happened this is critical!!", lanelet 44311): a vehicle ahead that stands facing ego or comes toward it in the free
// width of a narrow street, its box reaching into ego's own lane (a car keeping to its own lane while ego is out is
// overtake v6's case) and within CONFLICT_M of ego's body on its planned path (#24: 0af3ae stood in the
// middle of the one-lane corridor between the parked rows; the robot drove on out in the oncoming lane, snapped back
// and stopped face to face, centred, leaving it no room; the operator took over). Ego keeps right: as far as the car
// needs (a coming car whose line clears ego's body by PASS_CLEAR_M: passed at most PASS_V, no stop; a standing one: its
// width + 2 side_m beside ego) and at least KEEP_RIGHT_M right of its lane centre, never past the rightmost lateral place
// on the way that keeps obstacle_stop's lateral margin + LAT_EXTRA from every standing object seen on its right (at most
// 0.15 m inside its lane's right bound), the rightmost itself only for a car running into ego. It stops with its front
// gap_m before the car (a moving car: before where it will be when ego stands), or earlier where the car has width +
// 2 side_m beside ego's body (a place of its own row it reaches going forward); approached at decel m/s2 where it can,
// harder only as needed. The plan is re-done every REPLAN_S. It holds until the car has passed / left ego's way for
// RELEASE_S (then back to the lane centre first if ego kept right of it), or both stood still park_s (a parked car:
// the module's own plan takes over). Giving way launches the module by itself.
// Ahmed 10-24: "it should keep right when it can"; 10-26 (FOLW): "maybe it should have gone right to dynamically let it
// pass instead of stopping fully in front of it". Upstream's dynamic avoidance shifts away from moving oncoming cars
// only and takes no standing car; nothing upstream gives way in a one-lane corridor.
struct Env
{
  bool on{false};
  double gap_m{6.0};
  double park_s{12.0};
  double side_m{0.3};
  double decel{1.0};
};
const Env & env()
{
  static const Env e = [] {
    Env r;
    if (const char * v = std::getenv("NERO_SOA_CORRIDOR_GIVEWAY"); v && *v && std::string(v) != "0") {
      double g = 6.0, p = 12.0, sd = 0.3, d = 1.0;
      std::sscanf(v, "%lf,%lf,%lf,%lf", &g, &p, &sd, &d);
      r.on = true;
      r.gap_m = std::max(2.0, g);
      r.park_s = std::max(1.0, p);
      r.side_m = std::max(0.0, sd);
      r.decel = std::clamp(d, 0.3, 2.5);
    }
    std::fprintf(
      stderr, "[NERO_SOA] corridor_giveway=%d (gap %.1f m, park %.1f s, side %.2f m, decel %.2f)\n", r.on, r.gap_m,
      r.park_s, r.side_m, r.decel);
    return r;
  }();
  return e;
}
constexpr double HORIZON = 45.0;        // cars whose nearest point is within this far ahead
constexpr double CONFLICT_M = 0.5;      // a car's box within this of ego's body on its planned path is in ego's way
constexpr double CONFLICT_MOVING_M = 1.0;  // ... a coming car within this (it passes ego that close)
constexpr double SQUEEZE_M = 0.2;       // a coming car passes the standing objects on its side this close
constexpr double PASS_V = 1.5;          // a coming car that clears ego's keep-right body: ego passes it at most this fast
constexpr double PASS_CLEAR_M = 0.5;    // ... by at least this
constexpr double PASS_LOOK_M = 10.0;    // ... its keep-right line clears the standing objects this far past the meeting
constexpr double KEEP_TRACK_M = 1.5;    // while holding, the car is tracked until its box is this far beyond ego's lane
constexpr double REPLAN_S = 1.0;        // while holding, the keep-right line / wait place are re-planned this often
constexpr double STANDING_V = 0.5;      // standing objects (as RETURN_CLEAR / the corridor check)
constexpr double UNKNOWN_MIN_V = 2.0;   // a moving UNKNOWN counts from here (a standing one never read 2.0 toward ego)
constexpr double FACING_MIN_LAT = 0.25; // a standing car facing ego counts only with its centre this far left of ego's lane
                                        // centre (c08's 58b04f, parked facing ego at the right kerb: -0.18)
constexpr double LAT_EXTRA = 0.15;      // as RETURN_CLEAR: path smoothing / tracking on top of obstacle_stop's margin
constexpr double IN_LANE_M = 0.15;      // keep right at most this far inside the lane's right bound
constexpr double KEEP_RIGHT_M = 0.3;    // meeting a car ego keeps at least this far right of its lane centre where the
                                        // objects seen allow (Ahmed "keep right when it can"; not the rightmost: kerb
                                        // objects that show up only on the way, sim c24 early v7)
constexpr double RELEASE_S = 1.0;       // the car gone / out of the way for this long: the hold ends
constexpr double REACT_S = 1.0;         // a moving car advances this long more while ego stops
constexpr double BACK_S = 4.0;          // back to the lane centre after the car passed: the hold ends by then anyway
constexpr double BACK_PAST_S = 10.0;    // ... on until past the objects hidden from the module while holding: at most
constexpr double STOP_DECEL = 2.5;      // the planning limit (overtake v6's stop at once)
constexpr double STEP = 0.5;
// the path's segment nearest to p, stepped off a degenerate one (two equal points: the offsets divide by its length;
// sim c24 squeeze v16 gave a box 1.2e19 m ahead, v20 lost the kerb fragment 8866c2 that way)
template <class T>
size_t seg_of(const T & points, const geometry_msgs::msg::Point & p)
{
  const size_t i = autoware::motion_utils::findNearestSegmentIndex(points, p);
  const auto ok = [&](const size_t k) {
    return k + 1 < points.size() && autoware_utils::calc_distance2d(points.at(k), points.at(k + 1)) >= 1.0e-3;
  };
  if (ok(i)) return i;
  for (size_t d = 1; d < 5; ++d) {
    if (ok(i + d)) return i + d;
    if (i >= d && ok(i - d)) return i - d;
  }
  return i;
}
}  // namespace nero_cgw

namespace nero_ovl
{
bool lane_blocked()
{
  static const bool on = [] {
    const char * v = std::getenv("NERO_SOA_LANE_BLOCKED");
    const bool r = v && std::atoi(v) > 0;
    std::fprintf(stderr, "[NERO_SOA] lane_blocked=%d\n", r);
    return r;
  }();
  return on;
}
constexpr double VEHICLE_MIN_V = 1.0;
constexpr double UNKNOWN_MIN_V = 2.5;
constexpr double SUSTAIN_S = 0.3;
constexpr double RESUME_SAFE_S = 1.0;
constexpr double SLOWING_DV = 1.0;
constexpr double SLOWING_WINDOW_S = 2.0;
constexpr double MIN_EASE_DISTANCE = 2.0;
constexpr double BESIDE_MARGIN = 1.0;
constexpr double STOP_DECEL = 2.5;
}  // namespace nero_ovl

namespace nero_ps
{
// NERO_SOA_PATHSTEP_DEBUG=1 (2026-10-28, pathstep job, QUEUE SPS; log only): one line per cycle with ego's real lateral
// place on the reference path, the shift the previous / new output path has at ego, the registered and new shift
// lines, the module's flags and the targets ahead ([NERO_SOA_PS]); a WARN line when the output path at ego moves
// more than 0.2 m in one cycle ([NERO_SOA_PS] STEP)
bool debug()
{
  static const bool r = [] {
    const char * v = std::getenv("NERO_SOA_PATHSTEP_DEBUG");
    return v && std::atoi(v) > 0;
  }();
  return r;
}
}  // namespace nero_ps

namespace nero_pfe
{
// NERO_SOA_PATH_FROM_EGO="tol_m,off_m" (car: "0.1,0.3"; 2026-10-28, pathstep job, QUEUE SPS, captures 10-26e #17 "why" (MRM) / #24 "what
// happened this is critical!!"): the planned path stepped sideways under the robot (#17: 1.83 m at 1.8 m/s, -0.36 rad,
// AEB -> MRM; #24: 0.76 m at 3.4 m/s, a 23 deg snap). The module's NERO freezes (overtake v6's pause incl.
// LANE_BLOCKED, RETURN_CLEAR, the corridor give-way: neroFreezeShift) drop the line ego is on and ease flat from ego,
// from the previous path's shift at ego and not the robot's place or sideways motion; when registered lines pass,
// the path at the robot is re-based. Sim on the car (c06s truck19: 0.57-0.83 m at 3.7 m/s, steer reversed to -0.23 rad;
// #17's placement: the pause pulled the plan into the truck beside the robot, stranded for good). Upstream #11627
// (isValidShiftLine: a new line's path within 0.1 m of ego's real position) guards new stock lines only.
// Each cycle the new plan is compared with the module's previous plan around the robot (S_BACK behind to S_FRONT
// ahead, exactly: the previous lines drawn over this reference): when it moved there by more than tol_m, it is blended in from the previous output (moved onto the robot's real place when
// that is more than off_m off it, and continued along the robot's heading) over the length the module's maximum lateral
// jerk allows at ego's speed, anchored on the map (a smooth S, slope-continuous at the robot). Stock lines start after
// the prepare distance (>= 1 m) and never trigger it. Overtake v6's pause (and LANE_BLOCKED's) also checks its way back into the lane
// against standing objects with obstacle_stop's margins: one it runs into that holding ego's shift clears makes it hold
// the shift (LANE_BLOCKED, never further out) or not pause (an oncoming car: v6 decides as when no way back fits).
struct Env
{
  bool on{false};
  double tol{0.1};  // a jump at the robot (m); a kink is SLOPE_TOL
  double off{0.3};  // the robot this far off the new plan: the blend starts from the robot's place (m)
};
const Env & env()
{
  static const Env e = [] {
    Env r;
    const char * v = std::getenv("NERO_SOA_PATH_FROM_EGO");
    if (v && *v && std::string(v) != "0") {
      r.on = true;
      double a = 0.0, b = 0.0;
      const int n = std::sscanf(v, "%lf,%lf", &a, &b);
      if (n >= 1 && a > 0.0) r.tol = a;
      if (n >= 2 && b > 0.0) r.off = b;
    }
    std::fprintf(stderr, "[NERO_SOA] path_from_ego=%d tol=%.2f off=%.2f\n", r.on, r.tol, r.off);
    return r;
  }();
  return e;
}
constexpr double S_BACK = 6.0;     // the stretch around ego checked for a change: from this far behind (the planned
                                   // trajectory starts ~5 m behind ego: a path re-based behind a lagging robot moves it
                                   // at the robot, e.g. RETURN_CLEAR's hold dropping the passed line) ...
constexpr double S_FRONT = 0.5;    // ... to this far ahead (stock lines start after the prepare distance, >= 1 m)
constexpr double S_FAR = 40.0;     // how far the start profile is kept
constexpr double MIN_LEN = 2.0;
constexpr double V_MIN = 1.0;      // the module's lowest speed in its jerk map
constexpr double MAX_DPSI = 0.3;   // the robot's heading against the reference, at most (rad)
constexpr double MAX_EXTRA = 0.2;  // its sideways rate against the profile it was led along, at most (m per m)
constexpr double ACC_SCALE = 1.5;  // the blend's lateral acceleration bound over the module's (comfort) maximum: a
                                   // give-way's keep-right that turns the robot back from moving out stays as prompt
                                   // as the module asks (corridor_1027's c24 late: 0.62 m instead of 0.79 at 1.0)
constexpr double SLOPE_TOL = 0.1;  // a kink at the robot in the plan (m per m, ~6 deg) the robot cannot follow
constexpr double CHANGED = 0.02;   // the plan changed in front of the robot (m): then a jump / kink / robot off decides
constexpr double MAX_S_H = 6.0;    // how far that sideways motion is carried on, at most (m)
}  // namespace nero_pfe

StaticObstacleAvoidanceModule::StaticObstacleAvoidanceModule(
  const std::string & name, rclcpp::Node & node, std::shared_ptr<AvoidanceParameters> parameters,
  const std::unordered_map<std::string, std::shared_ptr<RTCInterface>> & rtc_interface_ptr_map,
  std::unordered_map<std::string, std::shared_ptr<ObjectsOfInterestMarkerInterface>> &
    objects_of_interest_marker_interface_ptr_map,
  const std::shared_ptr<PlanningFactorInterface> & planning_factor_interface)
: SceneModuleInterface{name, node, rtc_interface_ptr_map, objects_of_interest_marker_interface_ptr_map, planning_factor_interface},  // NOLINT
  helper_{std::make_shared<AvoidanceHelper>(parameters)},
  parameters_{parameters},
  generator_{parameters}
{
}

bool StaticObstacleAvoidanceModule::isExecutionRequested() const
{
  RCLCPP_DEBUG(getLogger(), "AVOIDANCE isExecutionRequested");

  // Check ego is in preferred lane
  updateInfoMarker(avoid_data_);
  updateDebugMarker(BehaviorModuleOutput{}, avoid_data_, path_shifter_, debug_data_);

  // there is object that should be avoid. return true.
  if (!!avoid_data_.stop_target_object) {
    return true;
  }

  // NERO_SOA_CORRIDOR_GIVEWAY: giving way launches the module by itself (its own targets may all be unavoidable, e.g.
  // a parked car registered only once ego moved: sim c24 late v9, where the keep-right line and the speed cap never
  // reached the path)
  if (nero_cgw_active_) {
    return true;
  }

  if (avoid_data_.new_shift_line.empty()) {
    return false;
  }

  return std::any_of(
    avoid_data_.target_objects.begin(), avoid_data_.target_objects.end(),
    [this](const auto & o) { return !helper_->isAbsolutelyNotAvoidable(o); });
}

bool StaticObstacleAvoidanceModule::isExecutionReady() const
{
  RCLCPP_DEBUG_STREAM(getLogger(), "---Avoidance GO/NO-GO status---");
  RCLCPP_DEBUG_STREAM(getLogger(), std::boolalpha << "SAFE:" << avoid_data_.safe);
  RCLCPP_DEBUG_STREAM(getLogger(), std::boolalpha << "COMFORTABLE:" << avoid_data_.comfortable);
  RCLCPP_DEBUG_STREAM(getLogger(), std::boolalpha << "VALID:" << avoid_data_.valid);
  RCLCPP_DEBUG_STREAM(getLogger(), std::boolalpha << "READY:" << avoid_data_.ready);
  return avoid_data_.safe && avoid_data_.comfortable && avoid_data_.valid && avoid_data_.ready;
}

AvoidanceState StaticObstacleAvoidanceModule::getCurrentModuleState(
  const AvoidancePlanningData & data) const
{
  const bool has_avoidance_target = std::any_of(
    data.target_objects.begin(), data.target_objects.end(),
    [this](const auto & o) { return !helper_->isAbsolutelyNotAvoidable(o); });

  if (has_avoidance_target) {
    return AvoidanceState::RUNNING;
  }

  // If the ego is on the shift line, keep RUNNING.
  {
    const size_t idx = planner_data_->findEgoIndex(path_shifter_.getReferencePath().points);
    const auto within = [](const auto & line, const size_t idx) {
      return line.start_idx < idx && idx < line.end_idx;
    };
    for (const auto & shift_line : path_shifter_.getShiftLines()) {
      if (within(shift_line, idx)) {
        return AvoidanceState::RUNNING;
      }
    }
  }

  const bool has_shift_point = !path_shifter_.getShiftLines().empty();
  const bool has_base_offset =
    std::abs(path_shifter_.getBaseOffset()) > parameters_->lateral_execution_threshold;

  if (has_base_offset) {
    return AvoidanceState::RUNNING;
  }

  // Nothing to do. -> EXIT.
  if (!has_shift_point) {
    return AvoidanceState::SUCCEEDED;
  }

  // Be able to canceling avoidance path. -> EXIT.
  if (!helper_->isShifted() && parameters_->enable_cancel_maneuver) {
    return AvoidanceState::CANCEL;
  }

  return AvoidanceState::RUNNING;
}

bool StaticObstacleAvoidanceModule::canTransitSuccessState()
{
  const auto & data = avoid_data_;

  // Change input lane. -> EXIT.
  if (!isDrivingSameLane(helper_->getPreviousDrivingLanes(), data.current_lanelets)) {
    RCLCPP_WARN_THROTTLE(getLogger(), *clock_, 3000, "Previous module lane is updated. Exit.");
    return true;
  }

  helper_->setPreviousDrivingLanes(data.current_lanelets);

  // Reach input path end point. -> EXIT.
  {
    const auto idx = planner_data_->findEgoIndex(data.reference_path.points);
    if (idx == data.reference_path.points.size() - 1) {
      arrived_path_end_ = true;
    }

    constexpr double THRESHOLD = 1.0;
    const auto is_further_than_threshold =
      calc_distance2d(getEgoPose(), autoware_utils::get_pose(data.reference_path.points.back())) >
      THRESHOLD;
    if (is_further_than_threshold && arrived_path_end_) {
      RCLCPP_WARN_THROTTLE(getLogger(), *clock_, 3000, "Reach path end point. Exit.");
      return true;
    }
  }

  return data.state == AvoidanceState::CANCEL || data.state == AvoidanceState::SUCCEEDED;
}

void StaticObstacleAvoidanceModule::fillFundamentalData(
  AvoidancePlanningData & data, DebugData & debug)
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);

  if (getPreviousModuleOutput().reference_path.points.empty()) {
    RCLCPP_WARN_THROTTLE(
      getLogger(), *clock_, 5000, "Previous module reference path is empty. Skip processing.");
    return;
  }

  // reference pose
  data.reference_pose =
    utils::getUnshiftedEgoPose(getEgoPose(), helper_->getPreviousSplineShiftPath());

  // lanelet info
  data.current_lanelets = utils::static_obstacle_avoidance::getCurrentLanesFromPath(
    getPreviousModuleOutput().reference_path, planner_data_);

  if (data.current_lanelets.empty()) {
    RCLCPP_WARN_THROTTLE(getLogger(), *clock_, 5000, "Current lanelets is empty. Skip processing.");
    return;
  }

  data.extend_lanelets = utils::static_obstacle_avoidance::getExtendLanes(
    data.current_lanelets, getEgoPose(), planner_data_);

  lanelet::ConstLanelet closest_lanelet{};
  if (lanelet::utils::query::getClosestLanelet(
        data.current_lanelets, getEgoPose(), &closest_lanelet))
    data.closest_lanelet = closest_lanelet;

  // expand drivable lanes
  const auto is_within_current_lane =
    utils::static_obstacle_avoidance::isWithinLanes(data.closest_lanelet, planner_data_);
  const auto red_signal_lane_itr = std::find_if(
    data.current_lanelets.begin(), data.current_lanelets.end(), [&](const auto & lanelet) {
      if (utils::traffic_light::isTrafficSignalStop({lanelet}, planner_data_)) {
        return true;
      }
      const auto next_lanes = planner_data_->route_handler->getNextLanelets(lanelet);
      return utils::traffic_light::isTrafficSignalStop(next_lanes, planner_data_);
    });
  const auto not_use_adjacent_lane =
    is_within_current_lane && red_signal_lane_itr != data.current_lanelets.end();

  std::for_each(
    data.current_lanelets.begin(), data.current_lanelets.end(), [&](const auto & lanelet) {
      if (!not_use_adjacent_lane || red_signal_lane_itr->id() != lanelet.id()) {
        data.drivable_lanes.push_back(
          utils::static_obstacle_avoidance::generateExpandedDrivableLanes(
            lanelet, planner_data_, parameters_));
      } else {
        data.drivable_lanes.push_back(
          utils::static_obstacle_avoidance::generateNotExpandedDrivableLanes(lanelet));
        data.red_signal_lane = lanelet;
      }
    });

  // calc drivable bound
  auto tmp_path = getPreviousModuleOutput().path;
  const auto shorten_lanes = utils::cutOverlappedLanes(tmp_path, data.drivable_lanes);
  const auto use_left_side_hatched_road_marking_area = [&]() {
    if (!not_use_adjacent_lane) {
      return true;
    }
    return !planner_data_->route_handler->getRoutingGraphPtr()->left(*red_signal_lane_itr);
  }();
  const auto use_right_side_hatched_road_marking_area = [&]() {
    if (!not_use_adjacent_lane) {
      return true;
    }
    return !planner_data_->route_handler->getRoutingGraphPtr()->right(*red_signal_lane_itr);
  }();
  data.left_bound = utils::calcBound(
    getPreviousModuleOutput().path, planner_data_, shorten_lanes,
    use_left_side_hatched_road_marking_area, parameters_->use_intersection_areas,
    parameters_->use_freespace_areas, true);
  data.right_bound = utils::calcBound(
    getPreviousModuleOutput().path, planner_data_, shorten_lanes,
    use_right_side_hatched_road_marking_area, parameters_->use_intersection_areas,
    parameters_->use_freespace_areas, false);

  // reference path
  if (isDrivingSameLane(helper_->getPreviousDrivingLanes(), data.current_lanelets)) {
    data.reference_path_rough = extendBackwardLength(getPreviousModuleOutput().path);
  } else {
    data.reference_path_rough = getPreviousModuleOutput().path;
    RCLCPP_WARN_THROTTLE(
      getLogger(), *clock_, 3000,
      "Previous module lane is updated. Don't use latest reference path.");
  }

  // resampled reference path
  data.reference_path = utils::resamplePathWithSpline(
    data.reference_path_rough, parameters_->resample_interval_for_planning);

  // closest index
  data.ego_closest_path_index = planner_data_->findEgoIndex(data.reference_path.points);

  // arclength from ego pose (used in many functions)
  data.arclength_from_ego = utils::calcPathArcLengthArray(
    data.reference_path, 0, data.reference_path.points.size(),
    autoware::motion_utils::calcSignedArcLength(data.reference_path.points, getEgoPosition(), 0));

  data.is_allowed_goal_modification =
    utils::isAllowedGoalModification(planner_data_->route_handler);
  data.distance_to_red_traffic_light = utils::traffic_light::calcDistanceToRedTrafficLight(
    data.current_lanelets, data.reference_path_rough, planner_data_);

  data.to_return_point = utils::static_obstacle_avoidance::calcDistanceToReturnDeadLine(
    data.current_lanelets, data.reference_path_rough, planner_data_, parameters_,
    data.distance_to_red_traffic_light, data.is_allowed_goal_modification);

  data.to_start_point = utils::static_obstacle_avoidance::calcDistanceToAvoidStartLine(
    data.current_lanelets, parameters_, data.distance_to_red_traffic_light);

  // filter only for the latest detected objects.
  fillAvoidanceTargetObjects(data, debug);

  // compensate lost object which was avoidance target. if the time hasn't passed more than
  // threshold since perception module lost the target yet, this module keeps it as avoidance
  // target.
  utils::static_obstacle_avoidance::compensateLostTargetObjects(
    registered_objects_, data, clock_->now(), planner_data_, parameters_);

  // once an object filtered for boundary clipping, this module keeps the information until the end
  // of execution.
  utils::static_obstacle_avoidance::updateClipObject(clip_objects_, data);

  // calculate various data for each target objects.
  fillAvoidanceTargetData(data.target_objects);

  // sort object order by longitudinal distance
  std::sort(data.target_objects.begin(), data.target_objects.end(), [](auto a, auto b) {
    return a.longitudinal < b.longitudinal;
  });

  // set base path
  path_shifter_.setPath(data.reference_path);
}

void StaticObstacleAvoidanceModule::fillAvoidanceTargetObjects(
  AvoidancePlanningData & data, DebugData & debug) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  using utils::static_obstacle_avoidance::filterTargetObjects;
  using utils::static_obstacle_avoidance::separateObjectsByPath;
  using utils::static_obstacle_avoidance::updateRoadShoulderDistance;

  // Separate dynamic objects based on whether they are inside or outside of the expanded lanelets.
  constexpr double MARGIN = 10.0;
  const auto forward_detection_range = [&]() {
    if (!data.distance_to_red_traffic_light.has_value()) {
      return helper_->getForwardDetectionRange(data.closest_lanelet);
    }
    return std::min(
      helper_->getForwardDetectionRange(data.closest_lanelet),
      data.distance_to_red_traffic_light.value());
  }();

  const auto [object_within_target_lane, object_outside_target_lane] = separateObjectsByPath(
    helper_->getPreviousReferencePath(), helper_->getPreviousSplineShiftPath().path, planner_data_,
    data, parameters_, forward_detection_range + MARGIN, debug);

  for (const auto & object : object_outside_target_lane.objects) {
    ObjectData other_object = createObjectData(data, object);
    other_object.info = ObjectInfo::OUT_OF_TARGET_AREA;
    data.other_objects.push_back(other_object);
  }

  ObjectDataArray objects;
  for (const auto & object : object_within_target_lane.objects) {
    objects.push_back(createObjectData(data, object));
  }

  // Filter out the objects to determine the ones to be avoided.
  filterTargetObjects(objects, data, forward_detection_range, planner_data_, parameters_);
  updateRoadShoulderDistance(data, planner_data_, parameters_);

  // debug
  {
    std::vector<AvoidanceDebugMsg> debug_info_array;
    const auto append = [&](const auto & o) {
      AvoidanceDebugMsg debug_info;
      debug_info.object_id = to_hex_string(o.object.object_id);
      debug_info.longitudinal_distance = o.longitudinal;
      debug_info.lateral_distance_from_centerline = o.to_centerline;
      debug_info_array.push_back(debug_info);
    };

    std::for_each(objects.begin(), objects.end(), [&](const auto & o) { append(o); });

    updateAvoidanceDebugData(debug_info_array);
  }
}

void StaticObstacleAvoidanceModule::fillAvoidanceTargetData(ObjectDataArray & objects) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  using utils::static_obstacle_avoidance::fillAvoidanceNecessity;
  using utils::static_obstacle_avoidance::fillObjectStoppableJudge;

  // Calculate the distance needed to safely decelerate the ego vehicle to a stop line.
  const auto & vehicle_width = planner_data_->parameters.vehicle_width;
  const auto feasible_stop_distance = helper_->getFeasibleDecelDistance(0.0, false);
  std::for_each(objects.begin(), objects.end(), [&, this](auto & o) {
    fillAvoidanceNecessity(o, registered_objects_, vehicle_width, parameters_);
    o.to_stop_line = calcDistanceToStopLine(o);
    fillObjectStoppableJudge(o, registered_objects_, feasible_stop_distance, parameters_);
  });
}

ObjectData StaticObstacleAvoidanceModule::createObjectData(
  const AvoidancePlanningData & data, const PredictedObject & object) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  using boost::geometry::return_centroid;

  const auto & path_points = data.reference_path.points;
  const auto & object_pose = object.kinematics.initial_pose_with_covariance.pose;
  const auto object_closest_index =
    autoware::motion_utils::findNearestIndex(path_points, object_pose.position);
  const auto object_closest_pose = path_points.at(object_closest_index).point.pose;
  const auto object_type = utils::getHighestProbLabel(object.classification);
  const auto object_parameter = parameters_->object_parameters.at(object_type);

  ObjectData object_data{};

  object_data.object = object;

  const auto lower = parameters_->lower_distance_for_polygon_expansion;
  const auto upper = parameters_->upper_distance_for_polygon_expansion;
  const auto clamp =
    std::clamp(calc_distance2d(getEgoPose(), object_pose) - lower, 0.0, upper) / upper;
  object_data.distance_factor = object_parameter.max_expand_ratio * clamp + 1.0;

  // Calc envelop polygon.
  utils::static_obstacle_avoidance::fillObjectEnvelopePolygon(
    object_data, registered_objects_, object_closest_pose, parameters_);

  // calc object centroid.
  object_data.centroid = return_centroid<Point2d>(object_data.envelope_poly);

  // Calc moving time.
  utils::static_obstacle_avoidance::fillObjectMovingTime(
    object_data, stopped_objects_, parameters_);

  // Calc lateral deviation from path to target object.
  object_data.direction = calc_lateral_deviation(object_closest_pose, object_pose.position) > 0.0
                            ? Direction::LEFT
                            : Direction::RIGHT;

  return object_data;
}

bool StaticObstacleAvoidanceModule::canYieldManeuver(const AvoidancePlanningData & data) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  // transit yield maneuver only when the avoidance maneuver is not initiated.
  if (helper_->isShifted()) {
    RCLCPP_DEBUG(getLogger(), "avoidance maneuver already initiated.");
    return false;
  }

  // prevent sudden steering.
  const auto registered_lines = path_shifter_.getShiftLines();
  if (!registered_lines.empty()) {
    const size_t idx = planner_data_->findEgoIndex(path_shifter_.getReferencePath().points);
    const auto prepare_distance = autoware::motion_utils::calcSignedArcLength(
      path_shifter_.getReferencePath().points, idx, registered_lines.front().start_idx);
    if (!helper_->isEnoughPrepareDistance(prepare_distance)) {
      // NERO_SOA_ONCOMING_START_YIELD: ego has not moved sideways yet (isShifted() above), so dropping the shift is no
      // sudden steering: an object ahead that makes the pass unsafe is waited for in the lane
      const auto who = nero_ovl::env().start_yield || nero_ovl::env().rear_start_yield
                         ? neroStartYieldObject(nero_ovl::env().start_yield, nero_ovl::env().rear_start_yield)
                         : std::string{};
      if (who.empty()) {
        if (nero_ovl::env().debug) {  // NERO_SOA_ONCOMING_DEBUG (log only)
          RCLCPP_INFO_THROTTLE(
            getLogger(), *clock_, 500,
            "[NERO_SOA_OVL] no yield: the registered shift starts in %.2f m (< %.2f m), ego shift %.3f",
            prepare_distance,
            std::max(getEgoSpeed() * parameters_->min_prepare_time, parameters_->min_prepare_distance),
            helper_->getEgoShift());
        }
        RCLCPP_DEBUG(
          getLogger(),
          "Distance to shift start point is less than minimum prepare distance. The distance is not "
          "enough.");
        return false;
      }
      RCLCPP_WARN_THROTTLE(
        getLogger(), *clock_, 500,
        "[NERO_SOA_OVL] %s makes the pass unsafe before the shift has started (it starts in %.2f m, ego shift %.3f, "
        "v %.2f): yield in the lane",
        who.c_str(), prepare_distance, helper_->getEgoShift(), getEgoSpeed());
    }
  }

  if (!data.stop_target_object) {
    RCLCPP_DEBUG(getLogger(), "can pass by the object safely without avoidance maneuver.");
    return true;
  }

  constexpr double TH_STOP_SPEED = 0.01;
  constexpr double TH_STOP_POSITION = 0.5;

  const auto stopped_for_the_object =
    getEgoSpeed() < TH_STOP_SPEED && std::abs(data.to_stop_line) < TH_STOP_POSITION;

  const auto id = data.stop_target_object.value().object.object_id;
  const auto same_id_obj = std::find_if(
    ego_stopped_objects_.begin(), ego_stopped_objects_.end(),
    [&id](const auto & o) { return o.object.object_id == id; });

  // if the ego already started avoiding for the object, never transit yield maneuver again.
  if (same_id_obj != ego_stopped_objects_.end()) {
    return stopped_for_the_object;
  }

  // registered objects whom the ego stopped for at the moment of stopping.
  if (stopped_for_the_object) {
    ego_stopped_objects_.push_back(data.stop_target_object.value());
  }

  return true;
}

void StaticObstacleAvoidanceModule::fillShiftLine(
  AvoidancePlanningData & data, DebugData & debug) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  auto path_shifter = path_shifter_;

  /**
   * STEP1: Create candidate shift lines.
   * Merge rough shift lines, and extract new shift lines.
   */
  data.new_shift_line = generator_.generate(data, debug);

  /**
   * Step2: Validate new shift lines.
   * Output new shift lines only when the avoidance path which is generated from them doesn't have
   * huge offset from ego.
   */
  data.valid = isValidShiftLine(data.new_shift_line, path_shifter);
  const auto found_new_sl = data.new_shift_line.size() > 0;
  const auto registered = path_shifter.getShiftLines().size() > 0;
  data.found_avoidance_path = found_new_sl || registered;

  /**
   * STEP3: Set new shift lines.
   * If there are new shift points, these shift points are registered in path_shifter in order to
   * generate candidate avoidance path.
   */
  if (!data.new_shift_line.empty()) {
    addNewShiftLines(path_shifter, data.new_shift_line);
  }

  /**
   * STEP4: Generate avoidance path.
   */
  ShiftedPath spline_shift_path =
    utils::static_obstacle_avoidance::toShiftedPath(data.reference_path);
  const auto success_spline_path_generation =
    path_shifter.generate(&spline_shift_path, true, SHIFT_TYPE::SPLINE);
  data.candidate_path = success_spline_path_generation
                          ? spline_shift_path
                          : utils::static_obstacle_avoidance::toShiftedPath(data.reference_path);

  /**
   * STEP5: Check avoidance path safety.
   * For each target objects and the objects in adjacent lanes,
   * check that there is a certain amount of margin in the lateral and longitudinal direction.
   */
  data.comfortable = helper_->isComfortable(data.new_shift_line);
  // NERO_SOA_STILL_SHARP_S: standing still behind the object for this long, the move may be sharper than the jerk
  // limit at standstill (lateral max_jerk[0]) — ego that stopped a little closer than the stop point (a hold for a
  // passing car) could otherwise never start again: safe, valid and ready, but "not comfortable" (60+ s deadlock)
  if (nero_soa::env().still_sharp_s > 0.0) {
    const double now = clock_->now().seconds();
    if (getEgoSpeed() > 0.1 || data.target_objects.empty()) {
      nero_still_since_ = -1.0;
    } else if (nero_still_since_ < 0.0) {
      nero_still_since_ = now;
    } else if (!data.comfortable && now - nero_still_since_ > nero_soa::env().still_sharp_s) {
      data.comfortable = true;
      RCLCPP_INFO_THROTTLE(
        getLogger(), *clock_, 2000, "[NERO_SOA] standing still behind the object: sharper move allowed");
    }
  }
  data.safe = isSafePath(data.candidate_path, debug);
  if (nero_soa::env().gap) {
    neroGapTiming(data, debug);  // NERO_SOA_GAP: unsafe now may be safe at a lower approach speed
  }
  const auto avoidance_ready = helper_->isReady(data.target_objects);
  data.ready = helper_->isReady(data.new_shift_line, path_shifter_.getLastShiftLength()) &&
               avoidance_ready.first;
  data.request_operator = avoidance_ready.second;

  // NERO_SOA_APPROVAL_KEEPS_SAFETY: an object the operator already approved is not asked about again
  if (nero_soa::env().approval_keeps_safety && data.request_operator && !data.target_objects.empty()) {
    const auto & u = data.target_objects.front().object.object_id.uuid;
    const auto it = nero_approved_.find(std::string(u.begin(), u.end()));
    const double now = clock_->now().seconds();
    if (it != nero_approved_.end() && now - it->second < 30.0) {
      it->second = now;
      data.request_operator = false;
      data.ready = helper_->isReady(data.new_shift_line, path_shifter_.getLastShiftLength());
    }
  }
}

void StaticObstacleAvoidanceModule::fillEgoStatus(
  AvoidancePlanningData & data, [[maybe_unused]] DebugData & debug) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  data.state = getCurrentModuleState(data);
  nero_hold_now_ = false;

  /**
   * Find the nearest object that should be avoid. When the ego follows reference path,
   * if the both of following two conditions are satisfied, the module surely avoid the object.
   * Condition1: there is risk to collide with object without avoidance.
   * Condition2: there is enough space to avoid.
   * In NEED_DECELERATION condition, it is possible to avoid object by deceleration even if the flag
   * is_avoidable is FALSE. So, the module inserts stop point for such a object.
   */
  for (const auto & o : data.target_objects) {
    if (o.avoid_required && !helper_->isAbsolutelyNotAvoidable(o)) {
      data.avoid_required = true;
      data.stop_target_object = o;
      data.to_stop_line = o.to_stop_line;
      break;
    }
  }

  const auto can_yield_maneuver = canYieldManeuver(data);

  /**
   * If the output path is locked by outside of this module, don't update output path.
   */
  if (isOutputPathLocked()) {
    data.safe_shift_line.clear();
    data.candidate_path = helper_->getPreviousSplineShiftPath();
    RCLCPP_DEBUG_THROTTLE(
      getLogger(), *clock_, 500, "this module is locked now. keep current path.");
    return;
  }

  const auto registered_sl_force_deactivated =
    [&](const std::string & direction, const RegisteredShiftLineArray & shift_line_array) {
      return std::any_of(
        shift_line_array.begin(), shift_line_array.end(), [&](const auto & shift_line) {
          return rtc_interface_ptr_map_.at(direction)->isForceDeactivated(shift_line.uuid);
        });
    };

  const auto is_force_deactivated = registered_sl_force_deactivated("left", left_shift_array_) ||
                                    registered_sl_force_deactivated("right", right_shift_array_);
  if (is_force_deactivated && can_yield_maneuver) {
    data.yield_required = true;
    data.safe_shift_line = data.new_shift_line;
    data.force_deactivated = true;
    RCLCPP_INFO_THROTTLE(
      getLogger(), *clock_, 3000, "this module is force deactivated. wait until reactivation");
    return;
  }

  // NERO_SOA_CORRIDOR_GIVEWAY: a car against ego in its way in a narrow street: keep right, wait where it can pass
  if (nero_cgw::env().on && neroCorridorGiveway(data)) {
    if (nero_cgw_start_) {
      nero_paused_ = false;  // the give-way plan replaces overtake v6's pause
    }
    // the module keeps running while it gives way (else, with nothing left to avoid, it hands back the lane centre and
    // exits: the keep-right line and the wait were dropped once the robot had passed its last target)
    data.state = AvoidanceState::RUNNING;
    data.yield_required = false;
    data.safe_shift_line.clear();
    return;
  }

  // NERO_SOA_ONCOMING_PAUSE / _COMMITTED: an oncoming car while the shift is under way (stock: the unsafe verdict is
  // overwritten with safe below once ego has started to move sideways)
  if (nero_ovl::active()) {
    if (nero_ovl::env().sight_m > 0.0) {
      const auto need = neroPassSightNeeded(data);
      if (need && need.value() > nero_ovl::env().sight_m) {
        RCLCPP_WARN_THROTTLE(
          getLogger(), *clock_, 2000,
          "[NERO_SOA_OVL] the pass needs %.1f m of oncoming lane in sight, have %.1f m: wait before the line",
          need.value(), nero_ovl::env().sight_m);
        data.safe = false;  // the stock yield (not yet moving sideways) / NERO_SOA_HOLD_IN_LANE (inside the lane)
      }
    }
    nero_corridor_blocker_ = nero_ovl::env().corridor ? neroCorridorBlocker(data) : "";
    nero_lane_block_.reset();
    if (nero_ovl::lane_blocked()) {
      neroUpdateLaneLatch(data);
    }
    if (nero_ovl::lane_blocked() && nero_corridor_blocker_.empty()) {
      nero_lane_block_ = neroLaneBlocker(data);
      if (nero_lane_block_) {
        nero_corridor_blocker_ = "own lane " + nero_lane_block_->first;
        RCLCPP_WARN_THROTTLE(
          getLogger(), *clock_, 2000,
          "[NERO_SOA_LANE] the pass ends before %s %.1f m ahead (not passed: traffic): %s", nero_lane_block_->first.c_str(),
          nero_lane_block_->second, helper_->isShifted() ? "pause / hold, stop before it" : "wait in the lane");
      }
    }
    if (!nero_corridor_blocker_.empty()) {
      RCLCPP_WARN_THROTTLE(
        getLogger(), *clock_, 2000,
        "[NERO_SOA_OVL] the pass in the oncoming lane is blocked (%s within obstacle_stop's margin): %s",
        nero_corridor_blocker_.c_str(), helper_->isShifted() ? "pause inside the lane" : "wait before the line");
      data.safe = false;  // not moving sideways yet: the stock yield; already moving sideways: the pause below
    }
    nero_onc_ = neroFindOncoming(data);
    if (neroOncomingPause(data)) {
      data.yield_required = false;
      data.safe_shift_line.clear();
      return;
    }
    neroOncomingCommitted(data);
    if (nero_ovl::env().committed && (nero_returning_ || (nero_onc_ && helper_->isShifted()))) {
      // the go-through / stop decision is made (nero_stop_now_); the shift itself is kept, or the quick return
      data.safe = true;
      data.yield_required = false;
      if (nero_returning_) {
        data.safe_shift_line.clear();
      } else {
        data.safe_shift_line = data.new_shift_line;
      }
      return;
    }
  }

  /**
   * If the avoidance path is safe, use unapproved_new_sl for avoidance path generation.
   */
  if (data.safe) {
    data.yield_required = false;
    data.safe_shift_line = data.new_shift_line;
    return;
  }

  /**
   * If the yield maneuver is disabled, use unapproved_new_sl for avoidance path generation even if
   * the shift line is unsafe.
   */
  if (!parameters_->enable_yield_maneuver) {
    data.yield_required = false;
    data.safe_shift_line = data.new_shift_line;
    return;
  }

  /**
   * TODO(Satoshi OTA) Think yield maneuver in the middle of avoidance.
   * Even if it is determined that a yield is necessary, the yield maneuver is not executed
   * if the avoidance has already been initiated.
   */
  if (!can_yield_maneuver) {
    // NERO_SOA_HOLD_IN_LANE: already moving sideways (or within the commit window) but the body is still inside its
    // own lane: keep the "unsafe" verdict, so that insertStopPoint (yield.enable_during_shifting) holds ego before it
    // leaves the lane until the car beside / behind has gone (stock: overwritten to safe, ego swings out in front of it)
    // ... and only if the stop before the line is reachable at <= 1 m/s² (with policy "reliable" insertStopPoint puts
    // it in regardless: at 18 km/h a few metres before the line that was a 3.4 m/s² stop and an MRM)
    const auto nero_hold_reachable = [&]() {
      const auto & pts = data.candidate_path.path.points;
      if (pts.empty()) {
        return false;
      }
      const auto ego_idx = planner_data_->findEgoIndex(pts);
      size_t stop_idx = pts.size() - 1;
      for (size_t idx = ego_idx; idx < pts.size(); ++idx) {
        if (!utils::isEgoWithinOriginalLane(
              data.current_lanelets, pts.at(idx).point.pose, planner_data_->parameters)) {
          stop_idx = idx > ego_idx ? idx - 1 : ego_idx;
          break;
        }
      }
      const double dist =
        autoware::motion_utils::calcSignedArcLength(pts, getEgoPosition(), stop_idx) - 1.0;
      const double v = getEgoSpeed();
      return dist > 0.0 && v * v <= 2.0 * 1.0 * dist;
    };
    // (nearly) standing: holding means stopping where ego is — allowed while its body is inside the lane or at most
    // 0.3 m over the line (it crept to the line and the old test let go there: into a 46 km/h car)
    const bool nero_slow = getEgoSpeed() < 0.6;
    const auto nero_overhang_ok = [&]() {
      lanelet::ConstLanelet ll;
      if (!lanelet::utils::query::getClosestLanelet(data.current_lanelets, getEgoPose(), &ll)) {
        return false;
      }
      const lanelet::BasicPoint2d p{getEgoPosition().x, getEgoPosition().y};
      const auto l = lanelet::geometry::project(ll.leftBound2d(), p);
      const auto r = lanelet::geometry::project(ll.rightBound2d(), p);
      const double half = 0.5 * std::hypot(l.x() - r.x(), l.y() - r.y());
      return std::abs(helper_->getEgoShift()) + 0.5 * planner_data_->parameters.vehicle_width <
             half + 0.3;
    };
    const bool nero_in_lane =
      utils::isEgoWithinOriginalLane(data.current_lanelets, getEgoPose(), planner_data_->parameters);
    if (
      nero_soa::env().hold_in_lane &&
      ((nero_in_lane && nero_hold_reachable()) || (nero_slow && (nero_in_lane || nero_overhang_ok())))) {
      nero_hold_now_ = nero_slow;
      data.yield_required = false;
      data.safe_shift_line = data.new_shift_line;
      if (nero_soa::env().return_away && neroReturnAwayOnly(data)) {
        // NERO_SOA_RETURN_AWAY: back toward the lane centre, away from the car that makes the pass unsafe
        nero_hold_now_ = false;
        data.safe = true;
        RCLCPP_WARN_THROTTLE(
          getLogger(), *clock_, 1000,
          "[NERO_SOA] unsafe for a car on the side ego is shifted to: the return to the lane centre goes ahead");
        return;
      }
      RCLCPP_WARN_THROTTLE(
        getLogger(), *clock_, 1000, "[NERO_SOA] unsafe while still in the lane: hold before the line");
      return;
    }
    data.safe = true;  // overwrite safety judge.
    data.yield_required = false;
    data.safe_shift_line = data.new_shift_line;
    RCLCPP_WARN_THROTTLE(getLogger(), *clock_, 500, "unsafe. but could not transit yield status.");
    return;
  }

  auto candidate_sl_force_activated = [&](const std::string & direction) {
    // If statement to avoid unnecessary warning occurring from isForceActivated function
    if (candidate_uuid_ == uuid_map_.at(direction)) {
      if (rtc_interface_ptr_map_.at(direction)->isForceActivated(candidate_uuid_)) {
        return true;
      }
    }
    return false;
  };

  auto registered_sl_force_activated =
    [&](const std::string & direction, const RegisteredShiftLineArray & shift_line_array) {
      return std::any_of(
        shift_line_array.begin(), shift_line_array.end(), [&](const auto & shift_line) {
          return rtc_interface_ptr_map_.at(direction)->isForceActivated(shift_line.uuid);
        });
    };

  /**
   * Check if the candidate avoidance path is force activated
   * NERO_SOA_APPROVAL_KEEPS_SAFETY: the operator's approval ("Approve overtake" on the dashboard) authorises the move
   * around an ambiguous car; it does not switch off the module's own check of the traffic next to ego — the move is
   * still timed (gap timing / yield) instead of starting in front of a car that is passing
   */
  if (nero_soa::env().approval_keeps_safety) {
    // fall through to the yield maneuver below
  } else if (candidate_sl_force_activated("left") || candidate_sl_force_activated("right")) {
    data.yield_required = false;
    data.safe_shift_line = data.new_shift_line;
    return;
  }

  /**
   * Check if any registered shift line is force activated
   */
  if (
    !nero_soa::env().approval_keeps_safety &&
    (registered_sl_force_activated("left", left_shift_array_) ||
     registered_sl_force_activated("right", right_shift_array_))) {
    data.yield_required = false;
    data.safe_shift_line = data.new_shift_line;
    RCLCPP_WARN_THROTTLE(getLogger(), *clock_, 5000, "unsafe but force executed");
    return;
  }

  /**
   * Transit yield maneuver. Clear shift lines and output yield path.
   */
  {
    data.yield_required = true;
    data.safe_shift_line = data.new_shift_line;
  }

  /**
   * Even if data.avoid_required is false, the module cancels registered shift point when the
   * approved avoidance path is not safe.
   */
  const auto approved_path_exist = !path_shifter_.getShiftLines().empty();
  if (approved_path_exist) {
    RCLCPP_WARN_THROTTLE(getLogger(), *clock_, 5000, "unsafe. canceling approved path...");
    return;
  }

  /**
   * If the avoidance path is not safe in situation where the ego should avoid object, the ego
   * stops in front of the front object with the necessary distance to avoid the object.
   */
  {
    RCLCPP_WARN_THROTTLE(getLogger(), *clock_, 5000, "unsafe. transit yield maneuver...");
  }
}

void StaticObstacleAvoidanceModule::fillDebugData(
  const AvoidancePlanningData & data, [[maybe_unused]] DebugData & debug) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  if (!data.stop_target_object) {
    return;
  }

  if (helper_->isShifted()) {
    return;
  }

  if (data.new_shift_line.empty()) {
    return;
  }

  const auto o_front = data.stop_target_object.value();
  const auto object_type = utils::getHighestProbLabel(o_front.object.classification);
  const auto object_parameter = parameters_->object_parameters.at(object_type);
  const auto constant_distance = helper_->getFrontConstantDistance(o_front);
  const auto & vehicle_width = planner_data_->parameters.vehicle_width;

  const auto lateral_hard_margin = o_front.is_parked
                                     ? object_parameter.lateral_hard_margin_for_parked_vehicle
                                     : object_parameter.lateral_hard_margin;
  const auto max_avoid_margin = lateral_hard_margin * o_front.distance_factor +
                                object_parameter.lateral_soft_margin + 0.5 * vehicle_width;

  const auto avoidance_distance = helper_->getSharpAvoidanceDistance(helper_->getShiftLength(
    o_front, utils::static_obstacle_avoidance::isOnRight(o_front), max_avoid_margin));
  const auto prepare_distance = helper_->getNominalPrepareDistance();
  const auto total_avoid_distance = prepare_distance + avoidance_distance + constant_distance;

  const auto dead_pose_opt = autoware::motion_utils::calcLongitudinalOffsetPose(
    data.reference_path.points, getEgoPosition(), o_front.longitudinal - total_avoid_distance);
  dead_pose_ = dead_pose_opt.has_value()
                 ? PoseWithDetail(dead_pose_opt.value())
                 : PoseWithDetail(autoware_utils::get_pose(data.reference_path.points.front()));
}

void StaticObstacleAvoidanceModule::updateEgoBehavior(
  const AvoidancePlanningData & data, ShiftedPath & path)
{
  if (parameters_->path_generation_method == "optimization_base") {
    return;
  }

  insertPrepareVelocity(path);
  insertAvoidanceVelocity(path);
  neroInsertGapVelocity(path);  // NERO_SOA_GAP (no-op without a plan)

  const auto insert_velocity = [this, &data, &path]() {
    if (nero_paused_) {  // NERO_SOA_ONCOMING_PAUSE: stop before the target, the body held inside the lane
      neroInsertPauseStop(path);
      return;
    }

    if (data.yield_required) {
      insertWaitPoint(isBestEffort(parameters_->policy_deceleration), path);
      return;
    }

    if (!data.avoid_required) {
      return;
    }

    if (!data.found_avoidance_path) {
      insertWaitPoint(isBestEffort(parameters_->policy_deceleration), path);
      return;
    }

    if (isWaitingApproval() && path_shifter_.getShiftLines().empty()) {
      insertWaitPoint(isBestEffort(parameters_->policy_deceleration), path);
      return;
    }

    insertStopPoint(isBestEffort(parameters_->policy_deceleration), path);
  };

  insert_velocity();

  if (nero_cgw_active_) {
    neroInsertCgwStop(path);  // NERO_SOA_CORRIDOR_GIVEWAY: the wait place
  }

  if (nero_lane_stop_) {
    // NERO_SOA_LANE_BLOCKED: the latched vehicle in ego's lane: stop where obstacle_stop does, approached smoothly
    utils::static_obstacle_avoidance::insertDecelPoint(
      getEgoPosition(), nero_lane_stop_.value(), 0.0, path.path, stop_pose_);
  }

  if (nero_hold_now_) {
    // NERO_SOA_HOLD_IN_LANE at standstill / creeping: stay where ego is until the car has gone
    utils::static_obstacle_avoidance::insertDecelPoint(getEgoPosition(), 0.3, 0.0, path.path, stop_pose_);
  }

  if (nero_stop_now_) {
    // NERO_SOA_ONCOMING_COMMITTED: an oncoming car keeps coming and ego cannot be back in its lane in time: stop at
    // once (the planning limit's deceleration)
    const double v = getEgoSpeed();
    utils::static_obstacle_avoidance::insertDecelPoint(
      getEgoPosition(), std::max(0.3, v * v / (2.0 * nero_ovl::STOP_DECEL)), 0.0, path.path, stop_pose_);
  }

  insertReturnDeadLine(isBestEffort(parameters_->policy_deceleration), path);

  set_longitudinal_planning_factor(path.path);
}

double StaticObstacleAvoidanceModule::neroBodyOverLaneBound() const
{
  lanelet::ConstLanelet ll;
  if (!lanelet::utils::query::getClosestLanelet(avoid_data_.current_lanelets, getEgoPose(), &ll)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const lanelet::BasicPoint2d p{getEgoPosition().x, getEgoPosition().y};
  const auto l = lanelet::geometry::project(ll.leftBound2d(), p);
  const auto r = lanelet::geometry::project(ll.rightBound2d(), p);
  const double half = 0.5 * std::hypot(l.x() - r.x(), l.y() - r.y());
  return std::abs(helper_->getEgoShift()) + 0.5 * planner_data_->parameters.vehicle_width - half;
}

bool StaticObstacleAvoidanceModule::neroReturnAwayOnly(const AvoidancePlanningData & data) const
{
  // NERO_SOA_RETURN_AWAY: ego is at its widest (nowhere ahead does the candidate path shift further out than ego is
  // now: the outward move is over, a hold before a lane line is not skipped), every new line only brings ego back
  // toward its lane centre (it ends between the centre and ego's shift now), and every object this cycle's safety check
  // found unsafe is on the side ego is shifted to
  if (data.new_shift_line.empty()) {
    return false;
  }
  const double ego_shift = helper_->getEgoShift();
  if (std::abs(ego_shift) <= parameters_->lateral_execution_threshold) {
    return false;
  }
  const double sign = ego_shift > 0.0 ? 1.0 : -1.0;
  const auto & cp = data.candidate_path;
  if (!cp.path.points.empty() && cp.shift_length.size() == cp.path.points.size()) {
    for (size_t i = planner_data_->findEgoIndex(cp.path.points); i < cp.shift_length.size(); ++i) {
      if (sign * cp.shift_length.at(i) > sign * ego_shift + 0.05) {
        return false;
      }
    }
  }
  for (const auto & l : data.new_shift_line) {
    const double end = sign * l.end_shift_length;
    if (end > sign * ego_shift + 0.05 || end < -0.05) {
      return false;
    }
  }
  bool any_unsafe = false;
  for (const auto & [id, c] : debug_data_.collision_check) {
    if (c.is_safe) {
      continue;
    }
    any_unsafe = true;
    const double lat =
      autoware::motion_utils::calcLateralOffset(data.reference_path.points, c.current_obj_pose.position);
    if (sign * lat <= 0.0) {
      return false;
    }
  }
  return any_unsafe;
}

std::string StaticObstacleAvoidanceModule::neroStartYieldObject(const bool front, const bool rear) const
{
  // this cycle's safety check (isSafePath stops at its first unsafe object)
  for (const auto & [id, c] : debug_data_.collision_check) {
    if (c.is_safe || (c.is_front && !front) || (!c.is_front && !rear)) {
      continue;
    }
    const auto & q = c.current_obj_pose.position;
    const auto & e = getEgoPosition();
    char buf[96];
    std::snprintf(
      buf, sizeof(buf), "object %.1f m %s (%.1f m/s, %s)", std::hypot(q.x - e.x, q.y - e.y),
      c.is_front ? "ahead" : "behind",
      std::hypot(c.current_twist.linear.x, c.current_twist.linear.y), c.unsafe_reason.c_str());
    return buf;
  }
  return "";
}

void StaticObstacleAvoidanceModule::neroOncomingDebugLog() const
{
  const auto & d = avoid_data_;
  const auto & ego = getEgoPosition();
  std::string objs;
  for (const auto & [uuid, c] : debug_data_.collision_check) {
    const double v = std::hypot(c.current_twist.linear.x, c.current_twist.linear.y);
    if (v < 0.5) continue;
    char buf[200];
    std::snprintf(
      buf, sizeof(buf), " | obj %02x%02x v %.2f d %.1f %s%s%s", uuid.data[0], uuid.data[1], v,
      std::hypot(c.current_obj_pose.position.x - ego.x, c.current_obj_pose.position.y - ego.y),
      c.is_safe ? "safe" : "UNSAFE", c.is_safe ? "" : " ", c.is_safe ? "" : c.unsafe_reason.c_str());
    objs += buf;
  }
  if (!helper_->isShifted() && objs.empty() && !nero_onc_ && !neroPassSightNeeded(d)) return;
  const auto need = neroPassSightNeeded(d);
  char onc[200] = "";
  if (need) std::snprintf(onc, sizeof(onc), " need_sight %.1f", need.value());
  if (nero_onc_) {
    const auto n = std::strlen(onc);
    std::snprintf(
      onc + n, sizeof(onc) - n, " | ONC %s v %.2f max %.2f gap %.1f slowing %d can_stop %d", nero_onc_->id.c_str(),
      nero_onc_->v, nero_onc_->v_recent_max, nero_onc_->gap, nero_onc_->slowing, nero_onc_->can_stop);
  }
  RCLCPP_INFO(
    getLogger(),
    "[NERO_SOA_OVL] v %.2f shift %.2f over %.2f shifted %d safe %d yield %d avoid_req %d stop_line %.1f paused %d "
    "stop_now %d%s%s",
    getEgoSpeed(), helper_->getEgoShift(), neroBodyOverLaneBound(), helper_->isShifted(), d.safe,
    d.yield_required, d.avoid_required, d.to_stop_line < 1e6 ? d.to_stop_line : -1.0, nero_paused_,
    nero_stop_now_, onc, objs.c_str());
}

std::optional<double> StaticObstacleAvoidanceModule::neroInLaneShiftLimit(
  const AvoidancePlanningData & data, const bool left, const double margin) const
{
  lanelet::ConstLanelet ll;
  if (
    data.reference_path.points.empty() ||
    !lanelet::utils::query::getClosestLanelet(data.current_lanelets, getEgoPose(), &ll)) {
    return std::nullopt;
  }
  const auto bound = left ? ll.leftBound2d() : ll.rightBound2d();
  if (bound.size() < 2) {
    return std::nullopt;
  }
  const auto & pose = data.reference_path.points.at(data.ego_closest_path_index).point.pose;
  const lanelet::BasicPoint2d point{pose.position.x, pose.position.y};
  const auto distance = boost::geometry::distance(point, bound.basicLineString());
  return std::max(0.0, distance - 0.5 * planner_data_->parameters.vehicle_width - margin);
}

std::optional<StaticObstacleAvoidanceModule::NeroOncoming>
StaticObstacleAvoidanceModule::neroFindOncoming(const AvoidancePlanningData & data) const
{
  const auto & points = data.reference_path.points;
  if (points.size() < 2 || !planner_data_->dynamic_object) {
    return std::nullopt;
  }
  const double now = clock_->now().seconds();
  const auto & vp = planner_data_->parameters;

  // the strip ego's body uses while it is out: up to its largest planned shift in the next 50 m (while paused, the
  // shift it was heading for)
  double max_shift = 0.0;
  {
    const auto & sp = helper_->getPreviousSplineShiftPath();
    if (!sp.path.points.empty() && sp.shift_length.size() == sp.path.points.size()) {
      const auto idx = planner_data_->findEgoIndex(sp.path.points);
      double s = 0.0;
      for (size_t i = idx; i < sp.path.points.size() && s < 50.0; ++i) {
        if (i > idx) s += autoware_utils::calc_distance2d(sp.path.points.at(i - 1), sp.path.points.at(i));
        if (std::abs(sp.shift_length.at(i)) > std::abs(max_shift)) max_shift = sp.shift_length.at(i);
      }
    }
    if (nero_paused_ && std::abs(nero_pause_desired_) > std::abs(max_shift)) {
      max_shift = nero_pause_desired_;
    }
  }

  std::optional<NeroOncoming> best;
  double best_lon = std::numeric_limits<double>::max();
  std::set<std::string> seen;
  for (const auto & o : planner_data_->dynamic_object->objects) {
    const auto label = utils::getHighestProbLabel(o.classification);
    const bool vehicle = label == ObjectClassification::CAR || label == ObjectClassification::TRUCK ||
                         label == ObjectClassification::BUS || label == ObjectClassification::TRAILER ||
                         label == ObjectClassification::MOTORCYCLE ||
                         label == ObjectClassification::BICYCLE;
    if (!vehicle && label != ObjectClassification::UNKNOWN) continue;
    const auto & pose = o.kinematics.initial_pose_with_covariance.pose;
    const auto & tw = o.kinematics.initial_twist_with_covariance.twist;
    const double v = std::hypot(tw.linear.x, tw.linear.y);
    const double lon = autoware::motion_utils::calcSignedArcLength(points, getEgoPosition(), pose.position);
    if (lon < 0.0 || lon > 80.0) continue;
    const double lat = autoware::motion_utils::calcLateralOffset(points, pose.position);
    const auto near_idx = autoware::motion_utils::findNearestIndex(points, pose.position);
    const double path_yaw = tf2::getYaw(points.at(near_idx).point.pose.orientation);
    const double motion_yaw = tf2::getYaw(pose.orientation) + std::atan2(tw.linear.y, tw.linear.x);
    const bool against = std::abs(autoware_utils::normalize_radian(motion_yaw - path_yaw)) > 0.75 * M_PI;

    // speed history by place (the tracker re-creates ids; the low UNKNOWN fragment becomes a CAR)
    const auto & u = o.object_id.uuid;
    const std::string id(u.begin(), u.end());
    auto it = nero_onc_hist_.find(id);
    if (it == nero_onc_hist_.end()) {
      for (auto h = nero_onc_hist_.begin(); h != nero_onc_hist_.end(); ++h) {
        if (
          !seen.count(h->first) && now - h->second.last < 0.6 &&
          std::hypot(h->second.x - pose.position.x, h->second.y - pose.position.y) < 4.0) {
          auto moved = h->second;
          moved.pos.clear();  // the position trigger uses one track id only (a hand-over jumps the box)
          nero_onc_hist_.erase(h);
          it = nero_onc_hist_.emplace(id, moved).first;
          break;
        }
      }
      if (it == nero_onc_hist_.end()) {
        it = nero_onc_hist_.emplace(id, NeroHist{0.0, 0.0, now, {}, -1.0, {}, false}).first;
      }
    }
    seen.insert(id);
    auto & h = it->second;
    h.x = pose.position.x;
    h.y = pose.position.y;
    h.last = now;
    h.v.emplace_back(now, v);
    while (!h.v.empty() && now - h.v.front().first > nero_ovl::SLOWING_WINDOW_S) h.v.pop_front();
    // (a) position trigger on this track id
    {
      const double path_yaw_c = std::cos(path_yaw), path_yaw_s = std::sin(path_yaw);
      const double v_toward = against ? v : -v;
      h.pos.push_back({now, pose.position.x, pose.position.y, v_toward});
      while (!h.pos.empty() && now - h.pos.front()[0] > 0.6) h.pos.pop_front();
      for (const auto & q : h.pos) {
        const double moved = -((pose.position.x - q[1]) * path_yaw_c + (pose.position.y - q[2]) * path_yaw_s);
        if (now - q[0] > 1.0e-3 && moved >= 0.5 && q[3] >= 1.0 && v_toward >= 1.0) {
          if (!h.latched) h.since = now - nero_ovl::SUSTAIN_S;
          h.latched = true;
          break;
        }
      }
    }
    // (b) sustained speed
    const double v_min = vehicle ? nero_ovl::VEHICLE_MIN_V : nero_ovl::UNKNOWN_MIN_V;
    if (against && v >= v_min) {
      if (h.since < 0.0) h.since = now;
      if (now - h.since >= nero_ovl::SUSTAIN_S) h.latched = true;
    } else if (!against && !h.latched) {
      h.since = -1.0;
    }
    double v_recent_max = v;
    for (const auto & [t, vv] : h.v) v_recent_max = std::max(v_recent_max, vv);
    const bool slowing = v_recent_max - v >= nero_ovl::SLOWING_DV;
    // a car judged oncoming stays one while it slows down (else it would vanish below the speed threshold)
    if (!h.latched || !against) continue;
    if (v < 0.3) continue;  // stopped: no longer coming (obstacle_stop / avoidance see a standing car)

    // its box reaches into the strip of ego's body at the largest shift (shift side only)
    const double side = max_shift >= 0.0 ? 1.0 : -1.0;
    const double reach = side * lat - 0.5 * o.shape.dimensions.y;
    if (std::abs(max_shift) < 0.1 || side * lat < 0.0 ||
        reach > std::abs(max_shift) + 0.5 * vp.vehicle_width + 0.3) {
      continue;
    }
    if (lon < best_lon) {
      best_lon = lon;
      NeroOncoming r;
      char buf[8];
      std::snprintf(buf, sizeof(buf), "%02x%02x", u[0], u[1]);
      r.id = buf;
      r.v = v;
      r.v_recent_max = v_recent_max;
      r.gap = lon - 0.5 * o.shape.dimensions.x - vp.base_link2front;
      r.slowing = slowing;
      const auto & e = nero_ovl::env();
      r.can_stop = r.gap >= v * e.driver_reaction_s + v * v / (2.0 * e.driver_decel) + 1.0;
      r.vehicle = vehicle;
      best = r;
    }
  }
  for (auto h = nero_onc_hist_.begin(); h != nero_onc_hist_.end();) {
    h = now - h->second.last > 2.0 ? nero_onc_hist_.erase(h) : std::next(h);
  }
  return best;
}

std::optional<double> StaticObstacleAvoidanceModule::neroPauseTargetDistance(const AvoidancePlanningData & data) const
{
  // the target the shift is for: the module's stop target, else the nearest target ahead (a shift may start before
  // the target "requires" avoidance, e.g. at 4-5 m/s 20 m before it)
  if (nero_lane_block_) return nero_lane_block_->second;  // NERO_SOA_LANE_BLOCKED: stop before the vehicle in the lane
  if (data.stop_target_object) return data.stop_target_object.value().longitudinal;
  std::optional<double> d;
  for (const auto & o : data.target_objects) {
    // NERO_SOA_RETURN_NO_NEED: a target that needs no shift at all (a kerb object) is not what a shift is for: a pause
    // that stops before it would leave ego standing out on its way back (sim: hit by a car that never slows)
    if (nero_ret::no_need() && o.info == ObjectInfo::SAME_DIRECTION_SHIFT) continue;
    if (o.longitudinal > 0.0 && (!d || o.longitudinal < d.value())) d = o.longitudinal;
  }
  return d;
}

bool StaticObstacleAvoidanceModule::neroOncomingPause(AvoidancePlanningData & data) const
{
  const auto & e = nero_ovl::env();
  if (e.pause_margin < 0.0) {
    return false;
  }
  const double now = clock_->now().seconds();
  if (nero_paused_) {
    if (!neroPauseTargetDistance(data)) {
      RCLCPP_WARN(getLogger(), "[NERO_SOA_OVL] pause ends: no target ahead");
      nero_paused_ = false;
      return false;
    }
    if (nero_onc_ || !nero_corridor_blocker_.empty()) {
      nero_pause_clear_since_ = -1.0;
      return true;
    }
    if (nero_pause_clear_since_ < 0.0) nero_pause_clear_since_ = now;
    if (now - nero_pause_clear_since_ < nero_ovl::RESUME_SAFE_S) return true;
    RCLCPP_WARN(
      getLogger(), "[NERO_SOA_OVL] no oncoming car and a free pass for %.1f s: the pass goes on",
      nero_ovl::RESUME_SAFE_S);
    nero_paused_ = false;
    return false;
  }
  const auto target_distance = neroPauseTargetDistance(data);
  if ((!nero_onc_ && nero_corridor_blocker_.empty()) || !helper_->isShifted() || !target_distance) {
    return false;
  }
  const std::string why = nero_onc_ ? "oncoming " + nero_onc_->id : "pass blocked by " + nero_corridor_blocker_;

  // the shift ego is heading for (largest planned in the next 50 m)
  double desired = 0.0;
  const auto & sp = helper_->getPreviousSplineShiftPath();
  if (!sp.path.points.empty() && sp.shift_length.size() == sp.path.points.size()) {
    const auto idx = planner_data_->findEgoIndex(sp.path.points);
    double s = 0.0;
    for (size_t i = idx; i < sp.path.points.size() && s < 50.0; ++i) {
      if (i > idx) s += autoware_utils::calc_distance2d(sp.path.points.at(i - 1), sp.path.points.at(i));
      if (std::abs(sp.shift_length.at(i)) > std::abs(desired)) desired = sp.shift_length.at(i);
    }
  }
  if (std::abs(desired) < 0.1) return false;
  const double sign = desired > 0.0 ? 1.0 : -1.0;
  const bool left = desired > 0.0;
  const double actual =
    sign * autoware::motion_utils::calcLateralOffset(data.reference_path.points, getEgoPosition());
  const double v = getEgoSpeed();
  const double room = target_distance.value();
  // the hold e.pause_margin inside the lane if the way back to it fits before the target, else 0.1 m inside
  // (the RC1 study's over_line_margin); the way back at ego's own speed and the module's sharpest lateral jerk
  std::optional<double> hold;
  double back = 0.0, needed = 0.0;
  for (const double margin : {e.pause_margin, std::min(e.pause_margin, 0.1)}) {
    const auto h = neroInLaneShiftLimit(data, left, margin);
    if (!h) return false;
    if (sign * desired <= h.value() + 0.02) return false;  // the shift stays inside the lane anyway
    back = actual > h.value() ? std::max(
                                  nero_ovl::MIN_EASE_DISTANCE,
                                  autoware::motion_utils::calc_longitudinal_dist_from_jerk(
                                    actual - h.value(), parameters_->lateral_max_jerk_map.back(), std::max(v, 1.0)))
                              : nero_ovl::MIN_EASE_DISTANCE;
    needed = std::max(back, v * v / (2.0 * nero_ovl::STOP_DECEL)) + planner_data_->parameters.base_link2front +
             nero_ovl::BESIDE_MARGIN;
    if (room >= needed) {
      hold = h;
      break;
    }
  }
  // NERO_SOA_PATH_FROM_EGO: a way back into the lane that runs into a standing object beside / ahead (the truck ego is
  // passing: #17's placement) which holding ego's shift clears is no way back
  if (hold && nero_pfe::env().on) {
    const auto & ref = data.reference_path.points;
    std::vector<geometry_msgs::msg::Pose> back_poses, hold_poses;
    for (size_t i = data.ego_closest_path_index; i < ref.size() && i < data.arclength_from_ego.size(); ++i) {
      const double s = data.arclength_from_ego.at(i);
      if (s > back + planner_data_->parameters.base_link2front) break;
      const double x = std::clamp(s / std::max(back, 0.1), 0.0, 1.0);
      const double lat = actual + (hold.value() - actual) * x * x * (3.0 - 2.0 * x);
      back_poses.push_back(autoware_utils::calc_offset_pose(ref.at(i).point.pose, 0.0, sign * lat, 0.0));
      hold_poses.push_back(autoware_utils::calc_offset_pose(ref.at(i).point.pose, 0.0, sign * actual, 0.0));
    }
    const auto blocker = neroBlocker(back_poses, data, false);
    if (!blocker.empty() && neroBlocker(hold_poses, data, false).empty()) {
      RCLCPP_WARN_THROTTLE(
        getLogger(), *clock_, 500,
        "[NERO_SOA_PFE] %s: the way back into the lane (%.2f -> %.2f m over %.1f m) runs into the standing %s beside: no "
        "way back",
        why.c_str(), actual, hold.value(), back, blocker.c_str());
      hold.reset();
    }
  }
  if (!hold && nero_lane_block_ && !nero_onc_ && room > planner_data_->parameters.base_link2front) {
    // NERO_SOA_LANE_BLOCKED: no way back before the vehicle: hold the shift ego has (never further out), stop before it
    hold = std::max(0.0, actual);
    back = nero_ovl::MIN_EASE_DISTANCE;
    RCLCPP_WARN_THROTTLE(
      getLogger(), *clock_, 500,
      "[NERO_SOA_LANE] %s: no way back into the lane before it (needs %.1f m, has %.1f m): holding %.2f m", why.c_str(),
      needed, room, hold.value());
  }
  if (!hold) {
    RCLCPP_WARN_THROTTLE(
      getLogger(), *clock_, 500,
      "[NERO_SOA_OVL] %s: no way back into the lane before the target (needs %.1f m, has %.1f m)", why.c_str(),
      needed, room);
    return false;
  }
  // ... and only if ego is back in the lane before a car that keeps coming arrives (else: stop at once, below)
  if (nero_onc_) {
    const auto & o = nero_onc_.value();
    const double t_ret = needed / std::max(v, 1.0);
    if (!o.slowing && o.gap - (o.v + v) * t_ret < 1.0) {
      RCLCPP_WARN_THROTTLE(
        getLogger(), *clock_, 500,
        "[NERO_SOA_OVL] oncoming %s %.1f m/s, gap %.1f m: it arrives before ego is back in the lane (%.1f s)",
        o.id.c_str(), o.v, o.gap, t_ret);
      return false;
    }
  }
  const double target = actual < hold.value() ? std::max(0.0, std::min(actual + 0.05, hold.value())) : hold.value();
  nero_pause_shift_ = sign * target;
  nero_pause_ease_ = back;
  nero_pause_desired_ = desired;
  nero_pause_start_ = true;
  nero_paused_ = true;
  nero_pause_clear_since_ = -1.0;
  RCLCPP_WARN(
    getLogger(),
    "[NERO_SOA_OVL] %s, during the shift: holding %.2f m (ego %.2f m, heading for %.2f m) within %.1f m, stop before "
    "the target %.1f m ahead",
    why.c_str(), nero_pause_shift_, sign * actual, desired, back, room);
  return true;
}

void StaticObstacleAvoidanceModule::neroOncomingCommitted(const AvoidancePlanningData & data) const
{
  nero_go_through_ = false;
  if (!nero_ovl::env().committed || !nero_onc_ || !helper_->isShifted()) {
    if (nero_stop_now_) {
      RCLCPP_WARN(getLogger(), "[NERO_SOA_OVL] stop released: no oncoming car any more");
    }
    nero_stop_now_ = false;
    return;
  }
  const auto & o = nero_onc_.value();
  const auto & vp = planner_data_->parameters;
  // the quick return's geometry: the rear clears every target that needed the shift (+1 m), then back into the lane
  // at the module's sharpest lateral jerk at ego's speed
  double clear = 0.0;
  bool target_ahead = false;
  for (const auto & t : data.target_objects) {
    if (!t.avoid_required) continue;
    if (t.longitudinal > vp.base_link2front) target_ahead = true;
    const auto poly = autoware_utils::to_polygon2d(t.object.kinematics.initial_pose_with_covariance.pose, t.object.shape);
    for (const auto & q : poly.outer()) {
      clear = std::max(
        clear, autoware::motion_utils::calcSignedArcLength(
                 data.reference_path.points, getEgoPosition(), autoware_utils::create_point(q.x(), q.y(), 0.0)) +
                 vp.base_link2rear + nero_ovl::BESIDE_MARGIN);
    }
  }
  const double v = std::max(getEgoSpeed(), 1.0);
  const double s_back = clear + autoware::motion_utils::calc_longitudinal_dist_from_jerk(
                                  std::abs(helper_->getEgoShift()), parameters_->lateral_max_jerk_map.back(), v);
  const double t_clear = s_back / v;
  const bool finish_ok = o.gap - o.v * t_clear > s_back + 2.0;
  const bool past = !target_ahead && clear < 0.5 &&
                    neroReturnBlocker(
                      data, helper_->getEgoShift(), clear,
                      autoware::motion_utils::calc_longitudinal_dist_from_jerk(
                        std::abs(helper_->getEgoShift()), parameters_->lateral_max_jerk_map.back(), v))
                      .empty();
  // it slows down (Ahmed 10-24: "it will start slowing down as a signal for the robot to go through"), or ego is
  // already past the targets (going on takes it out of the car's way sooner than standing), or ego is back in its
  // lane before the car arrives: go through, with the quick return. Else, beside the target with a car that keeps
  // coming: stop at once (a robot standing in the car's lane, the least harm if it does not stop)
  const char * why = o.slowing ? "it slows down" : past ? "ego is past the targets"
                     : finish_ok ? "ego is back in its lane before it arrives" : nullptr;
  if (why) {
    if (nero_stop_now_ || !nero_returning_) {
      RCLCPP_WARN_THROTTLE(
        getLogger(), *clock_, 1000, "[NERO_SOA_OVL] oncoming %s %.1f m/s (was %.1f), gap %.1f m: %s, going through",
        o.id.c_str(), o.v, o.v_recent_max, o.gap, why);
    }
    nero_stop_now_ = false;
    nero_go_through_ = true;
    return;
  }
  if (!nero_stop_now_) {
    RCLCPP_WARN(
      getLogger(),
      "[NERO_SOA_OVL] oncoming %s %.1f m/s keeps coming, gap %.1f m, ego back in its lane only in %.1f m (%.1f s): "
      "stop now",
      o.id.c_str(), o.v, o.gap, s_back, t_clear);
  }
  nero_stop_now_ = true;
}

std::string StaticObstacleAvoidanceModule::neroBlocker(
  const std::vector<geometry_msgs::msg::Pose> & body_poses, const AvoidancePlanningData & data,
  const bool skip_targets) const
{
  if (!planner_data_->dynamic_object || body_poses.empty()) return "";
  const auto & vp = planner_data_->parameters;
  std::vector<autoware_utils::Polygon2d> fps;
  for (const auto & p : body_poses) {
    fps.push_back(autoware_utils::to_footprint(p, vp.base_link2front, vp.base_link2rear, vp.vehicle_width));
  }
  for (const auto & o : planner_data_->dynamic_object->objects) {
    const auto & tw = o.kinematics.initial_twist_with_covariance.twist.linear;
    if (std::hypot(tw.x, tw.y) >= 0.5) continue;  // standing objects only
    if (skip_targets) {
      // the corridor check: only objects outside ego's own lane (in the oncoming lane), and not the targets the
      // planned path avoids (it keeps their margins by design); a target the module cannot avoid counts
      if (std::any_of(data.target_objects.begin(), data.target_objects.end(), [&](const auto & t) {
            return t.is_avoidable && t.object.object_id == o.object_id;
          })) {
        continue;
      }
      const auto & c = o.kinematics.initial_pose_with_covariance.pose.position;
      const lanelet::BasicPoint2d cp{c.x, c.y};
      if (std::any_of(data.current_lanelets.begin(), data.current_lanelets.end(), [&](const auto & ll) {
            return boost::geometry::within(cp, ll.polygon2d().basicPolygon());
          })) {
        continue;
      }
    }
    const auto label = utils::getHighestProbLabel(o.classification);
    const bool vehicle = label == ObjectClassification::CAR || label == ObjectClassification::TRUCK ||
                         label == ObjectClassification::BUS || label == ObjectClassification::TRAILER ||
                         label == ObjectClassification::MOTORCYCLE;
    const double margin = vehicle ? 0.1 : 0.3;  // obstacle_stop's lateral margins on the car
    const auto poly = autoware_utils::to_polygon2d(o.kinematics.initial_pose_with_covariance.pose, o.shape);
    for (const auto & fp : fps) {
      if (boost::geometry::distance(fp, poly) < margin) {
        char buf[48];
        std::snprintf(
          buf, sizeof(buf), "%s %02x%02x", vehicle ? "vehicle" : "object", o.object_id.uuid[0], o.object_id.uuid[1]);
        return buf;
      }
    }
  }
  return "";
}

std::string StaticObstacleAvoidanceModule::neroCorridorBlocker(const AvoidancePlanningData & data) const
{
  // the planned pass (the candidate path: also while waiting before the line) where the body is over the line, before
  // the body has left the lane
  const auto & sp = data.candidate_path.path.points.empty() ? helper_->getPreviousSplineShiftPath() : data.candidate_path;
  if (sp.path.points.empty() || sp.shift_length.size() != sp.path.points.size()) return "";
  const auto idx = planner_data_->findEgoIndex(sp.path.points);
  double max_shift = 0.0;
  for (size_t i = idx; i < sp.path.points.size(); ++i) {
    if (std::abs(sp.shift_length.at(i)) > std::abs(max_shift)) max_shift = sp.shift_length.at(i);
  }
  const auto room = neroInLaneShiftLimit(data, max_shift > 0.0, 0.0);
  if (!room || std::abs(max_shift) <= room.value() || std::abs(helper_->getEgoShift()) > room.value()) return "";
  std::vector<geometry_msgs::msg::Pose> poses;
  double s = 0.0;
  for (size_t i = idx; i < sp.path.points.size() && s < 40.0; ++i) {
    if (i > idx) s += autoware_utils::calc_distance2d(sp.path.points.at(i - 1), sp.path.points.at(i));
    if (std::abs(sp.shift_length.at(i)) > room.value()) poses.push_back(sp.path.points.at(i).point.pose);
  }
  return neroBlocker(poses, data, true);
}

std::optional<std::pair<std::string, double>> StaticObstacleAvoidanceModule::neroLaneBlocker(
  const AvoidancePlanningData & data) const
{
  // the planned pass (the candidate path: also while waiting / paused) and how far the body is out of the lane along it
  const auto & sp = data.candidate_path.path.points.empty() ? helper_->getPreviousSplineShiftPath() : data.candidate_path;
  if (
    !planner_data_->dynamic_object || sp.path.points.empty() ||
    sp.shift_length.size() != sp.path.points.size() || data.reference_path.points.size() < 2) {
    return std::nullopt;
  }
  const auto idx = planner_data_->findEgoIndex(sp.path.points);
  std::vector<std::pair<double, double>> plan;  // (arc length from ego, shift)
  double max_shift = 0.0, s = 0.0;
  for (size_t i = idx; i < sp.path.points.size() && s < 60.0; ++i) {
    if (i > idx) s += autoware_utils::calc_distance2d(sp.path.points.at(i - 1), sp.path.points.at(i));
    plan.emplace_back(s, sp.shift_length.at(i));
    if (std::abs(sp.shift_length.at(i)) > std::abs(max_shift)) max_shift = sp.shift_length.at(i);
  }
  const auto room = neroInLaneShiftLimit(data, max_shift > 0.0, 0.0);
  if (!room || std::abs(max_shift) <= room.value()) return std::nullopt;  // the pass stays in the lane
  const auto & vp = planner_data_->parameters;
  constexpr double OBSTACLE_STOP_MARGIN = 5.0;  // obstacle_stop's stop margin on the car
  std::optional<std::pair<std::string, double>> best;
  for (const auto & o : planner_data_->dynamic_object->objects) {
    const auto label = utils::getHighestProbLabel(o.classification);
    const bool vehicle = label == ObjectClassification::CAR || label == ObjectClassification::TRUCK ||
                         label == ObjectClassification::BUS || label == ObjectClassification::TRAILER ||
                         label == ObjectClassification::MOTORCYCLE;
    if (!vehicle) continue;
    const auto & tw = o.kinematics.initial_twist_with_covariance.twist.linear;
    if (std::hypot(tw.x, tw.y) >= 0.5) continue;  // standing only
    if (!neroNeverPassed(data, o)) continue;  // only a vehicle the module has judged traffic (never avoided)
    const auto & c = o.kinematics.initial_pose_with_covariance.pose.position;
    const lanelet::BasicPoint2d cp{c.x, c.y};
    if (!std::any_of(data.current_lanelets.begin(), data.current_lanelets.end(), [&](const auto & ll) {
          return boost::geometry::within(cp, ll.polygon2d().basicPolygon());
        })) {
      continue;  // not in ego's own lane
    }
    double s_near = 1e9, s_far = -1e9;
    const auto poly = autoware_utils::to_polygon2d(o.kinematics.initial_pose_with_covariance.pose, o.shape);
    if (std::any_of(data.target_objects.begin(), data.target_objects.end(), [&](const auto & t) {
          return t.is_avoidable && boost::geometry::intersects(t.envelope_poly, poly);
        })) {
      continue;  // the place of a target the module passes (e.g. the vehicle's own low fragment): not a blocker
    }
    for (const auto & q : poly.outer()) {
      const double sq = autoware::motion_utils::calcSignedArcLength(
        data.reference_path.points, getEgoPosition(), autoware_utils::create_point(q.x(), q.y(), 0.0));
      s_near = std::min(s_near, sq);
      s_far = std::max(s_far, sq);
    }
    if (s_near < vp.base_link2front || s_near > 60.0) continue;  // already beside ego (passed by now), or far
    // the body out of the lane anywhere from where obstacle_stop stops ego for it to its far end: ego would stand out
    const double from = std::max(0.0, s_near - vp.base_link2front - OBSTACLE_STOP_MARGIN);
    const bool out = std::any_of(plan.begin(), plan.end(), [&](const auto & p) {
      return p.first >= from && p.first <= s_far && std::abs(p.second) > room.value();
    });
    if (!out || (best && best->second <= s_near)) continue;
    nero_lane_latch_ = std::string(o.object_id.uuid.begin(), o.object_id.uuid.end());
    char buf[48];
    std::snprintf(
      buf, sizeof(buf), "vehicle %02x%02x%02x%02x", o.object_id.uuid[0], o.object_id.uuid[1], o.object_id.uuid[2],
      o.object_id.uuid[3]);
    best = std::make_pair(std::string(buf), s_near);
  }
  return best;
}

bool StaticObstacleAvoidanceModule::neroNeverPassed(
  const AvoidancePlanningData & data, const PredictedObject & o) const
{
  // not a target, and judged "not a parked object" (traffic: near a stop line / junction, queued, not parked): the
  // module never avoids it. A stopped vehicle it waits and sees about (ambiguous) may still be passed: not counted.
  if (std::any_of(data.target_objects.begin(), data.target_objects.end(), [&](const auto & t) {
        return t.object.object_id == o.object_id;
      })) {
    return false;
  }
  const auto it = std::find_if(data.other_objects.begin(), data.other_objects.end(), [&](const auto & t) {
    return t.object.object_id == o.object_id && t.info == ObjectInfo::IS_NOT_PARKING_OBJECT;
  });
  // narrowweave (10-26 #7 / #8, 44311): a row of cars parked in the lane is "queued" by the queue rule; only a vehicle
  // that waits for something is traffic (else ego waited behind the parked row for good)
  return it != data.other_objects.end() && neroWaiting(data, *it);
}

std::vector<double> StaticObstacleAvoidanceModule::neroStopsAhead(const AvoidancePlanningData & data) const
{
  std::vector<double> out;
  const auto & ref = data.reference_path.points;
  if (ref.size() < 2) return out;
  const auto add = [&](const double x, const double y) {
    out.push_back(
      autoware::motion_utils::calcSignedArcLength(ref, getEgoPosition(), autoware_utils::create_point(x, y, 0.0)));
  };
  for (const auto * lls : {&data.current_lanelets, &data.extend_lanelets}) {
    for (const auto & llt : *lls) {
      if (llt.hasAttribute("turn_direction") && !llt.centerline().empty()) {
        add(llt.centerline().front().x(), llt.centerline().front().y());
      }
      for (const auto & ts : llt.regulatoryElementsAs<lanelet::TrafficSign>()) {
        for (const auto & l : ts->refLines()) {
          if (!l.empty()) add(0.5 * (l.front().x() + l.back().x()), 0.5 * (l.front().y() + l.back().y()));
        }
      }
      for (const auto & rm : llt.regulatoryElementsAs<lanelet::autoware::RoadMarking>()) {
        const auto l = rm->roadMarking();
        if (l.attributeOr(lanelet::AttributeName::Type, "") == std::string("stop_line") && !l.empty()) {
          add(0.5 * (l.front().x() + l.back().x()), 0.5 * (l.front().y() + l.back().y()));
        }
      }
    }
  }
  return out;
}

bool StaticObstacleAvoidanceModule::neroWaiting(const AvoidancePlanningData & data, const ObjectData & od) const
{
  static const double stop_line_m = [] {
    const char * v = std::getenv("NERO_SOA_AMBIGUOUS_STOP_LINE_M");
    return v ? std::atof(v) : 20.0;
  }();
  static const double queue_m = [] {
    const char * v = std::getenv("NERO_SOA_AMBIGUOUS_QUEUE_M");
    return v ? std::atof(v) : 15.0;
  }();
  const auto & ref = data.reference_path.points;
  if (ref.size() < 2 || data.arclength_from_ego.size() != ref.size() || !planner_data_->dynamic_object) return true;
  const auto stops = neroStopsAhead(data);
  // the standing / slow vehicles near ego's lane, each projected once: (s of its centre, half length, speed, id)
  struct Veh
  {
    double s, half, v;
    unique_identifier_msgs::msg::UUID id;
  };
  std::vector<Veh> vehs;
  for (const auto & o : planner_data_->dynamic_object->objects) {
    const auto label = utils::getHighestProbLabel(o.classification);
    if (
      label != ObjectClassification::CAR && label != ObjectClassification::TRUCK &&
      label != ObjectClassification::BUS && label != ObjectClassification::TRAILER &&
      label != ObjectClassification::MOTORCYCLE) {
      continue;
    }
    const auto & tw = o.kinematics.initial_twist_with_covariance.twist.linear;
    const double v = std::hypot(tw.x, tw.y);
    if (v > 2.0) continue;
    const auto & p = o.kinematics.initial_pose_with_covariance.pose.position;
    const size_t seg = autoware::motion_utils::findNearestSegmentIndex(ref, p);
    if (std::abs(autoware::motion_utils::calcLateralOffset(ref, p, seg)) > 1.5) continue;
    vehs.push_back(
      {data.arclength_from_ego.at(seg) + autoware::motion_utils::calcLongitudinalOffsetToSegment(ref, seg, p),
       0.5 * o.shape.dimensions.x, v, o.object_id});
  }
  const auto stop_factor = [&](const unique_identifier_msgs::msg::UUID & id) {  // the module's own distance, if judged
    for (const auto * lst : {&data.target_objects, &data.other_objects}) {
      for (const auto & t : *lst) {
        if (t.object.object_id == id) return t.to_stop_factor_distance;
      }
    }
    return std::numeric_limits<double>::infinity();
  };
  const auto it0 = std::find_if(vehs.begin(), vehs.end(), [&](const auto & q) { return q.id == od.object.object_id; });
  if (it0 == vehs.end()) return true;  // not near ego's lane: as before
  const Veh * cur = &(*it0);
  for (int k = 0; k < 12; ++k) {
    if (k > 0 && cur->v >= 0.5) return true;  // a vehicle ahead in the row moves: a queue in traffic
    const double s_front = cur->s + cur->half;
    if (stop_factor(cur->id) < stop_line_m) return true;
    for (const double l : stops) {
      if (l >= s_front - 1.0 && l - s_front < stop_line_m) return true;  // waiting before a stop line / junction
    }
    const Veh * next = nullptr;  // the next vehicle ahead in ego's lane within the queue gap
    for (const auto & q : vehs) {
      if (q.id == cur->id || q.s <= cur->s || q.s - q.half - s_front > queue_m) continue;
      if (!next || q.s < next->s) next = &q;
    }
    if (!next) return false;  // the head of the row waits for nothing: parked
    cur = next;
  }
  return true;
}

void StaticObstacleAvoidanceModule::neroUpdateLaneLatch(const AvoidancePlanningData & data) const
{
  nero_lane_stop_.reset();
  if (nero_lane_latch_.empty() || !planner_data_->dynamic_object || data.reference_path.points.size() < 2) {
    nero_lane_latch_.clear();
    return;
  }
  const auto & vp = planner_data_->parameters;
  constexpr double OBSTACLE_STOP_MARGIN = 5.0;  // obstacle_stop's stop margin on the car
  for (const auto & o : planner_data_->dynamic_object->objects) {
    if (std::string(o.object_id.uuid.begin(), o.object_id.uuid.end()) != nero_lane_latch_) continue;
    const auto & tw = o.kinematics.initial_twist_with_covariance.twist.linear;
    const auto & c = o.kinematics.initial_pose_with_covariance.pose.position;
    const lanelet::BasicPoint2d cp{c.x, c.y};
    const bool in_lane = std::any_of(data.current_lanelets.begin(), data.current_lanelets.end(), [&](const auto & ll) {
      return boost::geometry::within(cp, ll.polygon2d().basicPolygon());
    });
    double s_near = 1e9;
    const auto poly = autoware_utils::to_polygon2d(o.kinematics.initial_pose_with_covariance.pose, o.shape);
    for (const auto & q : poly.outer()) {
      s_near = std::min(
        s_near, autoware::motion_utils::calcSignedArcLength(
                  data.reference_path.points, getEgoPosition(), autoware_utils::create_point(q.x(), q.y(), 0.0)));
    }
    if (std::hypot(tw.x, tw.y) < 0.5 && in_lane && s_near >= vp.base_link2front && neroNeverPassed(data, o)) {
      nero_lane_stop_ = std::max(0.0, s_near - vp.base_link2front - OBSTACLE_STOP_MARGIN);
      return;
    }
    break;
  }
  nero_lane_latch_.clear();  // gone, moving, out of the lane or beside ego: released
}

std::optional<double> StaticObstacleAvoidanceModule::neroPassSightNeeded(const AvoidancePlanningData & data) const
{
  // only before the body leaves the lane, while the planned path (the candidate: also while waiting) leaves it
  const auto & sp = data.candidate_path.path.points.empty() ? helper_->getPreviousSplineShiftPath() : data.candidate_path;
  if (sp.path.points.empty() || sp.shift_length.size() != sp.path.points.size()) return std::nullopt;
  const auto idx = planner_data_->findEgoIndex(sp.path.points);
  double s = 0.0, max_shift = 0.0;
  size_t i_max = idx;
  for (size_t i = idx; i < sp.path.points.size() && s < 60.0; ++i) {
    if (i > idx) s += autoware_utils::calc_distance2d(sp.path.points.at(i - 1), sp.path.points.at(i));
    if (std::abs(sp.shift_length.at(i)) > std::abs(max_shift)) {
      max_shift = sp.shift_length.at(i);
      i_max = i;
    }
  }
  const auto room = neroInLaneShiftLimit(data, max_shift > 0.0, 0.0);
  if (!room || std::abs(max_shift) <= room.value() || std::abs(helper_->getEgoShift()) > room.value()) {
    return std::nullopt;
  }
  double s_back = -1.0;
  for (size_t i = i_max; i < sp.path.points.size(); ++i) {
    if (std::abs(sp.shift_length.at(i)) <= room.value()) {
      s_back = autoware::motion_utils::calcSignedArcLength(sp.path.points, idx, i);
      break;
    }
  }
  if (s_back < 0.0) {
    // the return is not planned yet (it is approved half-way through the shift): the rear clears the farthest target
    // that needs the shift (+ its longitudinal margin), then the return at the nominal distance
    const auto & vp = planner_data_->parameters;
    double clear = -1.0;
    for (const auto & o : data.target_objects) {
      if (!o.avoid_required) continue;
      const auto poly =
        autoware_utils::to_polygon2d(o.object.kinematics.initial_pose_with_covariance.pose, o.object.shape);
      for (const auto & q : poly.outer()) {
        clear = std::max(
          clear, autoware::motion_utils::calcSignedArcLength(
                   data.reference_path.points, getEgoPosition(), autoware_utils::create_point(q.x(), q.y(), 0.0)) +
                   vp.base_link2rear + 1.0);
      }
    }
    if (clear < 0.0) return std::numeric_limits<double>::infinity();
    s_back = clear + helper_->getNominalAvoidanceDistance(std::abs(max_shift) - room.value());
  }
  // time to get there: ego's speed, +0.5 m/s2 (the module's max_acceleration) up to 4.17 m/s
  const double v0 = std::max(getEgoSpeed(), 0.3), a = 0.5, vmax = std::max(v0, 4.17);
  const double s_acc = (vmax * vmax - v0 * v0) / (2.0 * a);
  const double t_back = s_back <= s_acc ? (std::sqrt(v0 * v0 + 2.0 * a * s_back) - v0) / a
                                        : (vmax - v0) / a + (s_back - s_acc) / vmax;
  // the unseen car's centre R m ahead of ego's front: its front must still be 2 m short of where ego's front is then
  return s_back + nero_ovl::env().sight_v * t_back + 2.3 + 2.0;
}

void StaticObstacleAvoidanceModule::neroFreezeShift(
  const AvoidancePlanningData & data, const double target_shift, const double ease_distance,
  const double return_start, const double return_length, const double return_to)
{
  // port of the RC1 study's freezeShift (rc1-cancel-during-shift): drop every registered line, make the shift at ego
  // the base offset, and ease from it to target_shift over ease_distance
  const auto & points = data.reference_path.points;
  const auto ego_idx = data.ego_closest_path_index;
  if (ego_idx + 2 >= points.size()) {
    return;
  }
  const auto ego_shift = helper_->getEgoShift();
  const auto end_idx =
    utils::static_obstacle_avoidance::findPathIndexFromArclength(data.arclength_from_ego, ease_distance);
  {
    ShiftLine passed;
    passed.start = points.at(ego_idx > 0 ? ego_idx - 1 : 0).point.pose;
    passed.end = points.at(ego_idx).point.pose;
    passed.start_shift_length = ego_shift;
    passed.end_shift_length = ego_shift;
    path_shifter_.setShiftLines(ShiftLineArray{passed});
    path_shifter_.removeBehindShiftLineAndSetBaseOffset(ego_idx);
  }
  ShiftLineArray lines;
  if (std::abs(target_shift - ego_shift) > 1.0e-2 && end_idx > ego_idx + 1) {
    ShiftLine line;
    line.start = points.at(ego_idx).point.pose;
    line.end = points.at(end_idx).point.pose;
    line.start_shift_length = ego_shift;
    line.end_shift_length = target_shift;
    lines.push_back(line);
  }
  if (return_start >= 0.0 && return_length > 0.0 && std::abs(target_shift - return_to) > 1.0e-2) {
    const auto start_idx = std::max(
      ego_idx, utils::static_obstacle_avoidance::findPathIndexFromArclength(data.arclength_from_ego, return_start));
    const auto back_idx = utils::static_obstacle_avoidance::findPathIndexFromArclength(
      data.arclength_from_ego, return_start + return_length);
    if (back_idx > start_idx + 1) {
      ShiftLine back;
      back.start = points.at(start_idx).point.pose;
      back.end = points.at(back_idx).point.pose;
      back.start_shift_length = target_shift;
      back.end_shift_length = return_to;
      lines.push_back(back);
    }
  }
  path_shifter_.setShiftLines(lines);
  generator_.reset();
}

std::string StaticObstacleAvoidanceModule::neroReturnBlocker(
  const AvoidancePlanningData & data, const double shift, const double start, const double length) const
{
  // the body along the reference path: at `shift` up to `start`, then linearly back to the centre over `length`
  const auto & points = data.reference_path.points;
  std::vector<geometry_msgs::msg::Pose> poses;
  for (size_t i = data.ego_closest_path_index; i < points.size() && i < data.arclength_from_ego.size(); ++i) {
    const double s = data.arclength_from_ego.at(i);
    if (s > start + length + 1.0) break;
    const double r = s <= start ? 1.0 : std::max(0.0, 1.0 - (s - start) / std::max(length, 0.1));
    poses.push_back(autoware_utils::calc_offset_pose(points.at(i).point.pose, 0.0, shift * r, 0.0));
  }
  return neroBlocker(poses, data, false);
}

void StaticObstacleAvoidanceModule::neroOncomingReturn(AvoidancePlanningData & data)
{
  if (!nero_ovl::env().committed) {
    nero_returning_ = false;
    return;
  }
  const auto ego_shift = helper_->getEgoShift();
  const auto left = ego_shift > 0.0;
  const auto lane_room = neroInLaneShiftLimit(data, left, 0.0);
  if (!helper_->isShifted() || !lane_room || std::abs(ego_shift) <= lane_room.value()) {
    nero_returning_ = false;  // back in the lane
    return;
  }
  if (nero_returning_ || nero_paused_ || nero_cgw_active_ || !nero_onc_ || !nero_go_through_) {
    return;
  }
  // no target ahead still needs the shift
  const auto & vp = planner_data_->parameters;
  const auto & points = data.reference_path.points;
  double clear = 0.0;
  for (const auto & o : data.target_objects) {
    if (!o.avoid_required) continue;
    if (o.longitudinal > vp.base_link2front) return;
    // where the rear clears it: its farthest point along the path + rear overhang + 1 m
    const auto poly = autoware_utils::to_polygon2d(o.object.kinematics.initial_pose_with_covariance.pose, o.object.shape);
    for (const auto & q : poly.outer()) {
      const double s = autoware::motion_utils::calcSignedArcLength(
        points, getEgoPosition(), autoware_utils::create_point(q.x(), q.y(), 0.0));
      clear = std::max(clear, s + vp.base_link2rear + nero_ovl::BESIDE_MARGIN);
    }
  }
  const double v = std::max(getEgoSpeed(), 1.0);
  const double length = std::max(
    nero_ovl::MIN_EASE_DISTANCE, autoware::motion_utils::calc_longitudinal_dist_from_jerk(
                                   std::abs(ego_shift), parameters_->lateral_max_jerk_map.back(), v));
  const auto blocker = neroReturnBlocker(data, ego_shift, clear, length);
  if (!blocker.empty()) {
    RCLCPP_WARN_THROTTLE(
      getLogger(), *clock_, 1000, "[NERO_SOA_OVL] the way back into the lane is blocked (%s): no quick return",
      blocker.c_str());
    return;
  }
  neroFreezeShift(data, ego_shift, 0.0, clear, length);
  data.new_shift_line.clear();
  data.safe_shift_line.clear();
  nero_returning_ = true;
  RCLCPP_WARN(
    getLogger(),
    "[NERO_SOA_OVL] oncoming %s %.1f m/s, gap %.1f m, beside / past the targets: back into the lane from %.1f m "
    "ahead over %.1f m (shift %.2f m)",
    nero_onc_->id.c_str(), nero_onc_->v, nero_onc_->gap, clear, length, ego_shift);
}

void StaticObstacleAvoidanceModule::neroReturnClear(AvoidancePlanningData & data)
{
  const auto ego_shift = helper_->getEgoShift();
  const auto room = neroInLaneShiftLimit(data, ego_shift > 0.0, 0.0);
  if (!room || std::abs(ego_shift) <= room.value() || nero_paused_ || nero_returning_ || nero_cgw_active_) {
    nero_rcl_holding_ = false;  // the body is in its lane, or overtake v6's pause / quick return / the give-way has it
    return;
  }
  const auto & sp = data.candidate_path;
  const auto & ref = data.reference_path.points;
  if (
    ref.size() < 2 || sp.path.points.size() != ref.size() || sp.shift_length.size() != ref.size() ||
    data.arclength_from_ego.size() != ref.size() || !planner_data_->dynamic_object) {
    return;
  }
  const auto t_start = std::chrono::steady_clock::now();
  const auto & vp = planner_data_->parameters;
  const double half_w = 0.5 * vp.vehicle_width;
  struct Standing
  {
    autoware_utils::Polygon2d poly;
    double margin;        // obstacle_stop's lateral margin for the class (UNKNOWN 0.3, others 0.1) + LAT_EXTRA
    double s_near, s_far; // its nearest / farthest point along the reference path from ego
    std::string id;
  };
  // standing objects near enough to matter: along the next HORIZON m, laterally between the lane centre and ego's
  // shift (plus the body and the margin)
  const double band_lo = std::min(0.0, ego_shift) - half_w - 1.0;
  const double band_hi = std::max(0.0, ego_shift) + half_w + 1.0;
  std::vector<Standing> objs;
  for (const auto & o : planner_data_->dynamic_object->objects) {
    const auto & tw = o.kinematics.initial_twist_with_covariance.twist.linear;
    if (std::hypot(tw.x, tw.y) >= 0.5) continue;  // standing objects only
    // its centre's place along / beside the reference path and its half diagonal: a conservative extent (one
    // projection per object; the exact far end only for the objects the plan runs into)
    const auto & c = o.kinematics.initial_pose_with_covariance.pose.position;
    if (std::hypot(c.x - getEgoPosition().x, c.y - getEgoPosition().y) > nero_rcl::HORIZON + 30.0) continue;
    const double r = 0.5 * std::hypot(o.shape.dimensions.x, o.shape.dimensions.y);
    const size_t seg = autoware::motion_utils::findNearestSegmentIndex(ref, c);
    const double s_c = data.arclength_from_ego.at(seg) + autoware::motion_utils::calcLongitudinalOffsetToSegment(ref, seg, c);
    const double l_c = autoware::motion_utils::calcLateralOffset(ref, c, seg);
    Standing st;
    st.margin =
      (utils::getHighestProbLabel(o.classification) == ObjectClassification::UNKNOWN ? 0.3 : 0.1) +
      nero_rcl::LAT_EXTRA;
    st.s_near = s_c - r;
    st.s_far = s_c + r;
    if (st.s_far < -vp.base_link2rear || st.s_near > nero_rcl::HORIZON + 30.0) continue;  // behind ego / far ahead
    if (l_c + r < band_lo - st.margin || l_c - r > band_hi + st.margin) continue;         // off to the side
    st.poly = autoware_utils::to_polygon2d(o.kinematics.initial_pose_with_covariance.pose, o.shape);
    char id[8];
    std::snprintf(id, sizeof(id), "%02x%02x", o.object_id.uuid[0], o.object_id.uuid[1]);
    st.id = id;
    objs.push_back(st);
  }
  const auto footprint = [&](const geometry_msgs::msg::Pose & p) {
    return autoware_utils::to_footprint(p, vp.base_link2front, vp.base_link2rear, vp.vehicle_width);
  };
  // the body at path arc length s can only come near objects overlapping [s - rear, s + front] (+ 0.5 m)
  const auto near_s = [&](const Standing & o, const double s) {
    return o.s_near <= s + vp.base_link2front + 0.5 && o.s_far >= s - vp.base_link2rear - 0.5;
  };
  const auto ego_idx = data.ego_closest_path_index;
  constexpr double STEP = 0.5;  // body poses every 0.5 m

  // the module's plan where it brings the body back toward the lane: the standing objects it runs into there that a
  // body holding ego's shift at the same place clears
  std::vector<size_t> hit;
  double last_s = -1e9;
  for (size_t i = ego_idx; i < ref.size(); ++i) {
    const double s = data.arclength_from_ego.at(i);
    if (s > nero_rcl::HORIZON) break;
    if (s < 0.0 || std::abs(sp.shift_length.at(i)) > std::abs(ego_shift) - nero_rcl::TOWARD_EPS) continue;
    if (s - last_s < STEP) continue;
    last_s = s;
    std::optional<autoware_utils::Polygon2d> f_plan;
    for (size_t k = 0; k < objs.size(); ++k) {
      if (!near_s(objs.at(k), s) || std::find(hit.begin(), hit.end(), k) != hit.end()) continue;
      if (!f_plan) f_plan = footprint(sp.path.points.at(i).point.pose);
      if (boost::geometry::distance(f_plan.value(), objs.at(k).poly) >= objs.at(k).margin) continue;
      const auto f_hold = footprint(autoware_utils::calc_offset_pose(ref.at(i).point.pose, 0.0, ego_shift, 0.0));
      if (boost::geometry::distance(f_hold, objs.at(k).poly) < objs.at(k).margin) continue;  // in the way out there too
      hit.push_back(k);
    }
  }
  const auto elapsed_ms = [&] {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
  };
  if (hit.empty()) {
    nero_rcl_holding_ = false;
    if (nero_rcl::debug()) {
      RCLCPP_INFO_THROTTLE(
        getLogger(), *clock_, 1000, "[NERO_SOA_RCL] out (shift %.2f): the plan back is clear (%zu objects, %.2f ms)",
        ego_shift, objs.size(), elapsed_ms());
    }
    return;
  }

  // the next stop line or junction lanelet ahead: the way back must be over before the front reaches it
  double limit = data.arclength_from_ego.back() - vp.base_link2front;
  for (const double d : neroStopsAhead(data)) {
    if (d > 0.0) limit = std::min(limit, d - vp.base_link2front - 0.5);
  }

  // hold ego's shift until the rear is past those objects, then back to the centre (smooth S); every standing object
  // in the way back pushes the start past it; the hold itself must be clear
  double start = 0.0;
  std::string who;
  for (const auto k : hit) {
    auto & o = objs.at(k);
    o.s_far = -1e9;  // the exact far end of the objects the plan ran into
    for (const auto & q : o.poly.outer()) {
      o.s_far = std::max(
        o.s_far, autoware::motion_utils::calcSignedArcLength(
                   ref, getEgoPosition(), autoware_utils::create_point(q.x(), q.y(), 0.0)));
    }
    start = std::max(start, o.s_far + vp.base_link2rear + nero_rcl::LON_MARGIN);
    who += (who.empty() ? "" : ",") + objs.at(k).id;
  }
  const double v = std::max(getEgoSpeed(), 1.0);
  const double len_nominal = helper_->getNominalAvoidanceDistance(std::abs(ego_shift));
  const double len_sharp = std::max(
    nero_ovl::MIN_EASE_DISTANCE, autoware::motion_utils::calc_longitudinal_dist_from_jerk(
                                   std::abs(ego_shift), parameters_->lateral_max_jerk_map.back(), v));
  const auto y = [&](const double s, const double st, const double len) {
    if (s <= st) return ego_shift;
    if (s >= st + len) return 0.0;
    const double x = (s - st) / std::max(len, 0.1);
    return ego_shift * (1.0 - x * x * (3.0 - 2.0 * x));
  };
  std::string why = "no way back before the stop line / junction";
  for (const double len : {len_nominal, len_sharp}) {
    double st = start;
    for (int iter = 0; iter < 20 && st + len <= limit; ++iter) {
      double far = -1e9;
      std::string blocker;
      bool hold_blocked = false;
      double last = -1e9;
      for (size_t i = ego_idx; i < ref.size(); ++i) {
        const double s = data.arclength_from_ego.at(i);
        if (s > st + len + vp.base_link2front) break;
        if (s < 0.0 || s - last < STEP) continue;
        last = s;
        std::optional<autoware_utils::Polygon2d> f;
        for (const auto & o : objs) {
          if (!near_s(o, s)) continue;
          if (!f) f = footprint(autoware_utils::calc_offset_pose(ref.at(i).point.pose, 0.0, y(s, st, len), 0.0));
          if (boost::geometry::distance(f.value(), o.poly) >= o.margin) continue;
          if (s <= st) {
            hold_blocked = true;
            blocker = o.id;
          } else if (o.s_far > far) {
            far = o.s_far;
            blocker = o.id;
          }
        }
        if (hold_blocked) break;
      }
      if (hold_blocked) {
        why = "holding the shift runs into " + blocker;
        break;
      }
      if (far < -1e8) {
        neroFreezeShift(data, ego_shift, 0.0, st, len);
        data.new_shift_line.clear();
        data.safe_shift_line.clear();
        RCLCPP_WARN_THROTTLE(
          getLogger(), *clock_, 1000,
          "[NERO_SOA_RCL] the way back toward the lane ran into %s: shift %.2f held, back from %.1f m over %.1f m "
          "(stop line / junction %.1f m; %zu objects, %.2f ms)",
          who.c_str(), ego_shift, st, len, limit, objs.size(), elapsed_ms());
        nero_rcl_holding_ = true;
        return;
      }
      st = std::max(st, far + vp.base_link2rear + nero_rcl::LON_MARGIN);
      why = "way back blocked up to " + blocker;
    }
  }
  nero_rcl_holding_ = false;
  RCLCPP_WARN_THROTTLE(
    getLogger(), *clock_, 1000, "[NERO_SOA_RCL] the way back toward the lane ran into %s, no hold: %s (%.2f ms)",
    who.c_str(), why.c_str(), elapsed_ms());
}

std::optional<StaticObstacleAvoidanceModule::NeroCgwCar> StaticObstacleAvoidanceModule::neroCgwFindCar(
  const AvoidancePlanningData & data, const bool beside) const
{
  // NERO_SOA_CORRIDOR_GIVEWAY: the nearest vehicle ahead standing facing ego, or coming toward it, whose box reaches into
  // ego's own lane (now, or ahead where the standing objects on its side push it there) and comes within CONFLICT_M
  // (standing) / CONFLICT_MOVING_M (coming) of ego's body on its planned path (the previous cycle's path up to the car)
  const auto & ref = data.reference_path.points;
  if (ref.size() < 2 || data.arclength_from_ego.size() != ref.size() || !planner_data_->dynamic_object) {
    return std::nullopt;
  }
  const auto & vp = planner_data_->parameters;
  const double half_w = 0.5 * vp.vehicle_width;
  const double now = clock_->now().seconds();
  const auto & sp = helper_->getPreviousSplineShiftPath();
  const bool sp_ok = !sp.path.points.empty() && sp.shift_length.size() == sp.path.points.size();
  const size_t sp_ego = sp_ok ? planner_data_->findEgoIndex(sp.path.points) : 0;
  const double ego_shift = helper_->getEgoShift();
  const double ego_yaw = tf2::getYaw(getEgoPose().orientation);
  // ego's lane's left bound (from the reference path at ego): a car whose box stays beyond it all the way keeps to its
  // own lane, and a pass against it while ego is out is overtake v6's (NERO_SOA_ONCOMING_PAUSE / _COMMITTED)
  const auto left_room = neroInLaneShiftLimit(data, true, 0.0);
  if (!left_room) {
    return std::nullopt;
  }
  const double lane_left = left_room.value() + half_w;
  // an object's extent in the reference path's frame (corners projected on its centre's segment)
  struct Ext
  {
    double s_near, s_far, lo, hi;
  };
  const auto extent = [&](const autoware_perception_msgs::msg::PredictedObject & o, const size_t seg) {
    const auto poly = autoware_utils::to_polygon2d(o.kinematics.initial_pose_with_covariance.pose, o.shape);
    Ext x{1e9, -1e9, 1e9, -1e9};
    for (const auto & q : poly.outer()) {
      const auto p = autoware_utils::create_point(q.x(), q.y(), 0.0);
      const double s =
        data.arclength_from_ego.at(seg) + autoware::motion_utils::calcLongitudinalOffsetToSegment(ref, seg, p);
      const double l = autoware::motion_utils::calcLateralOffset(ref, p, seg);
      x.s_near = std::min(x.s_near, s);
      x.s_far = std::max(x.s_far, s);
      x.lo = std::min(x.lo, l);
      x.hi = std::max(x.hi, l);
    }
    return x;
  };
  std::set<std::string> seen;
  std::optional<NeroCgwCar> best;
  for (const auto & o : planner_data_->dynamic_object->objects) {
    const auto label = utils::getHighestProbLabel(o.classification);
    const bool vehicle = label == ObjectClassification::CAR || label == ObjectClassification::TRUCK ||
                         label == ObjectClassification::BUS || label == ObjectClassification::TRAILER ||
                         label == ObjectClassification::MOTORCYCLE;
    if (!vehicle && label != ObjectClassification::UNKNOWN) continue;
    const auto & pose = o.kinematics.initial_pose_with_covariance.pose;
    const auto & tw = o.kinematics.initial_twist_with_covariance.twist.linear;
    const double v = std::hypot(tw.x, tw.y);
    const auto & c = pose.position;
    if (std::hypot(c.x - getEgoPosition().x, c.y - getEgoPosition().y) > nero_cgw::HORIZON + 10.0) continue;
    // cheap first: against ego's heading (a parked row along the lane is skipped before any path search)
    const double box_yaw = tf2::getYaw(pose.orientation);
    const bool moving = v >= nero_cgw::STANDING_V;
    const double yaw_obj = moving ? box_yaw + std::atan2(tw.y, tw.x) : box_yaw;
    if (std::abs(autoware_utils::normalize_radian(yaw_obj - ego_yaw)) <= 0.5 * M_PI) continue;
    if (!moving && !vehicle) continue;
    char buf[8];
    std::snprintf(buf, sizeof(buf), "%02x%02x", o.object_id.uuid[0], o.object_id.uuid[1]);
    const std::string id(buf);
    const size_t seg = nero_cgw::seg_of(ref, c);
    const double s_c =
      data.arclength_from_ego.at(seg) + autoware::motion_utils::calcLongitudinalOffsetToSegment(ref, seg, c);
    if (s_c < (beside ? -6.0 : 0.0)) continue;
    const double path_yaw = tf2::getYaw(ref.at(seg).point.pose.orientation);
    if (std::abs(autoware_utils::normalize_radian(yaw_obj - path_yaw)) <= 0.75 * M_PI) continue;
    double v_pos = 0.0;  // a moving UNKNOWN's speed toward ego from its own positions (its tracked speed ramps up late)
    if (moving && !vehicle) {
      // a moving UNKNOWN below 2 m/s (a car's first fragment, its tracked speed still ramping): only once its own
      // positions show it coming (>= 0.5 m toward ego within 0.6 s at >= 1 m/s at both ends: overtake v5's trigger, never
      // on 986 standing UNKNOWN tracks of 49 field captures)
      // (its own displacement along the path, not its distance from the moving ego)
      auto & h = nero_cgw_hist_[std::string(o.object_id.uuid.begin(), o.object_id.uuid.end())];
      h.push_back({now, c.x, c.y, v});
      while (!h.empty() && now - h.front()[0] > 0.6) h.pop_front();
      const double py_c = std::cos(path_yaw), py_s = std::sin(path_yaw);
      const auto toward = [&](const std::array<double, 4> & q) { return -((c.x - q[1]) * py_c + (c.y - q[2]) * py_s); };
      bool came = false;
      for (const auto & q : h) {
        if (now - q[0] > 1.0e-3 && toward(q) >= 0.5 && q[3] >= 1.0 && v >= 1.0) came = true;
      }
      if (now - h.front()[0] >= 0.2) v_pos = std::max(0.0, toward(h.front()) / (now - h.front()[0]));
      seen.insert(std::string(o.object_id.uuid.begin(), o.object_id.uuid.end()));
      if (v < nero_cgw::UNKNOWN_MIN_V && !came) continue;
    }
    const auto x = extent(o, seg);
    // beside / behind ego (while holding: only once fully behind its rear), or far
    if ((!beside && x.s_near < vp.base_link2front) || x.s_far < -vp.base_link2rear - 0.5 || x.s_near > nero_cgw::HORIZON) {
      continue;
    }
    const double lat_c = 0.5 * (x.lo + x.hi);
    if (lat_c <= ego_shift) continue;                                     // on ego's right: not an oncoming car's side
    if (!moving && lat_c < nero_cgw::FACING_MIN_LAT) continue;            // parked facing ego at the right kerb of its lane
    // ego's body on its planned path up to the car: its leftmost edge
    double e_hi = ego_shift + half_w;
    if (sp_ok) {
      double s = 0.0;
      for (size_t i = sp_ego; i < sp.path.points.size() && s <= x.s_near; ++i) {
        if (i > sp_ego) s += autoware_utils::calc_distance2d(sp.path.points.at(i - 1), sp.path.points.at(i));
        e_hi = std::max(e_hi, sp.shift_length.at(i) + half_w);
      }
    }
    // a coming car in its own lane is pushed into ego's lane ahead by the standing objects on its side (its parked row)
    double lo_pred = x.lo;
    if (beside) {
      // holding: the car stays the one ego gives way to until it has passed ego or keeps well off ego's lane
      if (x.lo >= lane_left + nero_cgw::KEEP_TRACK_M) continue;
    } else if (moving && x.lo >= lane_left) {
      for (const auto & q : planner_data_->dynamic_object->objects) {
        const auto & qt = q.kinematics.initial_twist_with_covariance.twist.linear;
        if (std::hypot(qt.x, qt.y) >= nero_cgw::STANDING_V) continue;
        const auto & qc = q.kinematics.initial_pose_with_covariance.pose.position;
        if (std::hypot(qc.x - getEgoPosition().x, qc.y - getEgoPosition().y) > x.s_near + 10.0) continue;
        const auto qx = extent(q, nero_cgw::seg_of(ref, qc));
        if (0.5 * (qx.lo + qx.hi) <= lane_left || qx.s_far < 0.0 || qx.s_near > x.s_near) continue;
        lo_pred = std::min(lo_pred, qx.lo - o.shape.dimensions.y - nero_cgw::SQUEEZE_M);
      }
      if (lo_pred >= lane_left) continue;  // its side leaves it room: it keeps to its own lane (overtake v6's case)
    } else if (x.lo >= lane_left) {
      continue;
    }
    if (!beside && lo_pred > e_hi + (moving ? nero_cgw::CONFLICT_MOVING_M : nero_cgw::CONFLICT_M)) continue;  // clear of ego
    if (best && best->s_near <= x.s_near) continue;
    NeroCgwCar r;
    r.id = id;
    r.s_near = x.s_near;
    r.lat_lo = x.lo;
    r.lat_hi = x.hi;
    r.lo_pred = lo_pred;
    r.v = moving ? std::max(v, v_pos) : 0.0;
    r.width = std::max(o.shape.dimensions.y, 1.0);
    best = r;
  }
  for (auto h = nero_cgw_hist_.begin(); h != nero_cgw_hist_.end();) {
    h = !h->second.empty() && now - h->second.back()[0] > 2.0 ? nero_cgw_hist_.erase(h) : std::next(h);
  }
  (void)seen;
  return best;
}

bool StaticObstacleAvoidanceModule::neroCorridorGiveway(AvoidancePlanningData & data) const
{
  const auto & e = nero_cgw::env();
  const double now = clock_->now().seconds();
  const auto car = neroCgwFindCar(data, nero_cgw_active_);
  const auto & vp = planner_data_->parameters;
  const double v_ego = getEgoSpeed();
  if (nero_cgw_active_ && nero_cgw_back_) {
    // the car has passed: back to the lane centre first, then the hold ends (another car: a new plan)
    if (!car || car->id == nero_cgw_id_) {
      // ... and never waits there: standing 1.5 s into it (obstacle_stop for an object the module's own plan would go
      // round) hands over at once (sim c24 squeeze v11 stood 10 s at s 39 before the kerb fragments its plan avoids)
      const double ego_shift = helper_->getEgoShift();
      if (nero_cgw_back_past_) {
        // objects hidden from the module while holding: keep the line until ego's rear is past them
        const double left = neroCgwBox(data, nero_cgw_back_past_.value())[1] + vp.base_link2rear + 0.5;
        if (left > 0.0 && now < nero_cgw_back_until_) {
          return true;
        }
      } else {
        const bool stood = v_ego < 0.1 && now - (nero_cgw_back_until_ - nero_cgw::BACK_S) > 1.5;
        if (std::abs(ego_shift - nero_cgw_shift_) > 0.05 && now < nero_cgw_back_until_ && !stood) {
          return true;
        }
      }
      RCLCPP_WARN(getLogger(), "[NERO_SOA_CGW] %s back at %.2f: the hold ends", nero_cgw_id_.c_str(), ego_shift);
      nero_cgw_active_ = false;
      nero_cgw_back_ = false;
      nero_cgw_back_past_.reset();
      nero_cgw_seen_.clear();
      nero_cgw_gone_since_ = nero_cgw_still_since_ = -1.0;
      return false;
    }
    nero_cgw_back_ = false;
    nero_cgw_gone_since_ = nero_cgw_still_since_ = -1.0;
    nero_cgw_plan_t_ = -1.0;
  }
  if (nero_cgw_active_) {
    if (now - nero_cgw_seen_t_ >= 0.2) {
      neroCgwRemember(data);
      nero_cgw_seen_t_ = now;
    }
    bool release = false;
    bool parked = false;
    std::string why;
    if (!car) {
      if (nero_cgw_gone_since_ < 0.0) nero_cgw_gone_since_ = now;
      if (now - nero_cgw_gone_since_ >= nero_cgw::RELEASE_S) {
        release = true;
        why = "the car has passed / left ego's way";
      }
    } else {
      nero_cgw_gone_since_ = -1.0;
      nero_cgw_id_ = car->id;
      if (car->v < 0.3 && v_ego < 0.1) {
        if (nero_cgw_still_since_ < 0.0) nero_cgw_still_since_ = now;
        if (now - nero_cgw_still_since_ >= e.park_s) {
          release = true;
          parked = true;
          why = "the car stood still as long as ego (parked)";
          nero_cgw_released_id_ = car->id;
          nero_cgw_until_ = now + 60.0;
        }
      } else {
        nero_cgw_still_since_ = -1.0;
      }
      // passing a coming car slowly, but its line now runs into ego's keep-right body: stop at once (it re-plans next cycle)
      if (!release && nero_cgw_pass_ && car->lo_pred < nero_cgw_shift_ + 0.5 * vp.vehicle_width + 0.5 * nero_cgw::PASS_CLEAR_M) {
        RCLCPP_WARN(getLogger(), "[NERO_SOA_CGW] %s runs into ego's way after all: stop", car->id.c_str());
        nero_cgw_pass_ = false;
        const auto p = autoware::motion_utils::calcLongitudinalOffsetPose(
          data.reference_path.points, getEgoPosition(), v_ego * v_ego / (2.0 * nero_cgw::STOP_DECEL));
        if (p) nero_cgw_stop_pose_ = p.value();
      }
      // a car coming closer: the wait place only ever moves back toward ego
      if (!release && !nero_cgw_pass_ && car->v > 0.0 && nero_cgw_stop_pose_) {
        const double s_stop = autoware::motion_utils::calcSignedArcLength(
          data.reference_path.points, getEgoPosition(), nero_cgw_stop_pose_->position);
        const double latest = car->s_near - e.gap_m - vp.base_link2front - car->v * (v_ego / e.decel + nero_cgw::REACT_S);
        if (latest < s_stop) {
          const double s_min = v_ego * v_ego / (2.0 * nero_cgw::STOP_DECEL);
          const auto p = autoware::motion_utils::calcLongitudinalOffsetPose(
            data.reference_path.points, getEgoPosition(), std::max(latest, s_min));
          if (p) nero_cgw_stop_pose_ = p.value();
        }
      }
    }
    if (release) {
      nero_cgw_pass_ = false;
      nero_cgw_stop_pose_.reset();
      const double ego_shift = helper_->getEgoShift();
      // standing objects on the right seen while holding and no longer published: the module never planned a line for
      // them (sim c24 stop v13: the kerb fragment 8866c2, published only while ego held 16-19 m before it, hit at
      // 1.9 m/s after the hand-back; the car's stack registers its line while it is seen and keeps it). The way back
      // clears them and every standing object on the right over the same stretch, and holds until ego is past them.
      const auto & ref = data.reference_path.points;
      const auto & arc = data.arclength_from_ego;
      const double half_w = 0.5 * vp.vehicle_width;
      double back = 0.0;
      double far = -1e9;
      double need_s = 1e9;  // where the first object that needs the way back begins (ego's front there)
      if (!parked && planner_data_->dynamic_object && arc.size() == ref.size() && ref.size() > 1) {
        std::unordered_set<std::string> shown;
        for (const auto & o : planner_data_->dynamic_object->objects) {
          shown.insert(std::string(o.object_id.uuid.begin(), o.object_id.uuid.end()));
        }
        struct Box
        {
          double s0, s1, hi, margin;
        };
        const auto box_of = [&](const std::vector<geometry_msgs::msg::Point> & cs, const double margin) {
          const auto b = neroCgwBox(data, cs);
          return Box{b[0], b[1], b[3], margin};
        };
        for (const auto & [k, m] : nero_cgw_seen_) {
          const auto b = box_of(m.corners, m.margin);
          if (b.s1 < -vp.base_link2rear - 0.5 || b.s0 > nero_cgw::HORIZON || shown.count(k)) continue;
          if (b.s1 > far) {
            far = b.s1;
            nero_cgw_back_past_ = m.corners;  // the way back holds until ego's rear is past this box
          }
        }
        if (far > -1e8) {
          neroCgwRemember(data);  // the ones still shown at their poses now
          for (const auto & [k, m] : nero_cgw_seen_) {
            const auto b = box_of(m.corners, m.margin);
            if (b.s1 < -vp.base_link2rear - 0.5 || b.s0 > far + vp.base_link2front) continue;
            const double need = b.hi + b.margin + half_w;
            back = std::max(back, need);
            // only one still ahead of ego's front sets how soon (one beside / behind is already cleared where ego is)
            if (need > ego_shift + 0.05 && b.s0 > vp.base_link2front) {
              need_s = std::min(need_s, b.s0 - vp.base_link2front - 0.5);
            }
          }
          const auto left_room = neroInLaneShiftLimit(data, true, nero_cgw::IN_LANE_M);
          if (left_room) back = std::min(back, left_room.value());
        }
      }
      if (!parked && (ego_shift < -0.1 || nero_cgw_back_past_)) {
        // kept right of the lane centre: back to it before the module's own plan takes over (from where ego kept right
        // its plan for the next objects starts later: sim c24 late v10 passed the kerb fragment 8866c2 at 0.0 m)
        nero_cgw_back_ = true;
        nero_cgw_on_ = -1e9;
        nero_cgw_back_until_ = now + (nero_cgw_back_past_ ? nero_cgw::BACK_PAST_S : nero_cgw::BACK_S);
        nero_cgw_shift_ = back;
        // at the module's nominal (minimum) lateral jerk at ego's own speed (not its nominal speed: 11 m at < 1 m/s, the
        // next objects come first), sharper only as the first object that needs it requires, never above its maximum
        // jerk (sim c24 squeeze_plag v14: 1.0 m within 2.5 m at the maximum jerk = a snap)
        const double d = std::abs(back - ego_shift);
        const double v_ref = std::max(v_ego, 1.0);
        const double soft =
          autoware::motion_utils::calc_longitudinal_dist_from_jerk(d, parameters_->lateral_min_jerk_map.front(), v_ref);
        const double sharp =
          autoware::motion_utils::calc_longitudinal_dist_from_jerk(d, parameters_->lateral_max_jerk_map.back(), v_ref);
        nero_cgw_ease_ = std::max({nero_ovl::MIN_EASE_DISTANCE, sharp, std::min(soft, need_s)});
        nero_cgw_start_ = true;
        RCLCPP_WARN(
          getLogger(), "[NERO_SOA_CGW] %s %s: back to %.2f (ego %.2f) within %.1f m%s (far %.1f, need %.1f)",
          nero_cgw_id_.c_str(), why.c_str(), back, ego_shift, nero_cgw_ease_,
          nero_cgw_back_past_ ? " and on until past the objects seen while holding" : "", far, need_s);
        return true;
      }
      nero_cgw_seen_.clear();
      RCLCPP_WARN(getLogger(), "[NERO_SOA_CGW] %s %s: the hold ends", nero_cgw_id_.c_str(), why.c_str());
      nero_cgw_active_ = false;
      nero_cgw_gone_since_ = nero_cgw_still_since_ = -1.0;
      return false;
    }
    if (!car || now - nero_cgw_plan_t_ < nero_cgw::REPLAN_S) {
      return true;
    }
    // re-plan below: objects seen since (the keep-right target), the car's progress (the wait place / pass)
  } else if (!car || (car->id == nero_cgw_released_id_ && now < nero_cgw_until_)) {
    return false;
  }
  const bool replan = nero_cgw_active_;

  // ---- the plan: where ego waits and how far right it keeps
  const auto & ref = data.reference_path.points;
  const auto & arc = data.arclength_from_ego;
  const double half_w = 0.5 * vp.vehicle_width;
  const double ego_shift = helper_->getEgoShift();
  // latest stop of ego's base_link: its front gap_m before the car (a moving car: before where it is once ego stands)
  const double t_stop = v_ego / e.decel;
  double s_stop =
    car->s_near - e.gap_m - vp.base_link2front - (car->v > 0.0 ? car->v * (t_stop + nero_cgw::REACT_S) : 0.0);
  const double s_min = v_ego * v_ego / (2.0 * nero_cgw::STOP_DECEL);
  const double s_cmf = v_ego * v_ego / (2.0 * e.decel);
  if (car->v > 0.0 && s_stop < s_cmf &&
      car->s_near - car->v * (t_stop + nero_cgw::REACT_S) - s_cmf - vp.base_link2front >= 1.0) {
    s_stop = s_cmf;  // a moving car passes: no need to keep gap_m from where it is now, only not to brake hard
  }
  s_stop = std::max(s_stop, s_min);
  // standing objects along the way (not the car): their extent along / beside the reference path
  struct Obj
  {
    double s0, s1, lo, hi, margin;
  };
  std::vector<Obj> objs;
  // a coming car: also the stretch ego drives at its keep-right line while the car goes by (PASS_LOOK_M past where they
  // meet: until the hold ends; sim c24 squeeze_plag v17 kept -0.30 into the kerb fragment bf50aa there, obstacle_stop
  // braked -2.94)
  const double s_meet0 = car->v > 0.0 ? std::max(0.0, car->s_near * v_ego / std::max(v_ego + car->v, 0.1)) : 0.0;
  const double s_end = std::max(s_stop + vp.base_link2front + 6.0,
                                car->v > 0.0 ? s_meet0 + vp.base_link2front + nero_cgw::PASS_LOOK_M : 0.0);
  for (const auto & o : planner_data_->dynamic_object->objects) {
    const auto & tw = o.kinematics.initial_twist_with_covariance.twist.linear;
    if (std::hypot(tw.x, tw.y) >= nero_cgw::STANDING_V) continue;
    char buf[8];
    std::snprintf(buf, sizeof(buf), "%02x%02x", o.object_id.uuid[0], o.object_id.uuid[1]);
    if (car->id == buf) continue;
    const auto & c = o.kinematics.initial_pose_with_covariance.pose.position;
    if (std::hypot(c.x - getEgoPosition().x, c.y - getEgoPosition().y) > s_end + 10.0) continue;
    const size_t seg = nero_cgw::seg_of(ref, c);
    const auto poly = autoware_utils::to_polygon2d(o.kinematics.initial_pose_with_covariance.pose, o.shape);
    Obj b{1e9, -1e9, 1e9, -1e9, 0.0};
    for (const auto & q : poly.outer()) {
      const auto p = autoware_utils::create_point(q.x(), q.y(), 0.0);
      const double s = arc.at(seg) + autoware::motion_utils::calcLongitudinalOffsetToSegment(ref, seg, p);
      const double l = autoware::motion_utils::calcLateralOffset(ref, p, seg);
      b.s0 = std::min(b.s0, s);
      b.s1 = std::max(b.s1, s);
      b.lo = std::min(b.lo, l);
      b.hi = std::max(b.hi, l);
    }
    if (b.s1 < -vp.base_link2rear - 1.0 || b.s0 > s_end) continue;
    b.margin = (utils::getHighestProbLabel(o.classification) == ObjectClassification::UNKNOWN ? 0.3 : 0.1) +
               nero_cgw::LAT_EXTRA;
    objs.push_back(b);
  }
  // the rightmost centre allowed at arc length s: clear of every standing object on the right (its centre right of the
  // car's lane side: below the car's lowest edge) by its margin, at most IN_LANE_M inside the lane's right bound
  const auto right_room = neroInLaneShiftLimit(data, false, nero_cgw::IN_LANE_M);
  const double floor_c = right_room ? -right_room.value() : ego_shift;
  const auto req = [&](const double s) {
    double r = floor_c;
    for (const auto & b : objs) {
      if (0.5 * (b.lo + b.hi) >= car->lat_lo || b.s1 < s - vp.base_link2rear - 0.3 || b.s0 > s + vp.base_link2front + 0.3) {
        continue;
      }
      r = std::max(r, b.hi + b.margin + half_w);
    }
    return r;
  };
  // the room the car has beside ego's body (keeping right) at arc length s: to the nearest standing object on the
  // other side along the car's length, or the drivable area's left bound
  const auto left_edge = [&](const double s) {
    double best = 1e9;
    if (!data.left_bound.empty()) {
      const auto p = autoware::motion_utils::calcLongitudinalOffsetPose(ref, getEgoPosition(), std::max(0.0, s));
      if (p) {
        double dmin = 1e9;
        for (const auto & q : data.left_bound) {
          const double d = autoware_utils::calc_distance2d(p->position, q);
          if (d < dmin) {
            dmin = d;
            best = autoware::motion_utils::calcLateralOffset(ref, q);
          }
        }
      }
    }
    for (const auto & b : objs) {
      if (0.5 * (b.lo + b.hi) < car->lat_lo || b.s1 < s - vp.base_link2rear - 2.5 || b.s0 > s + vp.base_link2front + 2.5) {
        continue;
      }
      best = std::min(best, b.lo);
    }
    return best;
  };
  double tgt = floor_c;
  for (double s = 0.0; s <= s_stop + vp.base_link2front; s += nero_cgw::STEP) tgt = std::max(tgt, req(s));
  const double need = car->width + 2.0 * e.side_m;
  const auto room_at = [&](const double s) { return left_edge(s) - (tgt + half_w); };
  double s_wait = s_stop;
  double room = room_at(s_wait);
  if (room < need) {
    // a place of its own row ego reaches going forward, before the latest stop, where the car has the room (never one
    // that needs more than the comfortable decel: no hard brake to find room)
    for (double s = s_stop - nero_cgw::STEP; s >= std::max(s_min, s_cmf); s -= nero_cgw::STEP) {
      double t = floor_c;
      for (double k = 0.0; k <= s + vp.base_link2front; k += nero_cgw::STEP) t = std::max(t, req(k));
      const double r = left_edge(s) - (t + half_w);
      if (r >= need) {
        s_wait = s;
        room = r;
        tgt = t;
        break;
      }
    }
  }
  // a coming car whose line (where it meets ego) clears ego's keep-right body by PASS_CLEAR_M: keep right and pass it at
  // most PASS_V, no stop (the keep-right line over the way to where they meet)
  bool pass = false;
  double on_need = -1.0;    // passing: where (arc length of ego's base_link) a standing object past the meeting first
  double on_shift = -1e9;   // needs ego further left, and how far
  if (car->v > 0.0) {
    const double s_meet = std::max(0.0, car->s_near * v_ego / std::max(v_ego + car->v, 0.1));
    double t = floor_c;
    for (double s = 0.0; s <= std::max(s_meet, s_stop) + vp.base_link2front; s += nero_cgw::STEP) t = std::max(t, req(s));
    if (car->lo_pred >= t + half_w + nero_cgw::PASS_CLEAR_M) {
      pass = true;
      // as far right as the car needs (its line PASS_CLEAR_M + 0.1 off ego's body), at least KEEP_RIGHT_M, never left of
      // where ego is
      tgt = std::max(
        t, std::min({ego_shift, car->lo_pred - half_w - nero_cgw::PASS_CLEAR_M - 0.1, -nero_cgw::KEEP_RIGHT_M}));
      // ... and ego drives that line on while the car goes by, until the hold ends: standing objects PASS_LOOK_M past
      // the meeting that need ego further left (sim c24 squeeze_plag v17: the kerb fragment bf50aa, obstacle_stop
      // braked -2.94) are cleared just in time by a second line that ends where the first of them begins, not while
      // the car still comes (v18 moved 0.2 m left at the meeting for fragments 6-10 m past it)
      const double s_from = std::max(s_meet, s_stop) + vp.base_link2front;
      for (double s = s_from; s <= s_from + nero_cgw::PASS_LOOK_M; s += nero_cgw::STEP) {
        if (req(s) > tgt + 0.05) {
          if (on_need < 0.0) on_need = s;
          on_shift = std::max(on_shift, req(s));
        }
      }
    }
  }
  if (!pass && car->v <= 0.0) {
    // a standing car: as far right as gives it its width + 2 side_m beside ego at the wait place (+ 0.1), at least
    // KEEP_RIGHT_M, never left of where ego is: the rightmost place runs the body toward kerb objects that may only show
    // up on the way (sim c24 early: one seen 2 m before it stranded the robot inside obstacle_stop's margin for good)
    tgt = std::max(tgt, std::min({ego_shift, left_edge(s_wait) - half_w - need - 0.1, -nero_cgw::KEEP_RIGHT_M}));
    room = left_edge(s_wait) - (tgt + half_w);
  }
  const auto stop_pose = autoware::motion_utils::calcLongitudinalOffsetPose(ref, getEgoPosition(), s_wait);
  if (!stop_pose) {
    return replan;  // holding: keep the plan there is
  }
  const double d = std::abs(tgt - ego_shift);
  double ease = std::max(nero_ovl::MIN_EASE_DISTANCE, helper_->getNominalAvoidanceDistance(d));
  if (ease > s_wait) {
    ease = std::max(
      nero_ovl::MIN_EASE_DISTANCE, autoware::motion_utils::calc_longitudinal_dist_from_jerk(
                                     d, parameters_->lateral_max_jerk_map.back(), std::max(v_ego, 1.0)));
  }
  if (!pass) {
    // the keep-right line ends 1 m before the wait place: a line still under way when ego stands stays undone (sim c24
    // stop v10: planned -0.30 over 4.2 m, stood after 3.3 m at -0.10)
    ease = std::max(nero_ovl::MIN_EASE_DISTANCE, std::min(ease, s_wait - 1.0));
  }
  // the second line (passing): from the keep-right target to on_shift, ending where it is needed, at the module's
  // nominal lateral jerk, shorter only as the room after the first line requires, never above its maximum jerk (sim c24
  // squeeze_plag v19: 9 m at the nominal jerk did not fit before the kerb fragment bf50aa); if even that does not fit,
  // one line to the farther-left target (as v18)
  double on = -1e9, on_start = -1.0, on_len = 0.0;
  if (pass && on_need >= 0.0) {
    const double d_on = std::abs(on_shift - tgt);
    const double soft = autoware::motion_utils::calc_longitudinal_dist_from_jerk(
      d_on, parameters_->lateral_min_jerk_map.front(), std::max(v_ego, 1.0));
    const double sharp = autoware::motion_utils::calc_longitudinal_dist_from_jerk(
      d_on, parameters_->lateral_max_jerk_map.back(), std::max(v_ego, 1.0));
    const double len = std::max({nero_ovl::MIN_EASE_DISTANCE, sharp, std::min(soft, on_need - ease)});
    if (on_need - len >= ease) {
      on = on_shift;
      on_start = on_need - len;
      on_len = len;
    } else {
      tgt = std::max(tgt, on_shift);
      ease = std::max(nero_ovl::MIN_EASE_DISTANCE, helper_->getNominalAvoidanceDistance(std::abs(tgt - ego_shift)));
      ease = std::min(ease, std::max(nero_ovl::MIN_EASE_DISTANCE, on_need));
    }
  }
  nero_cgw_plan_t_ = now;
  if (replan) {
    // the line only when the target moved; the wait place only ever back toward ego (or a switch to / from passing)
    const bool moved = std::abs(tgt - nero_cgw_shift_) > 0.1 || std::abs(on - nero_cgw_on_) > 0.1;
    if (moved) {
      nero_cgw_start_ = true;
      nero_cgw_shift_ = tgt;
      nero_cgw_ease_ = ease;
      nero_cgw_on_ = on;
      nero_cgw_on_start_ = on_start;
      nero_cgw_on_len_ = on_len;
    }
    const double s_old = nero_cgw_stop_pose_ ? autoware::motion_utils::calcSignedArcLength(
                                                 ref, getEgoPosition(), nero_cgw_stop_pose_->position)
                                             : 1e9;
    if (pass != nero_cgw_pass_ || (!pass && s_wait < s_old)) {
      nero_cgw_stop_pose_ = stop_pose.value();
    }
    if (moved || pass != nero_cgw_pass_) {
      RCLCPP_WARN(
        getLogger(), "[NERO_SOA_CGW] %s re-plan: keep right %.2f m (ego %.2f) within %.1f m, %s %.1f m ahead; v %.2f",
        car->id.c_str(), tgt, ego_shift, ease, pass ? "pass slowly from" : "wait", s_wait, v_ego);
    }
    nero_cgw_pass_ = pass;
    return true;
  }
  nero_cgw_active_ = true;
  nero_cgw_start_ = true;
  nero_cgw_pass_ = pass;
  nero_cgw_shift_ = tgt;
  nero_cgw_ease_ = ease;
  nero_cgw_on_ = on;
  nero_cgw_on_start_ = on_start;
  nero_cgw_on_len_ = on_len;
  nero_cgw_stop_pose_ = stop_pose.value();
  nero_cgw_id_ = car->id;
  nero_cgw_gone_since_ = nero_cgw_still_since_ = -1.0;
  nero_cgw_seen_.clear();
  nero_cgw_seen_t_ = -1.0;
  nero_cgw_back_past_.reset();
  RCLCPP_WARN(
    getLogger(),
    "[NERO_SOA_CGW] %s %s %.1f m ahead (lat %.2f..%.2f, meets ego at %.2f, %.1f m/s toward ego): keep right %.2f m (ego "
    "%.2f) within %.1f m, %s %.1f m ahead (latest %.1f), room beside %.2f m (needs %.2f); v %.2f",
    car->id.c_str(), car->v > 0.0 ? "coming" : "standing", car->s_near, car->lat_lo, car->lat_hi, car->lo_pred, car->v,
    tgt, ego_shift, ease, pass ? "it clears ego: pass it slowly, no stop, from" : "wait", s_wait, s_stop, room, need,
    v_ego);
  return true;
}

std::array<double, 4> StaticObstacleAvoidanceModule::neroCgwBox(
  const AvoidancePlanningData & data, const std::vector<geometry_msgs::msg::Point> & corners) const
{
  // every corner on the segment of the box's centre (as the plan's own objects): a corner's own nearest segment can lie
  // on a far part of a path that comes back near it (sim c24 stop v15: a remembered box ended 84706.9 m ahead)
  const auto & ref = data.reference_path.points;
  const auto & arc = data.arclength_from_ego;
  std::array<double, 4> b{1e9, -1e9, 1e9, -1e9};
  if (corners.empty() || ref.size() < 2 || arc.size() != ref.size()) return b;
  geometry_msgs::msg::Point c0;
  for (const auto & p : corners) {
    c0.x += p.x / corners.size();
    c0.y += p.y / corners.size();
  }
  const size_t seg = nero_cgw::seg_of(ref, c0);
  for (const auto & p : corners) {
    const double s = arc.at(seg) + autoware::motion_utils::calcLongitudinalOffsetToSegment(ref, seg, p);
    const double l = autoware::motion_utils::calcLateralOffset(ref, p, seg);
    // a degenerate segment (two equal points: the offset divides by its length) gives nonsense (sim c24 squeeze v16:
    // far 1.2e19 held the way back 20 s): such a box counts as out of reach
    if (!std::isfinite(s) || !std::isfinite(l) || std::abs(s) > 1.0e4 || std::abs(l) > 1.0e3) {
      return {1e9, -1e9, 1e9, -1e9};
    }
    b[0] = std::min(b[0], s);
    b[1] = std::max(b[1], s);
    b[2] = std::min(b[2], l);
    b[3] = std::max(b[3], l);
  }
  return b;
}

void StaticObstacleAvoidanceModule::neroCgwRemember(const AvoidancePlanningData & data) const
{
  // NERO_SOA_CORRIDOR_GIVEWAY: while it holds the module plans no line of its own, so a standing object on ego's right
  // that is published only now (sim c24: the kerb fragment 8866c2, seen 3 times from 16-19 m) would be gone from the
  // module's plan at the hand-back; keep its box (map frame) until the hold ends
  const auto & ref = data.reference_path.points;
  const auto & arc = data.arclength_from_ego;
  if (ref.size() < 2 || arc.size() != ref.size() || !planner_data_->dynamic_object) return;
  const auto & vp = planner_data_->parameters;
  for (const auto & o : planner_data_->dynamic_object->objects) {
    const auto & tw = o.kinematics.initial_twist_with_covariance.twist.linear;
    if (std::hypot(tw.x, tw.y) >= nero_cgw::STANDING_V) continue;
    char buf[8];
    std::snprintf(buf, sizeof(buf), "%02x%02x", o.object_id.uuid[0], o.object_id.uuid[1]);
    if (nero_cgw_id_ == buf) continue;
    const auto & c = o.kinematics.initial_pose_with_covariance.pose.position;
    if (std::hypot(c.x - getEgoPosition().x, c.y - getEgoPosition().y) > nero_cgw::HORIZON) continue;
    const size_t seg = nero_cgw::seg_of(ref, c);
    const double lat_c = autoware::motion_utils::calcLateralOffset(ref, c, seg);
    const double s_c = arc.at(seg) + autoware::motion_utils::calcLongitudinalOffsetToSegment(ref, seg, c);
    if (lat_c >= 0.0 || s_c < -vp.base_link2rear) continue;
    NeroCgwSeen m;
    // the polygon held in a variable: a range-for over .outer() of the temporary read freed memory (the corners were
    // garbage: the hand-back lost the kerb fragment 8866c2, sim c24 squeeze v20)
    const auto poly = autoware_utils::to_polygon2d(o.kinematics.initial_pose_with_covariance.pose, o.shape);
    for (const auto & q : poly.outer()) {
      m.corners.push_back(autoware_utils::create_point(q.x(), q.y(), 0.0));
    }
    m.margin = (utils::getHighestProbLabel(o.classification) == ObjectClassification::UNKNOWN ? 0.3 : 0.1) +
               nero_cgw::LAT_EXTRA;
    nero_cgw_seen_[std::string(o.object_id.uuid.begin(), o.object_id.uuid.end())] = m;
  }
}

void StaticObstacleAvoidanceModule::neroInsertCgwStop(ShiftedPath & shifted_path) const
{
  const double v = getEgoSpeed();
  if (nero_cgw_pass_) {
    // a coming car that clears ego's keep-right body: at most PASS_V while it comes, reached at the comfortable decel
    PoseWithDetailOpt unused;
    const double d = std::max(0.5, (v * v - nero_cgw::PASS_V * nero_cgw::PASS_V) / (2.0 * nero_cgw::env().decel));
    utils::static_obstacle_avoidance::insertDecelPoint(getEgoPosition(), d, nero_cgw::PASS_V, shifted_path.path, unused);
    return;
  }
  if (!nero_cgw_stop_pose_) {
    return;
  }
  double dist = autoware::motion_utils::calcSignedArcLength(
    shifted_path.path.points, getEgoPosition(), nero_cgw_stop_pose_->position);
  dist = std::max(dist, std::max(0.3, v * v / (2.0 * nero_cgw::STOP_DECEL)));
  utils::static_obstacle_avoidance::insertDecelPoint(getEgoPosition(), dist, 0.0, shifted_path.path, stop_pose_);
}

void StaticObstacleAvoidanceModule::neroInsertPauseStop(ShiftedPath & shifted_path) const
{
  const auto & data = avoid_data_;
  const auto target_distance = neroPauseTargetDistance(data);
  if (!target_distance) {
    return;
  }
  const auto stop_distance =
    std::max(0.0, target_distance.value() - planner_data_->parameters.base_link2front - nero_ovl::BESIDE_MARGIN);
  utils::static_obstacle_avoidance::insertDecelPoint(
    getEgoPosition(), stop_distance, 0.0, shifted_path.path, stop_pose_);
}

bool StaticObstacleAvoidanceModule::isSafePath(
  ShiftedPath & shifted_path, [[maybe_unused]] DebugData & debug) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  const auto & p = planner_data_->parameters;

  if (force_deactivated_) {
    return false;
  }

  if (!parameters_->enable_safety_check) {
    return true;  // if safety check is disabled, it always return safe.
  }

  const auto ego_idx = planner_data_->findEgoIndex(shifted_path.path.points);

  const auto has_left_shift = [&]() {
    for (size_t i = ego_idx; i < shifted_path.shift_length.size(); i++) {
      const auto length = shifted_path.shift_length.at(i);

      if (parameters_->lateral_execution_threshold < length) {
        return true;
      }
    }

    return false;
  }();

  const auto has_right_shift = [&]() {
    for (size_t i = ego_idx; i < shifted_path.shift_length.size(); i++) {
      const auto length = shifted_path.shift_length.at(i);

      if (parameters_->lateral_execution_threshold < -1.0 * length) {
        return true;
      }
    }

    return false;
  }();

  if (!has_left_shift && !has_right_shift) {
    return true;
  }

  const auto hysteresis_factor = safe_ ? 1.0 : parameters_->hysteresis_factor_expand_rate;

  const auto safety_check_target_objects =
    utils::static_obstacle_avoidance::getSafetyCheckTargetObjects(
      avoid_data_, planner_data_, parameters_, has_left_shift, has_right_shift, debug);

  if (safety_check_target_objects.empty()) {
    return true;
  }

  const bool limit_to_max_velocity = false;
  const auto ego_predicted_path_params =
    std::make_shared<utils::path_safety_checker::EgoPredictedPathParams>(
      parameters_->ego_predicted_path_params);
  const size_t ego_seg_idx = planner_data_->findEgoSegmentIndex(shifted_path.path.points);
  // NERO_SOA_EGO_PRED: ego predicted along its planned speed, capped at max_vel
  const bool nero_pred = nero_soa::env().ego_pred;
  const auto ego_predicted_path_for_front_object =
    nero_pred ? neroEgoPredictedPath(shifted_path, true, planner_data_->parameters.max_vel, 0.0)
              : utils::path_safety_checker::createPredictedPath(
                  ego_predicted_path_params, shifted_path.path.points, getEgoPose(), getEgoSpeed(),
                  ego_seg_idx, true, limit_to_max_velocity);
  const auto ego_predicted_path_for_rear_object =
    nero_pred ? neroEgoPredictedPath(shifted_path, false, planner_data_->parameters.max_vel, 0.0)
              : utils::path_safety_checker::createPredictedPath(
                  ego_predicted_path_params, shifted_path.path.points, getEgoPose(), getEgoSpeed(),
                  ego_seg_idx, false, limit_to_max_velocity);

  for (const auto & object : safety_check_target_objects) {
    auto current_debug_data = utils::path_safety_checker::createObjectDebug(object);

    const auto obj_polygon = autoware_utils::to_polygon2d(object.initial_pose, object.shape);

    const auto is_object_front = utils::path_safety_checker::isTargetObjectFront(
      getEgoPose(), obj_polygon, p.vehicle_info.max_longitudinal_offset_m);

    const auto & object_twist = object.initial_twist;
    const auto v_norm = std::hypot(object_twist.linear.x, object_twist.linear.y);
    const auto object_type = object.classification;
    const auto object_parameter = parameters_->object_parameters.at(object_type.label);
    const auto is_object_moving = v_norm > object_parameter.moving_speed_threshold;

    const auto is_object_oncoming =
      is_object_moving &&
      utils::path_safety_checker::isTargetObjectOncoming(getEgoPose(), object.initial_pose);

    const auto obj_predicted_paths = utils::path_safety_checker::getPredictedPathFromObj(
      object, parameters_->check_all_predicted_path);

    const auto & ego_predicted_path = is_object_front && !is_object_oncoming
                                        ? ego_predicted_path_for_front_object
                                        : ego_predicted_path_for_rear_object;

    for (const auto & obj_path : obj_predicted_paths) {
      if (!utils::path_safety_checker::checkCollision(
            shifted_path.path, ego_predicted_path, object, obj_path, p, parameters_->rss_params,
            hysteresis_factor, parameters_->collision_check_yaw_diff_threshold,
            current_debug_data.second)) {
        utils::path_safety_checker::updateCollisionCheckDebugMap(
          debug.collision_check, current_debug_data, false);

        safe_count_ = 0;
        return false;
      }
    }
    utils::path_safety_checker::updateCollisionCheckDebugMap(
      debug.collision_check, current_debug_data, true);
  }

  safe_count_++;

  // NERO_SOA_START_SAFE_CYCLES: from (nearly) standstill the move starts only after N safe cycles in a row (one cycle
  // of "safe" between two cars was enough to start it, the next cycle was unsafe again)
  if (nero_soa::env().start_safe_cycles > 0 && !safe_ && getEgoSpeed() < 0.6) {
    return safe_count_ > static_cast<size_t>(nero_soa::env().start_safe_cycles);
  }

  return safe_ || safe_count_ > parameters_->hysteresis_factor_safe_count;
}

std::vector<std::pair<double, double>> StaticObstacleAvoidanceModule::neroBumpsOnPath(
  const ShiftedPath & shifted_path) const
{
  const auto & points = shifted_path.path.points;
  std::vector<std::pair<double, double>> bumps;  // [start, end] arc length from ego
  if (points.empty()) {
    return bumps;
  }
  const double s0 = autoware::motion_utils::calcSignedArcLength(points, 0, getEgoPosition());
  // speed bumps on the path (the vehicle crosses them at ~1 m/s, behavior_velocity_planner's speed_bump): ego is
  // predicted that slow there and braking toward it — a move that ends on a bump keeps ego beside the line longer
  // (av_x_two_field:1: back over the bump at 2.4 km/h while straddling, the car behind braked 7 m/s²)
  if (nero_soa::env().bump_v > 0.0) {
    std::set<lanelet::Id> seen_ids;
    for (const auto & pt : points) {
      for (const auto id : pt.lane_ids) {
        if (!seen_ids.insert(id).second) {
          continue;
        }
        const auto ll = planner_data_->route_handler->getLaneletsFromId(id);
        for (const auto & reg : ll.regulatoryElementsAs<lanelet::autoware::SpeedBump>()) {
          double lo = std::numeric_limits<double>::max();
          double hi = std::numeric_limits<double>::lowest();
          for (const auto & q : reg->speedBump()) {
            geometry_msgs::msg::Point g;
            g.x = q.x();
            g.y = q.y();
            g.z = q.z();
            const double sq = autoware::motion_utils::calcSignedArcLength(points, 0, g) - s0;
            lo = std::min(lo, sq);
            hi = std::max(hi, sq);
          }
          if (hi > 0.0 && lo < 200.0) {
            bumps.emplace_back(lo - 1.3, hi);  // speed_bump slow_start_margin
          }
        }
      }
    }
  }
  return bumps;
}

std::vector<utils::path_safety_checker::PoseWithVelocityStamped>
StaticObstacleAvoidanceModule::neroEgoPredictedPath(
  const ShiftedPath & shifted_path, const bool for_front_object, const double v_hold,
  const double s_hold, const std::vector<std::pair<double, double>> * bumps_in) const
{
  std::vector<utils::path_safety_checker::PoseWithVelocityStamped> out;
  const auto & points = shifted_path.path.points;
  if (points.empty()) {
    return out;
  }
  const auto & pp = parameters_->ego_predicted_path_params;
  const double horizon =
    for_front_object ? pp.time_horizon_for_front_object : pp.time_horizon_for_rear_object;
  const double dt = std::max(0.1, pp.time_resolution);
  const double v_cap = planner_data_->parameters.max_vel;
  const double a_acc = std::max(0.1, pp.acceleration);
  const double a_dec = nero_soa::env().gap_a_dec;
  const double s0 = autoware::motion_utils::calcSignedArcLength(points, 0, getEgoPosition());
  // NERO_SOA_GAP_FAST: the caller's bumps (the same path within one cycle)
  std::vector<std::pair<double, double>> own_bumps;
  if (!bumps_in) {
    own_bumps = neroBumpsOnPath(shifted_path);
  }
  const auto & bumps = bumps_in ? *bumps_in : own_bumps;
  const auto bump_cap = [&](const double sq) {
    double c = std::numeric_limits<double>::max();
    for (const auto & [lo, hi] : bumps) {
      const double vb = nero_soa::env().bump_v;
      if (sq >= lo && sq <= hi) {
        c = std::min(c, vb);
      } else if (sq < lo) {
        c = std::min(c, std::sqrt(vb * vb + 2.0 * 0.8 * (lo - sq)));
      }
    }
    return c;
  };
  double v = getEgoSpeed();
  double s = 0.0;
  for (double t = 0.0; t < horizon; t += dt) {
    out.emplace_back(t, autoware::motion_utils::calcInterpolatedPose(points, s0 + s), v);
    const double target = std::min(v_cap, s < s_hold ? v_hold : v_cap);
    double v_next =
      v > target ? std::max(target, v - a_dec * dt) : std::min(target, v + a_acc * dt);
    v_next = std::min(v_next, std::max(bump_cap(s), 0.0));
    s += 0.5 * (v + v_next) * dt;
    v = v_next;
  }
  return out;
}

bool StaticObstacleAvoidanceModule::neroIsSafeRaw(
  const ShiftedPath & shifted_path, const double v_hold, const double s_hold, DebugData & debug,
  const NeroGapCtx * ctx) const
{
  const auto & p = planner_data_->parameters;
  bool has_left_shift = false;
  bool has_right_shift = false;
  if (ctx) {  // NERO_SOA_GAP_FAST: shift sides, targets and bumps of this cycle's candidate path, computed once
    has_left_shift = ctx->has_left_shift;
    has_right_shift = ctx->has_right_shift;
  } else {
    const auto ego_idx = planner_data_->findEgoIndex(shifted_path.path.points);
    for (size_t i = ego_idx; i < shifted_path.shift_length.size(); i++) {
      const auto length = shifted_path.shift_length.at(i);
      has_left_shift |= parameters_->lateral_execution_threshold < length;
      has_right_shift |= parameters_->lateral_execution_threshold < -1.0 * length;
    }
  }
  if (!has_left_shift && !has_right_shift) {
    return true;
  }
  const auto hysteresis_factor = safe_ ? 1.0 : parameters_->hysteresis_factor_expand_rate;
  std::vector<utils::path_safety_checker::ExtendedPredictedObject> own_targets;
  if (!ctx) {
    own_targets = utils::static_obstacle_avoidance::getSafetyCheckTargetObjects(
      avoid_data_, planner_data_, parameters_, has_left_shift, has_right_shift, debug);
  }
  const auto & targets = ctx ? ctx->targets : own_targets;
  if (targets.empty()) {
    return true;
  }
  const auto * bumps = ctx ? &ctx->bumps : nullptr;
  const auto ego_front = neroEgoPredictedPath(shifted_path, true, v_hold, s_hold, bumps);
  const auto ego_rear = neroEgoPredictedPath(shifted_path, false, v_hold, s_hold, bumps);
  for (const auto & object : targets) {
    auto dbg = utils::path_safety_checker::createObjectDebug(object);
    const auto obj_polygon = autoware_utils::to_polygon2d(object.initial_pose, object.shape);
    const auto is_object_front = utils::path_safety_checker::isTargetObjectFront(
      getEgoPose(), obj_polygon, p.vehicle_info.max_longitudinal_offset_m);
    const auto v_norm = std::hypot(object.initial_twist.linear.x, object.initial_twist.linear.y);
    const auto object_parameter = parameters_->object_parameters.at(object.classification.label);
    const auto is_object_oncoming =
      v_norm > object_parameter.moving_speed_threshold &&
      utils::path_safety_checker::isTargetObjectOncoming(getEgoPose(), object.initial_pose);
    const auto obj_predicted_paths = utils::path_safety_checker::getPredictedPathFromObj(
      object, parameters_->check_all_predicted_path);
    const auto & ego_path = is_object_front && !is_object_oncoming ? ego_front : ego_rear;
    for (const auto & obj_path : obj_predicted_paths) {
      if (!utils::path_safety_checker::checkCollision(
            shifted_path.path, ego_path, object, obj_path, p, parameters_->rss_params,
            hysteresis_factor, parameters_->collision_check_yaw_diff_threshold, dbg.second)) {
        return false;
      }
    }
  }
  return true;
}

double StaticObstacleAvoidanceModule::neroGapFloor(
  const ShiftedPath & shifted_path, const double s_hold,
  const std::vector<std::pair<double, double>> & bumps) const
{
  // NERO_SOA_GAP_FAST: neroEgoPredictedPath brakes toward v_hold at a_dec while s < s_hold. A v_hold below every speed
  // this braking (with the bump caps) reaches before s_hold is never reached there: the braking never stops at it, the
  // predicted path is the same for every such v_hold, and so is the verdict. The lowest of those speeds = the floor
  // (the longer of the two horizons: the front object's path is a prefix of it)
  const auto & points = shifted_path.path.points;
  if (points.empty()) {
    return 0.0;
  }
  const auto & pp = parameters_->ego_predicted_path_params;
  const double horizon = std::max(pp.time_horizon_for_front_object, pp.time_horizon_for_rear_object);
  const double dt = std::max(0.1, pp.time_resolution);
  const double a_dec = nero_soa::env().gap_a_dec;
  const double vb = nero_soa::env().bump_v;
  const auto bump_cap = [&](const double sq) {  // as in neroEgoPredictedPath
    double c = std::numeric_limits<double>::max();
    for (const auto & [lo, hi] : bumps) {
      if (sq >= lo && sq <= hi) {
        c = std::min(c, vb);
      } else if (sq < lo) {
        c = std::min(c, std::sqrt(vb * vb + 2.0 * 0.8 * (lo - sq)));
      }
    }
    return c;
  };
  double v = getEgoSpeed();
  double s = 0.0;
  double floor_v = v;
  for (double t = 0.0; t < horizon && s < s_hold; t += dt) {
    double v_next = std::max(0.0, v - a_dec * dt);
    v_next = std::min(v_next, std::max(bump_cap(s), 0.0));
    floor_v = std::min(floor_v, v_next);
    s += 0.5 * (v + v_next) * dt;
    v = v_next;
  }
  return floor_v;
}

double StaticObstacleAvoidanceModule::neroShiftStartDistance(const ShiftedPath & shifted_path) const
{
  const auto & pts = shifted_path.path.points;
  if (pts.empty() || shifted_path.shift_length.size() != pts.size()) {
    return -1.0;
  }
  const auto ego_idx = planner_data_->findEgoIndex(pts);
  const double base = shifted_path.shift_length.at(ego_idx);
  const double th = std::max(0.05, parameters_->lateral_execution_threshold);
  for (size_t i = ego_idx; i < pts.size(); ++i) {
    if (std::abs(shifted_path.shift_length.at(i) - base) > th) {
      return autoware::motion_utils::calcSignedArcLength(pts, getEgoPosition(), i);
    }
  }
  return -1.0;
}

void StaticObstacleAvoidanceModule::neroGapTiming(
  AvoidancePlanningData & data, [[maybe_unused]] DebugData & debug) const
{
  const auto & E = nero_soa::env();
  const auto reset = [this]() {
    nero_gap_v_ = -1.0;
    nero_gap_safe_n_ = 0;
    nero_gap_up_n_ = 0;
    nero_cruise_safe_n_ = 0;
  };
  if (!E.gap || !parameters_->enable_safety_check || helper_->isShifted() || !data.valid) {
    reset();
    return;
  }
  const double s_hold = neroShiftStartDistance(data.candidate_path);
  if (s_hold < 0.0) {
    reset();
    return;
  }
  const double v_cap = planner_data_->parameters.max_vel;
  const double v_ego = getEgoSpeed();
  if (!data.safe) {
    // which object fails the stock check (debug: "what it thinks")
    for (const auto & [id, c] : debug.collision_check) {
      if (c.is_safe) {
        continue;
      }
      const auto & q = c.current_obj_pose.position;
      const auto & e = getEgoPosition();
      RCLCPP_INFO_THROTTLE(
        getLogger(), *clock_, 1000,
        "[NERO_SOA_GAP] unsafe: object at %.1f m (dx %.1f dy %.1f) v %.1f m/s %s, %s, rss %.1f, lat %.2f",
        std::hypot(q.x - e.x, q.y - e.y), q.x - e.x, q.y - e.y,
        std::hypot(c.object_twist.linear.x, c.object_twist.linear.y), c.is_front ? "front" : "rear",
        c.unsafe_reason.c_str(), c.rss_longitudinal, c.lat_offset);
      break;
    }
  }
  if (data.safe) {
    // safe at the current speed (stock verdict with the capped prediction): no plan needed; an active plan is let go
    // after up_cycles such cycles in a row (no flicker between the two)
    if (nero_gap_v_ < 0.0 || ++nero_cruise_safe_n_ >= E.gap_up_cycles) {
      reset();
      return;
    }
  } else {
    nero_cruise_safe_n_ = 0;
  }
  DebugData tmp;
  // NERO_SOA_GAP_FAST: what every check below shares, once (none of it depends on the speed tried)
  std::optional<NeroGapCtx> ctx;
  double floor_v = -1.0;
  if (E.gap_fast) {
    const auto & sp = data.candidate_path;
    ctx.emplace();
    const auto ego_idx = planner_data_->findEgoIndex(sp.path.points);
    for (size_t i = ego_idx; i < sp.shift_length.size(); i++) {
      const auto length = sp.shift_length.at(i);
      ctx->has_left_shift |= parameters_->lateral_execution_threshold < length;
      ctx->has_right_shift |= parameters_->lateral_execution_threshold < -1.0 * length;
    }
    if (ctx->has_left_shift || ctx->has_right_shift) {
      ctx->targets = utils::static_obstacle_avoidance::getSafetyCheckTargetObjects(
        avoid_data_, planner_data_, parameters_, ctx->has_left_shift, ctx->has_right_shift, tmp);
    }
    ctx->bumps = neroBumpsOnPath(sp);
    floor_v = neroGapFloor(sp, s_hold, ctx->bumps);
  }
  const NeroGapCtx * c = ctx ? &ctx.value() : nullptr;
  if (nero_gap_v_ >= 0.0 && neroIsSafeRaw(data.candidate_path, nero_gap_v_, s_hold, tmp, c)) {
    // the plan still works; one step up after up_cycles cycles in which that step also worked
    const double up = std::min(v_cap, nero_gap_v_ + E.gap_dv);
    if (up > nero_gap_v_ + 1e-3 && neroIsSafeRaw(data.candidate_path, up, s_hold, tmp, c)) {
      if (++nero_gap_up_n_ >= E.gap_up_cycles) {
        nero_gap_v_ = up;
        nero_gap_up_n_ = 0;
      }
    } else {
      nero_gap_up_n_ = 0;
    }
    ++nero_gap_safe_n_;
  } else {
    double found = -1.0;
    const double start =
      (nero_gap_v_ >= 0.0 ? nero_gap_v_ : std::min(v_ego, v_cap)) - E.gap_dv;
    for (double v = start; v > E.gap_v_min - E.gap_dv + 1e-6; v -= E.gap_dv) {
      const double vv = std::max(v, E.gap_v_min);
      if (neroIsSafeRaw(data.candidate_path, vv, s_hold, tmp, c)) {
        found = vv;
        break;
      }
      if (c && vv < floor_v - 1e-9) {
        break;  // NERO_SOA_GAP_FAST: every lower speed gives this same predicted path (below the floor): unsafe too
      }
    }
    if (found < 0.0) {
      // no approach speed down to v_min gets ego there after the gap opens: the stock yield takes over
      if (nero_gap_v_ >= 0.0) {
        RCLCPP_INFO(getLogger(), "[NERO_SOA_GAP] no approach speed works any more: stock yield");
      }
      reset();
      return;
    }
    if (nero_gap_v_ < 0.0) {
      nero_gap_safe_n_ = 0;
    }
    nero_gap_v_ = found;
    nero_gap_up_n_ = 0;
    ++nero_gap_safe_n_;
  }
  nero_gap_s_hold_ = s_hold;
  if (nero_gap_safe_n_ >= E.gap_safe_cycles) {
    data.safe = true;  // safe at the planned approach speed: the move is approved, ego eases off to it
  }
  RCLCPP_INFO_THROTTLE(
    getLogger(), *clock_, 1000,
    "[NERO_SOA_GAP] approach at %.2f m/s (ego %.2f), the shift starts in %.1f m, safe cycles %d%s",
    nero_gap_v_, v_ego, s_hold, nero_gap_safe_n_, data.safe ? " -> approved" : "");
}

void StaticObstacleAvoidanceModule::neroInsertGapVelocity(ShiftedPath & shifted_path) const
{
  auto & pts = shifted_path.path.points;
  if (pts.size() < 2 || !nero_soa::env().gap) {
    return;
  }
  // no approach speed works (a stream with no usable gap): the stock yield stops ego before the object; make that
  // stop gentle and early — v <= sqrt(2 a_dec (d_stop - s)) — so ego rolls toward the stop while the gap search keeps
  // looking (from the lower speed a gap often works and ego never stops). Only if that ramp is still reachable.
  const auto & data = avoid_data_;
  if (
    nero_gap_v_ < 0.0 && data.yield_required && !helper_->isShifted() && data.to_stop_line > 0.5 &&
    data.to_stop_line < 200.0) {
    const double a_dec = nero_soa::env().gap_a_dec;
    const double v0 = getEgoSpeed();
    if (v0 * v0 <= 2.0 * a_dec * data.to_stop_line + 1.0) {
      const auto ego_idx = planner_data_->findEgoIndex(pts);
      double s = autoware::motion_utils::calcSignedArcLength(pts, getEgoPosition(), ego_idx);
      for (size_t i = ego_idx; i < pts.size(); ++i) {
        if (i > ego_idx) {
          s += autoware_utils::calc_distance2d(pts.at(i - 1), pts.at(i));
        }
        const double rest = data.to_stop_line - std::max(0.0, s);
        auto & v = pts.at(i).point.longitudinal_velocity_mps;
        v = std::min(v, static_cast<float>(std::sqrt(2.0 * a_dec * std::max(0.0, rest))));
        if (rest <= 0.0) {
          break;
        }
      }
    }
    return;
  }
  if (nero_gap_v_ < 0.0) {
    return;
  }
  const double a_dec = nero_soa::env().gap_a_dec;
  const double a_acc = std::max(0.1, parameters_->ego_predicted_path_params.acceleration);
  const double v0 = getEgoSpeed();
  const auto ego_idx = planner_data_->findEgoIndex(pts);
  double s = autoware::motion_utils::calcSignedArcLength(pts, getEgoPosition(), ego_idx);
  for (size_t i = ego_idx; i < pts.size(); ++i) {
    if (i > ego_idx) {
      s += autoware_utils::calc_distance2d(pts.at(i - 1), pts.at(i));
    }
    const double d = std::max(0.0, s);
    const double v_lim =
      d < nero_gap_s_hold_
        ? std::max(nero_gap_v_, std::sqrt(std::max(0.0, v0 * v0 - 2.0 * a_dec * d)))
        : std::sqrt(nero_gap_v_ * nero_gap_v_ + 2.0 * a_acc * (d - nero_gap_s_hold_));
    auto & v = pts.at(i).point.longitudinal_velocity_mps;
    v = std::min(v, static_cast<float>(v_lim));
  }
}

PathWithLaneId StaticObstacleAvoidanceModule::extendBackwardLength(
  const PathWithLaneId & original_path) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  const auto previous_path = helper_->getPreviousReferencePath();

  const auto longest_dist_to_shift_point = [&]() {
    double max_dist = 0.0;
    auto lines = path_shifter_.getShiftLines();
    if (lines.empty()) {
      return max_dist;
    }
    std::sort(lines.begin(), lines.end(), [](const auto & a, const auto & b) {
      return a.start_idx < b.start_idx;
    });
    return std::max(
      max_dist, autoware::motion_utils::calcSignedArcLength(
                  previous_path.points, lines.front().start.position, getEgoPosition()));
  }();

  const auto extra_margin = 10.0;  // Since distance does not consider arclength, but just line.
  const auto backward_length = std::max(
    planner_data_->parameters.backward_path_length, longest_dist_to_shift_point + extra_margin);

  const size_t orig_ego_idx = planner_data_->findEgoIndex(original_path.points);
  const auto prev_ego_idx = autoware::motion_utils::findNearestSegmentIndex(
    previous_path.points, autoware_utils::get_pose(original_path.points.at(orig_ego_idx)),
    std::numeric_limits<double>::max(), planner_data_->parameters.ego_nearest_yaw_threshold);
  if (!prev_ego_idx) {
    return original_path;
  }

  size_t clip_idx = 0;
  double accumulated_length = 0.0;
  for (size_t i = prev_ego_idx.value(); i > 0; i--) {
    accumulated_length +=
      autoware_utils::calc_distance2d(previous_path.points.at(i - 1), previous_path.points.at(i));
    if (accumulated_length > backward_length) {
      clip_idx = i;
      break;
    }
  }

  PathWithLaneId extended_path{};
  {
    extended_path.points.insert(
      extended_path.points.end(), previous_path.points.begin() + clip_idx,
      previous_path.points.begin() + *prev_ego_idx);
  }

  // overwrite backward path velocity by latest one.
  std::for_each(extended_path.points.begin(), extended_path.points.end(), [&](auto & p) {
    p.point.longitudinal_velocity_mps =
      original_path.points.at(orig_ego_idx).point.longitudinal_velocity_mps;
  });

  {
    extended_path.points.insert(
      extended_path.points.end(), original_path.points.begin() + orig_ego_idx,
      original_path.points.end());
  }

  return extended_path;
}

auto StaticObstacleAvoidanceModule::getTurnSignal(
  const ShiftedPath & spline_shift_path, const ShiftedPath & linear_shift_path) -> TurnSignalInfo
{
  using autoware::motion_utils::calcSignedArcLength;

  const auto is_ignore_signal = [this](const UUID & uuid) {
    return ignore_signal_ids_.find(to_hex_string(uuid)) != ignore_signal_ids_.end();
  };

  const auto update_ignore_signal = [this](const UUID & uuid, const bool is_ignore) {
    if (is_ignore) {
      ignore_signal_ids_.insert(to_hex_string(uuid));
    }
  };

  const auto is_large_deviation = [this](const auto & path) {
    constexpr double threshold = 1.0;
    const auto current_seg_idx = planner_data_->findEgoSegmentIndex(path.points);
    const auto lateral_deviation =
      autoware::motion_utils::calcLateralOffset(path.points, getEgoPosition(), current_seg_idx);
    return std::abs(lateral_deviation) > threshold;
  };

  auto shift_lines = path_shifter_.getShiftLines();
  if (shift_lines.empty()) {
    return getPreviousModuleOutput().turn_signal_info;
  }

  if (is_large_deviation(spline_shift_path.path)) {
    return getPreviousModuleOutput().turn_signal_info;
  }

  const auto target_shift_line = [&]() {
    const auto & s1 = shift_lines.front();

    for (size_t i = 1; i < shift_lines.size(); i++) {
      const auto & s2 = shift_lines.at(i);

      const auto s1_relative_length = s1.start_shift_length - s1.end_shift_length;
      const auto s2_relative_length = s2.start_shift_length - s2.end_shift_length;

      // same side shift
      if (s1_relative_length > 0.0 && s2_relative_length > 0.0) {
        continue;
      }

      // same side shift
      if (s1_relative_length < 0.0 && s2_relative_length < 0.0) {
        continue;
      }

      // different side shift
      const auto & points = path_shifter_.getReferencePath().points;
      const size_t idx = planner_data_->findEgoIndex(points);

      // output turn signal for near shift line.
      if (calcSignedArcLength(points, idx, s1.start_idx) > 0.0) {
        return s1;
      }

      // output turn signal for far shift line.
      if (
        calcSignedArcLength(points, idx, s2.start_idx) <
        getEgoSpeed() * parameters_->max_prepare_time) {
        return s2;
      }

      // output turn signal for near shift line.
      return s1;
    }

    return s1;
  }();

  if (is_ignore_signal(target_shift_line.id)) {
    return getPreviousModuleOutput().turn_signal_info;
  }

  const auto original_signal = getPreviousModuleOutput().turn_signal_info;

  constexpr bool is_driving_forward = true;
  constexpr bool egos_lane_is_shifted = true;

  const auto [new_signal, is_ignore] = planner_data_->getBehaviorTurnSignalInfo(
    linear_shift_path, target_shift_line, avoid_data_.current_lanelets, helper_->getEgoShift(),
    is_driving_forward, egos_lane_is_shifted);

  update_ignore_signal(target_shift_line.id, is_ignore);

  const auto current_seg_idx = planner_data_->findEgoSegmentIndex(spline_shift_path.path.points);
  return planner_data_->turn_signal_decider.overwrite_turn_signal(
    spline_shift_path.path, getEgoPose(), current_seg_idx, original_signal, new_signal,
    planner_data_->parameters.ego_nearest_dist_threshold,
    planner_data_->parameters.ego_nearest_yaw_threshold);
}

BehaviorModuleOutput StaticObstacleAvoidanceModule::plan()
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  const auto & data = avoid_data_;

  resetPathCandidate();
  resetPathReference();

  updatePathShifter(data.safe_shift_line);

  if (data.state == AvoidanceState::SUCCEEDED) {
    removeRegisteredShiftLines(State::SUCCEEDED);
    return getPreviousModuleOutput();
  }

  if (data.state == AvoidanceState::CANCEL) {
    removeRegisteredShiftLines(State::FAILED);
    return getPreviousModuleOutput();
  }

  if (data.yield_required) {
    removeRegisteredShiftLines(State::FAILED);
  }

  // generate path with shift points that have been inserted.
  ShiftedPath linear_shift_path =
    utils::static_obstacle_avoidance::toShiftedPath(data.reference_path);
  ShiftedPath spline_shift_path =
    utils::static_obstacle_avoidance::toShiftedPath(data.reference_path);
  const auto success_spline_path_generation =
    path_shifter_.generate(&spline_shift_path, true, SHIFT_TYPE::SPLINE);
  const auto success_linear_path_generation =
    path_shifter_.generate(&linear_shift_path, true, SHIFT_TYPE::LINEAR);

  if (nero_ps::debug() && success_spline_path_generation && !spline_shift_path.path.points.empty()) {
    const auto & pts = spline_shift_path.path.points;
    const auto ego_pos = getEgoPosition();
    const size_t i_ego = planner_data_->findEgoIndex(pts);
    const double out_shift = spline_shift_path.shift_length.at(i_ego);
    const double prev_shift = helper_->getEgoShift();
    const double ego_lat = autoware::motion_utils::calcLateralOffset(data.reference_path.points, ego_pos);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << "[NERO_SOA_PS] v " << getEgoSpeed() << " ego_lat " << ego_lat
       << " prev " << prev_shift << " out " << out_shift << " base " << path_shifter_.getBaseOffset() << " state "
       << static_cast<int>(data.state) << " valid " << data.valid << " safe " << data.safe << " comf "
       << data.comfortable << " yield " << data.yield_required << " act " << isActivated() << " | reg:";
    for (const auto & sl : path_shifter_.getShiftLines()) {
      ss << " [" << autoware::motion_utils::calcSignedArcLength(pts, ego_pos, sl.start.position) << ">"
         << autoware::motion_utils::calcSignedArcLength(pts, ego_pos, sl.end.position) << " " << sl.start_shift_length
         << ">" << sl.end_shift_length << "]";
    }
    ss << " | new:";
    for (const auto & sl : data.new_shift_line) {
      ss << " [" << sl.start_longitudinal << ">" << sl.end_longitudinal << " " << sl.start_shift_length << ">"
         << sl.end_shift_length << "]";
    }
    ss << " | safe_sl " << data.safe_shift_line.size() << " | tg:";
    int n = 0;
    for (const auto & o : data.target_objects) {
      if (++n > 5) break;
      ss << " " << std::hex << static_cast<int>(o.object.object_id.uuid[0]) << static_cast<int>(o.object.object_id.uuid[1])
         << std::dec << ":" << magic_enum::enum_name(o.info) << (o.is_avoidable ? ":A" : ":a") << ":lon "
         << o.longitudinal << ":m " << (o.avoid_margin ? o.avoid_margin.value() : -1.0);
    }
    RCLCPP_INFO(getLogger(), "%s", ss.str().c_str());
    static bool nero_ps_first = true;  // the first cycle has no previous output path
    if (!nero_ps_first && std::abs(out_shift - prev_shift) > 0.2) {
      RCLCPP_WARN(
        getLogger(), "[NERO_SOA_PS] STEP the output path at ego moved %.2f m in one cycle (prev %.2f -> %.2f, ego %.2f)",
        out_shift - prev_shift, prev_shift, out_shift, ego_lat);
    }
    nero_ps_first = false;
  }

  // set previous data
  if (success_spline_path_generation && success_linear_path_generation) {
    helper_->setPreviousLinearShiftPath(linear_shift_path);
    helper_->setPreviousSplineShiftPath(spline_shift_path);
    helper_->setPreviousReferencePath(path_shifter_.getReferencePath());
  } else {
    spline_shift_path = helper_->getPreviousSplineShiftPath();
  }

  // NERO_SOA_PATH_FROM_EGO: the output only; the module keeps its own plan as its previous path (its decisions, e.g.
  // RETURN_CLEAR's hold of the shift at ego, stay on its plan)
  if (nero_pfe::env().on && success_spline_path_generation && success_linear_path_generation) {
    neroPathFromEgo(data, spline_shift_path);
  }

  BehaviorModuleOutput output;

  // turn signal
  {
    output.turn_signal_info = getTurnSignal(spline_shift_path, linear_shift_path);
  }

  // sparse resampling for computational cost
  {
    spline_shift_path.path = utils::resamplePathWithSpline(
      spline_shift_path.path, parameters_->resample_interval_for_output);
  }

  // update output data
  {
    updateEgoBehavior(data, spline_shift_path);
    updateInfoMarker(avoid_data_);
    updateDebugMarker(output, avoid_data_, path_shifter_, debug_data_);
  }

  if (isDrivingSameLane(helper_->getPreviousDrivingLanes(), data.current_lanelets)) {
    output.path = spline_shift_path.path;
  } else {
    output.path = getPreviousModuleOutput().path;
    RCLCPP_WARN_THROTTLE(
      getLogger(), *clock_, 3000, "Previous module lane is updated. Do nothing.");
  }

  output.reference_path = getPreviousModuleOutput().reference_path;
  path_reference_ = std::make_shared<PathWithLaneId>(getPreviousModuleOutput().reference_path);

  const size_t ego_idx = planner_data_->findEgoIndex(output.path.points);
  utils::clipPathLength(
    output.path, ego_idx, planner_data_->parameters.forward_path_length,
    planner_data_->parameters.backward_path_length);

  // Drivable area generation.
  {
    DrivableAreaInfo current_drivable_area_info;
    // generate drivable lanes
    std::for_each(
      data.current_lanelets.begin(), data.current_lanelets.end(), [&](const auto & lanelet) {
        current_drivable_area_info.drivable_lanes.push_back(
          utils::static_obstacle_avoidance::generateExpandedDrivableLanes(
            lanelet, planner_data_, parameters_));
      });
    // expand hatched road markings
    current_drivable_area_info.enable_expanding_hatched_road_markings =
      parameters_->use_hatched_road_markings;
    // expand intersection areas
    current_drivable_area_info.enable_expanding_intersection_areas =
      parameters_->use_intersection_areas;
    // expand freespace areas
    current_drivable_area_info.enable_expanding_freespace_areas = parameters_->use_freespace_areas;
    // generate obstacle polygons
    current_drivable_area_info.obstacles.clear();

    if (
      parameters_->path_generation_method == "optimization_base" ||
      parameters_->path_generation_method == "both") {
      current_drivable_area_info.obstacles =
        utils::static_obstacle_avoidance::generateObstaclePolygonsForDrivableArea(
          clip_objects_, parameters_, planner_data_->parameters.vehicle_width / 2.0);
    }

    output.drivable_area_info = utils::combineDrivableAreaInfo(
      current_drivable_area_info, getPreviousModuleOutput().drivable_area_info);

    setDrivableLanes(output.drivable_area_info.drivable_lanes);
  }

  return output;
}

void StaticObstacleAvoidanceModule::neroPathFromEgo(
  const AvoidancePlanningData & data, ShiftedPath & shifted_path)
{
  const auto & ref = data.reference_path.points;
  const auto & prev = helper_->getPreviousSplineShiftPath();
  const auto & arc = data.arclength_from_ego;
  const ShiftedPath raw = shifted_path;  // the module's own plan this cycle
  const bool usable = ref.size() >= 2 && shifted_path.path.points.size() == ref.size() &&
                      shifted_path.shift_length.size() == ref.size() && arc.size() == ref.size() &&
                      prev.path.points.size() >= 2;
  // the module's plan unchanged since the previous cycle (the same lines and base offset): nothing moved (most cycles)
  const auto same_plan = [&] {
    const auto & a = nero_pfe_shifter_.getShiftLines();
    const auto & b = path_shifter_.getShiftLines();
    if (!nero_pfe_have_ || a.size() != b.size() ||
        std::abs(nero_pfe_shifter_.getBaseOffset() - path_shifter_.getBaseOffset()) > 1.0e-4) {
      return false;
    }
    for (size_t k = 0; k < a.size(); ++k) {
      if (
        autoware_utils::calc_distance2d(a.at(k).start, b.at(k).start) > 1.0e-3 ||
        autoware_utils::calc_distance2d(a.at(k).end, b.at(k).end) > 1.0e-3 ||
        std::abs(a.at(k).end_shift_length - b.at(k).end_shift_length) > 1.0e-4) {
        return false;
      }
    }
    return true;
  }();
  // the previous cycle's plan (its lines and base offset) drawn over this cycle's reference path: exact, no geometry
  // (drawn over the previous cycle's own reference, which it was output on: a line starting before this cycle's
  // reference would be squeezed into its first point), read at this cycle's reference points along the lane; NaN where
  // that reference did not reach
  std::optional<std::vector<double>> old;
  if (usable && nero_pfe_have_ && same_plan) {
    old = raw.shift_length;  // the same plan: no move (only an active blend is applied below)
  } else if (usable && nero_pfe_have_) {
    auto sh = nero_pfe_shifter_;
    const auto & op = sh.getReferencePath().points;
    ShiftedPath o = utils::static_obstacle_avoidance::toShiftedPath(sh.getReferencePath());
    if (op.size() >= 2 && sh.generate(&o, true, SHIFT_TYPE::SPLINE) && o.shift_length.size() == op.size()) {
      std::vector<double> v(ref.size(), std::numeric_limits<double>::quiet_NaN());
      for (size_t i = 0; i < ref.size(); ++i) {
        if (arc.at(i) < -nero_pfe::S_BACK - 1.0) continue;
        if (arc.at(i) > nero_pfe::S_FAR) break;
        const auto & p = ref.at(i).point.pose.position;
        const size_t seg = autoware::motion_utils::findNearestSegmentIndex(op, p);
        const double len = autoware_utils::calc_distance2d(op.at(seg), op.at(seg + 1));
        if (len < 1.0e-3) continue;
        const double lon = autoware::motion_utils::calcLongitudinalOffsetToSegment(op, seg, p);
        if ((seg == 0 && lon < -0.05) || (seg + 2 == op.size() && lon > len + 0.05)) continue;
        if (std::abs(autoware::motion_utils::calcLateralOffset(op, p, seg)) > 0.3) continue;  // not the same line
        const double r = std::clamp(lon / len, 0.0, 1.0);
        v.at(i) = o.shift_length.at(seg) + r * (o.shift_length.at(seg + 1) - o.shift_length.at(seg));
      }
      old = v;
    }
  }
  nero_pfe_shifter_ = path_shifter_;
  nero_pfe_have_ = true;
  if (!usable || !old) {
    nero_pfe_active_ = false;
    return;
  }
  const auto & e = nero_pfe::env();
  const auto ego = getEgoPosition();
  const double v = std::max(getEgoSpeed(), nero_pfe::V_MIN);
  const double jerk = std::max(helper_->getLateralMaxJerkLimit(), 0.1);
  const auto smooth = [](const double x) {
    const double c = std::clamp(x, 0.0, 1.0);
    return c * c * (3.0 - 2.0 * c);
  };
  const auto from_at = [&](const double s) {  // the stored start profile at arc length s from the blend start
    const auto & fs = nero_pfe_fs_;
    if (fs.empty()) return 0.0;
    if (s <= fs.front()) return nero_pfe_fv_.front();
    if (s >= fs.back()) return nero_pfe_fv_.back();
    const size_t k = std::upper_bound(fs.begin(), fs.end(), s) - fs.begin();
    const double r = (s - fs.at(k - 1)) / std::max(fs.at(k) - fs.at(k - 1), 1.0e-6);
    return nero_pfe_fv_.at(k - 1) + r * (nero_pfe_fv_.at(k) - nero_pfe_fv_.at(k - 1));
  };
  // what the robot was led along last cycle at reference point i: the old plan, blended as the active blend did
  const double s_bs = nero_pfe_active_ ? autoware::motion_utils::calcSignedArcLength(ref, ego, nero_pfe_start_) : 0.0;
  const auto led = [&](const size_t i) {
    const double p_old = std::isfinite(old->at(i)) ? old->at(i) : raw.shift_length.at(i);
    if (!nero_pfe_active_) return p_old;
    const double s_rel = arc.at(i) - s_bs;
    if (s_rel >= nero_pfe_length_) return p_old;
    const double f = from_at(s_rel);
    return f + (p_old - f) * smooth(s_rel / nero_pfe_length_);
  };
  // the module's plan moved around the robot (against its own plan of the previous cycle)
  const size_t i0 = data.ego_closest_path_index;
  double moved = 0.0, moved_behind = 0.0;  // at / in front of the robot; behind it only
  for (size_t i = 0; i < ref.size(); ++i) {
    if (arc.at(i) < -nero_pfe::S_BACK) continue;
    if (arc.at(i) > nero_pfe::S_FRONT) break;
    if (!std::isfinite(old->at(i))) continue;
    const double d = std::abs(raw.shift_length.at(i) - old->at(i));
    if (arc.at(i) < 0.0) {
      moved_behind = std::max(moved_behind, d);
    } else {
      moved = std::max(moved, d);
    }
  }
  // at the robot: the new plan against what it was led along (value and slope), and the robot against the new plan
  const size_t j = std::min(i0, ref.size() - 2);
  const double ds = std::max(arc.at(j + 1) - arc.at(j), 1.0e-3);
  const double r_ego = std::clamp(-arc.at(j) / ds, 0.0, 1.0);
  const double led_ego = led(j) + (led(j + 1) - led(j)) * r_ego;
  const double led_slope = (led(j + 1) - led(j)) / ds;
  const double raw_ego = raw.shift_length.at(j) + (raw.shift_length.at(j + 1) - raw.shift_length.at(j)) * r_ego;
  const double raw_slope = (raw.shift_length.at(j + 1) - raw.shift_length.at(j)) / ds;
  const double e_real = autoware::motion_utils::calcLateralOffset(ref, ego);
  // a change the robot can follow (it starts at the robot's place and direction: a stock line ahead, a freeze easing
  // flat from where a robot driving straight is) is the module's own; a jump, a kink, or a robot off the new plan is not
  const bool jump = std::abs(raw_ego - led_ego) > e.tol || std::abs(raw_slope - led_slope) > nero_pfe::SLOPE_TOL ||
                    (std::isfinite(e_real) && std::abs(e_real - raw_ego) > e.off);
  if (!jump || moved <= nero_pfe::CHANGED) {
    moved = 0.0;  // nothing to blend in front of the robot (behind it: below)
  }
  if (moved <= nero_pfe::CHANGED && moved_behind > e.tol) {
    // the plan was re-based only behind the robot (passed lines dropped, a freeze's flat start from the plan's own shift
    // at ego): keep what the robot was led along behind it, the module's plan from the robot on; during a blend, the
    // blend goes on from where it has led the robot (else the robot, still on the blend, would be off the plan at once)
    const double rem = nero_pfe_active_ ? nero_pfe_length_ + s_bs : 0.0;  // the active blend's length still ahead
    const double s_keep = rem > 0.0 ? nero_pfe::S_FAR : nero_pfe::S_FRONT;
    std::vector<double> fs, fv;
    for (size_t i = 0; i < ref.size(); ++i) {
      if (arc.at(i) < -nero_pfe::S_BACK - 1.0) continue;
      if (arc.at(i) > s_keep) break;
      fs.push_back(arc.at(i) - arc.at(i0));
      fv.push_back(led(i));
    }
    if (fs.size() >= 2) {
      nero_pfe_fs_ = fs;
      nero_pfe_fv_ = fv;
      nero_pfe_length_ = rem > 0.0 ? std::max(nero_pfe::MIN_LEN, rem) : 0.01;
      nero_pfe_start_ = ref.at(i0).point.pose.position;
      nero_pfe_active_ = true;
      RCLCPP_INFO(
        getLogger(), "[NERO_SOA_PFE] the plan moved %.2f m behind the robot only (v %.2f): the old part kept behind it",
        moved_behind, getEgoSpeed());
    }
  } else if (moved > nero_pfe::CHANGED) {
    // the profile the robot was led along, moved onto the robot when it is more than off_m off it, and continued along
    // the robot's heading against the reference (slope-continuous with its sideways motion)
    const double c = std::isfinite(e_real) && std::abs(e_real - led_ego) > e.off ? e_real - led_ego : 0.0;
    const double ref_yaw = tf2::getYaw(ref.at(j).point.pose.orientation);
    const double dpsi = std::clamp(
      autoware_utils::normalize_radian(tf2::getYaw(getEgoPose().orientation) - ref_yaw), -nero_pfe::MAX_DPSI,
      nero_pfe::MAX_DPSI);
    // the robot's sideways rate beyond the profile's (per metre); it carries on only as far as the module's maximum
    // lateral acceleration needs to cancel it: h(s) = extra s (1 - s / 2 s_h) up to s_h = |extra| v^2 / a, then flat
    double extra = std::clamp(std::tan(dpsi) - led_slope, -nero_pfe::MAX_EXTRA, nero_pfe::MAX_EXTRA);
    if (std::abs(extra) < 0.03) extra = 0.0;
    const double s_h = std::min(
      nero_pfe::MAX_S_H, std::abs(extra) * v * v / std::max(helper_->getLateralMaxAccelLimit(), 0.1));
    const auto carry = [&](const double s) {
      if (s <= 0.0 || s_h < 1.0e-3) return 0.0;
      const double q = std::min(s, s_h);
      return extra * q * (1.0 - q / (2.0 * s_h));
    };
    std::vector<double> fs, fv;
    for (size_t i = 0; i < ref.size(); ++i) {
      if (arc.at(i) < -nero_pfe::S_BACK - 1.0) continue;
      if (arc.at(i) > nero_pfe::S_FAR) break;
      fs.push_back(arc.at(i) - arc.at(i0));
      fv.push_back(led(i) + c + carry(arc.at(i)));
    }
    if (fs.size() >= 2) {
      nero_pfe_fs_ = fs;
      nero_pfe_fv_ = fv;
      // the blend length: a smooth S (w = 3x^2 - 2x^3) whose lateral jerk dmax * 12 v^3 / L^3 and lateral acceleration
      // v^2 (dmax * 6 / L^2 + sigma * 3 / L) stay within the module's maxima at ego's speed; dmax = the largest change
      // over it, sigma = the slope the robot must turn away from at its start (its sideways motion against the plan)
      const double acc = nero_pfe::ACC_SCALE * std::max(helper_->getLateralMaxAccelLimit(), 0.1);
      const size_t k = std::min(i0 + 1, ref.size() - 1);
      const double dk = std::max(arc.at(k) - arc.at(i0), 1.0e-3);
      const double sigma = std::abs(
        ((raw.shift_length.at(k) - from_at(arc.at(k) - arc.at(i0))) -
         (raw.shift_length.at(i0) - from_at(0.0))) / dk);
      double len = nero_pfe::MIN_LEN, dmax = 0.0;
      for (int it = 0; it < 6; ++it) {
        dmax = 0.0;
        for (size_t i = i0; i < ref.size(); ++i) {
          const double s_rel = arc.at(i) - arc.at(i0);
          if (s_rel > len) break;
          dmax = std::max(dmax, std::abs(raw.shift_length.at(i) - from_at(s_rel)));
        }
        const double by_jerk = v * std::cbrt(12.0 * dmax / jerk);
        const double b = 3.0 * sigma * v * v;
        const double by_acc = (b + std::sqrt(b * b + 24.0 * acc * dmax * v * v)) / (2.0 * acc);
        len = std::max({len, by_jerk, by_acc});
      }
      nero_pfe_length_ = len;
      nero_pfe_start_ = ref.at(i0).point.pose.position;
      nero_pfe_active_ = true;
      RCLCPP_WARN(
        getLogger(),
        "[NERO_SOA_PFE] the plan moved %.2f m around the robot (v %.2f): blended in from %s (robot %.2f m off it, "
        "%.2f sideways) over %.1f m (largest change %.2f m, slope %.2f, jerk %.2f, acc %.2f)",
        moved, getEgoSpeed(), std::abs(c) > 0.0 ? "the robot's place" : "where the robot was led", e_real - led_ego,
        extra, nero_pfe_length_, dmax, sigma, jerk, acc);
    }
  }
  if (!nero_pfe_active_) return;
  const double s_start = autoware::motion_utils::calcSignedArcLength(ref, ego, nero_pfe_start_);  // < 0: behind ego
  if (-s_start > nero_pfe_length_ + nero_pfe::S_BACK) {
    nero_pfe_active_ = false;  // ego is past the blend by the trajectory's backward reach: the module's plan as it is
    return;
  }
  bool changed = false;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double s_rel = arc.at(i) - s_start;
    if (s_rel >= nero_pfe_length_) break;
    if (s_rel < -nero_pfe::S_BACK - 1.0) continue;
    const double f = from_at(s_rel);
    const double out = f + (raw.shift_length.at(i) - f) * smooth(s_rel / nero_pfe_length_);
    shifted_path.shift_length.at(i) = out;
    shifted_path.path.points.at(i).point.pose = autoware_utils::calc_offset_pose(ref.at(i).point.pose, 0.0, out, 0.0);
    changed = true;
  }
  if (changed) {
    autoware::motion_utils::insertOrientation(shifted_path.path.points, true);
  }
}

CandidateOutput StaticObstacleAvoidanceModule::planCandidate() const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  const auto & data = avoid_data_;

  CandidateOutput output;

  auto shifted_path = data.candidate_path;

  if (data.safe_shift_line.empty()) {
    const size_t ego_idx = planner_data_->findEgoIndex(shifted_path.path.points);
    utils::clipPathLength(
      shifted_path.path, ego_idx, planner_data_->parameters.forward_path_length,
      planner_data_->parameters.backward_path_length);

    output.path_candidate = shifted_path.path;
    return output;
  }

  const auto sl = helper_->getMainShiftLine(data.safe_shift_line);
  const auto sl_front = data.safe_shift_line.front();
  const auto sl_back = data.safe_shift_line.back();

  utils::clipPathLength(
    shifted_path.path, sl_front.start_idx, std::numeric_limits<double>::max(), 0.0);

  output.lateral_shift = helper_->getRelativeShiftToPath(sl);
  output.start_distance_to_path_change = sl_front.start_longitudinal;
  output.finish_distance_to_path_change = sl_back.end_longitudinal;

  const uint16_t planning_factor_direction = std::invoke([&output]() {
    return output.lateral_shift > 0.0 ? PlanningFactor::SHIFT_LEFT : PlanningFactor::SHIFT_RIGHT;
  });

  planning_factor_interface_->add(
    output.start_distance_to_path_change, output.finish_distance_to_path_change, sl_front.start,
    sl_back.end, planning_factor_direction,
    utils::path_safety_checker::to_safety_factor_array(debug_data_.collision_check), true, 0.0,
    output.lateral_shift);

  output.path_candidate = shifted_path.path;
  return output;
}

BehaviorModuleOutput StaticObstacleAvoidanceModule::planWaitingApproval()
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  BehaviorModuleOutput out = plan();

  if (path_shifter_.getShiftLines().empty()) {
    out.turn_signal_info = getPreviousModuleOutput().turn_signal_info;
  }

  path_candidate_ = std::make_shared<PathWithLaneId>(planCandidate().path_candidate);
  path_reference_ = std::make_shared<PathWithLaneId>(getPreviousModuleOutput().reference_path);

  return out;
}

void StaticObstacleAvoidanceModule::updatePathShifter(const AvoidLineArray & shift_lines)
{
  if (parameters_->path_generation_method == "optimization_base") {
    return;
  }

  if (shift_lines.empty()) {
    return;
  }

  if (!isActivated()) {
    return;
  }

  if (
    nero_soa::env().approval_keeps_safety && avoid_data_.request_operator &&
    !avoid_data_.target_objects.empty()) {
    const auto & u = avoid_data_.target_objects.front().object.object_id.uuid;
    nero_approved_[std::string(u.begin(), u.end())] = clock_->now().seconds();
    RCLCPP_INFO(
      getLogger(), "[NERO_SOA] the operator approved moving around this object: not asked again");
  }

  addNewShiftLines(path_shifter_, shift_lines);

  generator_.setRawRegisteredShiftLine(shift_lines, avoid_data_);

  const auto sl = helper_->getMainShiftLine(shift_lines);
  const auto & sl_front = shift_lines.front();
  const auto & sl_back = shift_lines.back();
  const auto relative_longitudinal = sl_back.end_longitudinal - sl_front.start_longitudinal;

  if (helper_->getRelativeShiftToPath(sl) > 0.0) {
    left_shift_array_.push_back(
      {uuid_map_.at("left"), sl_front.start, sl_back.end, relative_longitudinal});
  } else if (helper_->getRelativeShiftToPath(sl) < 0.0) {
    right_shift_array_.push_back(
      {uuid_map_.at("right"), sl_front.start, sl_back.end, relative_longitudinal});
  }

  uuid_map_.at("left") = generate_uuid();
  uuid_map_.at("right") = generate_uuid();
  candidate_uuid_ = generate_uuid();

  lockNewModuleLaunch();
}

/**
 * set new shift points. remove old shift points if it has a conflict.
 */
void StaticObstacleAvoidanceModule::addNewShiftLines(
  PathShifter & path_shifter, const AvoidLineArray & new_shift_lines) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  ShiftLineArray future = utils::static_obstacle_avoidance::toShiftLineArray(new_shift_lines);

  size_t min_start_idx = std::numeric_limits<size_t>::max();
  for (const auto & sl : new_shift_lines) {
    min_start_idx = std::min(min_start_idx, sl.start_idx);
  }

  const auto current_shift_lines = path_shifter.getShiftLines();
  const auto front_new_shift_line = new_shift_lines.front();
  const auto new_shift_length = front_new_shift_line.end_shift_length;
  const auto new_shift_end_idx = front_new_shift_line.end_idx;

  RCLCPP_DEBUG(getLogger(), "min_start_idx = %lu", min_start_idx);

  // Remove shift points that starts later than the new_shift_line from path_shifter.
  //
  // Why? Because shifter sorts by start position and applies shift points, so if there is a
  // shift point that starts after the one you are going to put in, new one will be affected
  // by the old one.
  //
  // Is it ok? This is a situation where the vehicle was originally going to avoid at the farther
  // point, but decided to avoid it at a closer point. In this case, it is reasonable to cancel the
  // farther avoidance.
  for (const auto & sl : current_shift_lines) {
    if (sl.start_idx >= min_start_idx) {
      RCLCPP_DEBUG(
        getLogger(), "sl.start_idx = %lu, this sl starts after new proposal. remove this one.",
        sl.start_idx);
      continue;
    }

    if (sl.end_idx > new_shift_end_idx) {
      if (
        sl.end_shift_length > -1e-3 && new_shift_length > -1e-3 &&
        sl.end_shift_length < new_shift_length) {
        continue;
      }

      if (
        sl.end_shift_length < 1e-3 && new_shift_length < 1e-3 &&
        sl.end_shift_length > new_shift_length) {
        continue;
      }
    }

    RCLCPP_DEBUG(getLogger(), "sl.start_idx = %lu, no conflict. keep this one.", sl.start_idx);
    future.push_back(sl);
  }

  const double road_velocity = avoid_data_.reference_path.points.at(front_new_shift_line.start_idx)
                                 .point.longitudinal_velocity_mps;
  const double shift_time = autoware::motion_utils::calc_shift_time_from_jerk(
    front_new_shift_line.getRelativeLength(), helper_->getLateralMaxJerkLimit(),
    helper_->getLateralMaxAccelLimit());
  const double longitudinal_acc =
    std::clamp(road_velocity / shift_time, 0.0, parameters_->max_acceleration);

  path_shifter.setShiftLines(future);
  path_shifter.setVelocity(getEgoSpeed());
  path_shifter.setLongitudinalAcceleration(longitudinal_acc);
  path_shifter.setLateralAccelerationLimit(helper_->getLateralMaxAccelLimit());
}

bool StaticObstacleAvoidanceModule::isValidShiftLine(
  const AvoidLineArray & shift_lines, const PathShifter & shifter) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  if (shift_lines.empty()) {
    return true;
  }

  auto shifter_for_validate = shifter;

  addNewShiftLines(shifter_for_validate, shift_lines);

  ShiftedPath proposed_shift_path;
  shifter_for_validate.generate(&proposed_shift_path);

  debug_data_.proposed_spline_shift = proposed_shift_path.shift_length;

  // check offset between new shift path and ego position.
  {
    const auto new_idx = planner_data_->findEgoIndex(proposed_shift_path.path.points);
    const auto new_shift_length = proposed_shift_path.shift_length.at(new_idx);

    constexpr double THRESHOLD = 0.1;
    const auto offset = std::abs(new_shift_length - helper_->getEgoShift());
    if (offset > THRESHOLD) {
      RCLCPP_DEBUG_THROTTLE(
        getLogger(), *clock_, 1000, "new shift line is invalid. [HUGE OFFSET (%.2f)]", offset);
      return false;
    }
  }

  const auto is_return_shift =
    [](const double start_shift_length, const double end_shift_length, const double threshold) {
      return std::abs(start_shift_length) > threshold && std::abs(end_shift_length) < threshold;
    };

  // check if the vehicle is in road. (yaw angle is not considered)
  {
    const auto minimum_distance = 0.5 * planner_data_->parameters.vehicle_width +
                                  parameters_->hard_drivable_bound_margin -
                                  parameters_->max_deviation_from_lane;

    for (const auto & shift_line : shift_lines) {
      const size_t start_idx = shift_line.start_idx;
      const size_t end_idx = shift_line.end_idx;

      if (is_return_shift(
            shift_line.start_shift_length, shift_line.end_shift_length,
            parameters_->lateral_small_shift_threshold)) {
        continue;
      }

      const auto path = shifter_for_validate.getReferencePath();
      const auto left_bound = lanelet::utils::to2D(toLineString3d(avoid_data_.left_bound));
      const auto right_bound = lanelet::utils::to2D(toLineString3d(avoid_data_.right_bound));
      for (size_t i = start_idx; i <= end_idx; ++i) {
        const auto p = get_point(path.points.at(i));
        lanelet::BasicPoint2d basic_point{p.x, p.y};

        const auto shift_length = proposed_shift_path.shift_length.at(i);
        const auto THRESHOLD = minimum_distance + std::abs(shift_length);

        if (
          boost::geometry::distance(basic_point, (shift_length > 0.0 ? left_bound : right_bound)) <
          THRESHOLD) {
          RCLCPP_DEBUG_THROTTLE(
            getLogger(), *clock_, 1000,
            "following latest new shift line may cause deviation from drivable area.");
          return false;
        }
      }
    }
  }

  return true;  // valid shift line.
}

void StaticObstacleAvoidanceModule::updateData()
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  using utils::static_obstacle_avoidance::toShiftedPath;

  helper_->setData(planner_data_);

  if (!helper_->isInitialized()) {
    helper_->setPreviousSplineShiftPath(toShiftedPath(getPreviousModuleOutput().path));
    helper_->setPreviousLinearShiftPath(toShiftedPath(getPreviousModuleOutput().path));
    helper_->setPreviousReferencePath(getPreviousModuleOutput().path);
    helper_->setPreviousDrivingLanes(utils::static_obstacle_avoidance::getCurrentLanesFromPath(
      getPreviousModuleOutput().reference_path, planner_data_));
  }

  debug_data_ = DebugData();
  avoid_data_ = AvoidancePlanningData();

  // update base path and target objects.
  fillFundamentalData(avoid_data_, debug_data_);

  // an empty path will kill further processing
  if (avoid_data_.reference_path.points.empty()) {
    return;
  }

  // update shift line generator.
  generator_.setData(planner_data_);
  generator_.setPathShifter(path_shifter_);
  generator_.setHelper(helper_);
  generator_.update(avoid_data_, debug_data_);

  // update shift line and check path safety.
  fillShiftLine(avoid_data_, debug_data_);

  // update ego behavior.
  fillEgoStatus(avoid_data_, debug_data_);

  if (nero_soa::env().gap && !avoid_data_.target_objects.empty()) {
    const auto & d = avoid_data_;
    RCLCPP_INFO_THROTTLE(
      getLogger(), *clock_, 500,
      "[NERO_SOA_STATE] v %.2f safe %d valid %d ready %d comf %d avoid_req %d yield %d shifted %d req_op %d "
      "gap_v %.2f s_hold %.1f stop_line %.1f targets %zu",
      getEgoSpeed(), d.safe, d.valid, d.ready, d.comfortable, d.avoid_required, d.yield_required,
      helper_->isShifted(), d.request_operator, nero_gap_v_, nero_gap_s_hold_,
      d.to_stop_line < 1e6 ? d.to_stop_line : -1.0, d.target_objects.size());
  }

  if (nero_pause_start_) {  // NERO_SOA_ONCOMING_PAUSE: the pause starts this cycle
    neroFreezeShift(avoid_data_, nero_pause_shift_, nero_pause_ease_);
    nero_pause_start_ = false;
  }
  if (nero_cgw_start_) {  // NERO_SOA_CORRIDOR_GIVEWAY: the keep-right line from ego's shift
    if (nero_cgw_on_ > -1e8 && !nero_cgw_back_) {
      neroFreezeShift(avoid_data_, nero_cgw_shift_, nero_cgw_ease_, nero_cgw_on_start_, nero_cgw_on_len_, nero_cgw_on_);
    } else {
      neroFreezeShift(avoid_data_, nero_cgw_shift_, nero_cgw_ease_);
    }
    avoid_data_.new_shift_line.clear();
    avoid_data_.safe_shift_line.clear();
    nero_cgw_start_ = false;
  }
  if (nero_ovl::active()) {
    neroOncomingReturn(avoid_data_);  // NERO_SOA_ONCOMING_COMMITTED
  }
  if (nero_rcl::on()) {
    neroReturnClear(avoid_data_);  // NERO_SOA_RETURN_CLEAR
  }

  if (nero_ovl::env().debug) {
    neroOncomingDebugLog();  // NERO_SOA_ONCOMING_DEBUG (log only)
  }

  // update debug data.
  fillDebugData(avoid_data_, debug_data_);

  // update rtc status.
  updateRTCData();

  // update interest objects data
  for (const auto & [uuid, data] : debug_data_.collision_check) {
    const auto color = data.is_safe ? ColorName::GREEN : ColorName::RED;
    setObjectsOfInterestData(data.current_obj_pose, data.obj_shape, color);
  }

  safe_ = avoid_data_.safe;

  if (!force_deactivated_) {
    last_deactivation_triggered_time_ = clock_->now();
    force_deactivated_ = avoid_data_.force_deactivated;
    return;
  }

  if (
    (clock_->now() - last_deactivation_triggered_time_).seconds() >
    parameters_->force_deactivate_duration_time) {
    RCLCPP_INFO_THROTTLE(getLogger(), *clock_, 3000, "The force deactivation is released");
    force_deactivated_ = false;
  }
}

void StaticObstacleAvoidanceModule::processOnEntry()
{
  initVariables();
  if (nero_cgw_active_) {
    nero_cgw_start_ = true;  // NERO_SOA_CORRIDOR_GIVEWAY launched the module: the keep-right line again (path_shifter_ reset)
  }
}

void StaticObstacleAvoidanceModule::processOnExit()
{
  initVariables();
  initRTCStatus();
}

void StaticObstacleAvoidanceModule::initVariables()
{
  helper_->reset();
  generator_.reset();
  path_shifter_ = PathShifter{};
  debug_data_ = DebugData();
  debug_marker_.markers.clear();
  resetPathCandidate();
  resetPathReference();
  arrived_path_end_ = false;
  ignore_signal_ids_.clear();
  nero_paused_ = false;
  nero_pause_start_ = false;
  nero_stop_now_ = false;
  nero_returning_ = false;
  nero_onc_.reset();
  nero_lane_latch_.clear();
  nero_lane_stop_.reset();
}

void StaticObstacleAvoidanceModule::initRTCStatus()
{
  left_shift_array_.clear();
  right_shift_array_.clear();
  uuid_map_.at("left") = generate_uuid();
  uuid_map_.at("right") = generate_uuid();
  candidate_uuid_ = generate_uuid();
}

void StaticObstacleAvoidanceModule::updateRTCData()
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  const auto & data = avoid_data_;

  updateRegisteredRTCStatus(helper_->getPreviousSplineShiftPath().path);

  const auto candidates = data.safe ? data.safe_shift_line : data.new_shift_line;

  if (candidates.empty()) {
    removeCandidateRTCStatus();
    return;
  }

  const auto shift_line = helper_->getMainShiftLine(candidates);
  if (helper_->getRelativeShiftToPath(shift_line) > 0.0) {
    removePreviousRTCStatusRight();
  } else if (helper_->getRelativeShiftToPath(shift_line) < 0.0) {
    removePreviousRTCStatusLeft();
  } else {
    RCLCPP_WARN_STREAM(getLogger(), "Direction is UNKNOWN");
  }

  CandidateOutput output;

  const auto & sl_front = candidates.front();
  const auto & sl_back = candidates.back();

  output.path_candidate = data.candidate_path.path;
  output.lateral_shift = helper_->getRelativeShiftToPath(shift_line);
  output.start_distance_to_path_change = sl_front.start_longitudinal;
  output.finish_distance_to_path_change = sl_back.end_longitudinal;

  updateCandidateRTCStatus(output);
}

void StaticObstacleAvoidanceModule::updateInfoMarker(const AvoidancePlanningData & data) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  using utils::static_obstacle_avoidance::createAmbiguousObjectsMarkerArray;
  using utils::static_obstacle_avoidance::createStopTargetObjectMarkerArray;
  using utils::static_obstacle_avoidance::createTargetObjectsMarkerArray;

  info_marker_.markers.clear();
  append_marker_array(
    createTargetObjectsMarkerArray(data.target_objects, "target_objects"), &info_marker_);
  append_marker_array(createStopTargetObjectMarkerArray(data), &info_marker_);
  append_marker_array(
    createAmbiguousObjectsMarkerArray(
      data.target_objects, getEgoPose(), parameters_->policy_ambiguous_vehicle),
    &info_marker_);
}

void StaticObstacleAvoidanceModule::updateDebugMarker(
  const BehaviorModuleOutput & output, const AvoidancePlanningData & data,
  const PathShifter & shifter, const DebugData & debug) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  debug_marker_.markers.clear();
  debug_marker_ = utils::static_obstacle_avoidance::createDebugMarkerArray(
    output, data, shifter, debug, parameters_);
}

void StaticObstacleAvoidanceModule::updateAvoidanceDebugData(
  std::vector<AvoidanceDebugMsg> & avoidance_debug_msg_array) const
{
  debug_data_.avoidance_debug_msg_array.avoidance_info.clear();
  auto & debug_data_avoidance = debug_data_.avoidance_debug_msg_array.avoidance_info;
  debug_data_avoidance = avoidance_debug_msg_array;
  if (!debug_avoidance_initializer_for_shift_line_.empty()) {
    const bool is_info_old_ =
      (clock_->now() - debug_avoidance_initializer_for_shift_line_time_).seconds() > 0.1;
    if (!is_info_old_) {
      debug_data_avoidance.insert(
        debug_data_avoidance.end(), debug_avoidance_initializer_for_shift_line_.begin(),
        debug_avoidance_initializer_for_shift_line_.end());
    }
  }
}

double StaticObstacleAvoidanceModule::calcDistanceToStopLine(const ObjectData & object) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  const auto & p = parameters_;
  const auto & vehicle_width = planner_data_->parameters.vehicle_width;

  //         D5
  //      |<---->|                               D4
  //      |<----------------------------------------------------------------------->|
  // +-----------+            D1                 D2                      D3         +-----------+
  // |           |        |<------->|<------------------------->|<----------------->|           |
  // |    ego    |======= x ======= x ========================= x ==================|    obj    |
  // |           |    stop_point  avoid                       avoid                 |           |
  // +-----------+                start                        end                  +-----------+
  //
  // D1: p.min_prepare_distance
  // D2: min_avoid_distance
  // D3: longitudinal_avoid_margin_front (margin + D5)
  // D4: o_front.longitudinal
  // D5: additional_buffer_longitudinal (base_link2front or 0 depending on the
  // use_conservative_buffer_longitudinal)

  const auto object_type = utils::getHighestProbLabel(object.object.classification);
  const auto object_parameter = parameters_->object_parameters.at(object_type);

  const auto lateral_hard_margin = object.is_parked
                                     ? object_parameter.lateral_hard_margin_for_parked_vehicle
                                     : object_parameter.lateral_hard_margin;
  const auto avoid_margin = lateral_hard_margin * object.distance_factor +
                            object_parameter.lateral_soft_margin + 0.5 * vehicle_width;
  const auto avoidance_distance = helper_->getMinAvoidanceDistance(helper_->getShiftLength(
    object, utils::static_obstacle_avoidance::isOnRight(object), avoid_margin));
  const auto constant_distance = helper_->getFrontConstantDistance(object);
  const auto prepare_distance = helper_->getNominalPrepareDistance(0.0);

  return object.longitudinal -
         std::min(
           avoidance_distance + constant_distance + prepare_distance + p->stop_buffer,
           p->stop_max_distance);
}

void StaticObstacleAvoidanceModule::insertReturnDeadLine(
  const bool use_constraints_for_decel, ShiftedPath & shifted_path) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  const auto & data = avoid_data_;

  if (data.to_return_point > planner_data_->parameters.forward_path_length) {
    RCLCPP_DEBUG(getLogger(), "return dead line is far enough.");
    return;
  }

  const auto shift_length = path_shifter_.getLastShiftLength();

  if (std::abs(shift_length) < 1e-3) {
    RCLCPP_DEBUG(getLogger(), "don't have to consider return shift.");
    return;
  }

  if (data.new_shift_line.empty()) {
    RCLCPP_WARN_THROTTLE(getLogger(), *clock_, 3000, "module doesn't have return shift line.");
    return;
  }

  if (!helper_->isFeasible(data.new_shift_line)) {
    RCLCPP_WARN_THROTTLE(
      getLogger(), *clock_, 3000, "return shift line is not feasible. do nothing..");
    return;
  }

  // Consider the difference in path length between the shifted path and original path (the path
  // that is shifted inward has a shorter distance to the end of the path than the other one.)
  const auto & to_reference_path_end = data.arclength_from_ego.back();
  const auto to_shifted_path_end = autoware::motion_utils::calcSignedArcLength(
    shifted_path.path.points, getEgoPosition(), shifted_path.path.points.size() - 1);
  const auto buffer = std::max(0.0, to_shifted_path_end - to_reference_path_end);

  const auto min_return_distance =
    helper_->getMinAvoidanceDistance(shift_length) + helper_->getNominalPrepareDistance(0.0);
  const auto to_stop_line = data.to_return_point - min_return_distance - buffer;
  if (to_stop_line < -1.0 * parameters_->stop_buffer) {
    RCLCPP_WARN_THROTTLE(
      getLogger(), *clock_, 3000, "ego overran return shift dead line. do nothing.");
    return;
  }

  // If we don't need to consider deceleration constraints, insert a deceleration point
  // and return immediately
  if (!use_constraints_for_decel) {
    utils::static_obstacle_avoidance::insertDecelPoint(
      getEgoPosition(), to_stop_line - parameters_->stop_buffer, 0.0, shifted_path.path,
      stop_pose_);
    return;
  }

  // If the stop distance is not enough for comfortable stop, don't insert wait point.
  const auto is_comfortable_stop = helper_->getFeasibleDecelDistance(0.0) < to_stop_line;
  if (!is_comfortable_stop) {
    RCLCPP_DEBUG(getLogger(), "stop distance is not enough.");
    return;
  }

  utils::static_obstacle_avoidance::insertDecelPoint(
    getEgoPosition(), to_stop_line - parameters_->stop_buffer, 0.0, shifted_path.path, stop_pose_);

  // insert slow down speed.
  const double current_target_velocity = autoware::motion_utils::calc_feasible_velocity_from_jerk(
    shift_length, helper_->getLateralMinJerkLimit(), to_stop_line);
  if (current_target_velocity < getEgoSpeed()) {
    RCLCPP_DEBUG(getLogger(), "current velocity exceeds target slow down speed.");
    return;
  }

  const auto start_idx = planner_data_->findEgoIndex(shifted_path.path.points);
  for (size_t i = start_idx; i < shifted_path.path.points.size(); ++i) {
    const auto distance_from_ego =
      autoware::motion_utils::calcSignedArcLength(shifted_path.path.points, start_idx, i);

    // slow down speed is inserted only in front of the object.
    const auto shift_longitudinal_distance = to_stop_line - distance_from_ego;
    if (shift_longitudinal_distance < 0.0) {
      break;
    }

    // target speed with nominal jerk limits.
    const double v_target = autoware::motion_utils::calc_feasible_velocity_from_jerk(
      shift_length, helper_->getLateralMinJerkLimit(), shift_longitudinal_distance);
    const double v_original = shifted_path.path.points.at(i).point.longitudinal_velocity_mps;
    const double v_insert =
      std::max(v_target - parameters_->buf_slow_down_speed, parameters_->min_slow_down_speed);

    shifted_path.path.points.at(i).point.longitudinal_velocity_mps = std::min(v_original, v_insert);
  }
}

void StaticObstacleAvoidanceModule::insertWaitPoint(
  const bool use_constraints_for_decel, ShiftedPath & shifted_path) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  const auto & data = avoid_data_;

  // If avoidance path is NOT valid, don't insert any stop points.
  if (!data.valid) {
    return;
  }

  if (!data.stop_target_object) {
    return;
  }

  if (helper_->isShifted()) {
    return;
  }

  if (data.stop_target_object.value().info == ObjectInfo::CLOSE_DISTANCE_AVOIDANCE) {
    utils::static_obstacle_avoidance::insertDecelPoint(
      getEgoPosition(), 0.0, 0.0, shifted_path.path, stop_pose_);
    return;
  }

  if (data.to_stop_line < -1.0 * parameters_->stop_buffer) {
    RCLCPP_WARN_THROTTLE(
      getLogger(), *clock_, 3000, "ego overran avoidance dead line. do nothing.");
    return;
  }

  // If we don't need to consider deceleration constraints, insert a deceleration point
  // and return immediately
  if (!use_constraints_for_decel) {
    utils::static_obstacle_avoidance::insertDecelPoint(
      getEgoPosition(), data.to_stop_line, 0.0, shifted_path.path, stop_pose_);
    return;
  }

  // If the stop distance is not enough for comfortable stop, don't insert wait point.
  const auto is_comfortable_stop = helper_->getFeasibleDecelDistance(0.0) < data.to_stop_line;
  const auto is_slow_speed = getEgoSpeed() < parameters_->min_slow_down_speed;
  if (!is_comfortable_stop && !is_slow_speed) {
    RCLCPP_WARN_THROTTLE(getLogger(), *clock_, 3000, "not execute uncomfortable deceleration.");
    return;
  }

  // If target object can be stopped for, insert a deceleration point and return
  if (data.stop_target_object.value().is_stoppable) {
    utils::static_obstacle_avoidance::insertDecelPoint(
      getEgoPosition(), data.to_stop_line, 0.0, shifted_path.path, stop_pose_);
    return;
  }

  // If the object cannot be stopped for, calculate a "mild" deceleration distance
  // and insert a deceleration point at that distance
  const auto stop_distance = helper_->getFeasibleDecelDistance(0.0, false);
  utils::static_obstacle_avoidance::insertDecelPoint(
    getEgoPosition(), stop_distance, 0.0, shifted_path.path, stop_pose_);
}

void StaticObstacleAvoidanceModule::insertStopPoint(
  const bool use_constraints_for_decel, ShiftedPath & shifted_path) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  const auto & data = avoid_data_;

  if (data.safe) {
    return;
  }

  if (!parameters_->enable_yield_maneuver_during_shifting) {
    return;
  }

  const auto stop_idx = [&]() {
    const auto ego_idx = planner_data_->findEgoIndex(shifted_path.path.points);
    for (size_t idx = ego_idx; idx < shifted_path.path.points.size(); ++idx) {
      const auto & estimated_pose = shifted_path.path.points.at(idx).point.pose;
      if (!utils::isEgoWithinOriginalLane(
            data.current_lanelets, estimated_pose, planner_data_->parameters)) {
        return idx - 1;
      }
    }

    return shifted_path.path.points.size() - 1;
  }();

  const auto stop_distance = autoware::motion_utils::calcSignedArcLength(
    shifted_path.path.points, getEgoPosition(), stop_idx);

  // If we don't need to consider deceleration constraints, insert a deceleration point
  // and return immediately
  if (!use_constraints_for_decel) {
    utils::static_obstacle_avoidance::insertDecelPoint(
      getEgoPosition(), stop_distance, 0.0, shifted_path.path, stop_pose_);
    return;
  }

  // Otherwise, consider deceleration constraints before inserting deceleration point
  const auto decel_distance = helper_->getFeasibleDecelDistance(0.0, false);
  if (stop_distance < decel_distance) {
    return;
  }

  constexpr double MARGIN = 1.0;
  utils::static_obstacle_avoidance::insertDecelPoint(
    getEgoPosition(), stop_distance - MARGIN, 0.0, shifted_path.path, stop_pose_);
}

void StaticObstacleAvoidanceModule::insertPrepareVelocity(ShiftedPath & shifted_path) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  const auto & data = avoid_data_;

  // If avoidance path is NOT safe, don't insert any slow down sections.
  if (!data.safe && !data.stop_target_object) {
    return;
  }

  // If avoidance path is NOT safe, don't insert any slow down sections.
  if (!data.valid) {
    return;
  }

  // do nothing if there is no avoidance target.
  if (data.target_objects.empty()) {
    return;
  }

  // insert slow down speed only when the avoidance maneuver is not initiated.
  if (helper_->isShifted()) {
    return;
  }

  // insert slow down speed only when no shift line is approved.
  if (!path_shifter_.getShiftLines().empty()) {
    return;
  }

  const auto itr = std::find_if(
    data.target_objects.begin(), data.target_objects.end(),
    [](const auto & o) { return o.avoid_required; });

  const auto object = [&]() -> std::optional<ObjectData> {
    if (!data.yield_required) {
      return data.target_objects.front();
    }

    if (itr == data.target_objects.end()) {
      return std::nullopt;
    }

    return *itr;
  }();

  // do nothing if it can't avoid at the moment and avoidance is NOT definitely necessary.
  if (!object.has_value()) {
    return;
  }

  if (helper_->isAbsolutelyNotAvoidable(object.value())) {
    return;
  }

  // calculate shift length for front object.
  const auto & vehicle_width = planner_data_->parameters.vehicle_width;
  const auto object_type = utils::getHighestProbLabel(object.value().object.classification);
  const auto object_parameter = parameters_->object_parameters.at(object_type);
  const auto lateral_hard_margin = object.value().is_parked
                                     ? object_parameter.lateral_hard_margin_for_parked_vehicle
                                     : object_parameter.lateral_hard_margin;
  const auto avoid_margin = lateral_hard_margin * object.value().distance_factor +
                            object_parameter.lateral_soft_margin + 0.5 * vehicle_width;
  const auto shift_length = helper_->getShiftLength(
    object.value(), utils::static_obstacle_avoidance::isOnRight(object.value()), avoid_margin);

  // check slow down feasibility
  const auto min_avoid_distance = helper_->getMinAvoidanceDistance(shift_length);
  const auto distance_to_object = object.value().longitudinal;
  const auto remaining_distance = distance_to_object - min_avoid_distance;
  const auto decel_distance = helper_->getFeasibleDecelDistance(parameters_->velocity_map.front());
  if (remaining_distance < decel_distance) {
    return;
  }

  // decide slow down lower limit.
  const auto lower_speed = object.value().avoid_required ? 0.0 : parameters_->min_slow_down_speed;

  // insert slow down speed.
  const double current_target_velocity = autoware::motion_utils::calc_feasible_velocity_from_jerk(
    shift_length, helper_->getLateralMinJerkLimit(), distance_to_object);
  // NERO_SOA_FIX_10794 (upstream #10794): the stock test is inverted
  const bool too_fast_for_nominal_shift =
    nero_soa::env().fix_10794
      ? current_target_velocity + parameters_->buf_slow_down_speed < getEgoSpeed()
      : current_target_velocity < getEgoSpeed() + parameters_->buf_slow_down_speed;
  if (too_fast_for_nominal_shift) {
    utils::static_obstacle_avoidance::insertDecelPoint(
      getEgoPosition(), decel_distance, parameters_->velocity_map.front(), shifted_path.path,
      slow_pose_);
    return;
  }

  const auto start_idx = planner_data_->findEgoIndex(shifted_path.path.points);
  for (size_t i = start_idx; i < shifted_path.path.points.size(); ++i) {
    const auto distance_from_ego =
      autoware::motion_utils::calcSignedArcLength(shifted_path.path.points, start_idx, i);

    // slow down speed is inserted only in front of the object.
    const auto shift_longitudinal_distance = distance_to_object - distance_from_ego;
    if (shift_longitudinal_distance < min_avoid_distance) {
      break;
    }

    // target speed with nominal jerk limits.
    const double v_target = autoware::motion_utils::calc_feasible_velocity_from_jerk(
      shift_length, helper_->getLateralMinJerkLimit(), shift_longitudinal_distance);
    const double v_original = shifted_path.path.points.at(i).point.longitudinal_velocity_mps;
    const double v_insert = std::max(v_target - parameters_->buf_slow_down_speed, lower_speed);

    shifted_path.path.points.at(i).point.longitudinal_velocity_mps = std::min(v_original, v_insert);
  }

  const auto slow_pose_opt = autoware::motion_utils::calcLongitudinalOffsetPose(
    shifted_path.path.points, start_idx, distance_to_object);
  slow_pose_ = slow_pose_opt.has_value() ? PoseWithDetailOpt(PoseWithDetail(slow_pose_opt.value()))
                                         : std::nullopt;
}

void StaticObstacleAvoidanceModule::insertAvoidanceVelocity(ShiftedPath & shifted_path) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  const auto & data = avoid_data_;

  // do nothing if no shift line is approved.
  if (path_shifter_.getShiftLines().empty()) {
    return;
  }

  // do nothing if there is no avoidance target.
  if (data.target_objects.empty()) {
    return;
  }

  const auto [distance_to_accel_end_point, v_max] =
    helper_->getDistanceToAccelEndPoint(shifted_path.path);
  if (distance_to_accel_end_point < 1e-3) {
    return;
  }

  const auto start_idx = planner_data_->findEgoIndex(shifted_path.path.points);
  for (size_t i = start_idx; i < shifted_path.path.points.size(); ++i) {
    const auto distance_from_ego =
      autoware::motion_utils::calcSignedArcLength(shifted_path.path.points, start_idx, i);

    // slow down speed is inserted only in front of the object.
    const auto accel_distance = distance_to_accel_end_point - distance_from_ego;
    if (accel_distance < 0.0) {
      break;
    }

    const double v_target_square =
      v_max * v_max - 2.0 * parameters_->max_acceleration * accel_distance;
    if (v_target_square < 1e-3) {
      break;
    }

    // target speed with nominal jerk limits.
    const double v_target = std::max(getEgoSpeed(), std::sqrt(v_target_square));
    const double v_original = shifted_path.path.points.at(i).point.longitudinal_velocity_mps;
    shifted_path.path.points.at(i).point.longitudinal_velocity_mps = std::min(v_original, v_target);
  }

  const auto slow_pose_opt = autoware::motion_utils::calcLongitudinalOffsetPose(
    shifted_path.path.points, start_idx, distance_to_accel_end_point);
  slow_pose_ = slow_pose_opt.has_value() ? PoseWithDetailOpt(PoseWithDetail(slow_pose_opt.value()))
                                         : std::nullopt;
}

std::shared_ptr<AvoidanceDebugMsgArray> StaticObstacleAvoidanceModule::get_debug_msg_array() const
{
  debug_data_.avoidance_debug_msg_array.header.stamp = clock_->now();
  return std::make_shared<AvoidanceDebugMsgArray>(debug_data_.avoidance_debug_msg_array);
}

void StaticObstacleAvoidanceModule::acceptVisitor(
  const std::shared_ptr<SceneModuleVisitor> & visitor) const
{
  if (visitor) {
    visitor->visitAvoidanceModule(this);
  }
}

void SceneModuleVisitor::visitAvoidanceModule(const StaticObstacleAvoidanceModule * module) const
{
  avoidance_visitor_ = module->get_debug_msg_array();
}
}  // namespace autoware::behavior_path_planner
