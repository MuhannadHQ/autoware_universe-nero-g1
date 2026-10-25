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
#include <autoware_lanelet2_extension/regulatory_elements/speed_bump.hpp>
#include <autoware_lanelet2_extension/utility/message_conversion.hpp>
#include <autoware_lanelet2_extension/utility/query.hpp>
#include <autoware_lanelet2_extension/utility/utilities.hpp>
#include <autoware_utils/geometry/geometry.hpp>
#include <autoware_utils/math/normalization.hpp>
#include <tf2/utils.h>
#include <lanelet2_core/geometry/LineString.h>
#include <autoware_utils/system/time_keeper.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
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
    return r;
  }();
  return e;
}
bool active()
{
  return env().pause_margin >= 0.0 || env().committed || env().sight_m > 0.0 || env().corridor;
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
      const auto who = nero_ovl::env().start_yield ? neroStartYieldObject() : std::string{};
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

std::string StaticObstacleAvoidanceModule::neroStartYieldObject() const
{
  // this cycle's safety check (isSafePath stops at its first unsafe object)
  for (const auto & [id, c] : debug_data_.collision_check) {
    if (c.is_safe || !c.is_front) {
      continue;
    }
    const auto & q = c.current_obj_pose.position;
    const auto & e = getEgoPosition();
    char buf[96];
    std::snprintf(
      buf, sizeof(buf), "object %.1f m ahead (%.1f m/s, %s)", std::hypot(q.x - e.x, q.y - e.y),
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
  if (data.stop_target_object) return data.stop_target_object.value().longitudinal;
  std::optional<double> d;
  for (const auto & o : data.target_objects) {
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
  const double return_start, const double return_length)
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
  if (return_start >= 0.0 && return_length > 0.0 && std::abs(target_shift) > 1.0e-2) {
    const auto start_idx = std::max(
      ego_idx, utils::static_obstacle_avoidance::findPathIndexFromArclength(data.arclength_from_ego, return_start));
    const auto back_idx = utils::static_obstacle_avoidance::findPathIndexFromArclength(
      data.arclength_from_ego, return_start + return_length);
    if (back_idx > start_idx + 1) {
      ShiftLine back;
      back.start = points.at(start_idx).point.pose;
      back.end = points.at(back_idx).point.pose;
      back.start_shift_length = target_shift;
      back.end_shift_length = 0.0;
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
  if (nero_returning_ || nero_paused_ || !nero_onc_ || !nero_go_through_) {
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

std::vector<utils::path_safety_checker::PoseWithVelocityStamped>
StaticObstacleAvoidanceModule::neroEgoPredictedPath(
  const ShiftedPath & shifted_path, const bool for_front_object, const double v_hold,
  const double s_hold) const
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
  // speed bumps on the path (the vehicle crosses them at ~1 m/s, behavior_velocity_planner's speed_bump): ego is
  // predicted that slow there and braking toward it — a move that ends on a bump keeps ego beside the line longer
  // (av_x_two_field:1: back over the bump at 2.4 km/h while straddling, the car behind braked 7 m/s²)
  std::vector<std::pair<double, double>> bumps;  // [start, end] arc length from ego
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
  const ShiftedPath & shifted_path, const double v_hold, const double s_hold, DebugData & debug) const
{
  const auto & p = planner_data_->parameters;
  const auto ego_idx = planner_data_->findEgoIndex(shifted_path.path.points);
  bool has_left_shift = false;
  bool has_right_shift = false;
  for (size_t i = ego_idx; i < shifted_path.shift_length.size(); i++) {
    const auto length = shifted_path.shift_length.at(i);
    has_left_shift |= parameters_->lateral_execution_threshold < length;
    has_right_shift |= parameters_->lateral_execution_threshold < -1.0 * length;
  }
  if (!has_left_shift && !has_right_shift) {
    return true;
  }
  const auto hysteresis_factor = safe_ ? 1.0 : parameters_->hysteresis_factor_expand_rate;
  const auto targets = utils::static_obstacle_avoidance::getSafetyCheckTargetObjects(
    avoid_data_, planner_data_, parameters_, has_left_shift, has_right_shift, debug);
  if (targets.empty()) {
    return true;
  }
  const auto ego_front = neroEgoPredictedPath(shifted_path, true, v_hold, s_hold);
  const auto ego_rear = neroEgoPredictedPath(shifted_path, false, v_hold, s_hold);
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
  if (nero_gap_v_ >= 0.0 && neroIsSafeRaw(data.candidate_path, nero_gap_v_, s_hold, tmp)) {
    // the plan still works; one step up after up_cycles cycles in which that step also worked
    const double up = std::min(v_cap, nero_gap_v_ + E.gap_dv);
    if (up > nero_gap_v_ + 1e-3 && neroIsSafeRaw(data.candidate_path, up, s_hold, tmp)) {
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
      if (neroIsSafeRaw(data.candidate_path, vv, s_hold, tmp)) {
        found = vv;
        break;
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

  // set previous data
  if (success_spline_path_generation && success_linear_path_generation) {
    helper_->setPreviousLinearShiftPath(linear_shift_path);
    helper_->setPreviousSplineShiftPath(spline_shift_path);
    helper_->setPreviousReferencePath(path_shifter_.getReferencePath());
  } else {
    spline_shift_path = helper_->getPreviousSplineShiftPath();
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
  if (nero_ovl::active()) {
    neroOncomingReturn(avoid_data_);  // NERO_SOA_ONCOMING_COMMITTED
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
