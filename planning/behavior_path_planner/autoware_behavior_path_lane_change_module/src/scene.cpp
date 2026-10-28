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

#include "autoware/behavior_path_lane_change_module/scene.hpp"
#include <cstdio>
#include <array>

#include <chrono>
#include <cstdlib>
#include <unordered_map>

#include "autoware/behavior_path_lane_change_module/utils/calculation.hpp"
#include "autoware/behavior_path_lane_change_module/utils/path.hpp"
#include "autoware/behavior_path_lane_change_module/utils/utils.hpp"
#include "autoware/behavior_path_planner_common/utils/drivable_area_expansion/static_drivable_area.hpp"
#include "autoware/behavior_path_planner_common/utils/path_safety_checker/objects_filtering.hpp"
#include "autoware/behavior_path_planner_common/utils/path_safety_checker/safety_check.hpp"
#include "autoware/behavior_path_planner_common/utils/path_utils.hpp"
#include "autoware/behavior_path_planner_common/utils/traffic_light_utils.hpp"
#include "autoware/behavior_path_planner_common/utils/utils.hpp"

#include <autoware/motion_utils/trajectory/path_shift.hpp>
#include <autoware/motion_utils/trajectory/trajectory.hpp>
#include <autoware_frenet_planner/frenet_planner.hpp>
#include <autoware_frenet_planner/structures.hpp>
#include <autoware_lanelet2_extension/utility/message_conversion.hpp>
#include <autoware_lanelet2_extension/utility/utilities.hpp>
#include <autoware_utils/geometry/boost_polygon_utils.hpp>
#include <autoware_utils/math/unit_conversion.hpp>
#include <autoware_utils/system/stop_watch.hpp>
#include <autoware_utils/system/time_keeper.hpp>
#include <range/v3/action/insert.hpp>
#include <range/v3/algorithm.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/view.hpp>

#include <boost/geometry/algorithms/buffer.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <tf2/utils.h>
#include <lanelet2_core/geometry/LineString.h>

#include <lanelet2_core/geometry/Point.h>
#include <lanelet2_core/geometry/Polygon.h>
#include <lanelet2_core/primitives/LineString.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <memory>
#include <typeinfo>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace
{
// NERO_LC_DEBUG=1: once-per-second diagnostics of the candidate search (why no lane change path)
bool nero_lc_debug_tick(const char * key)
{
  static const bool on = std::getenv("NERO_LC_DEBUG") != nullptr;
  if (!on) return false;
  static std::unordered_map<std::string, std::chrono::steady_clock::time_point> last;
  const auto now = std::chrono::steady_clock::now();
  auto & l = last[key];
  if (now - l < std::chrono::milliseconds(1000)) return false;
  l = now;
  return true;
}

// NERO_LC_DEBUG=1: the same with its own period (streamjoin: per-candidate lines from a stand)
bool nero_lc_debug_every(const std::string & key, const int period_ms)
{
  static const bool on = std::getenv("NERO_LC_DEBUG") != nullptr;
  if (!on) return false;
  static std::unordered_map<std::string, std::chrono::steady_clock::time_point> last;
  const auto now = std::chrono::steady_clock::now();
  auto & l = last[key];
  if (now - l < std::chrono::milliseconds(period_ms)) return false;
  l = now;
  return true;
}

// NERO_LC_DEBUG (log only, lca job 2026-10-28): what became of every candidate of one search (path shifter or frenet),
// printed once per 0.5 s per module and direction: the 10-27 #6 phase-A question (why no change into a gap) had no
// record on the car. Counts: prepare samples used / failed, lane-changing samples, candidates rejected while built,
// unsafe, safety check thrown; the first message / unsafe object of each kind.
struct NeroSearchCount
{
  int prep{0};
  int prep_fail{0};
  int lc{0};
  int no_lc{0};
  int skip{0};
  int rej{0};
  int unsafe{0};
  int thrown{0};
  std::string rej_msg{"-"};
  std::string unsafe_msg{"-"};
  std::string thrown_msg{"-"};
};

std::string nero_first_unsafe(
  const autoware::behavior_path_planner::utils::path_safety_checker::CollisionCheckDebugMap & objs)
{
  for (const auto & [uuid, d] : objs) {
    if (d.is_safe) continue;
    char b[200];
    std::snprintf(
      b, sizeof(b), "%s %s front %d rss %.1f min_lon %.1f gap %.1f",
      boost::uuids::to_string(uuid).substr(0, 6).c_str(), d.unsafe_reason.c_str(), d.is_front ? 1 : 0,
      d.rss_longitudinal, d.inter_vehicle_distance,
      autoware_utils::calc_distance2d(d.expected_ego_pose, d.expected_obj_pose));
    return b;
  }
  return "?";
}

// streamjoin (2026-10-27): per module instance, "this change starts from the robot standing at its waiting stop with two
// or more changes due" (set while waiting, kept from the approval on until the instance ends). Kept outside the class so
// its layout (inherited by avoidance_by_lane_change) is unchanged.
std::unordered_map<const void *, bool> & nero_stand_start_map()
{
  static std::unordered_map<const void *, bool> m;
  return m;
}

// streamjoin (NERO_LC_SIGNALLED_PREPARE): when a change finished, per direction; per module instance whether it continues
// such a change
struct NeroFinish
{
  std::chrono::steady_clock::time_point t{};
  const void * by{nullptr};
};
std::unordered_map<int, NeroFinish> & nero_finish_times()
{
  static std::unordered_map<int, NeroFinish> m;
  return m;
}
bool nero_signalled_prepare()
{
  static const bool v = [] {
    const char * e = std::getenv("NERO_LC_SIGNALLED_PREPARE");
    return e != nullptr && std::atoi(e) > 0;
  }();
  return v;
}

// NERO_LC_CONT_SHORT_PREPARE=<s> (streamjoin2): the continuing change's prepare duration; < 0 = unset
double nero_cont_short_prepare()
{
  static const double v = [] {
    const char * e = std::getenv("NERO_LC_CONT_SHORT_PREPARE");
    return e != nullptr ? std::atof(e) : -1.0;
  }();
  return v;
}

// consecutive safe cycles per module instance (NERO_LC_MIN_SAFE_CYCLES; namespace scope so the stop placement can read it)
struct SafeCount
{
  int n{0};
  std::chrono::steady_clock::time_point last{};
};
std::unordered_map<const void *, SafeCount> & nero_safe_counts()
{
  static std::unordered_map<const void *, SafeCount> m;
  return m;
}

struct NeroUnsafeSeen
{
  bool seen{false};
  std::chrono::steady_clock::time_point last{};
};
std::unordered_map<const void *, NeroUnsafeSeen> & nero_unsafe_seen_map()
{
  static std::unordered_map<const void *, NeroUnsafeSeen> m;
  return m;
}

// streamjoin2: per module instance, "this change waits for a gap in a passing stream" (NERO_LC_MULTI_STAND_OFFSET /
// NERO_LC_MULTI_ROLL act only then)
struct NeroGapWait
{
  bool seen{false};
  std::chrono::steady_clock::time_point last{};
};
std::unordered_map<const void *, NeroGapWait> & nero_gap_wait_map()
{
  static std::unordered_map<const void *, NeroGapWait> m;
  return m;
}

struct NeroContinuation
{
  bool continues{false};
  std::chrono::steady_clock::time_point last{};
};
std::unordered_map<const void *, NeroContinuation> & nero_continuation_map()
{
  static std::unordered_map<const void *, NeroContinuation> m;
  return m;
}

// NERO_LC_MULTI_STAND_PREPARE=1: the prepare-phase RSS of upstream #11307 (autoware_universe, merged 2025-09-05; launch
// #1631 defaults): front / rear deceleration -1.0 / -1.0, rear reaction 1.0 s + margin 0.8 s, lateral 0.5 m, longitudinal
// 1.0 m, no velocity-delta headway, along the path.
autoware::behavior_path_planner::utils::path_safety_checker::RSSparams nero_prepare_rss()
{
  autoware::behavior_path_planner::utils::path_safety_checker::RSSparams p;
  p.extended_polygon_policy = "along_path";
  p.rear_vehicle_reaction_time = 1.0;
  p.rear_vehicle_safety_time_margin = 0.8;
  p.lateral_distance_max_threshold = 0.5;
  p.longitudinal_distance_min_threshold = 1.0;
  p.longitudinal_velocity_delta_time = 0.0;
  p.front_vehicle_deceleration = -1.0;
  p.rear_vehicle_deceleration = -1.0;
  return p;
}
}  // namespace

namespace autoware::behavior_path_planner
{
using autoware::motion_utils::calcSignedArcLength;
using utils::lane_change::create_lanes_polygon;
namespace calculation = utils::lane_change::calculation;
using utils::path_safety_checker::filter::velocity_filter;

namespace
{
// streamjoin job (NERO_LC_MULTI_STAND_PREPARE): the predicted time at which ego reaches the candidate's lane-changing
// start (the end of its prepare segment) along the given ego prediction; 0 once ego is past it
double nero_time_to_lc_start(
  const LaneChangePath & path, const std::vector<PoseWithVelocityStamped> & ego_predicted_path,
  const geometry_msgs::msg::Point & ego_position)
{
  if (path.path.points.size() < 2 || ego_predicted_path.size() < 2) {
    return 0.0;
  }
  const double s_start =
    calcSignedArcLength(path.path.points, ego_position, path.info.lane_changing_start.position);
  if (s_start <= 0.0) {
    return 0.0;
  }
  double cum = 0.0;
  for (size_t i = 1; i < ego_predicted_path.size(); ++i) {
    cum += autoware_utils::calc_distance2d(ego_predicted_path.at(i - 1).pose, ego_predicted_path.at(i).pose);
    if (cum >= s_start - 1e-3) {
      return ego_predicted_path.at(i).time;
    }
  }
  return ego_predicted_path.back().time;
}
}  // namespace

NormalLaneChange::NormalLaneChange(
  const std::shared_ptr<LaneChangeParameters> & parameters, LaneChangeModuleType type,
  Direction direction)
: LaneChangeBase(parameters, type, direction)
{
  stop_watch_.tic(getModuleTypeStr());
  stop_watch_.tic("stop_time");
}

void NormalLaneChange::update_lanes(const bool is_approved)
{
  if (is_approved) {
    return;
  }

  const auto current_lanes =
    utils::getCurrentLanesFromPath(prev_module_output_.path, planner_data_);

  if (current_lanes.empty()) {
    return;
  }

  const auto target_lanes = get_lane_change_lanes(current_lanes);
  if (target_lanes.empty()) {
    return;
  }

  lanelet::ConstLanelet current_lane;
  if (!common_data_ptr_->route_handler_ptr->getClosestLaneletWithinRoute(
        common_data_ptr_->get_ego_pose(), &current_lane)) {
    return;
  }

  common_data_ptr_->lanes_ptr->ego_lane = current_lane;
  const auto is_same_lanes_with_prev_iteration =
    utils::lane_change::is_same_lane_with_prev_iteration(
      common_data_ptr_, current_lanes, target_lanes);

  if (is_same_lanes_with_prev_iteration) {
    return;
  }

  common_data_ptr_->lanes_ptr->current = current_lanes;
  common_data_ptr_->lanes_ptr->target = target_lanes;

  const auto & route_handler_ptr = common_data_ptr_->route_handler_ptr;
  common_data_ptr_->current_lanes_path =
    route_handler_ptr->getCenterLinePath(current_lanes, 0.0, std::numeric_limits<double>::max());

  common_data_ptr_->target_lanes_path =
    route_handler_ptr->getCenterLinePath(target_lanes, 0.0, std::numeric_limits<double>::max());

  common_data_ptr_->lanes_ptr->target_neighbor = utils::lane_change::get_target_neighbor_lanes(
    *route_handler_ptr, current_lanes, common_data_ptr_->lc_type);

  common_data_ptr_->current_lanes_path =
    route_handler_ptr->getCenterLinePath(current_lanes, 0.0, std::numeric_limits<double>::max());
  common_data_ptr_->lanes_ptr->current_lane_in_goal_section =
    route_handler_ptr->isInGoalRouteSection(current_lanes.back());
  common_data_ptr_->lanes_ptr->target_lane_in_goal_section =
    route_handler_ptr->isInGoalRouteSection(target_lanes.back());

  common_data_ptr_->lanes_ptr->preceding_target =
    utils::lane_change::get_preceding_lanes(common_data_ptr_);

  lane_change_debug_.current_lanes = common_data_ptr_->lanes_ptr->current;
  lane_change_debug_.target_lanes = common_data_ptr_->lanes_ptr->target;

  lane_change_debug_.target_backward_lanes.clear();
  ranges::for_each(
    common_data_ptr_->lanes_ptr->preceding_target,
    [&](const lanelet::ConstLanelets & preceding_lanes) {
      ranges::insert(
        lane_change_debug_.target_backward_lanes, lane_change_debug_.target_backward_lanes.end(),
        preceding_lanes);
    });

  *common_data_ptr_->lanes_polygon_ptr = create_lanes_polygon(common_data_ptr_);
}

void NormalLaneChange::update_transient_data(const bool is_approved)
{
  if (
    !common_data_ptr_ || !common_data_ptr_->is_data_available() ||
    !common_data_ptr_->is_lanes_available()) {
    return;
  }

  auto & transient_data = common_data_ptr_->transient_data;

  const auto & p = *common_data_ptr_->bpp_param_ptr;
  const auto nearest_seg_idx =
    autoware::motion_utils::findFirstNearestSegmentIndexWithSoftConstraints(
      prev_module_output_.path.points, common_data_ptr_->get_ego_pose(),
      p.ego_nearest_dist_threshold, p.ego_nearest_yaw_threshold);
  transient_data.current_path_velocity =
    prev_module_output_.path.points.at(nearest_seg_idx).point.longitudinal_velocity_mps;
  transient_data.current_path_seg_idx = nearest_seg_idx;

  const auto active_signal_duration = std::invoke([&]() {
    const double own =
      signal_activation_time_ ? (clock_.now() - signal_activation_time_.value()).seconds() : 0.0;
    // NERO_LC_SIGNALLED_PREPARE (see get_prepare_metrics): the blinker has been on through the wait at the waiting place,
    // or since the change this one continues (same direction): upstream's blinker-time rule counts that time
    return nero_signalled_prepare() && (nero_stand_start() || nero_continues_change())
             ? std::max(own, lane_change_parameters_->trajectory.max_prepare_duration)
             : own;
  });
  transient_data.lane_change_prepare_duration =
    is_approved ? status_.lane_change_path.info.duration.prepare
                : calculation::calc_actual_prepare_duration(
                    common_data_ptr_, common_data_ptr_->get_ego_speed(), active_signal_duration);
  // NERO_LC_CONT_SHORT_PREPARE=<s> (streamjoin2, see get_prepare_metrics): the change continuing a change made from the
  // waiting place prepares this long (the blinker has been on since the first change)
  if (!is_approved && nero_cont_short_prepare() >= 0.0 && nero_continues_change() && !nero_stand_start()) {
    transient_data.lane_change_prepare_duration =
      std::min(transient_data.lane_change_prepare_duration, nero_cont_short_prepare());
  }

  std::tie(transient_data.lane_changing_length, transient_data.current_dist_buffer) =
    calculation::calc_lc_length_and_dist_buffer(common_data_ptr_, get_current_lanes());

  transient_data.next_dist_buffer.min =
    transient_data.current_dist_buffer.min - transient_data.lane_changing_length.min -
    common_data_ptr_->lc_param_ptr->lane_change_finish_judge_buffer;

  transient_data.dist_to_terminal_end = calculation::calc_dist_from_pose_to_terminal_end(
    common_data_ptr_, common_data_ptr_->lanes_ptr->current, common_data_ptr_->get_ego_pose());
  transient_data.dist_to_terminal_start =
    transient_data.dist_to_terminal_end - transient_data.current_dist_buffer.min;

  transient_data.dist_to_target_end = calculation::calc_dist_from_pose_to_terminal_end(
    common_data_ptr_, common_data_ptr_->lanes_ptr->target, common_data_ptr_->get_ego_pose());

  transient_data.max_prepare_length = calculation::calc_maximum_prepare_length(common_data_ptr_);

  transient_data.target_lane_length =
    lanelet::utils::getLaneletLength2d(common_data_ptr_->lanes_ptr->target);

  transient_data.current_lanes_ego_arc = lanelet::utils::getArcCoordinates(
    common_data_ptr_->lanes_ptr->current, common_data_ptr_->get_ego_pose());

  transient_data.target_lanes_ego_arc = lanelet::utils::getArcCoordinates(
    common_data_ptr_->lanes_ptr->target, common_data_ptr_->get_ego_pose());

  transient_data.is_ego_near_current_terminal_start =
    transient_data.dist_to_terminal_start < transient_data.max_prepare_length;

  transient_data.current_footprint = utils::lane_change::get_ego_footprint(
    common_data_ptr_->get_ego_pose(), common_data_ptr_->bpp_param_ptr->vehicle_info);

  const auto & ego_lane = common_data_ptr_->lanes_ptr->ego_lane;
  const auto & route_handler_ptr = common_data_ptr_->route_handler_ptr;
  transient_data.in_intersection = utils::lane_change::is_within_intersection(
    route_handler_ptr, ego_lane, transient_data.current_footprint);
  transient_data.in_turn_direction_lane =
    utils::lane_change::is_within_turn_direction_lanes(ego_lane, transient_data.current_footprint);

  update_dist_from_intersection();

  // streamjoin job (NERO_LC_MULTI_STAND_PREPARE): while waiting, "standing at the waiting stop with two or more changes
  // due" (below the stuck-detection speed 0.5 m/s); from the approval on the flag is kept (the robot then rolls through
  // the prepare phase it was approved with) until the module instance ends
  // With NERO_LC_MULTI_ROLL the robot rolls into its waiting place at the two-change crawl (NERO_LC_MULTI_WAIT_CRAWL)
  // instead of standing: the flag also covers that crawl.
  {
    static const double roll_v = [] {
      const char * r = std::getenv("NERO_LC_MULTI_ROLL");
      const char * c = std::getenv("NERO_LC_MULTI_WAIT_CRAWL");
      return (r != nullptr && std::atoi(r) > 0 && c != nullptr) ? std::atof(c) + 0.3 : 0.0;
    }();
    auto & stand = nero_stand_start_map()[this];
    // streamjoin2 (NERO_LC_MULTI_STAND_OFFSET / NERO_LC_MULTI_ROLL): the change waits for a gap in a passing stream =
    // unsafe (valid path) while target-lane cars faster than the crawl pass ego before its stand stop (the arrival cap's
    // own test, nero_passing_need), latched for the module instance. Sim c56c: unsafe for a car cutting in from the other
    // side, no stream: the stand stop 8 m earlier and the rolling wait slowed the approach, the change was approved 0.3 m/s
    // slower and later ran on past the car merging in behind it (511f24 0.38 m; the car / ma7 0.92 m).
    {
      static const double stand_offset = [] {
        const char * v = std::getenv("NERO_LC_MULTI_STAND_OFFSET");
        return v != nullptr ? std::atof(v) : 0.0;
      }();
      auto & gw = nero_gap_wait_map()[this];
      const auto now_g = std::chrono::steady_clock::now();
      if (now_g - gw.last > std::chrono::milliseconds(500)) {
        gw.seen = false;  // a new instance / a pause
      }
      gw.last = now_g;
      if (
        !is_approved && !gw.seen && !status_.is_safe && status_.is_valid_path && nero_two_or_more_due() &&
        !nero_rolling_wait_dir() &&
        nero_passing_need(std::max(0.0, transient_data.dist_to_terminal_start - stand_offset)) > 0.0) {
        gw.seen = true;
        if (nero_lc_debug_every("gap_wait", 500)) {
          RCLCPP_INFO(logger_, "[NERO_LC_GAP_WAIT] the change waits for a gap in a passing stream (%.1f m to the stand stop)",
            transient_data.dist_to_terminal_start - stand_offset);
        }
      }
    }
    if (!is_approved) {
      // only in a direction without the car's own rolling wait (on the car: the right changes; the left changes keep the
      // car's Saudi wait, hug and checks)
      // and only near the waiting stop (the rolling zone NERO_LC_WAIT_ROLL + 5 m): a robot slowed far before it (sim c56c:
      // braking for a car cutting in 35 m before the stop) is not at its waiting place
      static const double near_stop = [] {
        const char * w = std::getenv("NERO_LC_WAIT_ROLL");
        return (w != nullptr ? std::atof(w) : 0.0) + 5.0;
      }();
      const double to_stop = transient_data.dist_to_terminal_start - nero_stop_offset();
      // ... and only while it waits for a gap: the change has been unsafe near the stop since this instance started
      // (sim c56 / c56c: a change that was safe all along, the robot slow after braking for a car cutting in, is not a
      // gap wait; with the flag it was carried through at 15 km/h past the car merging in behind it, 0.32-0.37 m)
      auto & unsafe_seen = nero_unsafe_seen_map()[this];
      const auto now_u = std::chrono::steady_clock::now();
      if (now_u - unsafe_seen.last > std::chrono::milliseconds(500)) {
        unsafe_seen.seen = false;  // a new instance / a pause
      }
      unsafe_seen.last = now_u;
      const bool near = nero_two_or_more_due() && !nero_rolling_wait_dir() && to_stop <= near_stop;
      if (near && !status_.is_safe && status_.is_valid_path) {
        unsafe_seen.seen = true;
      }
      stand = near && unsafe_seen.seen &&
              std::abs(common_data_ptr_->get_ego_speed()) < std::max(lane_change_parameters_->th_stop_velocity, roll_v);
    }
    // NERO_LC_SIGNALLED_PREPARE: a module instance waiting within 2 s after a change in the same direction finished
    // continues it (the blinker stayed on)
    // (the next change's instance can already exist while the previous one finishes: latched while not approved)
    auto & cont = nero_continuation_map()[this];
    const auto now = std::chrono::steady_clock::now();
    if (now - cont.last > std::chrono::milliseconds(500)) {
      cont.continues = false;  // a new instance / a pause
    }
    cont.last = now;
    if (!cont.continues) {
      const auto & fin = nero_finish_times();
      const auto it = fin.find(static_cast<int>(direction_));
      cont.continues = it != fin.end() && it->second.by != this &&
                       now - it->second.t < std::chrono::milliseconds(2000);
      if (cont.continues && nero_lc_debug_every("continues", 500)) {
        RCLCPP_INFO(logger_, "[NERO_LC_SIGNALLED_PREPARE] this change continues the last one (approved %d)", is_approved);
      }
    }
  }

  updateStopTime();
  transient_data.is_ego_stuck = is_ego_stuck();

  // NERO: inching by lateral shift (NERO_LC_INCH_SHIFT=<m over the line>). While the lane change waits for
  // a gap, the lane path this module works from is shifted toward the target lane with a smooth ramp
  // (NERO_LC_INCH_RAMP, default 12 m) that ends at the waiting stop point, until ego's side is <m> over the
  // line — how drivers here show they want in. The waiting path, the stop point and every lane change
  // candidate are then built from the inched path, so ego neither swerves in nor S-curves out. The offset
  // only grows while every moving car behind in the target lane could still stop comfortably before ego
  // (NERO_LC_INCH_REACT 1.0 s + v^2 / 2 NERO_LC_INCH_DECEL 2.0 + 2 m) and nobody is alongside; otherwise
  // it holds where ego already is (never pulls back right while waiting).
  {
    static const double inch_over = [] {
      const char * v = std::getenv("NERO_LC_INCH_SHIFT");
      return v != nullptr ? std::atof(v) : -1.0;
    }();
    static const double inch_ramp = [] {
      const char * v = std::getenv("NERO_LC_INCH_RAMP");
      return v != nullptr ? std::max(3.0, std::atof(v)) : 12.0;
    }();
    static const double inch_react = [] {
      const char * v = std::getenv("NERO_LC_INCH_REACT");
      return v != nullptr ? std::atof(v) : 1.0;
    }();
    static const double inch_decel = [] {
      const char * v = std::getenv("NERO_LC_INCH_DECEL");
      return v != nullptr ? std::max(0.5, std::atof(v)) : 2.0;
    }();
    static std::unordered_map<const NormalLaneChange *, double> inch_cmd_held;
    auto & path = prev_module_output_.path;
    // NERO_LC_SAUDI_DIR (session 4): 0 = both directions (session 3), 1 = left lane changes only, 2 = right only. The
    // hug / nose-in is how a driver here shows he must get left before the U-turn; for the keep-right change into a
    // 20-30 km/h right lane it only made the cars in that lane brake (kr_field:1, a car needed 15 m/s^2).
    static const int saudi_dir = [] {
      const char * v = std::getenv("NERO_LC_SAUDI_DIR");
      return v != nullptr ? std::atoi(v) : 0;
    }();
    const bool saudi_dir_ok = saudi_dir == 0 || (saudi_dir == 1 && direction_ == Direction::LEFT) ||
                              (saudi_dir == 2 && direction_ == Direction::RIGHT);
    if (
      inch_over >= 0.0 && saudi_dir_ok && !is_approved && path.points.size() >= 2 &&
      direction_ != Direction::NONE) {
      const auto & ego_pose = common_data_ptr_->get_ego_pose();
      const auto & vi = common_data_ptr_->bpp_param_ptr->vehicle_info;
      const double sgn = (direction_ == Direction::LEFT) ? 1.0 : -1.0;
      const size_t ego_idx = autoware::motion_utils::findNearestIndex(path.points, ego_pose.position);
      const double s_stop = transient_data.dist_to_terminal_start - nero_stop_offset();
      // lateral room from the lane path to the line at the stop point
      const auto & cur_lanes = common_data_ptr_->lanes_ptr->current;
      const size_t stop_idx = std::invoke([&]() -> size_t {
        double acc = 0.0;
        for (size_t i = ego_idx + 1; i < path.points.size(); ++i) {
          acc += autoware_utils::calc_distance2d(path.points.at(i - 1), path.points.at(i));
          if (acc >= s_stop) return i;
        }
        return path.points.size() - 1;
      });
      const auto & sp = path.points.at(stop_idx).point.pose.position;
      double room = std::numeric_limits<double>::max();
      for (const auto & lane : cur_lanes) {
        const auto bound = (direction_ == Direction::LEFT) ? lane.leftBound2d() : lane.rightBound2d();
        room = std::min(room, static_cast<double>(lanelet::geometry::distance2d(
                                bound, lanelet::BasicPoint2d(sp.x, sp.y))));
      }
      // level 1, always while waiting: left side at the line (inside ego's own lane) — "I am coming over"
      // level 2, only while it is safe to: side inch_over over the line ("nosed in")
      static const double inch_signal_gap = [] {
        const char * v = std::getenv("NERO_LC_INCH_SIGNAL_GAP");  // body this far inside the line when hugging
        return v != nullptr ? std::atof(v) : 0.25;                // (the front corner leads while drifting)
      }();
      const double signal = std::max(0.0, room - 0.5 * vi.vehicle_width_m - inch_signal_gap);
      const double full = std::max(0.0, room - 0.5 * vi.vehicle_width_m + inch_over);
      // is it OK to (further) intrude now?
      // NERO_LC_INCH_BACK_DECEL (default 4.5): nosed in, but a moving car behind would now need more than this to
      // stop short of ego's nose (it is not going to let ego in), or a moving car is alongside -> give the room
      // back: pull back to the hug level over NERO_LC_INCH_BACK_LEN (default 4 m) — on 2.85 m lanes a car passing
      // at its lane centre would otherwise clear the nosed-in ego by ~0.2 m
      static const double inch_back_decel = [] {
        const char * v = std::getenv("NERO_LC_INCH_BACK_DECEL");
        return v != nullptr ? std::max(0.5, std::atof(v)) : 4.5;
      }();
      static const double inch_back_len = [] {
        const char * v = std::getenv("NERO_LC_INCH_BACK_LEN");
        return v != nullptr ? std::max(1.0, std::atof(v)) : 4.0;
      }();
      bool inch_safe = true;
      bool inch_threat = false;
      const auto & tgt_lanes = common_data_ptr_->lanes_ptr->target;
      const double s_ego_t = lanelet::utils::getArcCoordinates(tgt_lanes, ego_pose).length;
      for (const auto & obj : filtered_objects_.target_lane_trailing) {
        const double v = std::max(0.0, obj.initial_twist.linear.x);
        if (v < 0.5) continue;
        const double s_obj = lanelet::utils::getArcCoordinates(tgt_lanes, obj.initial_pose).length;
        const double gap = s_ego_t - vi.rear_overhang_m - (s_obj + 0.5 * obj.shape.dimensions.x);
        if (gap < v * inch_react + v * v / (2.0 * inch_decel) + 2.0) {
          inch_safe = false;
        }
        // ego's nose is ~max_longitudinal_offset ahead of its rear: the car must stop short of the nose's side
        if (gap + vi.max_longitudinal_offset_m < v * inch_react + v * v / (2.0 * inch_back_decel) + 1.0) {
          inch_threat = true;
        }
      }
      for (const auto & obj : filtered_objects_.target_lane_leading.moving) {
        const double s_obj = lanelet::utils::getArcCoordinates(tgt_lanes, obj.initial_pose).length;
        if (s_obj - 0.5 * obj.shape.dimensions.x < s_ego_t + vi.max_longitudinal_offset_m + 1.0) {
          inch_safe = false;  // alongside
          inch_threat = true;
          break;
        }
      }
      const double ego_lat =
        std::max(0.0, sgn * autoware::motion_utils::calcLateralOffset(path.points, ego_pose.position));
      double & held = inch_cmd_held[this];
      static std::unordered_map<const NormalLaneChange *, bool> inch_pulling_back;
      bool & pulling = inch_pulling_back[this];
      double cmd = signal;
      if (inch_safe) {
        cmd = full;
        pulling = false;
      } else if ((held > signal + 0.02 || pulling) && inch_threat && ego_lat > signal + 0.05) {
        cmd = signal;  // nosed in and a car behind will not stop / one is alongside: back to the hug level
        pulling = true;
      } else if (held > signal) {
        cmd = std::max(signal, std::min(held, ego_lat + 0.02));  // nosed in already: hold (no flicker)
        pulling = pulling && ego_lat > signal + 0.05;
      } else {
        pulling = false;
      }
      // NERO_LC_INCH_CLEAR=<m> (routelc job 2026-10-26): while a moving car is in the target lane within 40 m behind
      // ego (or alongside / just ahead), the hug keeps ego's side <m> inside the line (pulling back if it is over).
      // The hug and nose-in assume the driver behind stops for them; drivers who do not (and the crazy ones) pass the
      // hugging robot close: the car's own hug level gave 0.84-0.90 m (sim o37), and 0.55 m inside still 0.56-0.81 m to
      // stream cars driving near the line (sim c02, 10-26 #2, at 93795's end); the package uses 1.0 (ego about centred
      // while such a car is near). With the target lane empty the car's hug and nose-in are unchanged. Unset = the car.
      static const double inch_clear = [] {
        const char * v = std::getenv("NERO_LC_INCH_CLEAR");
        return v != nullptr ? std::atof(v) : -1.0;
      }();
      if (inch_clear >= 0.0) {
        bool traffic = false;
        for (const auto & obj : filtered_objects_.target_lane_trailing) {
          if (std::max(0.0, obj.initial_twist.linear.x) < 0.5) continue;
          const double s_obj = lanelet::utils::getArcCoordinates(tgt_lanes, obj.initial_pose).length;
          if (s_ego_t - (s_obj + 0.5 * obj.shape.dimensions.x) < 40.0) traffic = true;
        }
        for (const auto & obj : filtered_objects_.target_lane_leading.moving) {
          const double s_obj = lanelet::utils::getArcCoordinates(tgt_lanes, obj.initial_pose).length;
          if (s_obj - 0.5 * obj.shape.dimensions.x < s_ego_t + vi.max_longitudinal_offset_m + 5.0) traffic = true;
        }
        const double cap = std::max(0.0, room - 0.5 * vi.vehicle_width_m - inch_clear);
        if (traffic && cmd > cap) {
          cmd = cap;
          if (ego_lat > cap + 0.05) pulling = true;
        }
      }
      held = cmd;
      double s_a = std::max(0.0, s_stop - inch_ramp);
      double s_b = std::max(s_a + 0.5 * inch_ramp, s_stop);
      if (pulling) {
        s_a = 0.0;
        s_b = inch_back_len;
      }
      double arc = 0.0;
      for (size_t i = 0; i < path.points.size(); ++i) {
        if (i > ego_idx) {
          arc += autoware_utils::calc_distance2d(path.points.at(i - 1), path.points.at(i));
        }
        double off = ego_lat;
        if (i > ego_idx && arc > s_a) {
          const double x = std::clamp((arc - s_a) / (s_b - s_a), 0.0, 1.0);
          off = ego_lat + (cmd - ego_lat) * x * x * (3.0 - 2.0 * x);
        }
        auto & pose = path.points.at(i).point.pose;
        const double yaw = tf2::getYaw(pose.orientation);
        pose.position.x += -std::sin(yaw) * off * sgn;
        pose.position.y += std::cos(yaw) * off * sgn;
      }
      autoware::motion_utils::insertOrientation(path.points, true);
      if (nero_lc_debug_tick("inch")) {
        RCLCPP_INFO(
          logger_, "[NERO_LC] inch safe %d threat %d back %d cmd %.2f full %.2f ego_lat %.2f s_stop %.1f room %.2f",
          inch_safe, inch_threat, pulling, cmd, full, ego_lat, s_stop, room);
      }
    } else if (is_approved) {
      inch_cmd_held.erase(this);
    }
  }

  RCLCPP_DEBUG(
    logger_, "lane_changing_length - min: %.4f, max: %.4f", transient_data.lane_changing_length.min,
    transient_data.lane_changing_length.max);
  RCLCPP_DEBUG(
    logger_, "current_dist_buffer - min: %.4f, max: %.4f", transient_data.current_dist_buffer.min,
    transient_data.current_dist_buffer.max);
  RCLCPP_DEBUG(
    logger_, "next_dist_buffer - min: %.4f, max: %.4f", transient_data.next_dist_buffer.min,
    transient_data.next_dist_buffer.max);
  RCLCPP_DEBUG(logger_, "dist_to_terminal_start: %.4f", transient_data.dist_to_terminal_start);
  RCLCPP_DEBUG(logger_, "dist_to_terminal_end: %.4f", transient_data.dist_to_terminal_end);
  RCLCPP_DEBUG(logger_, "max_prepare_length: %.4f", transient_data.max_prepare_length);
  RCLCPP_DEBUG(
    logger_, "is_ego_near_current_terminal_start: %s",
    (transient_data.is_ego_near_current_terminal_start ? "true" : "false"));
}

void NormalLaneChange::update_filtered_objects()
{
  filtered_objects_ = filter_objects();
}

void NormalLaneChange::updateLaneChangeStatus()
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  const auto [found_valid_path, found_safe_path] = getSafePath(status_.lane_change_path);

  // Update status
  status_.is_valid_path = found_valid_path;
  // NERO: approve a lane change only after the path has been judged safe in N consecutive cycles,
  // so one optimistic cycle (a predicted-path flicker, an object missing for a cycle) cannot start
  // it. N from NERO_LC_MIN_SAFE_CYCLES (default 1 = upstream behaviour).
  static const int min_safe_cycles = [] {
    const char * v = std::getenv("NERO_LC_MIN_SAFE_CYCLES");
    return v ? std::max(1, std::atoi(v)) : 1;
  }();
  // kept outside the class so its layout (inherited by avoidance_by_lane_change) is unchanged
  // (a gap of > 0.5 s means a new module instance / a pause: start counting again)
  auto & c = nero_safe_counts()[this];
  const auto now = std::chrono::steady_clock::now();
  if (now - c.last > std::chrono::milliseconds(500)) {
    c.n = 0;
  }
  c.last = now;
  // NERO_LC_MULTI_ROLL (streamjoin2, 2026-10-27): while the change waits for a gap in a passing stream (nero_gap_wait),
  // the arrival cap slows ego so that the passing target-lane cars get ahead of it; a change is then not approved while
  // one of those cars clearly passing (>= 1.5 m/s faster than ego / the crawl, within 40 m behind, past the nose within
  // 8 s) is not yet past ego's nose - ego goes behind them, as the cap timed it. Sim c28 slow_alongside: the cap slowed the robot to 7 km/h for d9d99b coming up the merging road,
  // the change was then approved in front of it (its predicted path not yet in the target lane) and the slowed robot
  // crossed 1.3-2.1 m ahead of it (hw 0.3-0.4 s; the car without the cap 9.8 m). Only ever holds a change. Unset = the car.
  bool nero_hold_for_passing = false;
  {
    static const bool multi_roll = [] {
      const char * v = std::getenv("NERO_LC_MULTI_ROLL");
      return v != nullptr && std::atoi(v) > 0;
    }();
    if (multi_roll && found_safe_path && nero_gap_wait()) {
      const std::string who = nero_passing_behind();
      nero_hold_for_passing = !who.empty();
      if (nero_hold_for_passing && nero_lc_debug_every("roll_hold", 500)) {
        RCLCPP_INFO(logger_, "[NERO_LC_ROLL] safe path held: %s passing, not yet past ego's nose", who.c_str());
      }
    }
  }
  c.n = (found_safe_path && !nero_hold_for_passing) ? c.n + 1 : 0;
  status_.is_safe = found_safe_path && !nero_hold_for_passing && c.n >= min_safe_cycles;
  status_.lane_change_path.path.header = getRouteHeader();
}

std::pair<bool, bool> NormalLaneChange::getSafePath(LaneChangePath & safe_path) const
{
  const auto & current_lanes = get_current_lanes();
  const auto & target_lanes = get_target_lanes();

  if (current_lanes.empty() || target_lanes.empty()) {
    return {false, false};
  }

  LaneChangePaths valid_paths{};
  bool found_safe_path = get_lane_change_paths(valid_paths);
  // if no safe path is found and ego is stuck, try to find a path with a small margin

  if (valid_paths.empty() && terminal_lane_change_path_) {
    valid_paths.push_back(terminal_lane_change_path_.value());
  }

  lane_change_debug_.valid_paths = valid_paths;

  if (valid_paths.empty()) {
    return {false, false};
  }

  if (found_safe_path) {
    safe_path = valid_paths.back();
  } else {
    // force candidate
    safe_path = valid_paths.front();
  }

  return {true, found_safe_path};
}

bool NormalLaneChange::isLaneChangeRequired()
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);

  if (
    !common_data_ptr_ || !common_data_ptr_->is_data_available() ||
    !common_data_ptr_->is_lanes_available()) {
    return false;
  }

  const auto & current_lanes = common_data_ptr_->lanes_ptr->current;
  const auto & target_lanes = common_data_ptr_->lanes_ptr->target;

  const auto ego_dist_to_target_start =
    calculation::calc_ego_dist_to_lanes_start(common_data_ptr_, current_lanes, target_lanes);
  const auto max_prepare_length = calculation::calc_maximum_prepare_length(common_data_ptr_);
  // NERO_LC_DEBUG (log only, lca job): which condition keeps the module from being requested (once per 0.5 s)
  const auto nero_req_log = [&](const char * why) {
    const std::string key = std::string("req_") + typeid(*this).name() + (direction_ == Direction::LEFT ? "L" : "R");
    if (!nero_lc_debug_every(key, 500)) return;
    const bool only_tl = getStopTime() >= lane_change_parameters_->th_stop_time;
    RCLCPP_INFO(
      logger_, "[NERO_LC_REQ] %s %s | cur %ld tgt %ld | to_tgt_start %.1f max_prep %.1f to_reg %.1f near_term %d",
      direction_ == Direction::LEFT ? "L" : (direction_ == Direction::RIGHT ? "R" : "-"), why,
      current_lanes.front().id(), target_lanes.front().id(), ego_dist_to_target_start, max_prepare_length,
      utils::lane_change::get_distance_to_next_regulatory_element(common_data_ptr_, only_tl, only_tl),
      common_data_ptr_->transient_data.is_ego_near_current_terminal_start ? 1 : 0);
  };

  if (ego_dist_to_target_start > max_prepare_length) {
    nero_req_log("no: target start beyond max prepare");
    return false;
  }

  if (is_near_regulatory_element()) {
    RCLCPP_DEBUG(logger_, "Ego is close to regulatory element, don't run LC module");
    nero_req_log("no: near regulatory element");
    return false;
  }

  if (nero_keep_right_gate_blocks()) {
    nero_req_log("no: keep-right gate");
    return false;
  }

  nero_req_log("yes");
  return true;
}

namespace
{
struct NeroKeepRightGate
{
  double min_end{-1.0};  // <= 0: off
  double lead_dv{1.0};
  double lead_m{50.0};
  double lead_t{30.0};
};

const NeroKeepRightGate & nero_keep_right_gate()
{
  static const NeroKeepRightGate g = [] {
    NeroKeepRightGate r;
    if (const char * e = std::getenv("NERO_LC_KEEP_RIGHT_GATE")) {
      std::sscanf(e, "%lf,%lf,%lf,%lf", &r.min_end, &r.lead_dv, &r.lead_m, &r.lead_t);
    }
    return r;
  }();
  return g;
}

// UN R79 5.6.4.7: the gap a rear vehicle needs to brake at 3 m/s^2 after 0.4 s and keep a 1 s gap
double nero_scrit(const double v_e, const double v_r)
{
  const double dv = std::max(v_r - v_e, 0.0);
  return dv * 0.4 + dv * dv / 6.0 + v_e * 1.0;
}

bool nero_kr_log_tick()
{
  static double last = 0.0;
  const double now =
    std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  if (now - last < 1.0) return false;
  last = now;
  return true;
}
}  // namespace

bool NormalLaneChange::nero_keep_right_discretionary() const
{
  const auto & g = nero_keep_right_gate();
  if (
    g.min_end <= 0.0 || type_ != LaneChangeModuleType::NORMAL || direction_ != Direction::RIGHT || !common_data_ptr_ ||
    !common_data_ptr_->is_data_available() || !common_data_ptr_->is_lanes_available()) {
    return false;
  }
  return common_data_ptr_->transient_data.dist_to_terminal_end > g.min_end;
}

bool NormalLaneChange::nero_keep_right_gate_blocks() const
{
  if (!nero_keep_right_discretionary()) {
    return false;
  }
  const auto & g = nero_keep_right_gate();
  const double v_e = std::max(0.0, getEgoVelocity());
  // the speed ego drives at: the planner's max_vel, or the external limit when lower (the car's 20 km/h cap)
  double v_max = common_data_ptr_->bpp_param_ptr->max_vel;
  if (planner_data_ && planner_data_->external_limit_max_velocity) {
    v_max = std::min(v_max, static_cast<double>(planner_data_->external_limit_max_velocity->max_velocity));
  }
  const auto speed = [](const auto & o) { return std::hypot(o.initial_twist.linear.x, o.initial_twist.linear.y); };
  const auto block = [&](const char * why, const double a, const double b) {
    if (nero_kr_log_tick()) {
      RCLCPP_INFO(
        logger_, "[NERO_LC_KEEP_RIGHT_GATE] keep-right change held (%s %.1f / %.1f), v_e %.1f (max %.1f), end in %.0f m",
        why, a, b, v_e, v_max, common_data_ptr_->transient_data.dist_to_terminal_end);
    }
    return true;
  };
  // (a) a slower or standing car ahead in the target lane: stay left and pass it first (a standing one within lead_m, a
  // moving one also when ego at its maximum would reach it within lead_t: merging behind it means following it)
  const auto & lead = filtered_objects_.target_lane_leading;
  for (const auto * set : {&lead.moving, &lead.stopped, &lead.stopped_at_bound}) {
    for (const auto & o : *set) {
      const double v = speed(o);
      if (v >= v_max - g.lead_dv) {
        continue;
      }
      const bool reach = v >= 0.5 && o.dist_from_ego / std::max(v_max - v, 0.1) < g.lead_t;
      if (o.dist_from_ego < g.lead_m || reach) {
        return block("slower car ahead in the target lane: m / m/s", o.dist_from_ego, v);
      }
    }
  }
  // (b) never merge blind: the rear view of the target lane must cover S_crit for an unseen car at the limit + 30 km/h
  // the rear view of the target lane: 40 m (the 10-19/20 rule). A car following ego does not shorten it: the robot's left /
  // right / top / front lidars see the next lane past it (Ahmed 10-24; the 10-19 rule's 31.5 m assumed the low rear lidar)
  const double view = 40.0;
  // the unseen car comes along the target lane BEHIND ego: the highest speed limit of the target lanelet beside ego and
  // the preceding target lanelets (a junction piece's own turn limit, e.g. 10 km/h, is not the limit of the road behind
  // it; run 1024-220631); a lanelet without a tag counts as 50 km/h
  const auto & target = common_data_ptr_->lanes_ptr->target;
  double limit_kmh = target.empty() ? 50.0 : target.front().attributeOr("speed_limit", 50.0);
  for (const auto & seq : common_data_ptr_->lanes_ptr->preceding_target) {
    for (const auto & ll : seq) {
      limit_kmh = std::max(limit_kmh, ll.attributeOr("speed_limit", 50.0));
    }
  }
  const double v_phantom = (limit_kmh + 30.0) / 3.6;
  if (view < nero_scrit(v_e, v_phantom)) {
    return block("rear view m < S_crit m for an unseen car", view, nero_scrit(v_e, v_phantom));
  }
  // every seen car behind in the target lane needs S_crit
  const double rear = common_data_ptr_->bpp_param_ptr->vehicle_info.rear_overhang_m;
  for (const auto & o : filtered_objects_.target_lane_trailing) {
    const double gap = -o.dist_from_ego - rear - 0.5 * o.shape.dimensions.x;
    if (gap < nero_scrit(v_e, speed(o))) {
      return block("car behind in the target lane: gap m < S_crit m", gap, nero_scrit(v_e, speed(o)));
    }
  }
  return false;
}

bool NormalLaneChange::is_near_regulatory_element() const
{
  if (!common_data_ptr_ || !common_data_ptr_->is_data_available()) return false;

  if (common_data_ptr_->transient_data.is_ego_near_current_terminal_start) return false;

  const bool only_tl = getStopTime() >= lane_change_parameters_->th_stop_time;

  if (only_tl) {
    RCLCPP_DEBUG(logger_, "Stop time is over threshold. Ignore crosswalk and intersection checks.");
  }

  return common_data_ptr_->transient_data.max_prepare_length >
         utils::lane_change::get_distance_to_next_regulatory_element(
           common_data_ptr_, only_tl, only_tl);
}

bool NormalLaneChange::isStoppedAtRedTrafficLight() const
{
  return utils::traffic_light::isStoppedAtRedTrafficLightWithinDistance(
    get_current_lanes(), status_.lane_change_path.path, planner_data_,
    status_.lane_change_path.info.length.sum());
}

TurnSignalInfo NormalLaneChange::get_current_turn_signal_info() const
{
  const auto original_turn_signal_info = prev_module_output_.turn_signal_info;

  if (getModuleType() != LaneChangeModuleType::NORMAL || get_current_lanes().empty()) {
    return original_turn_signal_info;
  }

  if (direction_ != Direction::LEFT && direction_ != Direction::RIGHT) {
    return original_turn_signal_info;
  }

  const auto & path = prev_module_output_.path;
  const auto & original_command = original_turn_signal_info.turn_signal.command;
  // NERO_LC_FIX_11545=1: port of upstream #11545: with no valid path the blinker points toward the end of the lane
  // path (before: toward a stale / empty lane-changing end -> no blinker at 93699, 10-26e #25 "should go left").
  // Also (routelc): a module instance with no change to make (no shift interval: an "infinite" distance buffer; the
  // lane_change_right instance that appears beside a running left change at 93699) passes the blinker through: its
  // terminal blinker, placed with that buffer, started at the path front and turned the left change's blinker right.
  static const bool fix_11545 = [] {
    const char * v = std::getenv("NERO_LC_FIX_11545");
    return v != nullptr && std::atoi(v) > 0;
  }();
  const bool has_change = common_data_ptr_ && common_data_ptr_->transient_data.current_dist_buffer.min < 1e6;
  if (fix_11545 && !has_change) {
    return original_turn_signal_info;
  }
  if (
    !path.points.empty() && original_command != TurnIndicatorsCommand::DISABLE &&
    original_command != TurnIndicatorsCommand::NO_COMMAND) {
    return get_terminal_turn_signal_info();
  }

  if (fix_11545 && !status_.is_valid_path && !path.points.empty()) {
    return get_turn_signal(getEgoPose(), path.points.back().point.pose);
  }

  set_signal_activation_time();

  return get_turn_signal(getEgoPose(), getLaneChangePath().info.lane_changing_end);
}

TurnSignalInfo NormalLaneChange::get_terminal_turn_signal_info() const
{
  const auto & lane_change_param = getLaneChangeParam();
  const auto & common_param = getCommonParam();
  const auto & current_pose = getEgoPose();
  const auto & path = prev_module_output_.path;

  const auto original_turn_signal_info = prev_module_output_.turn_signal_info;

  const auto buffer = common_data_ptr_->transient_data.current_dist_buffer.min +
                      lane_change_param.min_length_for_turn_signal_activation +
                      common_param.base_link2front;
  const double path_length = autoware::motion_utils::calcArcLength(path.points);
  const auto start_pose = autoware::motion_utils::calcLongitudinalOffsetPose(
    path.points, 0, std::max(path_length - buffer, 0.0));

  if (!start_pose) return original_turn_signal_info;

  const auto terminal_turn_signal_info =
    get_turn_signal(*start_pose, getLaneChangePath().info.lane_changing_end);

  const auto nearest_dist_threshold = common_param.ego_nearest_dist_threshold;
  const auto nearest_yaw_threshold = common_param.ego_nearest_yaw_threshold;
  const auto current_nearest_seg_idx = common_data_ptr_->transient_data.current_path_seg_idx;

  const auto turn_signal_info = getTurnSignalDecider().overwrite_turn_signal(
    path, current_pose, current_nearest_seg_idx, original_turn_signal_info,
    terminal_turn_signal_info, nearest_dist_threshold, nearest_yaw_threshold);

  set_signal_activation_time(
    turn_signal_info.turn_signal.command != terminal_turn_signal_info.turn_signal.command);

  return turn_signal_info;
}

LaneChangePath NormalLaneChange::getLaneChangePath() const
{
  return status_.lane_change_path;
}

BehaviorModuleOutput NormalLaneChange::getTerminalLaneChangePath() const
{
  if (
    !lane_change_parameters_->terminal_path.enable ||
    !common_data_ptr_->transient_data.is_ego_near_current_terminal_start) {
    return prev_module_output_;
  }

  const auto is_near_goal = lane_change_parameters_->terminal_path.disable_near_goal &&
                            common_data_ptr_->lanes_ptr->target_lane_in_goal_section &&
                            common_data_ptr_->transient_data.dist_to_target_end <
                              common_data_ptr_->transient_data.lane_changing_length.max;
  if (is_near_goal) {
    return prev_module_output_;
  }

  const auto & current_lanes = get_current_lanes();
  if (current_lanes.empty()) {
    RCLCPP_DEBUG(logger_, "Current lanes not found. Returning previous module's path as output.");
    return prev_module_output_;
  }

  const auto terminal_lc_path = compute_terminal_lane_change_path();

  if (!terminal_lc_path) {
    RCLCPP_DEBUG(logger_, "Terminal path not found. Returning previous module's path as output.");
    return prev_module_output_;
  }

  auto output = prev_module_output_;

  output.path = *terminal_lc_path;
  output.turn_signal_info = get_current_turn_signal_info();

  extendOutputDrivableArea(output);

  return output;
}

BehaviorModuleOutput NormalLaneChange::generateOutput()
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  if (!status_.is_valid_path) {
    RCLCPP_DEBUG(logger_, "No valid path found. Returning previous module's path as output.");
    insert_stop_point(get_current_lanes(), prev_module_output_.path);
    return prev_module_output_;
  }

  auto output = prev_module_output_;
  if (isAbortState() && abort_path_) {
    output.path = abort_path_->path;
    insert_stop_point(get_current_lanes(), output.path);
  } else {
    output.path = status_.lane_change_path.path;

    const auto found_extended_path = extendPath();
    if (found_extended_path) {
      output.path = utils::combinePath(output.path, *found_extended_path);
    }
    output.reference_path = getReferencePath();

    if (isStopState()) {
      const auto current_velocity = getEgoVelocity();
      const auto current_dist = calcSignedArcLength(
        output.path.points, output.path.points.front().point.pose.position, getEgoPosition());
      const auto stop_dist =
        -(current_velocity * current_velocity / (2.0 * planner_data_->parameters.min_acc));
      set_stop_pose(stop_dist + current_dist, output.path, "incoming rear object");
    } else {
      insert_stop_point(get_target_lanes(), output.path);
    }
  }

  extendOutputDrivableArea(output);

  const auto turn_signal_info =
    get_turn_signal(getEgoPose(), status_.lane_change_path.info.lane_changing_end);
  const auto current_seg_idx = planner_data_->findEgoSegmentIndex(output.path.points);
  output.turn_signal_info = planner_data_->turn_signal_decider.overwrite_turn_signal(
    output.path, getEgoPose(), current_seg_idx, prev_module_output_.turn_signal_info,
    turn_signal_info, planner_data_->parameters.ego_nearest_dist_threshold,
    planner_data_->parameters.ego_nearest_yaw_threshold);

  set_signal_activation_time(
    output.turn_signal_info.turn_signal.command != turn_signal_info.turn_signal.command);

  return output;
}

void NormalLaneChange::extendOutputDrivableArea(BehaviorModuleOutput & output) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  const auto & dp = planner_data_->drivable_area_expansion_parameters;

  const auto drivable_lanes = utils::lane_change::generateDrivableLanes(
    *getRouteHandler(), get_current_lanes(), get_target_lanes());
  const auto shorten_lanes = utils::cutOverlappedLanes(output.path, drivable_lanes);
  const auto expanded_lanes = utils::expandLanelets(
    shorten_lanes, dp.drivable_area_left_bound_offset, dp.drivable_area_right_bound_offset,
    dp.drivable_area_types_to_skip);

  // for new architecture
  DrivableAreaInfo current_drivable_area_info;
  current_drivable_area_info.drivable_lanes = expanded_lanes;
  output.drivable_area_info = utils::combineDrivableAreaInfo(
    current_drivable_area_info, prev_module_output_.drivable_area_info);

  // NERO_LC_DRIVABLE_DEDUP=1 (lcfinish job 2026-10-27): combineDrivableLanes matches lanes by lanelet id AND bound ids;
  // this module's lanes are expanded copies (new bound ids), so nothing matches and the previous module's lanes after its
  // first one are appended behind ours: [93795|93828] [140954|140953] [140954|140954]. The left bound then runs along
  // 140953 and steps back to 140954's left edge, so past the end of the first lanelet pair the drivable area is the
  // current lane only; the path optimizer pulled the trajectory back toward it and the robot stopped for the next change
  // across the 93795 / 93828 line (sim o04, 10-26e #4-#7: lat -1.33, a car in 93795 passed at 0.24 m). Drop the appended
  // elements whose lanelets this module's own elements already hold; lanes beyond them stay. Upstream main the same.
  // Not while the change is held / stopping (the car's NERO hold, upstream's 'incoming rear object' stop): the robot then
  // stops on its merge path, and there the old bounds kept it nearer its own lane (sim c25: +0.30 m vs +0.51 m with the
  // dedupe, a car coming up beside it); that stays as on the car. Unset = the car.
  static const bool nero_dedup = [] {
    const char * v = std::getenv("NERO_LC_DRIVABLE_DEDUP");
    return v != nullptr && std::atoi(v) > 0;
  }();
  if (
    nero_dedup && !isStopState() &&
    output.drivable_area_info.drivable_lanes.size() > expanded_lanes.size()) {
    auto & combined = output.drivable_area_info.drivable_lanes;
    const auto own_end = combined.begin() + static_cast<std::ptrdiff_t>(expanded_lanes.size());
    std::unordered_set<lanelet::Id> own_ids;
    for (auto it = combined.begin(); it != own_end; ++it) {
      own_ids.insert(it->right_lane.id());
      own_ids.insert(it->left_lane.id());
      for (const auto & m : it->middle_lanes) own_ids.insert(m.id());
    }
    const auto covered = [&own_ids](const DrivableLanes & d) {
      return own_ids.count(d.right_lane.id()) > 0 && own_ids.count(d.left_lane.id()) > 0 &&
             std::all_of(d.middle_lanes.begin(), d.middle_lanes.end(), [&own_ids](const auto & m) {
               return own_ids.count(m.id()) > 0;
             });
    };
    combined.erase(std::remove_if(own_end, combined.end(), covered), combined.end());
  }

  // NERO_LC_DEBUG (log only, lcfinish job): which lanes the drivable area is built from (o04: past the end of 93828 the
  // bounds held 140954 only, the path optimizer pulled the robot back toward it)
  if (
    direction_ != Direction::NONE &&
    nero_lc_debug_tick(direction_ == Direction::LEFT ? "drivable_lanes_l" : "drivable_lanes_r")) {
    const auto ids = [](const lanelet::ConstLanelets & ls) {
      std::string s;
      for (const auto & l : ls) s += std::to_string(l.id()) + " ";
      return s;
    };
    const auto dl = [](const std::vector<DrivableLanes> & v) {
      std::string s;
      for (const auto & d : v) {
        s += "[" + std::to_string(d.right_lane.id()) + "|";
        for (const auto & m : d.middle_lanes) s += std::to_string(m.id()) + ",";
        s += "|" + std::to_string(d.left_lane.id()) + "] ";
      }
      return s;
    };
    RCLCPP_INFO(
      logger_, "[NERO_LC] drivable (%s): current %s target %s lc %s short %s prev %s out %s",
      direction_ == Direction::LEFT ? "left" : "right", ids(get_current_lanes()).c_str(),
      ids(get_target_lanes()).c_str(), dl(drivable_lanes).c_str(), dl(shorten_lanes).c_str(),
      dl(prev_module_output_.drivable_area_info.drivable_lanes).c_str(),
      dl(output.drivable_area_info.drivable_lanes).c_str());
  }
}

void NormalLaneChange::insert_stop_point(
  const lanelet::ConstLanelets & lanelets, PathWithLaneId & path, const bool is_waiting_approval)
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  if (lanelets.empty()) {
    return;
  }

  const auto & route_handler = getRouteHandler();

  if (route_handler->getNumLaneToPreferredLane(lanelets.back()) == 0) {
    return;
  }

  const auto & current_lanes = get_current_lanes();
  const auto is_current_lane = lanelets.front().id() == current_lanes.front().id() &&
                               lanelets.back().id() == current_lanes.back().id();

  // if input is not current lane, we can just insert the points at terminal.
  if (!is_current_lane) {
    if (common_data_ptr_->transient_data.next_dist_buffer.min < calculation::eps) {
      return;
    }
    const auto arc_length_to_stop_pose = motion_utils::calcArcLength(path.points) -
                                         common_data_ptr_->transient_data.next_dist_buffer.min;
    set_stop_pose(
      nero_comfort_stop_arc(
        arc_length_to_stop_pose, path,
        motion_utils::calcArcLength(path.points) - common_data_ptr_->bpp_param_ptr->base_link2front),
      path, "next lc terminal");
    return;
  }

  insert_stop_point_on_current_lanes(path, is_waiting_approval);
}

void NormalLaneChange::insert_stop_point_on_current_lanes(
  PathWithLaneId & path, const bool is_waiting_approval)
{
  const auto & path_front_pose = path.points.front().point.pose;
  const auto & ego_pose = common_data_ptr_->get_ego_pose();
  const auto dist_from_path_front =
    motion_utils::calcSignedArcLength(path.points, path_front_pose.position, ego_pose.position);

  const auto & transient_data = common_data_ptr_->transient_data;
  const auto & lanes_ptr = common_data_ptr_->lanes_ptr;
  const auto & lc_param_ptr = common_data_ptr_->lc_param_ptr;

  const auto dist_to_terminal_start = transient_data.dist_to_terminal_start - nero_stop_offset();

  const auto & bpp_param_ptr = common_data_ptr_->bpp_param_ptr;
  const auto dist_to_terminal_stop_raw = std::invoke([&]() -> double {
    const auto & curr_lanes_poly = common_data_ptr_->lanes_polygon_ptr->current;
    if (!utils::isEgoWithinOriginalLane(curr_lanes_poly, getEgoPose(), *bpp_param_ptr)) {
      return dist_from_path_front + dist_to_terminal_start;
    }

    if (
      terminal_lane_change_path_ && is_waiting_approval &&
      lc_param_ptr->terminal_path.stop_at_boundary) {
      return calculation::calc_dist_to_last_fit_width(common_data_ptr_, path);
    }

    const auto dist_to_last_fit_width = calculation::calc_dist_to_last_fit_width(
      lanes_ptr->current, path.points.front().point.pose, *bpp_param_ptr);

    return std::min(dist_from_path_front + dist_to_terminal_start, dist_to_last_fit_width);
  });

  const auto terminal_stop_reason = status_.is_valid_path ? "no safe path" : "no valid path";
  // NERO_LC_STOP_AT_TARGET=<m> (routelc job 2026-10-26): the terminal stop is never placed before the point <m> past
  // where the target lane begins (as long as that point is still before the latest lane-changing start). Before it,
  // where the window is short (KRS with both changes counted: the stop 5.5 m before 93732 begins; 93699), the robot
  // stood where no prepare from standstill (<= 3 m) reaches the target lane: no candidate for good. Unset = the car.
  const auto dist_to_terminal_stop_placed = std::invoke([&]() -> double {
    static const double at_target = [] {
      const char * v = std::getenv("NERO_LC_STOP_AT_TARGET");
      return v != nullptr ? std::atof(v) : -1.0;
    }();
    if (at_target < 0.0 || lanes_ptr->target_neighbor.empty()) {
      return dist_to_terminal_stop_raw;
    }
    const auto & front_lane = lanes_ptr->target_neighbor.front();
    if (front_lane.centerline2d().size() < 2) {
      return dist_to_terminal_stop_raw;
    }
    const auto target_front = utils::to_geom_msg_pose(front_lane.centerline2d().front(), front_lane);
    const double s_target_start = motion_utils::calcSignedArcLength(
      common_data_ptr_->current_lanes_path.points, path_front_pose.position, target_front.position);
    const double s_clamp = s_target_start + at_target;
    const double s_latest_start = dist_from_path_front + transient_data.dist_to_terminal_start;
    if (dist_to_terminal_stop_raw >= s_clamp || s_clamp > s_latest_start) {
      return dist_to_terminal_stop_raw;
    }
    if (nero_lc_debug_tick("stop_at_target")) {
      RCLCPP_INFO(
        logger_, "[NERO_LC_STOP_AT_TARGET] terminal stop %.1f m ahead -> %.1f m (target lane begins %.1f m ahead)",
        dist_to_terminal_stop_raw - dist_from_path_front, s_clamp - dist_from_path_front,
        s_target_start - dist_from_path_front);
    }
    return s_clamp;
  });
  const auto dist_to_terminal_stop = nero_comfort_stop_arc(
    dist_to_terminal_stop_placed, path,
    dist_from_path_front + transient_data.dist_to_terminal_end - bpp_param_ptr->base_link2front);
  if (
    filtered_objects_.current_lane.empty() ||
    !lane_change_parameters_->enable_stopped_vehicle_buffer) {
    set_stop_pose(dist_to_terminal_stop, path, terminal_stop_reason);
    return;
  }

  const auto & center_line = common_data_ptr_->current_lanes_path.points;
  const auto dist_to_target_lane_start = std::invoke([&]() -> double {
    const auto & front_lane = lanes_ptr->target_neighbor.front();
    const auto target_front =
      utils::to_geom_msg_pose(front_lane.centerline2d().front(), front_lane);
    return motion_utils::calcSignedArcLength(
      center_line, path_front_pose.position, target_front.position);
  });

  const auto arc_length_to_current_obj = utils::lane_change::get_min_dist_to_current_lanes_obj(
    common_data_ptr_, filtered_objects_, dist_to_target_lane_start, path);

  // margin with leading vehicle
  // consider rss distance when the LC need to avoid obstacles
  const auto rss_dist = calcRssDistance(
    0.0, lc_param_ptr->trajectory.min_lane_changing_velocity,
    lc_param_ptr->safety.rss_params_for_parked);

  const auto stop_margin = transient_data.lane_changing_length.min +
                           lc_param_ptr->backward_length_buffer_for_blocking_object + rss_dist +
                           bpp_param_ptr->base_link2front;
  const auto stop_arc_length_behind_obj = arc_length_to_current_obj - stop_margin;
  // NERO_LC_STOP_COMFORT also for the module's waiting place behind a car ahead in the current lane (room to pull out
  // behind it; the car itself is obstacle_stop's / obstacle_cruise's): never past 2 m behind that car (sim o37: this
  // stop appeared 2.4 m ahead at 10.9 km/h behind 1f9640 at 2-3 m/s -> -2.68)
  const double obj_hard_end = std::min(
    dist_from_path_front + transient_data.dist_to_terminal_end - bpp_param_ptr->base_link2front,
    arc_length_to_current_obj - bpp_param_ptr->base_link2front - 2.0);
  const auto comfort_obj = [&](const double arc) { return nero_comfort_stop_arc(arc, path, obj_hard_end); };

  if (stop_arc_length_behind_obj < dist_to_target_lane_start) {
    set_stop_pose(comfort_obj(dist_to_target_lane_start), path, "maintain distance to front object");
    return;
  }

  if (stop_arc_length_behind_obj > dist_to_terminal_stop) {
    set_stop_pose(dist_to_terminal_stop, path, terminal_stop_reason);
    return;
  }

  //  If the target lane in the lane change section is blocked by a stationary obstacle, there
  //  is no reason for stopping with a lane change margin. Instead, stop right behind the
  //  obstacle.
  //  ----------------------------------------------------------
  //                            [obj]>
  //  ----------------------------------------------------------
  //    [ego]>          | <--- stop margin --->  [obj]>
  //  ----------------------------------------------------------
  const auto has_blocking_target_lane_obj = utils::lane_change::has_blocking_target_object(
    filtered_objects_.target_lane_leading, stop_arc_length_behind_obj, path);

  if (has_blocking_target_lane_obj || stop_arc_length_behind_obj <= 0.0) {
    set_stop_pose(dist_to_terminal_stop, path, terminal_stop_reason);
    return;
  }

  set_stop_pose(comfort_obj(stop_arc_length_behind_obj), path, "maintain distance to front object");
}

PathWithLaneId NormalLaneChange::getReferencePath() const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  lanelet::ConstLanelet closest_lanelet;
  if (!lanelet::utils::query::getClosestLanelet(
        get_target_lanes(), getEgoPose(), &closest_lanelet)) {
    return prev_module_output_.reference_path;
  }
  auto reference_path = utils::getCenterLinePathFromLanelet(closest_lanelet, planner_data_);
  if (reference_path.points.empty()) {
    return prev_module_output_.reference_path;
  }
  return reference_path;
}

std::optional<PathWithLaneId> NormalLaneChange::extendPath()
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  const auto path = status_.lane_change_path.path;

  auto & target_lanes = common_data_ptr_->lanes_ptr->target;
  const auto & transient_data = common_data_ptr_->transient_data;

  const auto forward_path_length = getCommonParam().forward_path_length;

  if (
    (transient_data.target_lane_length - transient_data.target_lanes_ego_arc.length) >
    forward_path_length) {
    return std::nullopt;
  }
  const auto dist_to_end_of_path =
    lanelet::utils::getArcCoordinates(target_lanes, path.points.back().point.pose).length;

  if (common_data_ptr_->lanes_ptr->target_lane_in_goal_section) {
    const auto goal_pose = getRouteHandler()->getGoalPose();

    const auto dist_to_goal = lanelet::utils::getArcCoordinates(target_lanes, goal_pose).length;

    return getRouteHandler()->getCenterLinePath(target_lanes, dist_to_end_of_path, dist_to_goal);
  }

  lanelet::ConstLanelet next_lane;
  if (!getRouteHandler()->getNextLaneletWithinRoute(target_lanes.back(), &next_lane)) {
    return getRouteHandler()->getCenterLinePath(
      target_lanes, dist_to_end_of_path, transient_data.target_lane_length);
  }

  target_lanes.push_back(next_lane);

  const auto target_pose = std::invoke([&]() {
    const auto is_goal_in_next_lane = getRouteHandler()->isInGoalRouteSection(next_lane);
    if (is_goal_in_next_lane) {
      return getRouteHandler()->getGoalPose();
    }

    return utils::to_geom_msg_pose(next_lane.centerline2d().back(), next_lane);
  });

  const auto dist_to_target_pose =
    lanelet::utils::getArcCoordinates(target_lanes, target_pose).length;

  return getRouteHandler()->getCenterLinePath(
    target_lanes, dist_to_end_of_path, dist_to_target_pose);
}

void NormalLaneChange::resetParameters()
{
  is_abort_path_approved_ = false;
  is_abort_approval_requested_ = false;
  current_lane_change_state_ = LaneChangeStates::Normal;
  abort_path_ = nullptr;
  status_ = LaneChangeStatus();
  unsafe_hysteresis_count_ = 0;
  lane_change_debug_.reset();

  RCLCPP_DEBUG(logger_, "reset all flags and debug information.");
}

TurnSignalInfo NormalLaneChange::updateOutputTurnSignal() const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  const auto & current_lanes = get_current_lanes();
  const auto & shift_line = status_.lane_change_path.info.shift_line;
  const auto & shift_path = status_.lane_change_path.shifted_path;
  const auto current_shift_length = common_data_ptr_->transient_data.current_lanes_ego_arc.distance;
  constexpr bool is_driving_forward = true;
  // The getBehaviorTurnSignalInfo method expects the shifted line to be generated off of the ego's
  // current lane, lane change is different, so we set this flag to false.
  constexpr bool egos_lane_is_shifted = false;
  constexpr bool is_pull_out = false;
  constexpr bool is_lane_change = true;

  const auto [new_signal, is_ignore] = planner_data_->getBehaviorTurnSignalInfo(
    shift_path, shift_line, current_lanes, current_shift_length, is_driving_forward,
    egos_lane_is_shifted, is_pull_out, is_lane_change);
  return new_signal;
}

lanelet::ConstLanelets NormalLaneChange::get_lane_change_lanes(
  const lanelet::ConstLanelets & current_lanes) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  if (current_lanes.empty()) {
    return {};
  }
  // Get lane change lanes
  const auto & route_handler = getRouteHandler();

  const auto lane_change_lane =
    utils::lane_change::get_lane_change_target_lane(common_data_ptr_, current_lanes);

  if (!lane_change_lane) {
    return {};
  }

  const auto forward_length = std::invoke([&]() {
    const auto front_pose =
      utils::to_geom_msg_pose(lane_change_lane->centerline().front(), *lane_change_lane);
    const auto signed_distance = utils::getSignedDistance(front_pose, getEgoPose(), current_lanes);
    const auto forward_path_length = planner_data_->parameters.forward_path_length;
    return forward_path_length + std::max(signed_distance, 0.0);
  });

  return route_handler->getLaneletSequence(
    lane_change_lane.value(), getEgoPose(), 0.0, forward_length);
}

bool NormalLaneChange::hasFinishedLaneChange() const
{
  const auto & current_pose = getEgoPose();
  const auto & lane_change_end = status_.lane_change_path.info.shift_line.end;
  const auto & target_lanes = get_target_lanes();
  const double dist_to_lane_change_end =
    utils::getSignedDistance(current_pose, lane_change_end, target_lanes);

  const auto finish_judge_buffer = std::invoke([&]() {
    const double ego_velocity = getEgoVelocity();
    // If ego velocity is low, relax finish judge buffer
    if (std::abs(ego_velocity) < 1.0) {
      return 0.0;
    }
    return lane_change_parameters_->lane_change_finish_judge_buffer;
  });

  const auto has_passed_end_pose = dist_to_lane_change_end + finish_judge_buffer < 0.0;

  lane_change_debug_.distance_to_lane_change_finished =
    dist_to_lane_change_end + finish_judge_buffer;

  if (has_passed_end_pose) {
    const auto & lanes_polygon = common_data_ptr_->lanes_polygon_ptr->target;
    return !boost::geometry::disjoint(
      lanes_polygon,
      lanelet::utils::to2D(lanelet::utils::conversion::toLaneletPoint(current_pose.position)));
  }

  // NERO_LC_CONT_FINISH=<lateral m>,<yaw deg> (streamjoin2, 2026-10-27): a change made from the waiting place with a
  // further change in the same direction due finishes once ego is within these of the target lane's centre line (instead
  // of th_finish_judge_lateral_diff / _yaw_diff, 0.1 m / 1 deg): the next change's module starts while ego is still
  // settling, and its own checks decide (10-26e #28 / #29: the operator's second change came straight out of the first,
  // 41986 s 30-31.6; sim c28: the second change's module started 0.7 s / ~2 m after the robot was within 0.3 m).
  // Unset = the car.
  static const std::pair<double, double> cont_finish = [] {
    const char * v = std::getenv("NERO_LC_CONT_FINISH");
    double lat = -1.0;
    double yaw_deg = -1.0;
    if (v != nullptr && std::sscanf(v, "%lf,%lf", &lat, &yaw_deg) != 2) {
      lat = yaw_deg = -1.0;
    }
    return std::make_pair(lat, yaw_deg);
  }();
  const bool cont_relaxed = cont_finish.first > 0.0 && nero_stand_start() && nero_two_or_more_due();
  const double th_yaw = cont_relaxed ? std::max(lane_change_parameters_->th_finish_judge_yaw_diff,
                                                autoware_utils::deg2rad(cont_finish.second))
                                     : lane_change_parameters_->th_finish_judge_yaw_diff;
  const double th_lat = cont_relaxed
                          ? std::max(lane_change_parameters_->th_finish_judge_lateral_diff, cont_finish.first)
                          : lane_change_parameters_->th_finish_judge_lateral_diff;

  const auto yaw_deviation_to_centerline =
    utils::lane_change::calc_angle_to_lanelet_segment(target_lanes, current_pose);

  if (yaw_deviation_to_centerline > th_yaw) {
    return false;
  }

  const auto & arc_length = common_data_ptr_->transient_data.target_lanes_ego_arc;
  const auto reach_target_lane = std::abs(arc_length.distance) < th_lat;

  lane_change_debug_.distance_to_lane_change_finished = arc_length.distance;

  return reach_target_lane;
}

bool NormalLaneChange::isAbleToReturnCurrentLane() const
{
  if (status_.lane_change_path.path.points.size() < 2) {
    lane_change_debug_.is_able_to_return_to_current_lane = false;
    return false;
  }

  const auto & curr_lanes_poly = common_data_ptr_->lanes_polygon_ptr->current;
  if (!utils::isEgoWithinOriginalLane(
        curr_lanes_poly, getEgoPose(), planner_data_->parameters,
        lane_change_parameters_->cancel.overhang_tolerance)) {
    lane_change_debug_.is_able_to_return_to_current_lane = false;
    return false;
  }

  if (common_data_ptr_->transient_data.in_turn_direction_lane) {
    return true;
  }

  const auto nearest_idx = autoware::motion_utils::findFirstNearestSegmentIndexWithSoftConstraints(
    status_.lane_change_path.path.points, getEgoPose(),
    planner_data_->parameters.ego_nearest_dist_threshold,
    planner_data_->parameters.ego_nearest_yaw_threshold);

  const double ego_velocity =
    std::max(getEgoVelocity(), lane_change_parameters_->trajectory.min_lane_changing_velocity);
  const double estimated_travel_dist = ego_velocity * lane_change_parameters_->cancel.delta_time;

  double dist = 0.0;
  for (size_t idx = nearest_idx; idx < status_.lane_change_path.path.points.size() - 1; ++idx) {
    dist += calcSignedArcLength(status_.lane_change_path.path.points, idx, idx + 1);
    if (dist > estimated_travel_dist) {
      const auto & estimated_pose = status_.lane_change_path.path.points.at(idx + 1).point.pose;
      auto is_ego_within_original_lane = utils::isEgoWithinOriginalLane(
        curr_lanes_poly, estimated_pose, planner_data_->parameters,
        lane_change_parameters_->cancel.overhang_tolerance);
      lane_change_debug_.is_able_to_return_to_current_lane = is_ego_within_original_lane;
      return is_ego_within_original_lane;
    }
  }

  lane_change_debug_.is_able_to_return_to_current_lane = true;
  return true;
}

bool NormalLaneChange::is_near_terminal() const
{
  if (!common_data_ptr_ || !common_data_ptr_->is_data_available()) {
    return true;
  }

  // TODO(Azu) fully change to transient data
  const auto & lc_param_ptr = common_data_ptr_->lc_param_ptr;
  const auto backward_buffer = calculation::calc_stopping_distance(lc_param_ptr);

  const auto current_min_dist_buffer = common_data_ptr_->transient_data.current_dist_buffer.min;
  const auto min_lc_dist_with_buffer = backward_buffer + current_min_dist_buffer;

  return common_data_ptr_->transient_data.dist_to_terminal_end < min_lc_dist_with_buffer;
}

bool NormalLaneChange::isEgoOnPreparePhase() const
{
  const auto & start_position = status_.lane_change_path.info.shift_line.start.position;
  const auto & path_points = status_.lane_change_path.path.points;
  return calcSignedArcLength(path_points, start_position, getEgoPosition()) < 0.0;
}

bool NormalLaneChange::isAbleToStopSafely() const
{
  if (status_.lane_change_path.path.points.size() < 2) {
    return false;
  }

  const auto nearest_idx = autoware::motion_utils::findFirstNearestSegmentIndexWithSoftConstraints(
    status_.lane_change_path.path.points, getEgoPose(),
    planner_data_->parameters.ego_nearest_dist_threshold,
    planner_data_->parameters.ego_nearest_yaw_threshold);

  const auto current_velocity = getEgoVelocity();
  const auto stop_dist =
    -(current_velocity * current_velocity / (2.0 * planner_data_->parameters.min_acc));

  const auto & curr_lanes_poly = common_data_ptr_->lanes_polygon_ptr->current;
  double dist = 0.0;
  for (size_t idx = nearest_idx; idx < status_.lane_change_path.path.points.size() - 1; ++idx) {
    dist += calcSignedArcLength(status_.lane_change_path.path.points, idx, idx + 1);
    if (dist > stop_dist) {
      const auto & estimated_pose = status_.lane_change_path.path.points.at(idx + 1).point.pose;
      return utils::isEgoWithinOriginalLane(
        curr_lanes_poly, estimated_pose, planner_data_->parameters);
    }
  }
  return true;
}

bool NormalLaneChange::hasFinishedAbort() const
{
  if (!abort_path_) {
    return true;
  }

  const auto distance_to_finish = calcSignedArcLength(
    abort_path_->path.points, getEgoPosition(), abort_path_->info.shift_line.end.position);
  lane_change_debug_.distance_to_abort_finished = distance_to_finish;

  const auto has_finished_abort = distance_to_finish < 0.0;

  return has_finished_abort;
}

bool NormalLaneChange::isAbortState() const
{
  if (!lane_change_parameters_->cancel.enable_on_lane_changing_phase) {
    return false;
  }

  if (current_lane_change_state_ != LaneChangeStates::Abort) {
    return false;
  }

  return abort_path_ != nullptr;
}

lane_change::TargetObjects NormalLaneChange::get_target_objects(
  const FilteredLanesObjects & filtered_objects,
  [[maybe_unused]] const lanelet::ConstLanelets & current_lanes) const
{
  ExtendedPredictedObjects leading_objects = filtered_objects.target_lane_leading.moving;
  auto insert_leading_objects = [&](const auto & objects) {
    ranges::actions::insert(leading_objects, leading_objects.end(), objects);
  };

  insert_leading_objects(filtered_objects.target_lane_leading.stopped);
  insert_leading_objects(filtered_objects.target_lane_leading.stopped_at_bound);
  const auto chk_obj_in_curr_lanes =
    lane_change_parameters_->safety.collision_check.check_current_lane;
  if (chk_obj_in_curr_lanes || common_data_ptr_->transient_data.is_ego_stuck) {
    insert_leading_objects(filtered_objects.current_lane);
  }

  const auto chk_obj_in_other_lanes =
    lane_change_parameters_->safety.collision_check.check_other_lanes;
  if (chk_obj_in_other_lanes) {
    insert_leading_objects(filtered_objects.others);
  }

  // NERO_LC_TRAILING_LIKELY=<c> (lcfinish job 2026-10-27): for the change's safety check (approval, the approved path's
  // re-check -> hold / cancel / 'incoming rear object' stop), a car behind ego whose body is not in the target lanes (nor
  // in the lanes leading into them) counts only through a predicted path of confidence >= <c> that enters the target
  // lanes. Upstream counts any predicted path (use_all_predicted_paths): map_based_prediction gives every car going
  // straight a left and a right change at 0.188 (lane keeping 0.625), so the cars queued behind the robot in the lane it
  // was leaving became target-lane trailing cars once the robot was half its width aside; they pulled away as it left
  // their way and the change was held / cancelled / stopped half-way (sims c02 / o04, 10-26 #2 / 10-26e #4-#7: the robot
  // stood 0.9 m off its lane's centre or across the 140953 / 140952 line while they passed at 0.65-0.81 m). A car that
  // really pulls over gets its change at ~0.77, and its body is in the target lane after ~0.5 m sideways. The hug
  // (NERO_LC_INCH*), the keep-right gate and the yield cap keep the full list. Unset = upstream (the car).
  static const double nero_trailing_likely = [] {
    const char * v = std::getenv("NERO_LC_TRAILING_LIKELY");
    return v != nullptr ? std::atof(v) : -1.0;
  }();
  if (nero_trailing_likely > 0.0) {
    const auto & lp = *common_data_ptr_->lanes_polygon_ptr;
    ExtendedPredictedObjects trailing;
    trailing.reserve(filtered_objects.target_lane_trailing.size());
    for (const auto & o : filtered_objects.target_lane_trailing) {
      const bool body_in = !boost::geometry::disjoint(o.initial_polygon, lp.target) ||
                           std::any_of(lp.preceding_target.begin(), lp.preceding_target.end(), [&](const auto & poly) {
                             return !boost::geometry::disjoint(o.initial_polygon, poly);
                           });
      const bool likely = std::any_of(o.predicted_paths.begin(), o.predicted_paths.end(), [&](const auto & pp) {
        if (pp.confidence < nero_trailing_likely) return false;
        autoware_utils::LineString2d ls;
        ls.reserve(pp.path.size());
        for (const auto & pt : pp.path) ls.push_back(autoware_utils::from_msg(pt.pose.position).to_2d());
        return !boost::geometry::disjoint(ls, lp.target);
      });
      if (body_in || likely) trailing.push_back(o);
    }
    return {leading_objects, trailing};
  }

  return {leading_objects, filtered_objects.target_lane_trailing};
}

FilteredLanesObjects NormalLaneChange::filter_objects() const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  auto objects = *planner_data_->dynamic_object;
  utils::path_safety_checker::filterObjectsByClass(
    objects, lane_change_parameters_->safety.target_object_types);

  if (objects.objects.empty()) {
    return {};
  }

  filterOncomingObjects(objects);

  if (objects.objects.empty()) {
    return {};
  }

  if (!common_data_ptr_->is_lanes_available()) {
    return {};
  }

  const auto & current_pose = common_data_ptr_->get_ego_pose();
  const auto & current_lanes = common_data_ptr_->lanes_ptr->current;

  const auto & current_lanes_ref_path = common_data_ptr_->current_lanes_path;

  const auto & lanes_polygon = *common_data_ptr_->lanes_polygon_ptr;
  const auto dist_ego_to_current_lanes_center =
    lanelet::utils::getLateralDistanceToClosestLanelet(current_lanes, current_pose);

  FilteredLanesObjects filtered_objects;
  const auto reserve_size = objects.objects.size();
  filtered_objects.current_lane.reserve(reserve_size);
  auto & target_lane_leading = filtered_objects.target_lane_leading;
  target_lane_leading.stopped.reserve(reserve_size);
  target_lane_leading.moving.reserve(reserve_size);
  target_lane_leading.stopped_at_bound.reserve(reserve_size);
  filtered_objects.target_lane_trailing.reserve(reserve_size);
  filtered_objects.others.reserve(reserve_size);

  for (const auto & object : objects.objects) {
    auto ext_object = utils::lane_change::transform(object, *lane_change_parameters_);
    const auto & ext_obj_pose = ext_object.initial_pose;
    ext_object.dist_from_ego = autoware::motion_utils::calcSignedArcLength(
      current_lanes_ref_path.points, current_pose.position, ext_obj_pose.position);

    const auto is_before_terminal =
      utils::lane_change::is_before_terminal(common_data_ptr_, current_lanes_ref_path, ext_object);

    const auto ahead_of_ego =
      utils::lane_change::is_ahead_of_ego(common_data_ptr_, current_lanes_ref_path, ext_object);

    if (utils::lane_change::filter_target_lane_objects(
          common_data_ptr_, ext_object, dist_ego_to_current_lanes_center, ahead_of_ego,
          is_before_terminal, target_lane_leading, filtered_objects.target_lane_trailing)) {
      continue;
    }

    if (
      ahead_of_ego && is_before_terminal &&
      !boost::geometry::disjoint(ext_object.initial_polygon, lanes_polygon.current)) {
      // check only the objects that are in front of the ego vehicle
      filtered_objects.current_lane.push_back(ext_object);
      continue;
    }

    filtered_objects.others.push_back(ext_object);
  }

  const auto dist_comparator = [](const auto & obj1, const auto & obj2) {
    return obj1.dist_from_ego < obj2.dist_from_ego;
  };

  // There are no use cases for other lane objects yet, so to save some computation time, we dont
  // have to sort them.
  ranges::sort(filtered_objects.current_lane, dist_comparator);
  ranges::sort(target_lane_leading.stopped_at_bound, dist_comparator);
  ranges::sort(target_lane_leading.stopped, dist_comparator);
  ranges::sort(target_lane_leading.moving, dist_comparator);
  ranges::sort(filtered_objects.target_lane_trailing, [&](const auto & obj1, const auto & obj2) {
    return !dist_comparator(obj1, obj2);
  });

  lane_change_debug_.filtered_objects = filtered_objects;

  return filtered_objects;
}

void NormalLaneChange::filterOncomingObjects(PredictedObjects & objects) const
{
  const auto & current_pose = getEgoPose();

  const auto is_same_direction = [&](const PredictedObject & object) {
    const auto & object_pose = object.kinematics.initial_pose_with_covariance.pose;
    return !utils::path_safety_checker::isTargetObjectOncoming(
      current_pose, object_pose,
      common_data_ptr_->lc_param_ptr->safety.collision_check.th_incoming_object_yaw);
  };

  //  Perception noise could make stationary objects seem opposite the ego vehicle; check the
  //  velocity to prevent this.
  const auto is_stopped_object = [](const auto & object) -> bool {
    constexpr double min_vel_th = -0.5;
    constexpr double max_vel_th = 0.5;
    return velocity_filter(
      object.kinematics.initial_twist_with_covariance.twist, min_vel_th, max_vel_th);
  };

  utils::path_safety_checker::filterObjects(objects, [&](const PredictedObject & object) {
    const auto same_direction = is_same_direction(object);
    if (same_direction) {
      return true;
    }

    return is_stopped_object(object);
  });
}

std::vector<LaneChangePhaseMetrics> NormalLaneChange::get_prepare_metrics() const
{
  const auto & current_lanes = common_data_ptr_->lanes_ptr->current;
  const auto & target_lanes = common_data_ptr_->lanes_ptr->target;
  const auto current_velocity = getEgoVelocity();
  // set speed limit to be current path velocity;
  const auto max_path_velocity = common_data_ptr_->transient_data.current_path_velocity;

  const auto dist_to_target_start =
    calculation::calc_ego_dist_to_lanes_start(common_data_ptr_, current_lanes, target_lanes);
  const double max_prepare_length = common_data_ptr_->transient_data.dist_to_terminal_start;
  auto nero_metrics = calculation::calc_prepare_phase_metrics(
    common_data_ptr_, current_velocity, max_path_velocity, dist_to_target_start,
    max_prepare_length);
  // NERO_LC_SIGNALLED_PREPARE=1 (streamjoin job 2026-10-27): upstream shortens the prepare phase by the time the blinker
  // has been on (calc_actual_prepare_duration: max_prepare_duration - signal time, at least min_prepare_duration), but near
  // the lane end samples max_prepare_duration .. 0 and takes the smoothest = the longest. For a change made from the
  // waiting place with two or more changes due (NERO_LC_MULTI_STAND_PREPARE's flag: the blinker has been on through the
  // wait) and for a change that continues one in the same direction (the blinker stayed on), only prepare durations up to
  // min_prepare_duration are kept (if any are left). Sim c28: change 1 steered 4.4 s after the robot moved (the operator
  // 1.2 s), change 2 ran 3-4 s straight after change 1 (the operator: straight out of change 1). Unset = the car.
  {
    if (nero_signalled_prepare() && (nero_stand_start() || nero_continues_change())) {
      const double cap = lane_change_parameters_->trajectory.min_prepare_duration + 1e-3;
      std::vector<LaneChangePhaseMetrics> kept;
      for (const auto & m : nero_metrics) {
        if (m.duration <= cap) kept.push_back(m);
      }
      if (!kept.empty()) {
        if (nero_lc_debug_every("signalled_prep", 1000)) {
          RCLCPP_INFO(
            logger_, "[NERO_LC_SIGNALLED_PREPARE] %zu of %zu prepare metrics <= %.1f s (%s)", kept.size(),
            nero_metrics.size(), cap, nero_stand_start() ? "from the waiting place" : "continues the last change");
        }
        nero_metrics = std::move(kept);
      }
    }
  }
  // NERO_LC_CONT_SHORT_PREPARE=<s> (streamjoin2, 2026-10-27): the change that continues a change made from the waiting
  // place (same direction, the blinker on since the first change) gets this prepare duration instead of
  // min_prepare_duration (update_transient_data; away from the lane end the module offers that one duration only) and,
  // near the lane end, tries its sampled durations from the shortest up: the first safe candidate wins, so it steers out
  // of the first change at once when that is safe (10-26e #28 / #29: the operator's second change began straight out of
  // the first, 41986 s 30-31.6; sim c28 ran 1 s / ~2.5 m straight first and crossed 3 s after the operator). Every
  // candidate keeps the full safety check (the follower in the target lane checked as before). Unset = the car.
  {
    if (nero_cont_short_prepare() >= 0.0 && nero_continues_change() && !nero_stand_start()) {
      std::stable_sort(nero_metrics.begin(), nero_metrics.end(), [](const auto & a, const auto & b) {
        return a.duration < b.duration;
      });
    }
  }
  if (nero_lc_debug_tick("prep")) {
    RCLCPP_INFO(
      logger_, "[NERO_LC] prepare metrics %zu (v %.2f, dist_to_terminal_start %.2f, max_prep %.2f, "
      "min_prep %.2f, near_terminal %d)", nero_metrics.size(), current_velocity,
      common_data_ptr_->transient_data.dist_to_terminal_start, max_prepare_length, dist_to_target_start,
      common_data_ptr_->transient_data.is_ego_near_current_terminal_start);
  }
  return nero_metrics;
}

namespace
{
// NERO_LC_ADJ_LIMIT=1 (session 5): the distance from ego to the end of the stretch of current lanes (from ego's lanelet
// on) whose lane-changeable neighbour on the change side is a target lane. A lane change has to end within it — beyond,
// the two lanes are not side by side for lanelet2 (93612 / 28800 share nodes but not a bound). Infinite = no limit.
double nero_dist_to_end_of_adjacency(
  const lane_change::CommonDataPtr & common_data_ptr, const Direction direction,
  const geometry_msgs::msg::Pose & ego_pose)
{
  static const bool on = [] {
    const char * v = std::getenv("NERO_LC_ADJ_LIMIT");
    return v && std::atoi(v) > 0;
  }();
  const double inf = std::numeric_limits<double>::max();
  if (!on || (direction != Direction::LEFT && direction != Direction::RIGHT)) {
    return inf;
  }
  const auto & cur = common_data_ptr->lanes_ptr->current;
  const auto & tgt = common_data_ptr->lanes_ptr->target;
  const auto rg = common_data_ptr->route_handler_ptr->getRoutingGraphPtr();
  if (cur.empty() || tgt.empty() || !rg) {
    return inf;
  }
  std::unordered_set<lanelet::Id> tgt_ids;
  for (const auto & t : tgt) {
    tgt_ids.insert(t.id());
  }
  const auto ego_arc = lanelet::utils::getArcCoordinates(cur, ego_pose).length;
  double acc = 0.0;
  bool in_stretch = false;
  for (const auto & ll : cur) {
    const double len = lanelet::utils::getLaneletLength2d(ll);
    const double end = acc + len;
    acc = end;
    if (end <= ego_arc) {
      continue;  // behind ego
    }
    const auto nb = direction == Direction::RIGHT ? rg->right(ll) : rg->left(ll);
    const bool adjacent = nb && tgt_ids.count(nb->id()) > 0;
    if (adjacent) {
      in_stretch = true;
      continue;
    }
    if (in_stretch) {
      return (end - len) - ego_arc;  // the stretch ended at the start of this lanelet
    }
  }
  return inf;
}
}  // namespace

std::vector<LaneChangePhaseMetrics> NormalLaneChange::get_lane_changing_metrics(
  const PathWithLaneId & prep_segment, const LaneChangePhaseMetrics & prep_metric,
  const double shift_length, const double dist_to_reg_element,
  lane_change::MetricsDebug & debug_metrics) const
{
  const auto & transient_data = common_data_ptr_->transient_data;
  const auto dist_lc_start_to_end_of_lanes = calculation::calc_dist_from_pose_to_terminal_end(
    common_data_ptr_, common_data_ptr_->lanes_ptr->target_neighbor,
    prep_segment.points.back().point.pose);

  const auto max_lane_changing_length = std::invoke([&]() {
    double max_length =
      transient_data.is_ego_near_current_terminal_start
        ? transient_data.dist_to_terminal_end - prep_metric.length
        : std::min(transient_data.dist_to_terminal_end, dist_to_reg_element) - prep_metric.length;
    max_length =
      std::min(max_length, dist_lc_start_to_end_of_lanes - transient_data.next_dist_buffer.min);
    // NERO_LC_ADJ_LIMIT: and within the stretch where the target lane is actually beside ego's lane
    const double d_adj = nero_dist_to_end_of_adjacency(common_data_ptr_, direction_, getEgoPose());
    if (d_adj < std::numeric_limits<double>::max()) {
      max_length = std::min(max_length, d_adj - prep_metric.length);
    }
    return max_length;
  });

  debug_metrics.max_lane_changing_length = max_lane_changing_length;
  const auto max_path_velocity = prep_segment.points.back().point.longitudinal_velocity_mps;
  return calculation::calc_shift_phase_metrics(
    common_data_ptr_, shift_length, prep_metric.velocity, max_path_velocity,
    prep_metric.sampled_lon_accel, max_lane_changing_length);
}

bool NormalLaneChange::get_lane_change_paths(LaneChangePaths & candidate_paths) const
{
  lane_change_debug_.collision_check_objects.clear();
  lane_change_debug_.lane_change_metrics.clear();

  if (!common_data_ptr_->is_lanes_available()) {
    RCLCPP_WARN(logger_, "lanes are not available. Not expected.");
    return false;
  }

  if (common_data_ptr_->lanes_polygon_ptr->target_neighbor.empty()) {
    RCLCPP_WARN(logger_, "target_lane_neighbors_polygon_2d is empty. Not expected.");
    return false;
  }

  const auto & current_lanes = get_current_lanes();

  const auto target_objects = get_target_objects(filtered_objects_, current_lanes);

  const auto prepare_phase_metrics = get_prepare_metrics();

  const auto sorted_lane_ids = utils::lane_change::get_sorted_lane_ids(common_data_ptr_);
  if (
    common_data_ptr_->lc_param_ptr->frenet.enable &&
    common_data_ptr_->transient_data.is_ego_near_current_terminal_start) {
    return get_path_using_frenet(
      prepare_phase_metrics, target_objects, sorted_lane_ids, candidate_paths);
  }

  return get_path_using_path_shifter(
    prepare_phase_metrics, target_objects, sorted_lane_ids, candidate_paths);
}

namespace
{
// NERO_LC_FRENET_CACHE=1 (MRM job 2026-10-25): reuse the frenet candidates of the previous cycle when everything they
// are generated from is exactly the same (ego pose and speed, lanes, transient data, the previous module's path, the
// prepare metrics, the generation parameters): the robot standing at the lane end waiting for a gap. The candidates
// are then identical; the safety check against the objects still runs on them every cycle (time_limit as before).
// At the lane end generate_frenet_candidates() ran before the time_limit check: 57 prepare metrics -> 90-118 ms per
// cycle and per module under the car's CPU load, behavior_path_planner under 5 Hz -> MRM (car 10-25 12:43).
// One slot per module type and direction (the field re-created the module every cycle). Unset = upstream.
bool nero_frenet_cache_on()
{
  static const bool on = [] {
    const char * v = std::getenv("NERO_LC_FRENET_CACHE");
    const bool b = v != nullptr && std::atoi(v) > 0;
    if (b) {
      RCLCPP_INFO(
        rclcpp::get_logger("lane_change"),
        "[NERO_LC_FRENET_CACHE] 1: frenet candidates reused while their inputs are unchanged");
    }
    return b;
  }();
  return on;
}

std::vector<int64_t> nero_frenet_key(
  const lane_change::CommonDataPtr & cd, const PathWithLaneId & prev,
  const std::vector<LaneChangePhaseMetrics> & metrics)
{
  std::vector<int64_t> k;
  k.reserve(64 + prev.points.size() * 8 + metrics.size() * 6);
  const auto q = [&k](const double v, const double res) {
    k.push_back(
      std::isfinite(v) ? static_cast<int64_t>(std::llround(v / res))
                       : (v > 0 ? std::numeric_limits<int64_t>::max() : std::numeric_limits<int64_t>::min()));
  };
  const auto ids = [&k](const lanelet::ConstLanelets & lls) {
    k.push_back(-1);
    for (const auto & ll : lls) k.push_back(ll.id());
  };
  const auto & ego = cd->get_ego_pose();
  q(ego.position.x, 1e-4); q(ego.position.y, 1e-4); q(ego.position.z, 1e-3);
  q(ego.orientation.x, 1e-7); q(ego.orientation.y, 1e-7); q(ego.orientation.z, 1e-7); q(ego.orientation.w, 1e-7);
  q(cd->get_ego_speed(), 1e-4);
  k.push_back(static_cast<int64_t>(cd->direction));
  ids(cd->lanes_ptr->current); ids(cd->lanes_ptr->target); ids(cd->lanes_ptr->target_neighbor);
  k.push_back(cd->lanes_ptr->current_lane_in_goal_section); k.push_back(cd->lanes_ptr->target_lane_in_goal_section);
  const auto & t = cd->transient_data;
  q(t.lane_changing_length.min, 1e-6); q(t.lane_changing_length.max, 1e-6);
  q(t.current_dist_buffer.min, 1e-6); q(t.current_dist_buffer.max, 1e-6);
  q(t.next_dist_buffer.min, 1e-6); q(t.next_dist_buffer.max, 1e-6);
  q(t.dist_to_terminal_end, 1e-6); q(t.dist_to_terminal_start, 1e-6); q(t.max_prepare_length, 1e-6);
  q(t.dist_from_prev_intersection, 1e-6); q(t.target_lane_length, 1e-6); q(t.dist_to_target_end, 1e-6);
  q(t.current_path_velocity, 1e-6); q(t.lane_change_prepare_duration, 1e-6);
  k.push_back(static_cast<int64_t>(t.current_path_seg_idx));
  k.push_back(t.is_ego_near_current_terminal_start); k.push_back(t.is_ego_stuck);
  k.push_back(t.in_turn_direction_lane); k.push_back(t.in_intersection);
  const auto & lp = *cd->lc_param_ptr;
  q(lp.lane_change_finish_judge_buffer, 1e-9); q(lp.trajectory.th_lane_changing_length_diff, 1e-9);
  q(lp.trajectory.lateral_jerk, 1e-9); q(lp.trajectory.min_lane_changing_velocity, 1e-9);
  k.push_back(lp.trajectory.lat_acc_sampling_num); k.push_back(lp.trajectory.lon_acc_sampling_num);
  q(lp.frenet.th_yaw_diff_deg, 1e-9); q(lp.frenet.th_curvature_smoothing, 1e-9);
  k.push_back(reinterpret_cast<int64_t>(cd->route_handler_ptr.get()));
  k.push_back(-2);
  for (const auto & m : metrics) {
    q(m.duration, 1e-9); q(m.length, 1e-9); q(m.velocity, 1e-9);
    q(m.sampled_lon_accel, 1e-9); q(m.actual_lon_accel, 1e-9); q(m.lat_accel, 1e-9);
  }
  k.push_back(-3);
  for (const auto & pt : prev.points) {
    const auto & ps = pt.point.pose;
    q(ps.position.x, 1e-4); q(ps.position.y, 1e-4); q(ps.position.z, 1e-3);
    q(ps.orientation.z, 1e-7); q(ps.orientation.w, 1e-7);
    q(pt.point.longitudinal_velocity_mps, 1e-4);
    k.push_back(-4);
    for (const auto id : pt.lane_ids) k.push_back(id);
  }
  return k;
}
}  // namespace

bool NormalLaneChange::get_path_using_frenet(
  const std::vector<LaneChangePhaseMetrics> & prepare_metrics,
  const lane_change::TargetObjects & target_objects,
  const std::vector<std::vector<int64_t>> & sorted_lane_ids,
  LaneChangePaths & candidate_paths) const
{
  stop_watch_.tic(__func__);
  constexpr auto found_safe_path = true;
  const auto frenet_candidates = std::invoke([&]() {
    if (!nero_frenet_cache_on()) {
      return utils::lane_change::generate_frenet_candidates(
        common_data_ptr_, prev_module_output_.path, prepare_metrics);
    }
    struct Slot
    {
      std::vector<int64_t> key;
      std::vector<lane_change::TrajectoryGroup> candidates;
      size_t hits{0};
      size_t misses{0};
    };
    static std::mutex mtx;
    static std::unordered_map<std::string, Slot> slots;
    auto key = nero_frenet_key(common_data_ptr_, prev_module_output_.path, prepare_metrics);
    const std::lock_guard<std::mutex> lock(mtx);
    auto & slot = slots[std::string(typeid(*this).name()) + (direction_ == Direction::LEFT ? "L" : "R")];
    if (slot.key == key) {
      ++slot.hits;
    } else {
      ++slot.misses;
      slot.candidates = utils::lane_change::generate_frenet_candidates(
        common_data_ptr_, prev_module_output_.path, prepare_metrics);
      slot.key = std::move(key);
    }
    if (nero_lc_debug_tick("frenet_cache")) {
      RCLCPP_INFO(
        logger_, "[NERO_LC_FRENET_CACHE] %zu candidates, hits %zu misses %zu", slot.candidates.size(),
        slot.hits, slot.misses);
    }
    return slot.candidates;
  });
  if (nero_lc_debug_tick("frenet")) {
    RCLCPP_INFO(logger_, "[NERO_LC] frenet candidates %zu from %zu prepare metrics",
      frenet_candidates.size(), prepare_metrics.size());
  }
  RCLCPP_DEBUG(
    logger_, "Generated %lu candidate paths in %2.2f[us]", frenet_candidates.size(),
    stop_watch_.toc(__func__));

  NeroSearchCount nc;
  const auto nero_report = [&](const char * outcome) {
    const std::string key = std::string("frenet_") + typeid(*this).name() + (direction_ == Direction::LEFT ? "L" : "R");
    if (!nero_lc_debug_every(key, 500)) return;
    const auto & td = common_data_ptr_->transient_data;
    RCLCPP_INFO(
      logger_,
      "[NERO_LC_FRENET] %s %s | v %.2f to_term_start %.1f to_term_end %.1f next_buf %.1f objs %zu | cands %zu "
      "from %zu prep | rej %d unsafe %d thrown %d | rej: %s | unsafe: %s | thrown: %s",
      direction_ == Direction::LEFT ? "L" : "R", outcome, common_data_ptr_->get_ego_speed(),
      td.dist_to_terminal_start, td.dist_to_terminal_end, td.next_dist_buffer.min,
      target_objects.leading.size() + target_objects.trailing.size(), frenet_candidates.size(),
      prepare_metrics.size(), nc.rej, nc.unsafe, nc.thrown, nc.rej_msg.c_str(), nc.unsafe_msg.c_str(),
      nc.thrown_msg.c_str());
  };
  candidate_paths.reserve(frenet_candidates.size());
  lane_change_debug_.frenet_states.clear();
  lane_change_debug_.frenet_states.reserve(frenet_candidates.size());
  for (const auto & frenet_candidate : frenet_candidates) {
    if (stop_watch_.toc(__func__) >= lane_change_parameters_->time_limit) {
      break;
    }

    lane_change_debug_.frenet_states.emplace_back(
      frenet_candidate.prepare_metric, frenet_candidate.lane_changing.sampling_parameter,
      frenet_candidate.max_lane_changing_length);

    std::optional<LaneChangePath> candidate_path_opt;
    try {
      candidate_path_opt =
        utils::lane_change::get_candidate_path(frenet_candidate, common_data_ptr_, sorted_lane_ids);
    } catch (const std::exception & e) {
      RCLCPP_DEBUG(logger_, "%s", e.what());
      if (nero_lc_debug_tick("frenet_cand")) RCLCPP_INFO(logger_, "[NERO_LC] frenet candidate rejected: %s", e.what());
      if (nc.rej++ == 0) nc.rej_msg = e.what();
    }

    if (!candidate_path_opt) {
      continue;
    }

    try {
      if (check_candidate_path_safety(*candidate_path_opt, target_objects)) {
        RCLCPP_DEBUG(
          logger_, "Found safe path after %lu candidate(s). Total time: %2.2f[us]",
          frenet_candidates.size(), stop_watch_.toc(__func__));
        utils::lane_change::append_target_ref_to_candidate(
          *candidate_path_opt, common_data_ptr_->lc_param_ptr->frenet.th_curvature_smoothing);
        candidate_paths.push_back(*candidate_path_opt);
        nero_report("ACCEPT");
        return found_safe_path;
      }
      if (nero_lc_debug_tick("frenet_unsafe")) RCLCPP_INFO(logger_, "[NERO_LC] frenet candidate unsafe");
      if (nc.unsafe++ == 0) nc.unsafe_msg = nero_first_unsafe(lane_change_debug_.collision_check_objects);
    } catch (const std::exception & e) {
      RCLCPP_DEBUG(logger_, "%s", e.what());
      if (nero_lc_debug_tick("frenet_safety")) RCLCPP_INFO(logger_, "[NERO_LC] frenet safety check threw: %s", e.what());
      if (nc.thrown++ == 0) nc.thrown_msg = e.what();
    }

    // appending all paths affect performance
    if (candidate_paths.empty()) {
      candidate_paths.push_back(*candidate_path_opt);
    }
  }

  RCLCPP_DEBUG(
    logger_, "No safe path after %lu candidate(s). Total time: %2.2f[us]", frenet_candidates.size(),
    stop_watch_.toc(__func__));
  nero_report("none");
  return !found_safe_path;
}

bool NormalLaneChange::get_path_using_path_shifter(
  const std::vector<LaneChangePhaseMetrics> & prepare_metrics,
  const lane_change::TargetObjects & target_objects,
  const std::vector<std::vector<int64_t>> & sorted_lane_ids,
  LaneChangePaths & candidate_paths) const
{
  stop_watch_.tic(__func__);
  const auto & target_lanes = get_target_lanes();
  candidate_paths.reserve(
    prepare_metrics.size() * lane_change_parameters_->trajectory.lat_acc_sampling_num);

  const bool only_tl = getStopTime() >= lane_change_parameters_->th_stop_time;
  const auto dist_to_next_regulatory_element =
    utils::lane_change::get_distance_to_next_regulatory_element(common_data_ptr_, only_tl, only_tl);

  auto check_length_diff =
    [&](const double prep_length, const double lc_length, const bool check_lc) {
      if (candidate_paths.empty()) return true;

      const auto prep_diff = std::abs(candidate_paths.back().info.length.prepare - prep_length);
      if (prep_diff > lane_change_parameters_->trajectory.th_prepare_length_diff) return true;

      if (!check_lc) return false;

      const auto lc_diff = std::abs(candidate_paths.back().info.length.lane_changing - lc_length);
      return lc_diff > lane_change_parameters_->trajectory.th_lane_changing_length_diff;
    };

  NeroSearchCount nc;
  const auto nero_report = [&](const char * outcome) {
    const std::string key = std::string("shifter_") + typeid(*this).name() + (direction_ == Direction::LEFT ? "L" : "R");
    if (!nero_lc_debug_every(key, 500)) return;
    const auto & td = common_data_ptr_->transient_data;
    RCLCPP_INFO(
      logger_,
      "[NERO_LC_SHIFTER] %s %s | v %.2f to_term_start %.1f to_term_end %.1f next_buf %.1f objs %zu | prep %zu used %d "
      "fail %d | lc %d no_lc %d skip %d rej %d unsafe %d thrown %d | rej: %s | unsafe: %s | thrown: %s",
      direction_ == Direction::LEFT ? "L" : "R", outcome, common_data_ptr_->get_ego_speed(),
      td.dist_to_terminal_start, td.dist_to_terminal_end, td.next_dist_buffer.min,
      target_objects.leading.size() + target_objects.trailing.size(), prepare_metrics.size(), nc.prep,
      nc.prep_fail, nc.lc, nc.no_lc, nc.skip, nc.rej, nc.unsafe, nc.thrown, nc.rej_msg.c_str(),
      nc.unsafe_msg.c_str(), nc.thrown_msg.c_str());
  };

  for (const auto & prep_metric : prepare_metrics) {
    const auto debug_print = [&](const std::string & s) {
      RCLCPP_DEBUG(
        logger_, "%s | prep_time: %.5f | lon_acc: %.5f | prep_len: %.5f", s.c_str(),
        prep_metric.duration, prep_metric.actual_lon_accel, prep_metric.length);
    };

    if (!check_length_diff(prep_metric.length, 0.0, false)) {
      RCLCPP_DEBUG(logger_, "Skip: Change in prepare length is less than threshold.");
      continue;
    }

    PathWithLaneId prepare_segment;
    try {
      if (!utils::lane_change::get_prepare_segment(
            common_data_ptr_, prev_module_output_.path, prep_metric.length, prepare_segment)) {
        debug_print("Reject: failed to get valid prepare segment!");
        ++nc.prep_fail;
        continue;
      }
    } catch (const std::exception & e) {
      debug_print(e.what());
      ++nc.prep_fail;
      break;
    }
    ++nc.prep;

    debug_print("Prepare path satisfy constraints");

    const auto & lane_changing_start_pose = prepare_segment.points.back().point.pose;

    const auto shift_length =
      lanelet::utils::getLateralDistanceToClosestLanelet(target_lanes, lane_changing_start_pose);

    lane_change_debug_.lane_change_metrics.emplace_back();
    auto & debug_metrics = lane_change_debug_.lane_change_metrics.back();
    debug_metrics.prep_metric = prep_metric;
    debug_metrics.max_prepare_length = common_data_ptr_->transient_data.dist_to_terminal_start;
    const auto lane_changing_metrics = get_lane_changing_metrics(
      prepare_segment, prep_metric, shift_length, dist_to_next_regulatory_element, debug_metrics);

    // set_prepare_velocity must only be called after computing lane change metrics, as lane change
    // metrics rely on the prepare segment's original velocity as max_path_velocity.
    utils::lane_change::set_prepare_velocity(
      prepare_segment, common_data_ptr_->get_ego_speed(), prep_metric.velocity);
    if (lane_changing_metrics.empty()) ++nc.no_lc;

    for (const auto & lc_metric : lane_changing_metrics) {
      if (stop_watch_.toc(__func__) >= lane_change_parameters_->time_limit) {
        RCLCPP_DEBUG(logger_, "Time limit reached and no safe path was found.");
        nero_report("time_limit");
        return false;
      }
      ++nc.lc;

      debug_metrics.lc_metrics.emplace_back(lc_metric, -1);

      const auto debug_print_lat = [&](const std::string & s) {
        RCLCPP_DEBUG(
          logger_, "%s | lc_time: %.5f | lon_acc: %.5f | lat_acc: %.5f | lc_len: %.5f", s.c_str(),
          lc_metric.duration, lc_metric.actual_lon_accel, lc_metric.lat_accel, lc_metric.length);
      };

      if (!check_length_diff(prep_metric.length, lc_metric.length, true)) {
        RCLCPP_DEBUG(logger_, "Skip: Change in lane changing length is less than threshold.");
        ++nc.skip;
        continue;
      }

      LaneChangePath candidate_path;
      try {
        candidate_path = utils::lane_change::get_candidate_path(
          common_data_ptr_, prep_metric, lc_metric, prepare_segment, sorted_lane_ids, shift_length);
      } catch (const std::exception & e) {
        debug_print_lat(std::string("Reject: ") + e.what());
        if (nc.rej++ == 0) nc.rej_msg = e.what();
        continue;
      }

      candidate_paths.push_back(candidate_path);
      debug_metrics.lc_metrics.back().second = static_cast<int>(candidate_paths.size()) - 1;

      try {
        if (check_candidate_path_safety(candidate_path, target_objects)) {
          debug_print_lat("ACCEPT!!!: it is valid and safe!");
          nero_report("ACCEPT");
          return true;
        }
      } catch (const std::exception & e) {
        debug_print_lat(std::string("Reject: ") + e.what());
        ++nc.thrown;
        nc.thrown_msg = e.what();
        nero_report("thrown");
        return false;
      }

      if (nc.unsafe++ == 0) {
        char b[96];
        std::snprintf(
          b, sizeof(b), "T %.1f/%.1f v_prep %.2f: ", prep_metric.duration, lc_metric.duration,
          candidate_path.info.velocity.prepare);
        nc.unsafe_msg = b + nero_first_unsafe(lane_change_debug_.collision_check_objects);
      }
      debug_print_lat("Reject: sampled path is not safe.");
    }
  }

  RCLCPP_DEBUG(logger_, "No safety path found.");
  nero_report("none");
  return false;
}

bool NormalLaneChange::check_candidate_path_safety(
  const LaneChangePath & candidate_path, const lane_change::TargetObjects & target_objects) const
{
  const auto is_stuck = common_data_ptr_->transient_data.is_ego_stuck;
  if (utils::lane_change::has_overtaking_turn_lane_object(
        common_data_ptr_, filtered_objects_.target_lane_trailing)) {
    throw std::logic_error("Ego is nearby intersection, and there might be overtaking vehicle.");
  }

  // NERO_LC_DELAY_NO_ROOM=<m> (streamjoin job 2026-10-27): for a change made from the waiting place with two or more
  // changes due (NERO_LC_MULTI_STAND_PREPARE's flag) or one continuing it, a stopped target-lane car delays the change
  // only if the change could still be made after passing it: its far end + <m> before the latest change start.
  // Upstream's exemption (is_near_end: the car's centre within one change's buffer of the lane end) misses a car just
  // before that: 10-26e #28 / #29 (sim c28) the curb-parked 97db75 6 m before the right turn held change 2 for 2.7-3.0 s
  // at 41986 s 40 in 5 of 5 runs (a knife edge on the candidate's length) although no change fits after it. The car stays
  // in the collision check (the parked RSS). Unset = the car.
  static const double no_room_margin = [] {
    const char * v = std::getenv("NERO_LC_DELAY_NO_ROOM");
    return v != nullptr ? std::atof(v) : -1.0;
  }();
  auto delay_objects = filtered_objects_.target_lane_leading.stopped;
  if (no_room_margin >= 0.0 && (nero_stand_start() || nero_continues_change())) {
    const double latest_start = common_data_ptr_->transient_data.dist_to_terminal_start;
    delay_objects.erase(
      std::remove_if(
        delay_objects.begin(), delay_objects.end(),
        [&](const auto & o) {
          return o.dist_from_ego + 0.5 * o.shape.dimensions.x + no_room_margin > latest_start;
        }),
      delay_objects.end());
  }
  if (
    !is_stuck && utils::lane_change::is_delay_lane_change(
                   common_data_ptr_, candidate_path, delay_objects,
                   lane_change_debug_.collision_check_objects)) {
    throw std::logic_error(
      "Ego is not stuck and parked vehicle exists in the target lane. Skip lane change.");
  }

  const auto lc_start_velocity = candidate_path.info.velocity.prepare;
  const auto min_lc_velocity = lane_change_parameters_->trajectory.min_lane_changing_velocity;
  constexpr double margin = 0.1;
  // path is unsafe if it exceeds target lane boundary with a high velocity
  if (
    lane_change_parameters_->safety.enable_target_lane_bound_check &&
    lc_start_velocity > min_lc_velocity + margin &&
    utils::lane_change::path_footprint_exceeds_target_lane_bound(
      common_data_ptr_, candidate_path.shifted_path.path, planner_data_->parameters.vehicle_info)) {
    throw std::logic_error("Path footprint exceeds target lane boundary. Skip lane change.");
  }

  if ((target_objects.empty()) || candidate_path.path.points.empty()) {
    RCLCPP_DEBUG(logger_, "There is nothing to check.");
    return true;
  }

  constexpr size_t decel_sampling_num = 1;
  const auto ego_predicted_paths = utils::lane_change::convert_to_predicted_paths(
    common_data_ptr_, candidate_path, decel_sampling_num);

  const auto safety_check_with_normal_rss = isLaneChangePathSafe(
    candidate_path, ego_predicted_paths, target_objects,
    common_data_ptr_->lc_param_ptr->safety.rss_params, lane_change_debug_.collision_check_objects);

  // NERO_LC_DEBUG (log only, streamjoin job): from a stand with two or more changes due, per prepare duration (0.3 s): the
  // verdict and the first unsafe object (who holds the start: front / rear, rss and minimum gaps, predicted gap)
  // (streamjoin2: also for the change continuing a flagged change: who holds the second change, tagged "cont")
  if ((nero_stand_start() || nero_continues_change()) && !ego_predicted_paths.empty()) {
    const auto & info = candidate_path.info;
    const std::string key = std::string(nero_stand_start() ? "stand_cand_" : "cont_cand_") +
                            std::to_string(static_cast<int>(info.duration.prepare * 10.0 + 0.5));
    if (nero_lc_debug_every(key, 300)) {
      std::string why = "-";
      for (const auto & [uuid, d] : lane_change_debug_.collision_check_objects) {
        if (d.is_safe) continue;
        char b2[200];
        std::snprintf(
          b2, sizeof(b2), "%s %s front %d rss %.1f min_lon %.1f gap %.1f", boost::uuids::to_string(uuid).substr(0, 6).c_str(),
          d.unsafe_reason.c_str(), d.is_front ? 1 : 0, d.rss_longitudinal, d.inter_vehicle_distance,
          autoware_utils::calc_distance2d(d.expected_ego_pose, d.expected_obj_pose));
        why = b2;
        break;
      }
      RCLCPP_INFO(
        logger_, "[NERO_LC_STAND] %s T %.1f v_prep %.2f lc_v %.2f t_split %.1f safe %d | %s",
        nero_stand_start() ? "stand" : "cont", info.duration.prepare,
        info.velocity.prepare, info.terminal_lane_changing_velocity,
        nero_time_to_lc_start(candidate_path, ego_predicted_paths.front(), getEgoPosition()),
        safety_check_with_normal_rss.is_safe ? 1 : 0, why.c_str());
    }
  }

  if (!safety_check_with_normal_rss.is_safe && is_stuck) {
    const auto safety_check_with_stuck_rss = isLaneChangePathSafe(
      candidate_path, ego_predicted_paths, target_objects,
      common_data_ptr_->lc_param_ptr->safety.rss_params_for_stuck,
      lane_change_debug_.collision_check_objects);
    return safety_check_with_stuck_rss.is_safe;
  }

  return safety_check_with_normal_rss.is_safe;
}

std::optional<PathWithLaneId> NormalLaneChange::compute_terminal_lane_change_path() const
{
  const auto & transient_data = common_data_ptr_->transient_data;
  const auto dist_to_terminal_start = transient_data.dist_to_terminal_start;
  const auto min_lc_velocity = lane_change_parameters_->trajectory.min_lane_changing_velocity;
  const auto current_velocity = getEgoVelocity();

  PathWithLaneId prepare_segment;
  try {
    if (!utils::lane_change::get_prepare_segment(
          common_data_ptr_, prev_module_output_.path, dist_to_terminal_start, prepare_segment)) {
      RCLCPP_DEBUG(logger_, "failed to get valid prepare segment for terminal LC path");
      return std::nullopt;
    }
  } catch (const std::exception & e) {
    RCLCPP_DEBUG(logger_, "failed to get terminal LC path: %s", e.what());
    return std::nullopt;
  }

  // t = 2 * d / (v1 + v2)
  const auto duration_to_lc_start =
    2.0 * dist_to_terminal_start / (current_velocity + min_lc_velocity);
  const auto lon_accel = std::invoke([&]() -> double {
    if (duration_to_lc_start < calculation::eps) {
      return 0.0;
    }
    return std::clamp(
      (min_lc_velocity - current_velocity) / duration_to_lc_start,
      lane_change_parameters_->trajectory.min_longitudinal_acc,
      lane_change_parameters_->trajectory.max_longitudinal_acc);
  });
  const auto vel_on_prep = current_velocity + lon_accel * duration_to_lc_start;
  const LaneChangePhaseMetrics prep_metric(
    duration_to_lc_start, dist_to_terminal_start, vel_on_prep, lon_accel, lon_accel, 0.0);

  if (terminal_lane_change_path_) {
    terminal_lane_change_path_->info.set_prepare(prep_metric);
    if (prepare_segment.points.empty()) {
      terminal_lane_change_path_->path = terminal_lane_change_path_->shifted_path.path;
      return terminal_lane_change_path_->path;
    }
    prepare_segment.points.pop_back();
    terminal_lane_change_path_->path =
      utils::combinePath(prepare_segment, terminal_lane_change_path_->shifted_path.path);
    return terminal_lane_change_path_->path;
  }

  const auto & lane_changing_start_pose = prepare_segment.points.back().point.pose;
  const auto & target_lanes = common_data_ptr_->lanes_ptr->target;
  const auto shift_length =
    lanelet::utils::getLateralDistanceToClosestLanelet(target_lanes, lane_changing_start_pose);

  const auto dist_lc_start_to_end_of_lanes = calculation::calc_dist_from_pose_to_terminal_end(
    common_data_ptr_, common_data_ptr_->lanes_ptr->target_neighbor,
    prepare_segment.points.back().point.pose);

  const auto max_lane_changing_length = std::invoke([&]() {
    double max_length = transient_data.dist_to_terminal_end - prep_metric.length;
    return std::min(
      max_length, dist_lc_start_to_end_of_lanes - transient_data.next_dist_buffer.min);
  });

  const auto max_path_velocity = prepare_segment.points.back().point.longitudinal_velocity_mps;
  constexpr double lane_changing_lon_accel{0.0};
  const auto lane_changing_metrics = calculation::calc_shift_phase_metrics(
    common_data_ptr_, shift_length, prep_metric.velocity, max_path_velocity,
    lane_changing_lon_accel, max_lane_changing_length);

  const auto sorted_lane_ids = utils::lane_change::get_sorted_lane_ids(common_data_ptr_);

  LaneChangePath candidate_path;
  for (const auto & lc_metric : lane_changing_metrics) {
    try {
      candidate_path = utils::lane_change::get_candidate_path(
        common_data_ptr_, prep_metric, lc_metric, prepare_segment, sorted_lane_ids, shift_length);
    } catch (const std::exception & e) {
      continue;
    }
    terminal_lane_change_path_ = candidate_path;
    return candidate_path.path;
  }

  return std::nullopt;
}

namespace
{
// NERO_LC_HOLD_V=<m/s> (+ NERO_LC_HOLD_OVERHANG, default 0.6 m; NERO_LC_HOLD_SAFE_CYCLES, default 5): after approval,
// while ego is slow and essentially still in its own lane, an approved lane change that turns unsafe is HELD — ego
// stays stopped on the merge path — and resumes only after the strict (execution) safety check has passed for N
// consecutive cycles. Upstream re-checks with the looser abort/cancel set and resumes on the first safe cycle (its
// stop "is not working properly" by its own comment): lc_mixed:1 stopped at the line for a closing car, resumed one
// cycle later and crawled into the path of a 20 km/h car 25 m behind -> collision.
struct NeroHold
{
  double v{-1.0};
  double overhang{0.6};
  int safe_cycles{5};
};
const NeroHold & nero_hold()
{
  static const NeroHold h = [] {
    NeroHold r;
    if (const char * v = std::getenv("NERO_LC_HOLD_V")) r.v = std::atof(v);
    if (const char * v = std::getenv("NERO_LC_HOLD_OVERHANG")) r.overhang = std::atof(v);
    if (const char * v = std::getenv("NERO_LC_HOLD_SAFE_CYCLES")) r.safe_cycles = std::max(1, std::atoi(v));
    return r;
  }();
  return h;
}
struct NeroHoldState
{
  bool holding{false};
  int unsafe_n{0};
  int safe_n{0};
  std::chrono::steady_clock::time_point last{};
};
std::unordered_map<const void *, NeroHoldState> & nero_hold_states()
{
  static std::unordered_map<const void *, NeroHoldState> m;
  return m;
}
}  // namespace

double NormalLaneChange::nero_yield_cap() const
{
  // NERO_LC_YIELD_FRONT="dv,v_min,dir,rear": while waiting for approval, a moving car in the target lane AHEAD of ego
  // (within 20 m, or alongside) -> ease off to dv below its speed (not below v_min) so it pulls ahead and ego can slot in
  // behind it; rear=1: the same for a car BEHIND within 30 m that is not slower than ego (let it by). Uses the module's
  // own filtered target-lane objects (the per-object safety debug map is assembled across candidates: not a reliable
  // "what blocks me").
  static const std::array<double, 4> p = [] {
    std::array<double, 4> r{-1.0, 1.0, 2.0, 0.0};
    if (const char * e = std::getenv("NERO_LC_YIELD_FRONT")) {
      std::sscanf(e, "%lf,%lf,%lf,%lf", &r[0], &r[1], &r[2], &r[3]);
    }
    return r;
  }();
  const double dv = p[0], v_min = p[1];
  const int dir = static_cast<int>(p[2]);
  if (dv <= 0.0 || direction_ == Direction::NONE) return -1.0;
  // NERO_LC_MULTI_YIELD_FRONT=1 (routelc job 2026-10-26): with two or more changes due the easing off applies to both
  // directions (the car: dir 2 = right only). At 93795 (two left changes) a 2-3 m/s car alongside in the target lane
  // kept pace with the robot rolling at 9 km/h (NERO_LC_MULTI_WAIT_CRAWL) and no gap opened behind it (sim o37);
  // easing off lets it pull ahead, then the change goes behind it (the operator's way). Unset = the car.
  static const bool multi_yield = [] {
    const char * v = std::getenv("NERO_LC_MULTI_YIELD_FRONT");
    return v != nullptr && std::atoi(v) > 0;
  }();
  const bool dir_ok = !((dir == 1 && direction_ != Direction::LEFT) || (dir == 2 && direction_ != Direction::RIGHT));
  if (!dir_ok && !(multi_yield && nero_two_or_more_due())) return -1.0;
  if (nero_keep_right_discretionary()) return -1.0;  // NERO_LC_KEEP_RIGHT_GATE: a keep-right change never eases off
  if (status_.is_safe && status_.is_valid_path) return -1.0;  // a gap is there: no easing
  const double ego_v = getEgoVelocity();
  double cap = std::numeric_limits<double>::max();
  for (const auto & o : filtered_objects_.target_lane_leading.moving) {
    const double v_obj = std::hypot(o.initial_twist.linear.x, o.initial_twist.linear.y);
    if (v_obj < 0.5 || o.dist_from_ego > 20.0) continue;
    cap = std::min(cap, std::max(v_min, v_obj - dv));
  }
  if (p[3] > 0.5) {
    for (const auto & o : filtered_objects_.target_lane_trailing) {
      const double v_obj = std::hypot(o.initial_twist.linear.x, o.initial_twist.linear.y);
      if (v_obj < std::max(0.5, ego_v - 0.5) || std::abs(o.dist_from_ego) > 30.0) continue;
      cap = std::min(cap, std::max(v_min, v_obj - dv));
    }
  }
  return cap == std::numeric_limits<double>::max() ? -1.0 : cap;
}

bool NormalLaneChange::nero_in_hold_mode() const
{
  const auto & h = nero_hold();
  // NERO_LC_MULTI_HOLD_V=<m/s>[,<room m>] (lca job 2026-10-28): for the changes of a double change (two or more due, or the change
  // continuing one) the hold below applies up to this speed instead of NERO_LC_HOLD_V: until the robot's body leaves its
  // lane, an approved change goes on only while it would still be approved (the strict check). Sim S1 (10-27 #6): the
  // change was approved behind bb521a with the target lane empty; 0.3 s later bf651d turned into it 17 m behind at
  // 3.4 m/s and ran up to 6.8 m/s; the robot rolled at the 2.5 m/s two-change crawl, a hair above NERO_LC_HOLD_V 2.5,
  // so only the loose abort check ran and the change went on in front of it (bf651d had to brake; the car's stack
  // without the parked car the same, 0.01 m). Unset = the car.
  // ,<room m> (default 3): only while the robot can still stop and then start the change from standstill before its
  // terminal start (dist_to_terminal_start > v^2 / 2 m/s2 + room): a hold closer than that cancels the change for good
  // (sim o04: held 2.9 m before it, cancelled, no prepare room left -> stood 50 s nosed 0.5 m toward the line; S1 park_s20
  // the same at the terminal start); there the car's own check decides as before.
  static const std::pair<double, double> multi_hold = [] {
    double v = -1.0, room = 3.0;
    if (const char * e = std::getenv("NERO_LC_MULTI_HOLD_V")) {
      std::sscanf(e, "%lf,%lf", &v, &room);
    }
    return std::make_pair(v, room);
  }();
  const double ego_v = std::max(0.0, getEgoVelocity());
  const bool multi_hold_room =
    common_data_ptr_ &&
    common_data_ptr_->transient_data.dist_to_terminal_start > ego_v * ego_v / 2.0 + multi_hold.second;
  const double hold_v = (multi_hold.first > h.v && h.v > 0.0 && multi_hold_room &&
                         (nero_two_or_more_due() || nero_continues_change()))
                          ? multi_hold.first
                          : h.v;
  if (h.v <= 0.0 || getEgoVelocity() > hold_v) {
    return false;
  }
  // NERO_LC_HOLD_DIR=1 (session 5): hold only LEFT lane changes (the merge before the U-turn, which has no other lane
  // to fall back to). A change to the right (keep right after the U-turn, attempted right out of the junction with the
  // inner lane kept in the route) is cancelled back into the inner lane instead and tried again further on — held, it
  // stood at the end of 18952 for good while junction traffic streamed into the right lane
  static const int hold_dir = [] {
    const char * v = std::getenv("NERO_LC_HOLD_DIR");
    return v ? std::atoi(v) : 0;
  }();
  if (hold_dir == 1 && direction_ != Direction::LEFT) {
    return false;
  }
  return utils::isEgoWithinOriginalLane(
    common_data_ptr_->lanes_polygon_ptr->current, getEgoPose(), planner_data_->parameters, h.overhang);
}

PathSafetyStatus NormalLaneChange::isApprovedPathSafe() const
{
  const auto & path = status_.lane_change_path;
  const auto & current_lanes = get_current_lanes();
  const auto & target_lanes = get_target_lanes();

  if (current_lanes.empty() || target_lanes.empty()) {
    return {true, true};
  }

  const auto target_objects = get_target_objects(filtered_objects_, current_lanes);

  CollisionCheckDebugMap debug_data;

  const auto has_overtaking_object = utils::lane_change::has_overtaking_turn_lane_object(
    common_data_ptr_, filtered_objects_.target_lane_trailing);

  if (has_overtaking_object) {
    return {false, true};
  }

  // NERO_LC_APPROVED_NO_DELAY=1 (session 4): once a lane change is approved, "delay lane change" (a stopped car in the
  // target lane ahead) is not a reason to call the approved path unsafe — it is a choice made before approval, not a
  // collision risk; with it, the hold (NERO_LC_HOLD_V) waited for a parked car to move and held ego for good.
  static const bool nero_approved_no_delay = [] {
    const char * e = std::getenv("NERO_LC_APPROVED_NO_DELAY");
    return e && std::atoi(e) != 0;
  }();
  if (
    !nero_approved_no_delay &&
    utils::lane_change::is_delay_lane_change(
      common_data_ptr_, path, filtered_objects_.target_lane_leading.stopped, debug_data)) {
    RCLCPP_DEBUG(logger_, "Lane change has been delayed.");
    return {false, false};
  }

  const auto decel_sampling_num =
    static_cast<size_t>(lane_change_parameters_->cancel.deceleration_sampling_num);
  const auto ego_predicted_paths =
    utils::lane_change::convert_to_predicted_paths(common_data_ptr_, path, decel_sampling_num);
  const auto safety_status = isLaneChangePathSafe(
    path, ego_predicted_paths, target_objects,
    nero_in_hold_mode() ? lane_change_parameters_->safety.rss_params  // strict while slow and in lane (NERO hold)
                        : lane_change_parameters_->safety.rss_params_for_abort,
    debug_data);
  {
    // only for debug purpose
    lane_change_debug_.collision_check_objects.clear();
    lane_change_debug_.collision_check_object_debug_lifetime +=
      (stop_watch_.toc(getModuleTypeStr()) / 1000);
    if (lane_change_debug_.collision_check_object_debug_lifetime > 2.0) {
      stop_watch_.toc(getModuleTypeStr(), true);
      lane_change_debug_.collision_check_object_debug_lifetime = 0.0;
      lane_change_debug_.collision_check_objects_after_approval.clear();
    }

    if (!safety_status.is_safe) {
      lane_change_debug_.collision_check_objects_after_approval = debug_data;
    }
  }

  return safety_status;
}

PathSafetyStatus NormalLaneChange::evaluateApprovedPathWithUnsafeHysteresis(
  PathSafetyStatus approved_path_safety_status)
{
  if (nero_hold().v > 0.0) {
    auto & st = nero_hold_states()[this];
    const auto now = std::chrono::steady_clock::now();
    if (now - st.last > std::chrono::milliseconds(500)) {
      st = NeroHoldState{};  // a new module instance / a pause
    }
    st.last = now;
    if (st.holding || nero_in_hold_mode()) {
      if (!approved_path_safety_status.is_safe) {
        st.safe_n = 0;
        if (++st.unsafe_n >= 2) {
          st.holding = true;
        }
      } else {
        st.unsafe_n = 0;
        if (st.holding && ++st.safe_n >= nero_hold().safe_cycles) {
          st.holding = false;
          st.safe_n = 0;
        }
      }
      if (st.holding) {
        if (nero_lc_debug_tick("hold")) {
          RCLCPP_INFO(logger_, "[NERO_LC] holding the approved lane change (strict check, safe %d/%d)",
            st.safe_n, nero_hold().safe_cycles);
        }
        PathSafetyStatus unsafe;
        unsafe.is_safe = false;
        unsafe.is_trailing_object = approved_path_safety_status.is_safe ? true
                                                                       : approved_path_safety_status.is_trailing_object;
        return unsafe;
      }
      if (!approved_path_safety_status.is_safe) {
        return {};  // first unsafe cycle: not yet
      }
      return approved_path_safety_status;
    }
  }
  if (!approved_path_safety_status.is_safe) {
    ++unsafe_hysteresis_count_;
    RCLCPP_DEBUG(
      logger_, "%s: Increasing hysteresis count to %d.", __func__, unsafe_hysteresis_count_);
  } else {
    if (unsafe_hysteresis_count_ > 0) {
      RCLCPP_DEBUG(logger_, "%s: Lane change is now SAFE. Resetting hysteresis count.", __func__);
    }
    unsafe_hysteresis_count_ = 0;
  }
  if (unsafe_hysteresis_count_ > lane_change_parameters_->cancel.th_unsafe_hysteresis) {
    RCLCPP_DEBUG(
      logger_, "%s: hysteresis count exceed threshold. lane change is now %s", __func__,
      (approved_path_safety_status.is_safe ? "safe" : "UNSAFE"));
    return approved_path_safety_status;
  }
  return {};
}

bool NormalLaneChange::isValidPath(const PathWithLaneId & path) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  const auto & route_handler = planner_data_->route_handler;
  const auto & dp = planner_data_->drivable_area_expansion_parameters;

  // check lane departure
  const auto drivable_lanes = utils::lane_change::generateDrivableLanes(
    *route_handler, utils::extendLanes(route_handler, get_current_lanes()),
    utils::extendLanes(route_handler, get_target_lanes()));
  const auto expanded_lanes = utils::expandLanelets(
    drivable_lanes, dp.drivable_area_left_bound_offset, dp.drivable_area_right_bound_offset,
    dp.drivable_area_types_to_skip);

  const auto lanelets = utils::transformToLanelets(expanded_lanes);

  // check path points are in any lanelets
  for (const auto & point : path.points) {
    bool is_in_lanelet = false;
    for (const auto & lanelet : lanelets) {
      if (lanelet::utils::isInLanelet(point.point.pose, lanelet)) {
        is_in_lanelet = true;
        break;
      }
    }
    if (!is_in_lanelet) {
      return false;
    }
  }

  // check relative angle
  return utils::checkPathRelativeAngle(path, M_PI);
}

bool NormalLaneChange::isRequiredStop(const bool is_trailing_object)
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  if (nero_hold().v > 0.0) {
    const auto it = nero_hold_states().find(this);
    if (it != nero_hold_states().end() && it->second.holding) {
      return true;  // NERO hold: stop on the merge path, whatever the zero-tolerance in-lane test says
    }
  }
  return common_data_ptr_->transient_data.is_ego_near_current_terminal_start &&
         isAbleToStopSafely() && is_trailing_object;
}

bool NormalLaneChange::calcAbortPath()
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  const auto & route_handler = getRouteHandler();
  const auto & common_param = getCommonParam();
  const auto current_velocity =
    std::max(lane_change_parameters_->trajectory.min_lane_changing_velocity, getEgoVelocity());
  const auto current_pose = getEgoPose();
  const auto & selected_path = status_.lane_change_path;

  const auto ego_nearest_dist_threshold = common_param.ego_nearest_dist_threshold;
  const auto ego_nearest_yaw_threshold = common_param.ego_nearest_yaw_threshold;

  const auto current_min_dist_buffer = common_data_ptr_->transient_data.current_dist_buffer.min;

  const auto & lane_changing_path = selected_path.path;
  const auto & reference_lanelets = get_current_lanes();
  const auto lane_changing_end_pose_idx = std::invoke([&]() {
    constexpr double s_start = 0.0;
    const double s_end = std::max(
      lanelet::utils::getLaneletLength2d(reference_lanelets) - current_min_dist_buffer, 0.0);

    const auto ref = route_handler->getCenterLinePath(reference_lanelets, s_start, s_end);
    return autoware::motion_utils::findFirstNearestIndexWithSoftConstraints(
      lane_changing_path.points, ref.points.back().point.pose, ego_nearest_dist_threshold,
      ego_nearest_yaw_threshold);
  });

  const auto ego_pose_idx = autoware::motion_utils::findFirstNearestIndexWithSoftConstraints(
    lane_changing_path.points, current_pose, ego_nearest_dist_threshold, ego_nearest_yaw_threshold);

  const auto get_abort_idx_and_distance = [&](const double param_time) {
    if (ego_pose_idx > lane_changing_end_pose_idx) {
      return std::make_pair(ego_pose_idx, 0.0);
    }

    const auto desired_distance = current_velocity * param_time;
    const auto & points = lane_changing_path.points;

    for (size_t idx = ego_pose_idx; idx < lane_changing_end_pose_idx; ++idx) {
      const double distance =
        utils::getSignedDistance(current_pose, points.at(idx).point.pose, reference_lanelets);
      if (distance > desired_distance) {
        return std::make_pair(idx, distance);
      }
    }

    return std::make_pair(ego_pose_idx, 0.0);
  };

  const auto [abort_start_idx, abort_start_dist] =
    get_abort_idx_and_distance(lane_change_parameters_->cancel.delta_time);
  const auto [abort_return_idx, abort_return_dist] = get_abort_idx_and_distance(
    lane_change_parameters_->cancel.delta_time + lane_change_parameters_->cancel.duration);

  if (abort_start_idx >= abort_return_idx) {
    RCLCPP_ERROR(logger_, "abort start idx and return idx is equal. can't compute abort path.");
    return false;
  }

  const auto enough_abort_dist =
    abort_start_dist + abort_return_dist +
      calculation::calc_stopping_distance(common_data_ptr_->lc_param_ptr) <=
    common_data_ptr_->transient_data.dist_to_terminal_start;

  if (!enough_abort_dist) {
    RCLCPP_ERROR(logger_, "insufficient distance to abort.");
    return false;
  }

  const auto abort_start_pose = lane_changing_path.points.at(abort_start_idx).point.pose;
  const auto abort_return_pose = lane_changing_path.points.at(abort_return_idx).point.pose;
  const auto shift_length =
    lanelet::utils::getArcCoordinates(reference_lanelets, abort_return_pose).distance;

  ShiftLine shift_line;
  shift_line.start = abort_start_pose;
  shift_line.end = abort_return_pose;
  shift_line.end_shift_length = -shift_length;
  shift_line.start_idx = abort_start_idx;
  shift_line.end_idx = abort_return_idx;

  PathShifter path_shifter;
  path_shifter.setPath(lane_changing_path);
  path_shifter.addShiftLine(shift_line);
  const auto lateral_jerk = autoware::motion_utils::calc_jerk_from_lat_lon_distance(
    shift_line.end_shift_length, abort_start_dist, current_velocity);
  path_shifter.setVelocity(current_velocity);
  const auto lateral_acc_range =
    lane_change_parameters_->trajectory.lat_acc_map.find(current_velocity);
  const double & max_lateral_acc = lateral_acc_range.second;
  path_shifter.setLateralAccelerationLimit(max_lateral_acc);

  if (lateral_jerk > lane_change_parameters_->cancel.max_lateral_jerk) {
    RCLCPP_ERROR(logger_, "Aborting jerk is too strong. lateral_jerk = %f", lateral_jerk);
    return false;
  }

  ShiftedPath shifted_path;
  // offset front side
  if (!path_shifter.generate(&shifted_path)) {
    RCLCPP_ERROR(logger_, "failed to generate abort shifted path.");
  }

  auto reference_lane_segment = prev_module_output_.path;
  {
    // const auto terminal_path =
    //   calcTerminalLaneChangePath(reference_lanelets, get_target_lanes());
    // if (terminal_path) {
    //   reference_lane_segment = terminal_path->path;
    // }
    const auto return_pose = shifted_path.path.points.at(abort_return_idx).point.pose;
    const auto seg_idx = autoware::motion_utils::findFirstNearestSegmentIndexWithSoftConstraints(
      reference_lane_segment.points, return_pose, common_param.ego_nearest_dist_threshold,
      common_param.ego_nearest_yaw_threshold);
    reference_lane_segment.points = autoware::motion_utils::cropPoints(
      reference_lane_segment.points, return_pose.position, seg_idx,
      common_param.forward_path_length, 0.0);
  }

  auto abort_path = selected_path;
  abort_path.shifted_path = shifted_path;
  abort_path.info.shift_line = shift_line;

  {
    PathWithLaneId aborting_path;
    aborting_path.points.insert(
      aborting_path.points.begin(), shifted_path.path.points.begin(),
      shifted_path.path.points.begin() + static_cast<int>(abort_return_idx));

    if (!reference_lane_segment.points.empty()) {
      abort_path.path = utils::combinePath(aborting_path, reference_lane_segment);
    } else {
      abort_path.path = aborting_path;
    }
  }

  abort_path_ = std::make_shared<LaneChangePath>(abort_path);
  return true;
}

PathSafetyStatus NormalLaneChange::isLaneChangePathSafe(
  const LaneChangePath & lane_change_path,
  const std::vector<std::vector<PoseWithVelocityStamped>> & ego_predicted_paths,
  const lane_change::TargetObjects & collision_check_objects,
  const utils::path_safety_checker::RSSparams & rss_params,
  CollisionCheckDebugMap & debug_data) const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  constexpr auto is_safe = true;
  constexpr auto is_object_behind_ego = true;

  const auto is_check_prepare_phase = check_prepare_phase();

  const auto all_paths_collide = [&](const auto & objects) {
    const auto stopped_obj_vel_th = lane_change_parameters_->safety.th_stopped_object_velocity;
    return ranges::all_of(ego_predicted_paths, [&](const auto & ego_predicted_path) {
      const auto check_for_collision = [&](const auto & object) {
        const auto selected_rss_param = (object.initial_twist.linear.x <= stopped_obj_vel_th)
                                          ? lane_change_parameters_->safety.rss_params_for_parked
                                          : rss_params;
        return is_colliding(
          lane_change_path, object, ego_predicted_path, selected_rss_param, is_check_prepare_phase,
          debug_data);
      };
      return ranges::any_of(objects, check_for_collision);
    });
  };

  if (all_paths_collide(collision_check_objects.trailing)) {
    return {!is_safe, is_object_behind_ego};
  }

  if (all_paths_collide(collision_check_objects.leading)) {
    return {!is_safe, !is_object_behind_ego};
  }

  return {is_safe, !is_object_behind_ego};
}

bool NormalLaneChange::is_colliding(
  const LaneChangePath & lane_change_path, const ExtendedPredictedObject & obj,
  const std::vector<PoseWithVelocityStamped> & ego_predicted_path,
  const RSSparams & selected_rss_param, const bool check_prepare_phase,
  CollisionCheckDebugMap & debug_data) const
{
  constexpr auto is_colliding{true};

  if (lane_change_path.path.points.empty()) {
    return !is_colliding;
  }

  if (ego_predicted_path.empty()) {
    return !is_colliding;
  }

  const auto & lanes_polygon_ptr = common_data_ptr_->lanes_polygon_ptr;
  const auto & current_polygon = lanes_polygon_ptr->current;
  const auto & expanded_target_polygon = lanes_polygon_ptr->target;

  if (current_polygon.empty() || expanded_target_polygon.empty()) {
    return !is_colliding;
  }

  constexpr auto is_safe{true};
  auto current_debug_data = utils::path_safety_checker::createObjectDebug(obj);
  constexpr auto hysteresis_factor{1.0};
  const auto obj_predicted_paths = utils::path_safety_checker::getPredictedPathFromObj(
    obj, lane_change_parameters_->safety.collision_check.use_all_predicted_paths);
  const auto safety_check_max_vel = get_max_velocity_for_safety_check();
  const auto & bpp_param = *common_data_ptr_->bpp_param_ptr;

  const double velocity_threshold = lane_change_parameters_->safety.th_stopped_object_velocity;
  const double prepare_duration = lane_change_path.info.duration.prepare;
  const double start_time = check_prepare_phase ? 0.0 : prepare_duration;

  // NERO_LC_MULTI_STAND_PREPARE=1 (streamjoin job 2026-10-27): a change that starts from the robot standing at its waiting
  // stop with two or more changes due is checked as upstream #11307 checks every change: the predicted poses while ego is
  // still in its prepare phase (in its own lane, before the lane-changing start) with #11307's prepare-phase RSS (lateral
  // 0.5 m, longitudinal 1.0 m), the rest with the car's lane-changing RSS as before (the follower in the target lane
  // checked unchanged). The car checks the prepare phase with the lane-changing RSS (lateral 2.0 m, 3 m + 1.0 s x speed):
  // a car passing in the target lane one lane over blocked the robot from rolling in its own lane until that car was
  // ~10 m ahead plus 5 safe cycles, then the start latency (sim c28: first motion 4.4-4.9 s after the second stream car,
  // the operator 0.8 s: 10-26e #28 / #29 "s"). The prepare phase is split where ego's predicted path reaches the
  // lane-changing start (from a stand = the prepare duration; for the approved path the remaining part). In the
  // lane-changing phase a car AHEAD of ego is checked with ego as far along as a launch at the planner's normal maximum
  // acceleration takes it (the prediction's NERO_LC_PRED_LAUNCH_ACC 0.33 m/s2 launch is only conservative for cars
  // behind). Unset = the car.
  static const bool nero_stand_prepare = [] {
    const char * v = std::getenv("NERO_LC_MULTI_STAND_PREPARE");
    return v != nullptr && std::atoi(v) > 0;
  }();
  const double t_light = (nero_stand_prepare && nero_stand_start())
                           ? nero_time_to_lc_start(lane_change_path, ego_predicted_path, getEgoPosition())
                           : -1.0;
  const auto lead_shift = [&](const auto & entry) {
    // how much further ego is at entry.time with a launch at bpp max_acc (capped at the path's change speed) than in the
    // prediction, if the object is ahead of the predicted ego then; 0 otherwise
    auto out = entry;
    if (ego_predicted_path.empty()) return out;
    const auto it = std::min_element(
      ego_predicted_path.begin(), ego_predicted_path.end(),
      [&](const auto & a, const auto & b) {
        return std::abs(a.time - entry.time) < std::abs(b.time - entry.time);
      });
    if (autoware_utils::calc_longitudinal_deviation(it->pose, entry.pose.position) <= 0.0) return out;
    double s_pred = 0.0;
    for (auto jt = ego_predicted_path.begin() + 1; jt <= it; ++jt) {
      s_pred += autoware_utils::calc_distance2d((jt - 1)->pose, jt->pose);
    }
    const double v0 = std::max(0.0, common_data_ptr_->get_ego_speed());
    const double a_f = std::max(0.1, common_data_ptr_->bpp_param_ptr->max_acc);
    const double v_cap = std::max(
      {v0, lane_change_path.info.velocity.prepare, lane_change_path.info.velocity.lane_changing,
       lane_change_path.info.terminal_lane_changing_velocity});
    const double t = it->time;
    const double t_acc = std::clamp((v_cap - v0) / a_f, 0.0, t);
    const double s_fast = v0 * t_acc + 0.5 * a_f * t_acc * t_acc + v_cap * (t - t_acc);
    const double d = std::max(0.0, s_fast - s_pred);
    if (d < 1e-3) return out;
    const double yaw = tf2::getYaw(entry.pose.orientation);
    const double dx = -d * std::cos(yaw);
    const double dy = -d * std::sin(yaw);
    out.pose.position.x += dx;
    out.pose.position.y += dy;
    for (auto & p : out.poly.outer()) {
      p.x() += dx;
      p.y() += dy;
    }
    return out;
  };

  // NERO_LC_MULTI_FRONT_HEADWAY=1 (streamjoin job 2026-10-27, with NERO_LC_MULTI_STAND_PREPARE's split): in the
  // lane-changing phase of the change made from the waiting place, a target-lane car fully AHEAD of ego (every corner
  // past ego's nose) and faster than ego gets the velocity headway at ego's speed: the minimum gap is the car's
  // longitudinal_distance_min_threshold + longitudinal_velocity_delta_time x ego's speed instead of x the faster speed
  // (calc_minimum_longitudinal_length takes max(front, rear)). The RSS term for that car braking (expected front
  // deceleration, ego's reaction + margin) is unchanged, so is every car beside or behind ego (the follower). Sim c28:
  // d9d99b pulling away at 5.8-7 m/s held change 1 until it was 3 m + 5.8 m ahead in every predicted step (approval 3 s
  // after it passed the nose); the operator steered 12 m centre to centre behind it (10-26e #28 / #29). Ego's speed is the
  // larger of the prediction and the launch the object shift (lead_shift) assumes. Unset = NERO_LC_MULTI_STAND_PREPARE.
  static const bool nero_front_headway = [] {
    const char * v = std::getenv("NERO_LC_MULTI_FRONT_HEADWAY");
    return v != nullptr && std::atoi(v) > 0;
  }();
  // the velocity-headway time for an entry fully ahead of ego and faster than it; < 0: not such an entry
  const auto front_headway_dt = [&](const auto & entry) -> double {
    if (!nero_front_headway || ego_predicted_path.empty()) return -1.0;
    const auto it = std::min_element(
      ego_predicted_path.begin(), ego_predicted_path.end(),
      [&](const auto & a, const auto & b) {
        return std::abs(a.time - entry.time) < std::abs(b.time - entry.time);
      });
    const auto nose = autoware_utils::calc_offset_pose(
      it->pose, bpp_param.vehicle_info.max_longitudinal_offset_m, 0.0, 0.0);
    for (const auto & p : entry.poly.outer()) {
      if (autoware_utils::calc_longitudinal_deviation(nose, autoware_utils::create_point(p.x(), p.y(), 0.0)) <= 0.0) {
        return -1.0;
      }
    }
    const double v0 = std::max(0.0, common_data_ptr_->get_ego_speed());
    const double a_f = std::max(0.1, common_data_ptr_->bpp_param_ptr->max_acc);
    const double v_cap = std::max(
      {v0, lane_change_path.info.velocity.prepare, lane_change_path.info.velocity.lane_changing,
       lane_change_path.info.terminal_lane_changing_velocity});
    const double v_ego = std::max(static_cast<double>(it->velocity), std::min(v_cap, v0 + a_f * it->time));
    const double v_obj = entry.velocity;
    if (v_obj <= v_ego + 0.1) return -1.0;
    return selected_rss_param.longitudinal_velocity_delta_time * v_ego / v_obj;
  };

  for (const auto & obj_path : obj_predicted_paths) {
    if (t_light > 0.0) {
      utils::path_safety_checker::PredictedPathWithPolygon prepare_part;
      utils::path_safety_checker::PredictedPathWithPolygon changing_part;
      prepare_part.confidence = changing_part.confidence = obj_path.confidence;
      std::vector<std::pair<utils::path_safety_checker::PoseWithVelocityAndPolygonStamped, double>> front_fast;
      for (const auto & entry : obj_path.path) {
        if (entry.time < t_light) {
          prepare_part.path.push_back(entry);
        } else {
          const auto shifted = lead_shift(entry);
          const double dt = front_headway_dt(shifted);
          if (dt >= 0.0) {
            front_fast.emplace_back(shifted, dt);
          } else {
            changing_part.path.push_back(shifted);
          }
        }
      }
      auto collided_polygons = utils::path_safety_checker::get_collided_polygons(
        lane_change_path.path, ego_predicted_path, obj, prepare_part, bpp_param.vehicle_info,
        nero_prepare_rss(), hysteresis_factor, safety_check_max_vel,
        common_data_ptr_->lc_param_ptr->safety.collision_check.th_yaw_diff, current_debug_data.second);
      const auto collided_changing = utils::path_safety_checker::get_collided_polygons(
        lane_change_path.path, ego_predicted_path, obj, changing_part, bpp_param.vehicle_info,
        selected_rss_param, hysteresis_factor, safety_check_max_vel,
        common_data_ptr_->lc_param_ptr->safety.collision_check.th_yaw_diff, current_debug_data.second);
      collided_polygons.insert(collided_polygons.end(), collided_changing.begin(), collided_changing.end());
      for (const auto & [entry, dt] : front_fast) {
        auto rss = selected_rss_param;
        rss.longitudinal_velocity_delta_time = dt;
        utils::path_safety_checker::PredictedPathWithPolygon one;
        one.confidence = obj_path.confidence;
        one.path.push_back(entry);
        const auto collided_front = utils::path_safety_checker::get_collided_polygons(
          lane_change_path.path, ego_predicted_path, obj, one, bpp_param.vehicle_info, rss, hysteresis_factor,
          safety_check_max_vel, common_data_ptr_->lc_param_ptr->safety.collision_check.th_yaw_diff,
          current_debug_data.second);
        collided_polygons.insert(collided_polygons.end(), collided_front.begin(), collided_front.end());
      }
      if (
        collided_polygons.empty() ||
        (!utils::lane_change::is_collided_polygons_in_lanelet(collided_polygons, current_polygon) &&
         !utils::lane_change::is_collided_polygons_in_lanelet(
           collided_polygons, expanded_target_polygon))) {
        utils::path_safety_checker::updateCollisionCheckDebugMap(debug_data, current_debug_data, is_safe);
        continue;
      }
      utils::path_safety_checker::updateCollisionCheckDebugMap(debug_data, current_debug_data, !is_safe);
      return is_colliding;
    }


    utils::path_safety_checker::PredictedPathWithPolygon predicted_obj_path;
    predicted_obj_path.confidence = obj_path.confidence;
    std::copy_if(
      obj_path.path.begin(), obj_path.path.end(), std::back_inserter(predicted_obj_path.path),
      [&](const auto & entry) {
        return !(
          entry.time < start_time ||
          (entry.time < prepare_duration && entry.velocity < velocity_threshold));
      });

    const auto collided_polygons = utils::path_safety_checker::get_collided_polygons(
      lane_change_path.path, ego_predicted_path, obj, predicted_obj_path, bpp_param.vehicle_info,
      selected_rss_param, hysteresis_factor, safety_check_max_vel,
      common_data_ptr_->lc_param_ptr->safety.collision_check.th_yaw_diff,
      current_debug_data.second);

    if (collided_polygons.empty()) {
      utils::path_safety_checker::updateCollisionCheckDebugMap(
        debug_data, current_debug_data, is_safe);
      continue;
    }

    const auto collision_in_current_lanes =
      utils::lane_change::is_collided_polygons_in_lanelet(collided_polygons, current_polygon);
    const auto collision_in_target_lanes = utils::lane_change::is_collided_polygons_in_lanelet(
      collided_polygons, expanded_target_polygon);

    if (!collision_in_current_lanes && !collision_in_target_lanes) {
      utils::path_safety_checker::updateCollisionCheckDebugMap(
        debug_data, current_debug_data, is_safe);
      continue;
    }

    utils::path_safety_checker::updateCollisionCheckDebugMap(
      debug_data, current_debug_data, !is_safe);
    return is_colliding;
  }
  utils::path_safety_checker::updateCollisionCheckDebugMap(debug_data, current_debug_data, is_safe);
  return !is_colliding;
}

double NormalLaneChange::get_max_velocity_for_safety_check() const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  const auto external_velocity_limit_ptr = planner_data_->external_limit_max_velocity;
  if (external_velocity_limit_ptr) {
    return std::min(
      static_cast<double>(external_velocity_limit_ptr->max_velocity), getCommonParam().max_vel);
  }

  return getCommonParam().max_vel;
}

bool NormalLaneChange::is_ego_stuck() const
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  const auto & lc_param_ptr = common_data_ptr_->lc_param_ptr;

  if (std::abs(common_data_ptr_->get_ego_speed()) > lc_param_ptr->th_stop_velocity) {
    RCLCPP_DEBUG(logger_, "Ego is still moving, not in stuck");
    return false;
  }

  // Ego is just stopped, not sure it is in stuck yet.
  if (getStopTime() < lc_param_ptr->th_stop_time) {
    RCLCPP_DEBUG(logger_, "Ego is just stopped, counting for stuck judge... (%f)", getStopTime());
    return false;
  }

  // Check if any stationary object exist in obstacle_check_distance
  const auto & current_lanes_path = common_data_ptr_->current_lanes_path;
  const auto & ego_pose = common_data_ptr_->get_ego_pose();
  const auto rss_dist = calcRssDistance(
    0.0, lc_param_ptr->trajectory.min_lane_changing_velocity, lc_param_ptr->safety.rss_params);

  // It is difficult to define the detection range. If it is too short, the stuck will not be
  // determined, even though you are stuck by an obstacle. If it is too long,
  // the ego will be judged to be stuck by a distant vehicle, even though the ego is only
  // stopped at a traffic light. Essentially, the calculation should be based on the information of
  // the stop reason, but this is outside the scope of one module. I keep it as a TODO.
  constexpr auto detection_distance_margin = 10.0;
  const auto obstacle_check_distance = common_data_ptr_->transient_data.lane_changing_length.max +
                                       rss_dist + common_data_ptr_->bpp_param_ptr->base_link2front +
                                       detection_distance_margin;
  const auto has_object_blocking = std::any_of(
    filtered_objects_.current_lane.begin(), filtered_objects_.current_lane.end(),
    [&](const auto & object) {
      // Note: it needs chattering prevention.
      if (
        std::abs(object.initial_twist.linear.x) >
        lc_param_ptr->safety.th_stopped_object_velocity) {  // check if stationary
        return false;
      }

      const auto ego_to_obj_dist =
        calcSignedArcLength(
          current_lanes_path.points, ego_pose.position, object.initial_pose.position) -
        obstacle_check_distance;
      return ego_to_obj_dist < 0.0;
    });

  lane_change_debug_.is_stuck = has_object_blocking;
  return has_object_blocking;
}

// NERO_LC_MULTI_STOP_OFFSET=<m> (routelc job 2026-10-26): with two or more changes still due (the distance buffer is
// more than one change), the waiting stop sits <m> before the latest start of the first change instead of
// backward_length_buffer_for_end_of_lane (10 m on the car, chosen for the single change before the U-turn: an approval
// near that lane end cannot be cancelled). With two changes due the whole two-change gate (~32 m) is still ahead after
// the first one starts there. Field / sim: 93643 stood 43 m before the end (10-26e #28), the gap behind the stream came
// ~1 s later; 93795 stood 47 / 42 m before the end (#4, #37). Single changes keep the 10 m. Unset = the car.
double NormalLaneChange::nero_stop_offset() const
{
  const double stock = calculation::calc_stopping_distance(lane_change_parameters_);
  static const double multi = [] {
    const char * v = std::getenv("NERO_LC_MULTI_STOP_OFFSET");
    return v != nullptr ? std::atof(v) : -1.0;
  }();
  // NERO_LC_MULTI_STAND_OFFSET=<m> (streamjoin job 2026-10-27): with two or more changes due, a direction without the
  // rolling wait (NERO_LC_WAIT_DECEL / NERO_LC_SAUDI_DIR: on the car the right changes) stops and starts both changes from
  // a stand at its waiting stop; that stop sits <m> before the first change's latest start instead of
  // NERO_LC_MULTI_STOP_OFFSET. From a stand 2 m before the latest start the first change can only be planned at its
  // minimum length (<= 2 m prepare) and ends at the second change's waiting stop (the 10 m end-of-lane buffer between
  // the changes): the second change then always starts from a second stand (sim c28, 93643: 41986 s 40; field 10-26e #57).
  // With <m> = 10 the stand is where the operator stood (10-26e #28 / #29: 93643 s 19.2, 42.7 m before the end) and the
  // second change comes straight out of the first. The left changes' rolling waits (93795, 93699) keep the 2 m. Unset =
  // NERO_LC_MULTI_STOP_OFFSET for every direction.
  static const double stand = [] {
    const char * v = std::getenv("NERO_LC_MULTI_STAND_OFFSET");
    return v != nullptr ? std::atof(v) : -1.0;
  }();
  if ((multi < 0.0 && stand < 0.0) || !common_data_ptr_ || !nero_two_or_more_due()) {
    return stock;
  }
  // (streamjoin2: only while the change waits for a gap in a passing stream: nero_gap_wait)
  if (stand >= 0.0 && !nero_rolling_wait_dir() && nero_gap_wait()) {
    // NERO_LC_MULTI_ROLL: while a gap is being confirmed (NERO_LC_MIN_SAFE_CYCLES: safe, not yet approved) the robot
    // rolling into its waiting place keeps rolling: the stop moves up to NERO_LC_MULTI_STOP_OFFSET before the latest
    // start (sim c28: braked from 7 to 3.6 km/h during the 0.8 s confirmation, the change then started 2 s later). If
    // the confirmation fails the stop goes back (a stop behind / just ahead of the moving robot: NERO_LC_STOP_COMFORT).
    static const bool roll = [] {
      const char * v = std::getenv("NERO_LC_MULTI_ROLL");
      return v != nullptr && std::atoi(v) > 0;
    }();
    if (roll && multi >= 0.0 && nero_stand_start()) {
      const auto & sc = nero_safe_counts();
      const auto it = sc.find(this);
      if (it != sc.end() && it->second.n >= 1 &&
          std::chrono::steady_clock::now() - it->second.last < std::chrono::milliseconds(500)) {
        return std::min(stock, multi);
      }
    }
    return std::min(stock, stand);
  }
  return multi >= 0.0 ? std::min(stock, multi) : stock;
}

// the car's rolling wait (interface.cpp nero_cap_waiting_velocity) acts for this direction
bool NormalLaneChange::nero_rolling_wait_dir() const
{
  static const int saudi_dir = [] {
    const char * v = std::getenv("NERO_LC_SAUDI_DIR");
    return v != nullptr ? std::atoi(v) : 0;
  }();
  static const double decel = [] {
    const char * v = std::getenv("NERO_LC_WAIT_DECEL");
    return v != nullptr ? std::atof(v) : 0.0;
  }();
  if (decel <= 0.0) {
    return false;
  }
  return saudi_dir == 0 || (saudi_dir == 1 && direction_ == Direction::LEFT) ||
         (saudi_dir == 2 && direction_ == Direction::RIGHT);
}

void NormalLaneChange::nero_note_finished() const
{
  // only a change made from the waiting place with two or more changes due is continued by the next one
  if (nero_stand_start()) {
    nero_finish_times()[static_cast<int>(direction_)] = NeroFinish{std::chrono::steady_clock::now(), this};
    if (nero_lc_debug_every("finish_note", 500)) {
      RCLCPP_INFO(logger_, "[NERO_LC_SIGNALLED_PREPARE] change from the waiting place finished (%s)",
        direction_ == Direction::LEFT ? "left" : "right");
    }
  }
}

bool NormalLaneChange::nero_continues_change() const
{
  const auto & m = nero_continuation_map();
  const auto it = m.find(this);
  return it != m.end() && it->second.continues;
}

double NormalLaneChange::nero_roll_arrival_cap(const double dist_to_stop) const
{
  // NERO_LC_MULTI_ROLL (streamjoin job 2026-10-27): while waiting with two or more changes due in a direction without the
  // car's rolling wait, arrive at the waiting stop when the passing target-lane cars (>= 0.5 m/s faster than ego, within
  // 40 m behind / alongside / just ahead) are the change's own front gap ahead of ego's nose there: the car's
  // lane-changing RSS minimum, longitudinal_distance_min_threshold + longitudinal_velocity_delta_time x its speed (3 m +
  // 1.0 s on the car = the operator's ~12 m centre to centre behind d9d99b at steering, 10-26e #28 / #29). The robot then
  // rolls in behind the last of them instead of braking to a stand at its stop (Ahmed 10-27: "without stops"). Every car
  // stays in the safety check; a car following too close behind still makes the robot wait at its stop.
  if (!nero_two_or_more_due() || nero_rolling_wait_dir() || dist_to_stop <= 0.5) {
    return -1.0;
  }
  // only while the change waits for its gap (no safe path, no safe cycle being counted): an approaching robot whose
  // change is already safe is never slowed (sim c56: capped from 15 to 5 km/h for cars the change was safe with, its
  // follower came within 0.28 m)
  {
    const auto & sc = nero_safe_counts();
    const auto it = sc.find(this);
    const bool counting = it != sc.end() && it->second.n >= 1 &&
                          std::chrono::steady_clock::now() - it->second.last < std::chrono::milliseconds(500);
    if (status_.is_safe || counting) {
      return -1.0;
    }
  }
  const double t_need = nero_passing_need(dist_to_stop);
  if (t_need <= 0.0) {
    return -1.0;
  }
  return dist_to_stop / t_need;
}

double NormalLaneChange::nero_passing_need(const double dist_to_stop) const
{
  const auto & rss = lane_change_parameters_->safety.rss_params;
  // a car "passes" when it is faster than the robot's crawl (not its current speed: a slowing robot would count more
  // and more cars as passing)
  static const double crawl = [] {
    const char * c = std::getenv("NERO_LC_MULTI_WAIT_CRAWL");
    return c != nullptr ? std::atof(c) : 0.0;
  }();
  const double v_ego = std::max({0.0, getEgoVelocity(), crawl});
  const double front = common_data_ptr_->bpp_param_ptr->base_link2front;
  double t_need = 0.0;
  const auto consider = [&](const auto & objs) {
    for (const auto & o : objs) {
      const double v = o.initial_twist.linear.x;
      if (v < v_ego + 0.5 || v < 1.0 || o.dist_from_ego < -40.0) continue;
      const double need = dist_to_stop + front + rss.longitudinal_distance_min_threshold +
                          rss.longitudinal_velocity_delta_time * v + 0.5 * o.shape.dimensions.x - o.dist_from_ego;
      if (need > 0.0) t_need = std::max(t_need, need / v);
    }
  };
  consider(filtered_objects_.target_lane_trailing);
  consider(filtered_objects_.target_lane_leading.moving);
  return t_need;
}

std::string NormalLaneChange::nero_passing_behind() const
{
  // the cars clearly passing (faster than max(ego, crawl) + 1.5 m/s, within 40 m behind) that are not yet fully past
  // ego's nose and pass it within 8 s; their id prefixes ("" = none). A car only slightly faster than the crawl is alongside, not passing: the
  // change's own check decides about it (sim c28 slow_alongside: a +0.5 margin held the robot behind a 3 m/s car for 12 s)
  static const double crawl = [] {
    const char * c = std::getenv("NERO_LC_MULTI_WAIT_CRAWL");
    return c != nullptr ? std::atof(c) : 0.0;
  }();
  const double v_ego = std::max({0.0, getEgoVelocity(), crawl});
  const double front = common_data_ptr_->bpp_param_ptr->base_link2front;
  std::string who;
  const auto consider = [&](const auto & objs) {
    for (const auto & o : objs) {
      const double v = o.initial_twist.linear.x;
      if (v < v_ego + 1.5 || o.dist_from_ego < -40.0) continue;
      // its rear still has to pass ego's nose: only a car that does so within 8 s at the speed difference (sim c28
      // follower_35: c519ee 30 m back at 4.1 m/s held a crawling robot into a stand)
      const double to_pass = front - (o.dist_from_ego - 0.5 * o.shape.dimensions.x);
      if (to_pass <= 0.0 || to_pass / (v - v_ego) > 8.0) continue;
      if (!who.empty()) who += ",";
      std::string id;
      for (const auto b : o.uuid.uuid) {
        char h[3];
        std::snprintf(h, sizeof(h), "%02x", b);
        id += h;
        if (id.size() >= 6) break;
      }
      who += id;
    }
  };
  consider(filtered_objects_.target_lane_trailing);
  consider(filtered_objects_.target_lane_leading.moving);
  return who;
}

bool NormalLaneChange::nero_gap_wait() const
{
  const auto & m = nero_gap_wait_map();
  const auto it = m.find(this);
  return it != m.end() && it->second.seen;
}

bool NormalLaneChange::nero_stand_start() const
{
  const auto & m = nero_stand_start_map();
  const auto it = m.find(this);
  return it != m.end() && it->second;
}

bool NormalLaneChange::nero_two_or_more_due() const
{
  if (!common_data_ptr_) {
    return false;
  }
  const auto & td = common_data_ptr_->transient_data;
  return td.current_dist_buffer.min < 1e6 &&
         td.current_dist_buffer.min >
           td.lane_changing_length.min + lane_change_parameters_->lane_change_finish_judge_buffer + 1e-3;
}

// NERO_LC_STOP_COMFORT=<a m/s2> (routelc job 2026-10-26): a lane-change terminal stop (no safe / no valid path, the
// next change's terminal) that would lie closer than v^2 / 2a ahead of the moving robot, or behind it, is placed at that
// distance instead (never past the hard end given). Field: change 2's waiting stop appeared 2.1 m ahead at 11.4 km/h
// (10-26e #6: -2.50 m/s2; sim o37 -2.65); after a re-engage past the stop it lay behind ego (#38: -2.60 / -3.40). The
// robot then stops comfortably within the 10 m buffer before the terminal start. Stops for objects are not touched.
// Unset = the car.
double NormalLaneChange::nero_comfort_stop_arc(
  const double arc_length_to_stop, const PathWithLaneId & path, const double arc_length_hard_end) const
{
  static const double a_c = [] {
    const char * v = std::getenv("NERO_LC_STOP_COMFORT");
    return v != nullptr ? std::atof(v) : -1.0;
  }();
  if (a_c <= 0.0 || path.points.size() < 2) {
    return arc_length_to_stop;
  }
  const double s_ego =
    motion_utils::calcSignedArcLength(path.points, path.points.front().point.pose.position, getEgoPosition());
  const double v = std::max(0.0, getEgoVelocity());
  const double want = s_ego + v * v / (2.0 * a_c);
  // only a stop ahead of the robot or at most 10 m behind it (a re-engage past it, a few cm of creep): a module with no
  // lane change to make (no shift interval) puts its stop at -inf, which upstream leaves off the path
  constexpr double max_behind = 10.0;
  if (arc_length_to_stop >= want || !std::isfinite(arc_length_to_stop) || arc_length_to_stop < s_ego - max_behind) {
    return arc_length_to_stop;
  }
  const double out = std::min(want, std::max(arc_length_to_stop, arc_length_hard_end));
  if (nero_lc_debug_tick("comfort_stop")) {
    RCLCPP_INFO(
      logger_, "[NERO_LC_STOP_COMFORT] stop %.1f m ahead at %.2f m/s -> %.1f m (hard end %.1f m)",
      arc_length_to_stop - s_ego, v, out - s_ego, arc_length_hard_end - s_ego);
  }
  return out;
}

void NormalLaneChange::set_stop_pose(
  const double arc_length_to_stop_pose, PathWithLaneId & path, const std::string & reason)
{
  // NERO_LC_DEBUG (log only, routelc job): who puts a stop just ahead of the moving robot
  static const bool nero_debug = std::getenv("NERO_LC_DEBUG") != nullptr;
  if (nero_debug && path.points.size() >= 2 && getEgoVelocity() > 1.0) {
    const double s_ego =
      motion_utils::calcSignedArcLength(path.points, path.points.front().point.pose.position, getEgoPosition());
    const double ahead = arc_length_to_stop_pose - s_ego;
    if (ahead < 6.0 && ahead > -20.0) {
      RCLCPP_INFO(
        logger_, "[NERO_LC] stop '%s' %.2f m ahead at %.2f m/s (%s, valid %d safe %d)", reason.c_str(), ahead,
        getEgoVelocity(), direction_ == Direction::LEFT ? "left" : "right", status_.is_valid_path, status_.is_safe);
    }
  }
  const auto stop_point = utils::insertStopPoint(arc_length_to_stop_pose, path);
  lane_change_stop_pose_ = PoseWithDetailOpt(PoseWithDetail(stop_point.point.pose, reason));
}

void NormalLaneChange::updateStopTime()
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  const auto current_vel = getEgoVelocity();

  if (std::abs(current_vel) > lane_change_parameters_->th_stop_velocity) {
    stop_time_ = 0.0;
  } else {
    const double duration = stop_watch_.toc("stop_time");
    // clip stop time
    if (stop_time_ + duration * 0.001 > lane_change_parameters_->th_stop_time) {
      constexpr double eps = 0.1;
      stop_time_ = lane_change_parameters_->th_stop_time + eps;
    } else {
      stop_time_ += duration * 0.001;
    }
  }

  stop_watch_.tic("stop_time");
}

bool NormalLaneChange::check_prepare_phase() const
{
  const auto & route_handler = getRouteHandler();

  const auto check_in_general_lanes =
    lane_change_parameters_->safety.collision_check.enable_for_prepare_phase_in_general_lanes;

  lanelet::ConstLanelet current_lane;
  if (!route_handler->getClosestLaneletWithinRoute(getEgoPose(), &current_lane)) {
    RCLCPP_DEBUG(
      logger_, "Unable to get current lane. Default to %s.",
      (check_in_general_lanes ? "true" : "false"));
    return check_in_general_lanes;
  }

  const auto check_in_intersection =
    lane_change_parameters_->safety.collision_check.enable_for_prepare_phase_in_intersection &&
    common_data_ptr_->transient_data.in_intersection;

  const auto check_in_turns =
    lane_change_parameters_->safety.collision_check.enable_for_prepare_phase_in_turns &&
    common_data_ptr_->transient_data.in_turn_direction_lane;

  return check_in_intersection || check_in_turns || check_in_general_lanes;
}

void NormalLaneChange::update_dist_from_intersection()
{
  auto & transient_data = common_data_ptr_->transient_data;
  const auto & route_handler_ptr = common_data_ptr_->route_handler_ptr;

  if (
    transient_data.in_intersection && transient_data.in_turn_direction_lane &&
    path_after_intersection_.empty()) {
    auto path_after_intersection = route_handler_ptr->getCenterLinePath(
      common_data_ptr_->lanes_ptr->target_neighbor, 0.0, std::numeric_limits<double>::max());
    path_after_intersection_ = std::move(path_after_intersection.points);
    transient_data.dist_from_prev_intersection = 0.0;
    return;
  }

  if (
    transient_data.in_intersection || transient_data.in_turn_direction_lane ||
    path_after_intersection_.empty()) {
    return;
  }

  const auto & path_points = path_after_intersection_;
  const auto & front_point = path_points.front().point.pose.position;
  const auto & ego_position = common_data_ptr_->get_ego_pose().position;
  transient_data.dist_from_prev_intersection =
    calcSignedArcLength(path_points, front_point, ego_position);

  if (
    transient_data.dist_from_prev_intersection >= 0.0 &&
    transient_data.dist_from_prev_intersection <=
      common_data_ptr_->lc_param_ptr->backward_length_from_intersection) {
    return;
  }

  path_after_intersection_.clear();
  transient_data.dist_from_prev_intersection = std::numeric_limits<double>::max();
}
}  // namespace autoware::behavior_path_planner
