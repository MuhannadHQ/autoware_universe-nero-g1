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
#ifndef AUTOWARE__BEHAVIOR_PATH_LANE_CHANGE_MODULE__SCENE_HPP_
#define AUTOWARE__BEHAVIOR_PATH_LANE_CHANGE_MODULE__SCENE_HPP_

#include "autoware/behavior_path_lane_change_module/base_class.hpp"
#include "autoware/behavior_path_lane_change_module/structs/data.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace autoware::behavior_path_planner
{
using autoware::behavior_path_planner::utils::path_safety_checker::CollisionCheckDebug;
using autoware::behavior_path_planner::utils::path_safety_checker::CollisionCheckDebugMap;
using autoware::behavior_path_planner::utils::path_safety_checker::ExtendedPredictedObject;
using autoware::behavior_path_planner::utils::path_safety_checker::
  PoseWithVelocityAndPolygonStamped;
using autoware::behavior_path_planner::utils::path_safety_checker::PoseWithVelocityStamped;
using autoware::behavior_path_planner::utils::path_safety_checker::PredictedPathWithPolygon;
using autoware::route_handler::Direction;
using autoware_internal_planning_msgs::msg::PathWithLaneId;
using geometry_msgs::msg::Point;
using geometry_msgs::msg::Pose;
using geometry_msgs::msg::Twist;
using lane_change::LanesPolygon;
using utils::path_safety_checker::ExtendedPredictedObjects;
using utils::path_safety_checker::RSSparams;

class NormalLaneChange : public LaneChangeBase
{
public:
  NormalLaneChange(
    const std::shared_ptr<LaneChangeParameters> & parameters, LaneChangeModuleType type,
    Direction direction);

  NormalLaneChange(const NormalLaneChange &) = delete;
  NormalLaneChange(NormalLaneChange &&) = delete;
  NormalLaneChange & operator=(const NormalLaneChange &) = delete;
  NormalLaneChange & operator=(NormalLaneChange &&) = delete;
  ~NormalLaneChange() override = default;

  void update_lanes(const bool is_approved) final;

  void update_transient_data(const bool is_approved) final;

  void update_filtered_objects() final;

  void updateLaneChangeStatus() override;

  std::pair<bool, bool> getSafePath(LaneChangePath & safe_path) const override;

  LaneChangePath getLaneChangePath() const override;

  BehaviorModuleOutput getTerminalLaneChangePath() const override;

  BehaviorModuleOutput generateOutput() override;

  void extendOutputDrivableArea(BehaviorModuleOutput & output) const override;

  void insert_stop_point(
    const lanelet::ConstLanelets & lanelets, PathWithLaneId & path,
    const bool is_waiting_approval = false) override;

  void insert_stop_point_on_current_lanes(
    PathWithLaneId & path, const bool is_waiting_approval = false);

  PathWithLaneId getReferencePath() const override;

  std::optional<PathWithLaneId> extendPath() override;

  void resetParameters() override;

  TurnSignalInfo updateOutputTurnSignal() const override;

  bool calcAbortPath() override;

  PathSafetyStatus isApprovedPathSafe() const override;

  bool nero_in_hold_mode() const;  // NERO hold (non-virtual: no layout change)

  // NERO_LC_YIELD_FRONT (session 4): the speed cap while waiting (< 0: none). Non-virtual: no layout change.
  double nero_yield_cap() const;

  // NERO_LC_FIX_10819 (routelc job): upstream #10819's blinker-time reset on a cycle with no valid path, kept out of
  // resetParameters() (also called on exit / success) so the car is unchanged with the switch unset. Non-virtual.
  void nero_reset_signal_time() const { set_signal_activation_time(true); }

  // NERO_LC_MULTI_STOP_OFFSET (routelc job): the waiting stop's offset before the terminal start. Non-virtual.
  double nero_stop_offset() const;
  bool nero_two_or_more_due() const;  // the distance buffer holds two or more changes (non-virtual)
  // streamjoin job: the car's rolling wait acts for this direction; this change starts from the robot standing at its
  // waiting stop with two or more changes due (NERO_LC_MULTI_STAND_*). Non-virtual.
  bool nero_rolling_wait_dir() const;
  bool nero_stand_start() const;
  void nero_note_finished() const;      // this change finished (time per direction)
  bool nero_continues_change() const;   // a change in the same direction finished just before this one started
  // NERO_LC_MULTI_ROLL: the speed at which ego reaches its waiting stop when the passing target-lane cars are the change's
  // gap ahead of it (-1: none). Non-virtual.
  double nero_roll_arrival_cap(const double dist_to_stop) const;
  // streamjoin2: the time ego needs so that the passing target-lane cars are the change's front gap ahead of its nose at
  // a stop dist_to_stop ahead (0: none passing); and "this change waits for a gap in a passing stream" (latched per module
  // instance: unsafe while such cars pass). Non-virtual.
  double nero_passing_need(const double dist_to_stop) const;
  bool nero_gap_wait() const;
  std::string nero_passing_behind() const;  // streamjoin2: passing cars not yet past ego's nose (ids, "" = none)

  // NERO_LC_STOP_COMFORT (routelc job): a terminal stop never closer than v^2 / 2a ahead of the moving robot. Non-virtual.
  double nero_comfort_stop_arc(
    const double arc_length_to_stop, const PathWithLaneId & path, const double arc_length_hard_end) const;

  // NERO_LC_KEEP_RIGHT_GATE="<min_end_m>,<lead_dv>,<lead_m>,<lead_t>" (trips job 2026-10-24): a change to the RIGHT that
  // the route does not need soon (ego's current lanes go on > min_end_m: a keep-right change of NERO_KEEP_RIGHT_LANES) is
  // not requested while (a) a car ahead in the target lane slower than ego's maximum by > lead_dv is within lead_m, or
  // (moving) would be reached within lead_t s (stay left and pass it), or (b) UN R79 5.6.4.7 / R157 5.2.6.7.2.2 do not hold: a car at the speed limit + 30 km/h at
  // the edge of the rear view (40 m; a car behind ego does not shorten it: side / top lidars) or any seen car behind in the target
  // lane closer than S_crit = dv 0.4 + dv^2 / 6 + v_e 1.0. Only blocks; unset = upstream. Non-virtual: no layout change.
  bool nero_keep_right_discretionary() const;
  bool nero_keep_right_gate_blocks() const;

  PathSafetyStatus evaluateApprovedPathWithUnsafeHysteresis(
    PathSafetyStatus approved_path_safety_status) override;

  bool isRequiredStop(const bool is_trailing_object) override;

  bool hasFinishedLaneChange() const override;

  bool isAbleToReturnCurrentLane() const override;

  bool is_near_terminal() const final;

  bool isEgoOnPreparePhase() const override;

  bool isAbleToStopSafely() const override;

  bool hasFinishedAbort() const override;

  bool isAbortState() const override;

  bool isLaneChangeRequired() override;

  bool isStoppedAtRedTrafficLight() const override;

  bool is_near_regulatory_element() const final;

  TurnSignalInfo get_current_turn_signal_info() const final;

protected:
  lanelet::ConstLanelets get_lane_change_lanes(const lanelet::ConstLanelets & current_lanes) const;

  TurnSignalInfo get_terminal_turn_signal_info() const final;

  lane_change::TargetObjects get_target_objects(
    const FilteredLanesObjects & filtered_objects,
    const lanelet::ConstLanelets & current_lanes) const;

  FilteredLanesObjects filter_objects() const;

  void filterOncomingObjects(PredictedObjects & objects) const;

  std::vector<LaneChangePhaseMetrics> get_prepare_metrics() const;
  std::vector<LaneChangePhaseMetrics> get_lane_changing_metrics(
    const PathWithLaneId & prep_segment, const LaneChangePhaseMetrics & prep_metrics,
    const double shift_length, const double dist_to_reg_element,
    lane_change::MetricsDebug & debug_metrics) const;

  bool get_lane_change_paths(LaneChangePaths & candidate_paths) const;

  bool get_path_using_frenet(
    const std::vector<LaneChangePhaseMetrics> & prepare_metrics,
    const lane_change::TargetObjects & target_objects,
    const std::vector<std::vector<int64_t>> & sorted_lane_ids,
    LaneChangePaths & candidate_paths) const;

  bool get_path_using_path_shifter(
    const std::vector<LaneChangePhaseMetrics> & prepare_metrics,
    const lane_change::TargetObjects & target_objects,
    const std::vector<std::vector<int64_t>> & sorted_lane_ids,
    LaneChangePaths & candidate_paths) const;

  bool check_candidate_path_safety(
    const LaneChangePath & candidate_path, const lane_change::TargetObjects & target_objects) const;

  std::optional<PathWithLaneId> compute_terminal_lane_change_path() const;

  bool isValidPath(const PathWithLaneId & path) const override;

  PathSafetyStatus isLaneChangePathSafe(
    const LaneChangePath & lane_change_path,
    const std::vector<std::vector<PoseWithVelocityStamped>> & ego_predicted_paths,
    const lane_change::TargetObjects & collision_check_objects,
    const utils::path_safety_checker::RSSparams & rss_params,
    CollisionCheckDebugMap & debug_data) const;

  bool is_colliding(
    const LaneChangePath & lane_change_path, const ExtendedPredictedObject & obj,
    const std::vector<PoseWithVelocityStamped> & ego_predicted_path,
    const RSSparams & selected_rss_param, const bool check_prepare_phase,
    CollisionCheckDebugMap & debug_data) const;

  double get_max_velocity_for_safety_check() const;

  bool is_ego_stuck() const;

  bool check_prepare_phase() const;

  void set_stop_pose(
    const double arc_length_to_stop_pose, PathWithLaneId & path, const std::string & reason = "");

  void updateStopTime();

  double getStopTime() const { return stop_time_; }

  const lanelet::ConstLanelets & get_target_lanes() const
  {
    return common_data_ptr_->lanes_ptr->target;
  }

  void update_dist_from_intersection();

  std::vector<PathPointWithLaneId> path_after_intersection_;
  double stop_time_{0.0};
};
}  // namespace autoware::behavior_path_planner
#endif  // AUTOWARE__BEHAVIOR_PATH_LANE_CHANGE_MODULE__SCENE_HPP_
