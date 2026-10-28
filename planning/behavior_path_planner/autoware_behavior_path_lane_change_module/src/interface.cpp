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

#include "autoware/behavior_path_lane_change_module/interface.hpp"
#include "autoware/behavior_path_lane_change_module/scene.hpp"
#include <cstdio>
#include <chrono>
#include <array>

#include "autoware/behavior_path_lane_change_module/utils/markers.hpp"
#include "autoware/behavior_path_lane_change_module/utils/utils.hpp"
#include "autoware/behavior_path_planner_common/interface/scene_module_interface.hpp"
#include "autoware/behavior_path_planner_common/interface/scene_module_visitor.hpp"
#include "autoware/behavior_path_planner_common/marker_utils/utils.hpp"
#include "autoware/behavior_path_planner_common/utils/path_safety_checker/safety_check.hpp"

#include <autoware_utils/ros/marker_helper.hpp>
#include <autoware_utils/system/time_keeper.hpp>

#include <autoware/motion_utils/trajectory/trajectory.hpp>
#include <autoware_utils/geometry/geometry.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <utility>

namespace autoware::behavior_path_planner
{
using autoware_utils::append_marker_array;
using utils::lane_change::assignToCandidate;

LaneChangeInterface::LaneChangeInterface(
  const std::string & name, rclcpp::Node & node, std::shared_ptr<LaneChangeParameters> parameters,
  const std::unordered_map<std::string, std::shared_ptr<RTCInterface>> & rtc_interface_ptr_map,
  std::unordered_map<std::string, std::shared_ptr<ObjectsOfInterestMarkerInterface>> &
    objects_of_interest_marker_interface_ptr_map,
  const std::shared_ptr<PlanningFactorInterface> & planning_factor_interface,
  std::unique_ptr<LaneChangeBase> && module_type)
: SceneModuleInterface{name, node, rtc_interface_ptr_map, objects_of_interest_marker_interface_ptr_map, planning_factor_interface},  // NOLINT
  parameters_{std::move(parameters)},
  module_type_{std::move(module_type)}
{
  module_type_->setTimeKeeper(getTimeKeeper());
  logger_ = utils::lane_change::getLogger(module_type_->getModuleTypeStr());
}

void LaneChangeInterface::processOnExit()
{
  module_type_->resetParameters();
  debug_marker_.markers.clear();
  post_process_safety_status_ = {};
  resetPathCandidate();
}

bool LaneChangeInterface::isExecutionRequested() const
{
  if (getCurrentStatus() == ModuleStatus::RUNNING) {
    return true;
  }

  return module_type_->isLaneChangeRequired();
}

bool LaneChangeInterface::isExecutionReady() const
{
  return module_type_->isSafe() && !module_type_->isAbortState();
}

void LaneChangeInterface::updateData()
{
  autoware_utils::ScopedTimeTrack st(__func__, *getTimeKeeper());
  module_type_->setPreviousModuleOutput(getPreviousModuleOutput());
  module_type_->update_lanes(getCurrentStatus() == ModuleStatus::RUNNING);
  module_type_->update_filtered_objects();
  module_type_->update_transient_data(getCurrentStatus() == ModuleStatus::RUNNING);
  module_type_->updateSpecialData();

  if (isWaitingApproval() || module_type_->isAbortState()) {
    if (!nero_abl_idle_skip()) {
      module_type_->updateLaneChangeStatus();
    }
  }
  // NERO_LC_DEBUG (log only, lca job): the instance's state each 0.5 s (is the path search running at all?)
  {
    static std::unordered_map<std::string, double> last;
    const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    double & l = last[name()];
    if (std::getenv("NERO_LC_DEBUG") != nullptr && now - l > 0.5) {
      l = now;
      RCLCPP_INFO(
        getLogger(), "[NERO_LC_STATE] %s status %d waiting %d abort %d valid %d safe %d", name().c_str(),
        static_cast<int>(getCurrentStatus()), isWaitingApproval() ? 1 : 0, module_type_->isAbortState() ? 1 : 0,
        module_type_->isValidPath() ? 1 : 0, module_type_->isSafe() ? 1 : 0);
    }
  }

  module_type_->resetStopPose();
  updateDebugMarker();
}

// NERO_LC_ABL_IDLE_GUARD=1 (MRM job 2026-10-25): the idle avoidance_by_lane_change module is always "waiting approval"
// (its processOnEntry calls waitApproval, and the manager enters the idle instance every cycle), so updateData ran the
// full lane change path search (frenet candidates near the lane end) every cycle only for isExecutionRequested() =
// isLaneChangeRequired() && specialRequiredCheck() && isValidPath() to come out false on one of the first two terms.
// At the 10-25 12:43 lane end that search cost 90-130 ms per cycle on top of the lane change's own -> MRM. Skip it
// while one of those two terms is false (the request is the same: false); a registered module is not affected.
// Its safe-cycle count (NERO_LC_MIN_SAFE_CYCLES) then starts when it has a target. Unset = upstream.
bool LaneChangeInterface::nero_abl_idle_skip()
{
  static const bool on = [this] {
    const char * v = std::getenv("NERO_LC_ABL_IDLE_GUARD");
    const bool b = v != nullptr && std::atoi(v) > 0;
    if (b) {
      RCLCPP_INFO(getLogger(), "[NERO_LC_ABL_IDLE_GUARD] 1: no path search in the idle avoidance_by_lane_change");
    }
    return b;
  }();
  if (!on || name() != "avoidance_by_lane_change" || getCurrentStatus() != ModuleStatus::IDLE) {
    return false;
  }
  if (module_type_->isAbortState()) {
    return false;
  }
  return !module_type_->isLaneChangeRequired() || !module_type_->specialRequiredCheck();
}

void LaneChangeInterface::postProcess()
{
  if (getCurrentStatus() == ModuleStatus::RUNNING) {
    const auto safety_status = module_type_->isApprovedPathSafe();
    post_process_safety_status_ =
      module_type_->evaluateApprovedPathWithUnsafeHysteresis(safety_status);
  }
  updateDebugMarker();
}

BehaviorModuleOutput LaneChangeInterface::plan()
{
  autoware_utils::ScopedTimeTrack st(__func__, *getTimeKeeper());
  resetPathCandidate();
  resetPathReference();

  // plan() should be called only when the module is in the RUNNING state, but
  // due to planner manager implementation, it can be called in the IDLE state.
  // TODO(Azu, Quda): consider a proper fix.
  if (getCurrentStatus() == ModuleStatus::IDLE) {
    auto output = getPreviousModuleOutput();
    path_reference_ = std::make_shared<PathWithLaneId>(output.reference_path);
    return output;
  }

  auto output = module_type_->generateOutput();
  path_reference_ = std::make_shared<PathWithLaneId>(output.reference_path);

  stop_pose_ = module_type_->getStopPose();

  const auto & lane_change_debug = module_type_->getDebugData();
  for (const auto & [uuid, data] : lane_change_debug.collision_check_objects_after_approval) {
    const auto color = data.is_safe ? ColorName::GREEN : ColorName::RED;
    setObjectsOfInterestData(data.current_obj_pose, data.obj_shape, color);
  }

  updateSteeringFactorPtr(output);
  if (module_type_->isAbortState()) {
    const auto candidate = planCandidate();
    path_candidate_ = std::make_shared<PathWithLaneId>(candidate.path_candidate);
    updateRTCStatus(
      std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), true,
      State::ABORTING);
  } else {
    const auto path =
      assignToCandidate(module_type_->getLaneChangePath(), module_type_->getEgoPosition());
    const auto is_registered = std::any_of(
      rtc_interface_ptr_map_.begin(), rtc_interface_ptr_map_.end(),
      [&](const auto & rtc) { return rtc.second->isRegistered(uuid_map_.at(rtc.first)); });

    if (!is_registered) {
      updateRTCStatus(
        path.start_distance_to_path_change, path.finish_distance_to_path_change, true,
        State::WAITING_FOR_EXECUTION);
    } else {
      const auto force_activated = std::any_of(
        rtc_interface_ptr_map_.begin(), rtc_interface_ptr_map_.end(),
        [&](const auto & rtc) { return rtc.second->isForceActivated(uuid_map_.at(rtc.first)); });
      updateRTCStatus(
        path.start_distance_to_path_change, path.finish_distance_to_path_change, !force_activated,
        State::RUNNING);
    }
  }

  set_longitudinal_planning_factor(output.path);

  return output;
}

namespace
{
double nero_env(const char * name, const double fallback)
{
  const char * v = std::getenv(name);
  return v != nullptr ? std::atof(v) : fallback;
}

// NERO: with NERO_LC_WAIT_DECEL=<m/s^2>, while a lane change waits for a gap the speed toward the
// waiting (stop) point is capped at sqrt(crawl^2 + 2 decel max(0, s - roll)): ego eases off early and
// gradually, rolls the last NERO_LC_WAIT_ROLL m (default 0) at NERO_LC_WAIT_CRAWL (default 0.8 m/s)
// signalling and hugging the line — time for drivers behind to let it in while it is still moving —
// and stops only at the stop point itself if no gap has come by then.
void nero_cap_waiting_velocity(
  PathWithLaneId & path, const PoseWithDetailOpt & stop, const Direction direction, const bool two_or_more,
  const geometry_msgs::msg::Point & ego_position, const double ego_velocity, const bool multi_roll_ok)
{
  static const int saudi_dir = static_cast<int>(nero_env("NERO_LC_SAUDI_DIR", 0.0));  // see scene.cpp
  // NERO_LC_MULTI_ROLL=1 (streamjoin job 2026-10-27): with two or more changes due the rolling wait acts in every
  // direction (NERO_LC_SAUDI_DIR keeps it to the left changes, the U-turn merge). The right changes otherwise braked to a
  // stand at their waiting stop and started both changes from it: 0.5 m/s2 launches, a stand at every stream car (10-26e
  // #28 / #29: the operator rolled in behind the stream; Ahmed 10-27 on the c28 sim: "the stops are too long it should be
  // smoother and without stops"). Unset = the car.
  // (streamjoin2: only while the change waits for a gap in a passing stream, or continues such a change: multi_roll_ok;
  // sim c56c: the rolling wait slowed a robot whose change was unsafe for a car cutting in from the other side, the
  // approval came 0.3 m/s slower and the approved change later ran on past the car merging in behind it, 0.38 m)
  static const bool multi_roll = nero_env("NERO_LC_MULTI_ROLL", 0.0) > 0.0;
  if (
    !(two_or_more && multi_roll && multi_roll_ok) &&
    ((saudi_dir == 1 && direction != Direction::LEFT) || (saudi_dir == 2 && direction != Direction::RIGHT))) {
    return;
  }
  static const double decel = nero_env("NERO_LC_WAIT_DECEL", 0.0);
  static const double crawl_single = nero_env("NERO_LC_WAIT_CRAWL", 0.8);
  // NERO_LC_MULTI_WAIT_CRAWL=<m/s> (routelc job 2026-10-26): with two or more changes due the rolling wait rolls at this
  // speed instead of NERO_LC_WAIT_CRAWL (1.2 m/s on the car, tuned for the single change before the U-turn: there the
  // crawl and the hug show the drivers behind ego wants in). At 93795 (two changes, 10-26e #6 "too slow", #37) the
  // operator kept 8-11 km/h beside the target lane and took the gaps. Single changes keep their crawl. Unset = the car.
  static const double crawl_multi = nero_env("NERO_LC_MULTI_WAIT_CRAWL", -1.0);
  const double crawl = (two_or_more && crawl_multi > 0.0) ? std::max(crawl_single, crawl_multi) : crawl_single;
  static const double roll = nero_env("NERO_LC_WAIT_ROLL", 0.0);
  if (decel <= 0.0 || !stop || path.points.size() < 2) {
    return;
  }
  const size_t stop_idx =
    autoware::motion_utils::findNearestIndex(path.points, stop->pose.position);
  // NERO_LC_WAIT_COMFORT=<m/s^2> (routelc job 2026-10-26): the cap never asks more than this deceleration from ego's
  // current speed. The cap is a profile back from the waiting stop, so a module that appears with ego already inside it
  // at speed stepped the path's speed at ego down: the second of two changes, the cycle the first one succeeded (the
  // crawl falls from the two-change 2.5 m/s to 1.2 m/s with the stop 10 m ahead) -> 2.6 -> 1.2 m/s at ego, the smoother
  // braked at -2.6 (sim o37; the car the same: -2.61, = the field's 10-26e #6 -2.50 with no factor). Unset = the car.
  static const double comfort = nero_env("NERO_LC_WAIT_COMFORT", -1.0);
  std::vector<double> v_comfort(path.points.size(), 0.0);
  if (comfort > 0.0) {
    const size_t ego_idx = autoware::motion_utils::findNearestIndex(path.points, ego_position);
    const double v0 = std::max(0.0, ego_velocity);
    double s_ego = 0.0;
    for (size_t i = 0; i < path.points.size(); ++i) {
      if (i > ego_idx) s_ego += autoware_utils::calc_distance2d(path.points.at(i - 1), path.points.at(i));
      v_comfort.at(i) = i <= ego_idx ? v0 : std::sqrt(std::max(0.0, v0 * v0 - 2.0 * comfort * s_ego));
    }
  }
  double s = 0.0;
  for (size_t i = stop_idx; i > 0; --i) {
    s += autoware_utils::calc_distance2d(path.points.at(i), path.points.at(i - 1));
    const double v = std::max(std::sqrt(crawl * crawl + 2.0 * decel * std::max(0.0, s - roll)), v_comfort.at(i - 1));
    auto & pv = path.points.at(i - 1).point.longitudinal_velocity_mps;
    pv = std::min(pv, static_cast<float>(v));
  }
}
// NERO_LC_YIELD_FRONT: see NormalLaneChange::nero_yield_cap() (scene.cpp), applied in planWaitingApproval below.
}  // namespace

BehaviorModuleOutput LaneChangeInterface::planWaitingApproval()
{
  BehaviorModuleOutput out = module_type_->getTerminalLaneChangePath();

  module_type_->insert_stop_point(module_type_->get_current_lanes(), out.path, true);
  const auto * nlc_wait = dynamic_cast<const NormalLaneChange *>(module_type_.get());
  nero_cap_waiting_velocity(
    out.path, module_type_->getStopPose(), module_type_->getDirection(),
    // NERO_LC_MULTI_ROLL: also the change that continues a double change made from the waiting place (the second of two
    // right changes: it otherwise braked toward its own waiting stop while waiting for a target-lane follower, sim c28
    // 9.8 -> 5.2 km/h)
    nlc_wait != nullptr && (nlc_wait->nero_two_or_more_due() ||
                            (nero_env("NERO_LC_MULTI_ROLL", 0.0) > 0.0 && nlc_wait->nero_continues_change())),
    module_type_->getEgoPose().position,
    module_type_->getEgoVelocity(),
    nlc_wait != nullptr && (nlc_wait->nero_gap_wait() || nlc_wait->nero_continues_change()));
  // NERO_LC_MULTI_ROLL (streamjoin): time the arrival at the waiting stop behind the passing target-lane cars (see
  // NormalLaneChange::nero_roll_arrival_cap): the path from ego to the stop is capped at that average speed, never below
  // 1.0 m/s (then the robot crawls up and waits), never above the crawl / the path, ramped down at 0.8 m/s2
  static const bool nero_multi_roll = nero_env("NERO_LC_MULTI_ROLL", 0.0) > 0.0;
  if (nlc_wait != nullptr && nero_multi_roll && module_type_->getStopPose() && out.path.points.size() >= 2) {
    const auto & ego_pos = module_type_->getEgoPosition();
    const double d_stop = autoware::motion_utils::calcSignedArcLength(
      out.path.points, ego_pos, module_type_->getStopPose()->pose.position);
    const double arr = nlc_wait->nero_roll_arrival_cap(d_stop);
    if (arr >= 0.0) {
      const double cap = std::max(1.0, arr);
      constexpr double a_ease = 0.8;
      const double v0 = std::max(0.0, module_type_->getEgoVelocity());
      const size_t ego_idx = autoware::motion_utils::findNearestIndex(out.path.points, ego_pos);
      double s_acc = 0.0;
      for (size_t i = ego_idx; i < out.path.points.size(); ++i) {
        if (i > ego_idx) s_acc += autoware_utils::calc_distance2d(out.path.points.at(i - 1), out.path.points.at(i));
        const double lim = std::max(cap, std::sqrt(std::max(0.0, v0 * v0 - 2.0 * a_ease * s_acc)));
        auto & pv = out.path.points.at(i).point.longitudinal_velocity_mps;
        pv = std::min(pv, static_cast<float>(lim));
      }
      static double last_log = 0.0;
      const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
      if (std::getenv("NERO_LC_DEBUG") != nullptr && now - last_log > 0.5) {
        last_log = now;
        RCLCPP_INFO(getLogger(), "[NERO_LC_ROLL] stop %.1f m ahead: arrive at %.2f m/s (cap %.2f)", d_stop, arr, cap);
      }
    }
  }
  if (const auto * nlc = dynamic_cast<const NormalLaneChange *>(module_type_.get())) {
    const double cap = nlc->nero_yield_cap();
    if (cap >= 0.0 && out.path.points.size() >= 2) {
      // ramp down from ego's speed at a comfortable 0.8 m/s^2 along the path (a step cap tripped the control /
      // planning validators: "over-speeding against the target", "deceleration is too large" -> MRM)
      constexpr double a_ease = 0.8;
      const double v0 = std::max(0.0, module_type_->getEgoVelocity());
      const size_t ego_idx = autoware::motion_utils::findNearestIndex(
        out.path.points, module_type_->getEgoPose().position);
      double s_acc = 0.0;
      for (size_t i = ego_idx; i < out.path.points.size(); ++i) {
        if (i > ego_idx) s_acc += autoware_utils::calc_distance2d(out.path.points.at(i - 1), out.path.points.at(i));
        const double lim = std::max(cap, std::sqrt(std::max(0.0, v0 * v0 - 2.0 * a_ease * s_acc)));
        auto & pv = out.path.points.at(i).point.longitudinal_velocity_mps;
        pv = std::min(pv, static_cast<float>(lim));
      }
      static double last = 0.0;
      const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
      if (now - last > 1.0) {
        last = now;
        RCLCPP_INFO(getLogger(), "[NERO_LC] easing off for a car in the target lane: cap %.1f m/s", cap);
      }
    }
  }
  out.turn_signal_info = module_type_->get_current_turn_signal_info();

  const auto & lane_change_debug = module_type_->getDebugData();
  for (const auto & [uuid, data] : lane_change_debug.collision_check_objects) {
    const auto color = data.is_safe ? ColorName::GREEN : ColorName::RED;
    setObjectsOfInterestData(data.current_obj_pose, data.obj_shape, color);
  }

  path_reference_ = std::make_shared<PathWithLaneId>(out.reference_path);
  stop_pose_ = module_type_->getStopPose();

  if (!module_type_->isValidPath()) {
    path_candidate_ = std::make_shared<PathWithLaneId>();
    // NERO_LC_FIX_10819=1: port of upstream #10819: with no valid path the RTC status is set to "waiting, unsafe"
    // and the module's state is reset (incl. the blinker time). Before it, the status of the last valid path stayed
    // un-updated; its entry dropped out after a few seconds and every new module instance then went RUNNING and was
    // cancelled for "InvalidPath" at once, its stop behind ego (sim 1026-214449: the robot standing at the 93643
    // waiting stop drove off straight on toward the lane end, the field's 10-26 #37). Unset = the car.
    static const bool fix_10819 = [] {
      const char * v = std::getenv("NERO_LC_FIX_10819");
      return v != nullptr && std::atoi(v) > 0;
    }();
    if (fix_10819) {
      updateRTCStatus(
        std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), false,
        State::WAITING_FOR_EXECUTION);
      module_type_->resetParameters();
      if (const auto * nlc = dynamic_cast<const NormalLaneChange *>(module_type_.get())) {
        nlc->nero_reset_signal_time();
      }
    }
    return out;
  }

  const auto candidate = planCandidate();
  path_candidate_ = std::make_shared<PathWithLaneId>(candidate.path_candidate);

  updateRTCStatus(
    candidate.start_distance_to_path_change, candidate.finish_distance_to_path_change,
    isExecutionReady(), State::WAITING_FOR_EXECUTION);
  is_abort_path_approved_ = false;

  set_longitudinal_planning_factor(out.path);

  return out;
}

CandidateOutput LaneChangeInterface::planCandidate() const
{
  autoware_utils::ScopedTimeTrack st(__func__, *getTimeKeeper());
  const auto selected_path = module_type_->getLaneChangePath();

  if (selected_path.path.points.empty()) {
    return {};
  }

  CandidateOutput output = assignToCandidate(selected_path, module_type_->getEgoPosition());

  updateSteeringFactorPtr(output, selected_path);
  return output;
}

void LaneChangeInterface::updateModuleParams(const std::any & parameters)
{
  parameters_ = std::any_cast<std::shared_ptr<LaneChangeParameters>>(parameters);
}

void LaneChangeInterface::setData(const std::shared_ptr<const PlannerData> & data)
{
  planner_data_ = data;
  module_type_->setData(data);
}

bool LaneChangeInterface::canTransitSuccessState()
{
  auto log_debug_throttled = [&](std::string_view message) -> void {
    RCLCPP_DEBUG(getLogger(), "%s", message.data());
  };
  updateDebugMarker();

  if (module_type_->specialExpiredCheck() && isWaitingApproval()) {
    log_debug_throttled("Run specialExpiredCheck.");
    return true;
  }

  if (module_type_->hasFinishedLaneChange()) {
    if (const auto * nlc = dynamic_cast<const NormalLaneChange *>(module_type_.get())) {
      nlc->nero_note_finished();  // streamjoin: NERO_LC_SIGNALLED_PREPARE (the next change in this direction continues it)
    }
    module_type_->resetParameters();
    log_debug_throttled("Lane change process has completed.");
    return true;
  }

  log_debug_throttled("Lane changing process is ongoing");
  return false;
}

bool LaneChangeInterface::canTransitFailureState()
{
  const auto force_activated = std::any_of(
    rtc_interface_ptr_map_.begin(), rtc_interface_ptr_map_.end(),
    [&](const auto & rtc) { return rtc.second->isForceActivated(uuid_map_.at(rtc.first)); });

  if (force_activated) {
    RCLCPP_WARN_THROTTLE(getLogger(), *clock_, 5000, "unsafe but force executed");
    return false;
  }

  const auto [state, reason] = check_transit_failure();

  interface_debug_.failing_reason = reason;
  interface_debug_.lc_state = state;

  updateDebugMarker();

  if (state == LaneChangeStates::Cancel) {
    updateRTCStatus(
      std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), true,
      State::FAILED);
    module_type_->toCancelState();
    return true;
  }

  if (state == LaneChangeStates::Abort) {
    module_type_->toAbortState();
    return false;
  }

  // Note: Ideally, if it is unsafe, but for some reason, we can't abort or cancel, then we should
  // stop. Note: This feature is not working properly for now.
  const auto [is_safe, unsafe_trailing_obj] = post_process_safety_status_;
  if (!is_safe && module_type_->isRequiredStop(unsafe_trailing_obj)) {
    module_type_->toStopState();
    return false;
  }

  module_type_->toNormalState();
  return false;
}

std::pair<LaneChangeStates, std::string_view> LaneChangeInterface::check_transit_failure()
{
  if (module_type_->isAbortState()) {
    if (module_type_->hasFinishedAbort()) {
      return {LaneChangeStates::Cancel, "Aborted"};
    }
    return {LaneChangeStates::Abort, "Aborting"};
  }

  if (isWaitingApproval()) {
    if (module_type_->is_near_regulatory_element()) {
      return {LaneChangeStates::Cancel, "CloseToRegElement"};
    }
    return {LaneChangeStates::Normal, "WaitingForApproval"};
  }

  if (!module_type_->isValidPath()) {
    return {LaneChangeStates::Cancel, "InvalidPath"};
  }

  const auto is_preparing = module_type_->isEgoOnPreparePhase();
  const auto can_return_to_current = module_type_->isAbleToReturnCurrentLane();

  // regardless of safe and unsafe, we want to cancel lane change.
  if (is_preparing) {
    const auto force_deactivated = std::any_of(
      rtc_interface_ptr_map_.begin(), rtc_interface_ptr_map_.end(),
      [&](const auto & rtc) { return rtc.second->isForceDeactivated(uuid_map_.at(rtc.first)); });

    if (force_deactivated && can_return_to_current) {
      return {LaneChangeStates::Cancel, "ForceDeactivation"};
    }
  }

  if (post_process_safety_status_.is_safe) {
    return {LaneChangeStates::Normal, "SafeToLaneChange"};
  }

  if (!module_type_->isCancelEnabled()) {
    return {LaneChangeStates::Warning, "CancelDisabled"};
  }

  // We also check if the ego can return to the current lane, as prepare segment might be out of the
  // lane, for example, during an evasive maneuver around a static object.
  if (is_preparing && can_return_to_current) {
    return {LaneChangeStates::Cancel, "SafeToCancel"};
  }

  if (module_type_->is_near_terminal()) {
    return {LaneChangeStates::Warning, "TooNearTerminal"};
  }

  if (!module_type_->isAbortEnabled()) {
    return {LaneChangeStates::Warning, "AbortDisabled"};
  }

  // To prevent the lane module from having to check rear objects in the current lane, we limit the
  // abort maneuver to cases where the ego vehicle is still in the current lane.
  if (!can_return_to_current) {
    return {LaneChangeStates::Warning, "TooLateToAbort"};
  }

  const auto found_abort_path = module_type_->calcAbortPath();
  if (!found_abort_path) {
    return {LaneChangeStates::Warning, "AbortPathNotFound"};
  }

  return {LaneChangeStates::Abort, "SafeToAbort"};
}

void LaneChangeInterface::updateDebugMarker() const
{
  debug_marker_.markers.clear();
  if (!parameters_->publish_debug_marker) {
    return;
  }
  using marker_utils::lane_change_markers::createDebugMarkerArray;
  debug_marker_ = createDebugMarkerArray(
    interface_debug_, module_type_->getDebugData(), module_type_->getEgoPose());
}

MarkerArray LaneChangeInterface::getModuleVirtualWall()
{
  using marker_utils::lane_change_markers::createLaneChangingVirtualWallMarker;
  MarkerArray marker;

  if (!parameters_->publish_debug_marker) {
    return marker;
  }

  if (isWaitingApproval() || getCurrentStatus() != ModuleStatus::RUNNING) {
    return marker;
  }
  const auto & start_pose = module_type_->getLaneChangePath().info.lane_changing_start;
  const auto start_marker =
    createLaneChangingVirtualWallMarker(start_pose, name(), clock_->now(), "lane_change_start");

  const auto & end_pose = module_type_->getLaneChangePath().info.lane_changing_end;
  const auto end_marker =
    createLaneChangingVirtualWallMarker(end_pose, name(), clock_->now(), "lane_change_end");
  marker.markers.reserve(start_marker.markers.size() + end_marker.markers.size());
  append_marker_array(start_marker, &marker);
  append_marker_array(end_marker, &marker);
  return marker;
}

void LaneChangeInterface::updateSteeringFactorPtr(const BehaviorModuleOutput & output)
{
  autoware_utils::ScopedTimeTrack st(__func__, *getTimeKeeper());

  const auto current_position = module_type_->getEgoPosition();
  const auto status = module_type_->getLaneChangeStatus();
  const auto start_distance = autoware::motion_utils::calcSignedArcLength(
    output.path.points, current_position, status.lane_change_path.info.shift_line.start.position);
  const auto finish_distance = autoware::motion_utils::calcSignedArcLength(
    output.path.points, current_position, status.lane_change_path.info.shift_line.end.position);

  const auto planning_factor_direction = std::invoke([&]() {
    if (module_type_->getDirection() == Direction::LEFT) {
      return PlanningFactor::SHIFT_LEFT;
    }
    if (module_type_->getDirection() == Direction::RIGHT) {
      return PlanningFactor::SHIFT_RIGHT;
    }
    return PlanningFactor::UNKNOWN;
  });

  const auto & lane_change_debug = module_type_->getDebugData();
  planning_factor_interface_->add(
    start_distance, finish_distance, status.lane_change_path.info.shift_line.start,
    status.lane_change_path.info.shift_line.end, planning_factor_direction,
    utils::path_safety_checker::to_safety_factor_array(lane_change_debug.collision_check_objects));
}

void LaneChangeInterface::updateSteeringFactorPtr(
  const CandidateOutput & output, const LaneChangePath & selected_path) const
{
  const uint16_t planning_factor_direction = std::invoke([&output]() {
    if (output.lateral_shift > 0.0) {
      return PlanningFactor::SHIFT_LEFT;
    }
    return PlanningFactor::SHIFT_RIGHT;
  });

  const auto & lane_change_debug = module_type_->getDebugData();
  planning_factor_interface_->add(
    output.start_distance_to_path_change, output.finish_distance_to_path_change,
    selected_path.info.shift_line.start, selected_path.info.shift_line.end,
    planning_factor_direction,
    utils::path_safety_checker::to_safety_factor_array(lane_change_debug.collision_check_objects));
}
}  // namespace autoware::behavior_path_planner
