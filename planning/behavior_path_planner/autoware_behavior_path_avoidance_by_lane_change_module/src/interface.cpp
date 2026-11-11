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

#include "interface.hpp"

#include "autoware/behavior_path_planner_common/interface/scene_module_interface.hpp"
#include "autoware/behavior_path_planner_common/interface/scene_module_visitor.hpp"

#include "autoware/behavior_path_static_obstacle_avoidance_module/utils.hpp"

#include <chrono>
#include <autoware_utils/geometry/geometry.hpp>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>

namespace autoware::behavior_path_planner
{
using autoware::behavior_path_planner::State;
using autoware::route_handler::Direction;

AvoidanceByLaneChangeInterface::AvoidanceByLaneChangeInterface(
  const std::string & name, rclcpp::Node & node,
  const std::shared_ptr<LaneChangeParameters> & parameters,
  const std::shared_ptr<AvoidanceByLCParameters> & avoidance_by_lane_change_parameters,
  const std::unordered_map<std::string, std::shared_ptr<RTCInterface>> & rtc_interface_ptr_map,
  std::unordered_map<std::string, std::shared_ptr<ObjectsOfInterestMarkerInterface>> &
    objects_of_interest_marker_interface_ptr_map,
  const std::shared_ptr<PlanningFactorInterface> & planning_factor_interface)
: LaneChangeInterface{
    name,
    node,
    parameters,
    rtc_interface_ptr_map,
    objects_of_interest_marker_interface_ptr_map,
    planning_factor_interface,
    std::make_unique<AvoidanceByLaneChange>(parameters, avoidance_by_lane_change_parameters)}
{
}

bool AvoidanceByLaneChangeInterface::isExecutionRequested() const
{
  const bool requested = module_type_->isLaneChangeRequired() && module_type_->specialRequiredCheck() &&
                         module_type_->isValidPath();
  // NERO_SOA_ABLC_HANDOFF (mainlane): tell static avoidance which objects this module requests a lane change for
  if (requested && utils::static_obstacle_avoidance::nero_ablc_handoff_hold() > 0.0) {
    if (const auto * m = dynamic_cast<const AvoidanceByLaneChange *>(module_type_.get())) {
      utils::static_obstacle_avoidance::nero_ablc_claim(m->nero_claim_ids(), clock_->now().seconds());
    }
  }
  return requested;
}

void AvoidanceByLaneChangeInterface::processOnEntry()
{
  waitApproval();
}

namespace
{
bool nero_ablc_lanes()
{
  static const bool on = [] {
    const char * v = std::getenv("NERO_ABLC_LANES");
    return v != nullptr && std::atoi(v) > 0;
  }();
  return on;
}
}  // namespace

void AvoidanceByLaneChangeInterface::updateData()
{
  // NERO_ABLC_LANES=1 (mainlane 2026-11-11): processOnEntry() sets is_waiting_approval_ (the idle instance needs it: its
  // request needs a valid path, and the path search runs only while waiting), but nothing clears it once the module is
  // RUNNING (after its auto approval): the manager then kept it a waiting candidate for good (sim: "status 1 waiting 1 valid
  // 1 safe 1" for 17 s, no lane change). Clear it once RUNNING, as the normal lane change (no such flag) is. Unset = upstream.
  if (nero_ablc_lanes() && getCurrentStatus() == ModuleStatus::RUNNING) {
    clearWaitingApproval();
  }
  LaneChangeInterface::updateData();
  // NERO_SOA_ABLC_HANDOFF: a launched instance (candidate or approved) keeps its claims fresh every cycle; the request
  // check (isExecutionRequested) only runs for the idle instance, so a change approved at once and then cancelled (the
  // strict hold) left static avoidance a claim older than the hold (sim r24b)
  if (
    getCurrentStatus() != ModuleStatus::IDLE &&
    utils::static_obstacle_avoidance::nero_ablc_handoff_hold() > 0.0) {
    if (const auto * m = dynamic_cast<const AvoidanceByLaneChange *>(module_type_.get())) {
      utils::static_obstacle_avoidance::nero_ablc_claim(m->nero_claim_ids(), clock_->now().seconds());
    }
  }
}

void AvoidanceByLaneChangeInterface::postProcess()
{
  LaneChangeInterface::postProcess();
  // NERO_ABLC_DEBUG (log only, mainlane): the LAUNCHED instance's state each 0.5 s (the manager's idle request instance is
  // skipped: its status is IDLE). Status 1 RUNNING (approved) / 2 WAITING_APPROVAL; the stop it outputs and why.
  static const bool on = std::getenv("NERO_ABLC_DEBUG") != nullptr;
  if (!on || getCurrentStatus() == ModuleStatus::IDLE) {
    return;
  }
  static std::unordered_map<const void *, std::chrono::steady_clock::time_point> last;
  auto & l = last[this];
  const auto now = std::chrono::steady_clock::now();
  if (now - l < std::chrono::milliseconds(500)) {
    return;
  }
  l = now;
  const auto sp = module_type_->getStopPose();
  double d_stop = -1.0;
  if (sp) {
    d_stop = autoware_utils::calc_distance2d(module_type_->getEgoPose(), sp->pose);
  }
  RCLCPP_INFO(
    getLogger(), "[NERO_ABLC_INST] %p status %d waiting %d safe %d valid %d abort %d | stop %.1f m (%s) | v %.2f",
    static_cast<const void *>(this), static_cast<int>(getCurrentStatus()), isWaitingApproval() ? 1 : 0,
    module_type_->isSafe() ? 1 : 0, module_type_->isValidPath() ? 1 : 0, module_type_->isAbortState() ? 1 : 0, d_stop,
    sp ? sp->detail.c_str() : "-", module_type_->getEgoVelocity());
}

void AvoidanceByLaneChangeInterface::updateRTCStatus(
  const double start_distance, const double finish_distance)
{
  const auto direction = std::invoke([&]() -> std::string {
    const auto dir = module_type_->getDirection();
    return (dir == Direction::LEFT) ? "left" : "right";
  });

  const auto state = isWaitingApproval() ? State::WAITING_FOR_EXECUTION : State::RUNNING;

  rtc_interface_ptr_map_.at(direction)->updateCooperateStatus(
    uuid_map_.at(direction), isExecutionReady(), state, start_distance, finish_distance,
    clock_->now());
}
}  // namespace autoware::behavior_path_planner
