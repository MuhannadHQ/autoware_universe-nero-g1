// Copyright 2022 TIER IV, Inc.
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

#include "arrival_checker.hpp"

#include <autoware_utils/geometry/geometry.hpp>
#include <autoware_utils/math/normalization.hpp>
#include <autoware_utils/math/unit_conversion.hpp>

#include <tf2/utils.h>

#include <cstdlib>

namespace autoware::mission_planner_universe
{

ArrivalChecker::ArrivalChecker(rclcpp::Node * node) : vehicle_stop_checker_(node)
{
  const double angle_deg = node->declare_parameter<double>("arrival_check_angle_deg");
  angle_ = autoware_utils::deg2rad(angle_deg);
  distance_ = node->declare_parameter<double>("arrival_check_distance");
  duration_ = node->declare_parameter<double>("arrival_check_duration");

  // NERO_MISSION_ARRIVAL_WAIT_PARKING=1 (nero, 2026-10-24; the arrival part of upstream
  // autoware_universe #11170): in the Parking scenario the route is ARRIVED only once the
  // freespace planner reports parking completed. Without it ARRIVED came while the vehicle stood
  // between two parking segments within 1 m / 45 deg of the goal (a re-park was cut off; parking
  // sims 2026-10-24). Unset = upstream (no subscriptions).
  const char * v = std::getenv("NERO_MISSION_ARRIVAL_WAIT_PARKING");
  if (v && std::atoi(v) == 1) {
    nero_sub_scenario_ = node->create_subscription<autoware_internal_planning_msgs::msg::Scenario>(
      "/planning/scenario_planning/scenario", rclcpp::QoS{1},
      [this](const autoware_internal_planning_msgs::msg::Scenario::ConstSharedPtr msg) {
        nero_scenario_ = msg->current_scenario;
      });
    nero_sub_parking_state_ = node->create_subscription<std_msgs::msg::Bool>(
      "/planning/scenario_planning/parking/is_completed", rclcpp::QoS{1},
      [this](const std_msgs::msg::Bool::ConstSharedPtr msg) {
        nero_parking_completed_ = msg->data;
      });
    RCLCPP_INFO(
      node->get_logger(), "[NERO_MISSION_ARRIVAL_WAIT_PARKING] arrival waits for parking completed");
  }
}

void ArrivalChecker::set_goal()
{
  // Ignore the modified goal after the route is cleared.
  goal_with_uuid_ = std::nullopt;
}

void ArrivalChecker::set_goal(const PoseWithUuidStamped & goal)
{
  // Ignore the modified goal for the previous route using uuid.
  goal_with_uuid_ = goal;
}

bool ArrivalChecker::is_arrived(const PoseStamped & pose) const
{
  if (!goal_with_uuid_) {
    return false;
  }
  const auto goal = goal_with_uuid_.value();

  // Check frame id
  if (goal.header.frame_id != pose.header.frame_id) {
    return false;
  }

  // Check distance.
  if (distance_ < autoware_utils::calc_distance2d(pose.pose, goal.pose)) {
    return false;
  }

  // Check angle.
  const double yaw_pose = tf2::getYaw(pose.pose.orientation);
  const double yaw_goal = tf2::getYaw(goal.pose.orientation);
  const double yaw_diff = autoware_utils::normalize_radian(yaw_pose - yaw_goal);
  if (angle_ < std::fabs(yaw_diff)) {
    return false;
  }

  if (
    nero_sub_parking_state_ &&
    nero_scenario_ == autoware_internal_planning_msgs::msg::Scenario::PARKING &&
    !nero_parking_completed_) {
    return false;
  }

  // Check vehicle stopped.
  return vehicle_stop_checker_.isVehicleStopped(duration_);
}

}  // namespace autoware::mission_planner_universe
