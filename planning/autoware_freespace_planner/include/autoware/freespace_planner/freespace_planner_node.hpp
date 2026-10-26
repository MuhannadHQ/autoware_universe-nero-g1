// Copyright 2020 Tier IV, Inc.
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

/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef AUTOWARE__FREESPACE_PLANNER__FREESPACE_PLANNER_NODE_HPP_
#define AUTOWARE__FREESPACE_PLANNER__FREESPACE_PLANNER_NODE_HPP_

#include "autoware_utils/ros/logger_level_configure.hpp"

#include <autoware/freespace_planning_algorithms/astar_search.hpp>
#include <autoware/freespace_planning_algorithms/rrtstar.hpp>
#include <autoware_utils/ros/polling_subscriber.hpp>
#include <autoware_vehicle_info_utils/vehicle_info_utils.hpp>
#include <rclcpp/rclcpp.hpp>

#include <autoware_adapi_v1_msgs/msg/operation_mode_state.hpp>
#include <autoware_control_msgs/msg/control.hpp>
#include <autoware_internal_debug_msgs/msg/float64_stamped.hpp>
#include <autoware_internal_planning_msgs/msg/scenario.hpp>
#include <autoware_perception_msgs/msg/predicted_objects.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/bool.hpp>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif

#include <autoware/route_handler/route_handler.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <deque>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class TestFreespacePlanner;

namespace autoware::freespace_planner
{
using autoware::freespace_planning_algorithms::AbstractPlanningAlgorithm;
using autoware::freespace_planning_algorithms::AstarParam;
using autoware::freespace_planning_algorithms::AstarSearch;
using autoware::freespace_planning_algorithms::PlannerCommonParam;
using autoware::freespace_planning_algorithms::RRTStar;
using autoware::freespace_planning_algorithms::RRTStarParam;
using autoware::freespace_planning_algorithms::VehicleShape;
using autoware_internal_planning_msgs::msg::Scenario;
using autoware_planning_msgs::msg::LaneletRoute;
using autoware_planning_msgs::msg::Trajectory;
using geometry_msgs::msg::PoseArray;
using geometry_msgs::msg::PoseStamped;
using geometry_msgs::msg::TransformStamped;
using geometry_msgs::msg::Twist;
using nav_msgs::msg::OccupancyGrid;
using nav_msgs::msg::Odometry;

struct NodeParam
{
  std::string planning_algorithm;
  double waypoints_velocity;  // constant velocity on planned waypoints [km/h]
  double update_rate;         // replanning and publishing rate [Hz]
  double th_arrived_distance_m;
  double th_stopped_time_sec;
  double th_stopped_velocity_mps;
  double th_course_out_distance_m;  // collision margin [m]
  double th_obstacle_time_sec;
  double vehicle_shape_margin_m;
  bool replan_when_obstacle_found;
  bool replan_when_course_out;
};

class FreespacePlannerNode : public rclcpp::Node
{
public:
  explicit FreespacePlannerNode(const rclcpp::NodeOptions & node_options);

private:
  // ros
  rclcpp::Publisher<Trajectory>::SharedPtr trajectory_pub_;
  rclcpp::Publisher<PoseArray>::SharedPtr debug_pose_array_pub_;
  rclcpp::Publisher<PoseArray>::SharedPtr debug_partial_pose_array_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr parking_state_pub_;
  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Float64Stamped>::SharedPtr
    processing_time_pub_;

  rclcpp::Subscription<LaneletRoute>::SharedPtr route_sub_;

  autoware_utils::InterProcessPollingSubscriber<OccupancyGrid> occupancy_grid_sub_{
    this, "~/input/occupancy_grid"};
  autoware_utils::InterProcessPollingSubscriber<Scenario> scenario_sub_{this, "~/input/scenario"};
  autoware_utils::InterProcessPollingSubscriber<Odometry, autoware_utils::polling_policy::All>
    odom_sub_{this, "~/input/odometry", rclcpp::QoS{100}};

  rclcpp::TimerBase::SharedPtr timer_;

  // NERO_FREESPACE_PREDICTED_HOLD_S (nero, 2026-10-24): created only with the switch set
  rclcpp::Subscription<autoware_perception_msgs::msg::PredictedObjects>::SharedPtr nero_objects_sub_;
  autoware_perception_msgs::msg::PredictedObjects::ConstSharedPtr nero_objects_;
  bool nero_holding_ = false;
  // NERO_FREESPACE_STRAIGHT_IN_WAIT_S / NERO_FREESPACE_REPARK_DEG (nero, 2026-10-24)
  double nero_straight_wait_since_ = -1.0;
  double nero_straight_wait_try_ = -1.0;
  int nero_reparks_ = 0;
  int nero_short_replans_ = 0;  // NERO_FREESPACE_STRAIGHT_FWD_MIN_M: re-plans for a segment too short to start
  double nero_failed_search_t_ = -1e9;  // NERO_FREESPACE_PREDICTED_HOLD_MOVING: when a full A* search last failed
  bool nero_repark_now_ = false;
  bool nero_repark_back_ = true;
  // NERO_FREESPACE_STALL_S (nero, parkfix 2026-10-25): created only with the switch set
  rclcpp::Subscription<autoware_adapi_v1_msgs::msg::OperationModeState>::SharedPtr nero_op_sub_;
  rclcpp::Subscription<autoware_control_msgs::msg::Control>::SharedPtr nero_cmd_sub_;
  autoware_adapi_v1_msgs::msg::OperationModeState::ConstSharedPtr nero_op_;
  autoware_control_msgs::msg::Control::ConstSharedPtr nero_cmd_;
  double nero_stall_since_ = -1.0;  // standing with a moving partial trajectory since (s)
  double nero_stall_last_t_ = -1.0;  // when the guard last saw a moving partial trajectory published (s)
  int nero_stall_recoveries_ = 0;   // since the last move
  bool nero_lead_in_now_ = false;   // the next plan starts with a straight move the vehicle can start
  double nero_start_blocked_since_ = -1.0;  // NERO_FREESPACE_START_ESCAPE hold: since when the start is blocked (s)
  int nero_lead_end_ = -1;          // index in trajectory_ where that straight move ends (-1: none)
  bool nero_lead_back_ = false;     // its direction
  int nero_unstartable_replans_ = 0;  // NERO_FREESPACE_MIN_SEGMENT: re-plans for a segment too short to start

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // params
  NodeParam node_param_;
  VehicleShape vehicle_shape_;

  // variables
  std::unique_ptr<AbstractPlanningAlgorithm> algo_;
  PoseStamped current_pose_;
  PoseStamped goal_pose_;

  Trajectory trajectory_;
  Trajectory partial_trajectory_;
  std::vector<size_t> reversing_indices_;
  size_t prev_target_index_;
  size_t target_index_;
  bool is_completed_ = false;
  bool reset_in_progress_ = false;
  bool is_new_parking_cycle_ = true;
  boost::optional<rclcpp::Time> obs_found_time_;

  LaneletRoute::ConstSharedPtr route_;
  OccupancyGrid::ConstSharedPtr occupancy_grid_;
  Scenario::ConstSharedPtr scenario_;
  Odometry::ConstSharedPtr odom_;
  std::shared_ptr<autoware::route_handler::RouteHandler> route_handler_;

  std::deque<Odometry::ConstSharedPtr> odom_buffer_;

  // functions used in the constructor
  PlannerCommonParam getPlannerCommonParam();

  // functions, callback
  void onRoute(const LaneletRoute::ConstSharedPtr msg);
  void onOdometry(const Odometry::ConstSharedPtr msg);

  void onTimer();
  void updateData();
  void reset();
  void planTrajectory();
  void initializePlanningAlgorithm();
  bool isDataReady();

  /**
   * @brief Checks if a new trajectory planning is required.
   * @details A new trajectory planning is required if:
   *           - Current trajectory points are empty, or
   *           - Current trajectory collides with an object, or
   *           - Ego deviates from current trajectory
   * @return true if any of the conditions are met.
   */
  bool isPlanRequired();

  /**
   * @brief Sets the target index along the current trajectory points
   * @details if Ego is stopped AND is near the current target index along the trajectory,
   *          then will get the next target index along the trajectory.
   *          If the new target index is the same as the current target index, then
   *          is_complete_ is set to true, and will publish is_completed_msg.
   *          Otherwise will update prev_target_index_ and target_index_, to continue
   *          following the trajectory.
   */
  void updateTargetIndex();

  // NERO_FREESPACE_PREDICTED_HOLD_S: true = do not start the current partial trajectory now
  bool neroPredictedHold();
  // NERO_FREESPACE_PARKED_BOX: the vehicle's footprint at `pose` lies within the box around `ref` (the planned end)
  bool neroParkedEnough(const geometry_msgs::msg::Pose & pose, const geometry_msgs::msg::Pose & ref, std::string & why);
  // NERO_FREESPACE_STALL_S / NERO_FREESPACE_START_ESCAPE: a straight move of `len` from `start` sweeps nothing new: the
  // real footprint's new area (cell centres) and the margin beyond it (grown half a cell) are free
  bool neroStraightFree(const geometry_msgs::msg::Pose & start, const bool back, const double len);
  // NERO_FREESPACE_STALL_S / NERO_FREESPACE_START_ESCAPE: a straight move from `start` the vehicle can start (forward
  // or reverse) whose swept area beyond the current footprint and end pose are free; false = none
  bool neroLeadIn(
    const geometry_msgs::msg::Pose & start, const bool start_blocked,
    std::vector<autoware::freespace_planning_algorithms::PlannerWaypoint> & lead, std::string & why);
  // NERO_FREESPACE_STALL_S: standing with a plan that says move -> recover (parked: done; else a lead-in plan);
  // true = recovered now (the caller publishes a stop this cycle)
  bool neroStallGuard();

  /**
   * @brief Checks if current trajectory is colliding with an object.
   * @details Will check if an obstacle exists along the current trajectory,
   *          if there is no obstacle along the current trajectory, will reset obs_found_time_.
   *          If an obstacle exists and the variable obs_found_time_ is not initialized,
   *          will initialize with the current time.
   * @return true if there is an obstacle along current trajectory, AND duration since
   *         obs_found_time_ exceeds the parameter th_obstacle_time_sec
   */
  bool checkCurrentTrajectoryCollision();

  TransformStamped getTransform(const std::string & from, const std::string & to);

  std::unique_ptr<autoware_utils::LoggerLevelConfigure> logger_configure_;

  friend class ::TestFreespacePlanner;
};
}  // namespace autoware::freespace_planner

#endif  // AUTOWARE__FREESPACE_PLANNER__FREESPACE_PLANNER_NODE_HPP_
