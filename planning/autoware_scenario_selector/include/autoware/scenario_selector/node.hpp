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

#ifndef AUTOWARE__SCENARIO_SELECTOR__NODE_HPP_
#define AUTOWARE__SCENARIO_SELECTOR__NODE_HPP_

#include <autoware_utils/ros/published_time_publisher.hpp>
#include <rclcpp/rclcpp.hpp>

#include <autoware_adapi_v1_msgs/msg/operation_mode_state.hpp>
#include <autoware_internal_debug_msgs/msg/float64_stamped.hpp>
#include <autoware_internal_planning_msgs/msg/scenario.hpp>
#include <autoware_map_msgs/msg/lanelet_map_bin.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/bool.hpp>

#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_routing/RoutingGraph.h>
#include <lanelet2_traffic_rules/TrafficRules.h>
#ifdef ROS_DISTRO_GALACTIC
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif
#include <autoware/route_handler/route_handler.hpp>
#include <autoware_utils/ros/polling_subscriber.hpp>
#include <autoware_utils/system/stop_watch.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <deque>
#include <memory>
#include <string>

namespace autoware::scenario_selector
{
class ScenarioSelectorNode : public rclcpp::Node
{
public:
  explicit ScenarioSelectorNode(const rclcpp::NodeOptions & node_options);

  void onOdom(const nav_msgs::msg::Odometry::ConstSharedPtr msg);

  bool isDataReady();
  void onTimer();
  void onMap(const autoware_map_msgs::msg::LaneletMapBin::ConstSharedPtr msg);
  void onRoute(const autoware_planning_msgs::msg::LaneletRoute::ConstSharedPtr msg);
  void onLaneDrivingTrajectory(const autoware_planning_msgs::msg::Trajectory::ConstSharedPtr msg);
  void onParkingTrajectory(const autoware_planning_msgs::msg::Trajectory::ConstSharedPtr msg);
  void publishTrajectory(const autoware_planning_msgs::msg::Trajectory::ConstSharedPtr msg);

  void updateCurrentScenario();
  std::string selectScenarioByPosition();
  autoware_planning_msgs::msg::Trajectory::ConstSharedPtr getScenarioTrajectory(
    const std::string & scenario);

  void updateData();

private:
  bool isAutonomous() const;
  bool isEmptyParkingTrajectory() const;
  bool isSwitchToLaneDriving();
  bool isSwitchToParking(const bool is_stopped);

  inline bool isCurrentLaneDriving() const
  {
    return current_scenario_ == autoware_internal_planning_msgs::msg::Scenario::LANEDRIVING;
  }

  inline bool isCurrentParking() const
  {
    return current_scenario_ == autoware_internal_planning_msgs::msg::Scenario::PARKING;
  }

  rclcpp::TimerBase::SharedPtr timer_;

  // subscribers
  rclcpp::Subscription<autoware_map_msgs::msg::LaneletMapBin>::SharedPtr sub_lanelet_map_;
  rclcpp::Subscription<autoware_planning_msgs::msg::LaneletRoute>::SharedPtr sub_route_;
  rclcpp::Subscription<autoware_planning_msgs::msg::Trajectory>::SharedPtr
    sub_lane_driving_trajectory_;
  rclcpp::Subscription<autoware_planning_msgs::msg::Trajectory>::SharedPtr sub_parking_trajectory_;
  rclcpp::Publisher<autoware_planning_msgs::msg::Trajectory>::SharedPtr pub_trajectory_;
  rclcpp::Publisher<autoware_internal_planning_msgs::msg::Scenario>::SharedPtr pub_scenario_;
  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Float64Stamped>::SharedPtr
    pub_processing_time_;

  // polling subscribers
  autoware_utils::InterProcessPollingSubscriber<
    nav_msgs::msg::Odometry, autoware_utils::polling_policy::All>::SharedPtr sub_odom_;
  autoware_utils::InterProcessPollingSubscriber<std_msgs::msg::Bool>::SharedPtr sub_parking_state_;
  autoware_utils::InterProcessPollingSubscriber<
    autoware_adapi_v1_msgs::msg::OperationModeState>::SharedPtr sub_operation_mode_state_;

  autoware_adapi_v1_msgs::msg::OperationModeState::ConstSharedPtr operation_mode_state_;
  autoware_planning_msgs::msg::Trajectory::ConstSharedPtr lane_driving_trajectory_;
  autoware_planning_msgs::msg::Trajectory::ConstSharedPtr parking_trajectory_;
  autoware_planning_msgs::msg::LaneletRoute::ConstSharedPtr route_;
  nav_msgs::msg::Odometry::ConstSharedPtr current_pose_;
  geometry_msgs::msg::TwistStamped::ConstSharedPtr twist_;

  std::string current_scenario_;
  std::deque<geometry_msgs::msg::TwistStamped::ConstSharedPtr> twist_buffer_;

  std::shared_ptr<autoware::route_handler::RouteHandler> route_handler_;
  std::unique_ptr<autoware_utils::PublishedTimePublisher> published_time_publisher_;

  // Parameters
  double update_rate_;
  double th_max_message_delay_sec_;
  double th_arrived_distance_m_;
  double th_stopped_time_sec_;
  double th_stopped_velocity_mps_;
  bool enable_mode_switching_;
  bool is_parking_completed_;

  boost::optional<rclcpp::Time> lane_driving_stop_time_;
  boost::optional<rclcpp::Time> empty_parking_trajectory_time_;

  static constexpr double lane_stopping_timeout_s = 5.0;
  static constexpr double empty_parking_trajectory_timeout_s = 3.0;

  // processing time
  autoware_utils::StopWatch<std::chrono::milliseconds> stop_watch;

  // [nero] far-stop comfort ceiling (NERO_SCSEL_FAR_STOP): a stop the comfort deceleration can still make from the
  // plan's state is approached at a constant, tapered deceleration; any other stop is the car's (unset = a relay)
  struct FarStop
  {
    bool on{false};
    double a_c{0.6};      // plateau deceleration for a stop seen far enough, at v_ref [m/s^2]
    double a_max{0.9};    // a stop needing more is left to the car as it is
    double v_taper{1.0};  // below this speed the deceleration tapers ...
    double a_end{0.35};   // ... down to this at the stop
    double j_ramp{1.0};   // jerk the plan needs to ramp into the plateau [m/s^3]
    double v_ref{0.0};    // > 0: the plateau is a_c * v / v_ref (at least a_end) at the plan's speed v
    double k_need{0.0};   // > 0: a stop first decided close gets at least k_need * v^2 / (2 d) (up to a_close)
    double n_hand{0.0};   // > 0: a stop first decided with v^2 / (2 d) above n_hand * v_ref / v keeps the car's braking
    double a_close{0.0};  // the cap of k_need's plateau (unset / out of range = a_c)
    // latch of the current stop
    bool has_stop{false};
    bool car{false};
    double a_use{0.0};
    double a_log{0.0};
    double sx{0.0}, sy{0.0};
    double seen_s{0.0};
    double rest_s{-1.0};
  } far_stop_;
  rclcpp::Subscription<autoware_planning_msgs::msg::Trajectory>::SharedPtr sub_far_stop_plan_;
  autoware_planning_msgs::msg::Trajectory::ConstSharedPtr far_stop_plan_;
  autoware_planning_msgs::msg::Trajectory::ConstSharedPtr shapeFarStop(
    const autoware_planning_msgs::msg::Trajectory::ConstSharedPtr & msg);
};
}  // namespace autoware::scenario_selector
#endif  // AUTOWARE__SCENARIO_SELECTOR__NODE_HPP_
