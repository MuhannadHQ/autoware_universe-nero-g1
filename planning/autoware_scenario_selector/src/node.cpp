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

#include "autoware/scenario_selector/node.hpp"

#include <autoware_lanelet2_extension/utility/message_conversion.hpp>
#include <autoware_lanelet2_extension/utility/query.hpp>
#include <autoware_utils/geometry/geometry.hpp>

#include <lanelet2_core/geometry/BoundingBox.h>
#include <lanelet2_core/geometry/Lanelet.h>
#include <lanelet2_core/geometry/LineString.h>
#include <lanelet2_core/geometry/Point.h>
#include <lanelet2_core/geometry/Polygon.h>

#include <tf2/utils.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
namespace autoware::scenario_selector
{
namespace
{

std::shared_ptr<lanelet::ConstPolygon3d> findNearestParkinglot(
  const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr,
  const lanelet::BasicPoint2d & current_position)
{
  const auto linked_parking_lot = std::make_shared<lanelet::ConstPolygon3d>();
  const auto result = lanelet::utils::query::getLinkedParkingLot(
    current_position, lanelet_map_ptr, linked_parking_lot.get());

  if (result) {
    return linked_parking_lot;
  } else {
    return {};
  }
}

bool isInLane(
  const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr,
  const geometry_msgs::msg::Point & current_pos)
{
  const lanelet::BasicPoint2d search_point(current_pos.x, current_pos.y);

  std::vector<std::pair<double, lanelet::Lanelet>> nearest_lanelets =
    lanelet::geometry::findNearest(lanelet_map_ptr->laneletLayer, search_point, 1);

  if (nearest_lanelets.empty()) return false;

  const auto dist_to_nearest_lanelet = nearest_lanelets.front().first;
  static constexpr double margin = 0.01;

  return dist_to_nearest_lanelet < margin;
}

bool isAlongLane(
  const std::shared_ptr<autoware::route_handler::RouteHandler> & route_handler,
  const geometry_msgs::msg::Pose & current_pose)
{
  lanelet::ConstLanelet closest_lanelet;
  if (!route_handler->getClosestLaneletWithConstrainsWithinRoute(
        current_pose, &closest_lanelet, 0.0, M_PI_4)) {
    return false;
  }
  const lanelet::BasicPoint2d src_point(current_pose.position.x, current_pose.position.y);
  const auto dist_to_centerline =
    lanelet::geometry::distanceToCenterline2d(closest_lanelet, src_point);
  static constexpr double margin = 1.0;
  return dist_to_centerline < margin;
}

bool isInParkingLot(
  const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr,
  const geometry_msgs::msg::Pose & current_pose)
{
  const auto & p = current_pose.position;
  const lanelet::Point3d search_point(lanelet::InvalId, p.x, p.y, p.z);

  const auto nearest_parking_lot =
    findNearestParkinglot(lanelet_map_ptr, search_point.basicPoint2d());

  if (!nearest_parking_lot) {
    return false;
  }

  return lanelet::geometry::within(search_point, nearest_parking_lot->basicPolygon());
}

bool isNearTrajectoryEnd(
  const autoware_planning_msgs::msg::Trajectory::ConstSharedPtr trajectory,
  const geometry_msgs::msg::Pose & current_pose, const double th_dist)
{
  if (!trajectory || trajectory->points.empty()) {
    return false;
  }

  const auto & p1 = current_pose.position;
  const auto & p2 = trajectory->points.back().pose.position;

  const auto dist = std::hypot(p1.x - p2.x, p1.y - p2.y);

  return dist < th_dist;
}

bool isStopped(
  const std::deque<geometry_msgs::msg::TwistStamped::ConstSharedPtr> & twist_buffer,
  const double th_stopped_velocity_mps)
{
  for (const auto & twist : twist_buffer) {
    if (std::abs(twist->twist.linear.x) > th_stopped_velocity_mps) {
      return false;
    }
  }
  return true;
}

// [nero] far-stop comfort ceiling: the speed profile of a stop braked at a_use, tapering linearly in speed below
// v_taper to a_end at the stop: v dv/dd = a_end + k v (k = (a_use - a_end) / v_taper)
double farStopTaperDist(const double v, const double a_use, const double v_taper, const double a_end)
{
  const double k = (a_use - a_end) / v_taper;
  if (k < 1e-6) return v * v / (2.0 * a_use);
  return v / k - a_end / (k * k) * std::log1p(k * v / a_end);
}

// distance the profile needs to stop from v
double farStopDist(const double v, const double a_use, const double v_taper, const double a_end)
{
  if (v <= v_taper) return farStopTaperDist(v, a_use, v_taper, a_end);
  return farStopTaperDist(v_taper, a_use, v_taper, a_end) +
         (v * v - v_taper * v_taper) / (2.0 * a_use);
}

// the profile's speed d before the stop
double farStopCeiling(const double d, const double a_use, const double v_taper, const double a_end)
{
  if (d <= 0.0) return 0.0;
  const double d_t = farStopTaperDist(v_taper, a_use, v_taper, a_end);
  if (d >= d_t) return std::sqrt(v_taper * v_taper + 2.0 * a_use * (d - d_t));
  double lo = 0.0, hi = v_taper;
  for (int i = 0; i < 30; ++i) {
    const double mid = 0.5 * (lo + hi);
    (farStopTaperDist(mid, a_use, v_taper, a_end) < d ? lo : hi) = mid;
  }
  return lo;
}

// distance the plan (v, a0) needs to stop: a jerk-limited ramp from a0 into -a_use, then the profile
double farStopNeed(
  const double v, const double a0, const double a_use, const double v_taper, const double a_end,
  const double j_ramp)
{
  constexpr double dt = 0.02;
  double s = 0.0, vv = v, a = a0;
  while (a > -a_use && vv > 0.0) {
    a = std::max(a - j_ramp * dt, -a_use);
    s += vv * dt;
    vv += a * dt;
  }
  return s + farStopDist(std::max(vv, 0.0), a_use, v_taper, a_end);
}

}  // namespace

autoware_planning_msgs::msg::Trajectory::ConstSharedPtr ScenarioSelectorNode::getScenarioTrajectory(
  const std::string & scenario)
{
  if (scenario == autoware_internal_planning_msgs::msg::Scenario::LANEDRIVING) {
    return lane_driving_trajectory_;
  }
  if (scenario == autoware_internal_planning_msgs::msg::Scenario::PARKING) {
    return parking_trajectory_;
  }
  RCLCPP_ERROR_STREAM(this->get_logger(), "invalid scenario argument: " << scenario);
  return lane_driving_trajectory_;
}

std::string ScenarioSelectorNode::selectScenarioByPosition()
{
  const auto is_in_lane =
    isInLane(route_handler_->getLaneletMapPtr(), current_pose_->pose.pose.position);
  const auto is_goal_in_lane =
    isInLane(route_handler_->getLaneletMapPtr(), route_->goal_pose.position);
  const auto is_in_parking_lot =
    isInParkingLot(route_handler_->getLaneletMapPtr(), current_pose_->pose.pose);

  if (current_scenario_ == autoware_internal_planning_msgs::msg::Scenario::EMPTY) {
    if (is_in_lane && is_goal_in_lane) {
      return autoware_internal_planning_msgs::msg::Scenario::LANEDRIVING;
    } else if (is_in_parking_lot) {
      return autoware_internal_planning_msgs::msg::Scenario::PARKING;
    }
    return autoware_internal_planning_msgs::msg::Scenario::LANEDRIVING;
  }

  if (current_scenario_ == autoware_internal_planning_msgs::msg::Scenario::LANEDRIVING) {
    if (is_in_parking_lot && !is_goal_in_lane) {
      return autoware_internal_planning_msgs::msg::Scenario::PARKING;
    }
  }

  if (current_scenario_ == autoware_internal_planning_msgs::msg::Scenario::PARKING) {
    if (is_parking_completed_ && is_in_lane) {
      is_parking_completed_ = false;
      return autoware_internal_planning_msgs::msg::Scenario::LANEDRIVING;
    }
  }

  return current_scenario_;
}

void ScenarioSelectorNode::updateCurrentScenario()
{
  const auto prev_scenario = current_scenario_;

  const auto scenario_trajectory = getScenarioTrajectory(current_scenario_);
  const auto is_near_trajectory_end =
    isNearTrajectoryEnd(scenario_trajectory, current_pose_->pose.pose, th_arrived_distance_m_);

  const auto is_stopped = isStopped(twist_buffer_, th_stopped_velocity_mps_);

  if (is_near_trajectory_end && is_stopped) {
    current_scenario_ = selectScenarioByPosition();
  }

  if (enable_mode_switching_) {
    if (isCurrentLaneDriving()) {
      current_scenario_ = isSwitchToParking(is_stopped)
                            ? autoware_internal_planning_msgs::msg::Scenario::PARKING
                            : current_scenario_;
    } else if (isCurrentParking()) {
      current_scenario_ = isSwitchToLaneDriving()
                            ? autoware_internal_planning_msgs::msg::Scenario::LANEDRIVING
                            : current_scenario_;
    }
  }

  if (current_scenario_ != prev_scenario) {
    lane_driving_stop_time_ = {};
    empty_parking_trajectory_time_ = {};
    RCLCPP_DEBUG_STREAM(
      this->get_logger(), "scenario changed: " << prev_scenario << " -> " << current_scenario_);
  }
}

bool ScenarioSelectorNode::isSwitchToParking(const bool is_stopped)
{
  const auto is_in_parking_lot =
    isInParkingLot(route_handler_->getLaneletMapPtr(), current_pose_->pose.pose);
  const auto is_goal_in_lane =
    isInLane(route_handler_->getLaneletMapPtr(), route_->goal_pose.position);

  if (!is_stopped || !isAutonomous() || !is_in_parking_lot || is_goal_in_lane) {
    lane_driving_stop_time_ = {};
    return false;
  }

  if (!lane_driving_stop_time_) {
    lane_driving_stop_time_ = this->now();
    return false;
  }

  return (this->now() - lane_driving_stop_time_.get()).seconds() > lane_stopping_timeout_s;
}

bool ScenarioSelectorNode::isSwitchToLaneDriving()
{
  const auto is_along_lane = isAlongLane(route_handler_, current_pose_->pose.pose);

  if (!isEmptyParkingTrajectory() || !is_along_lane) {
    empty_parking_trajectory_time_ = {};
    return false;
  }

  if (!empty_parking_trajectory_time_) {
    empty_parking_trajectory_time_ = this->now();
    return false;
  }

  const auto duration = (this->now() - empty_parking_trajectory_time_.get()).seconds();

  return duration > empty_parking_trajectory_timeout_s;
}

bool ScenarioSelectorNode::isAutonomous() const
{
  return operation_mode_state_->mode ==
           autoware_adapi_v1_msgs::msg::OperationModeState::AUTONOMOUS &&
         operation_mode_state_->is_autoware_control_enabled;
}

bool ScenarioSelectorNode::isEmptyParkingTrajectory() const
{
  if (parking_trajectory_) {
    return parking_trajectory_->points.size() <= 1;
  }
  return false;
}

void ScenarioSelectorNode::onMap(const autoware_map_msgs::msg::LaneletMapBin::ConstSharedPtr msg)
{
  route_handler_ = std::make_shared<autoware::route_handler::RouteHandler>(*msg);
}

void ScenarioSelectorNode::onRoute(
  const autoware_planning_msgs::msg::LaneletRoute::ConstSharedPtr msg)
{
  // When the route id is the same (e.g. rerouting with modified goal) keep the current scenario.
  // Otherwise, reset the scenario.
  if (!route_handler_ || route_handler_->getRouteUuid() != msg->uuid) {
    current_scenario_ = autoware_internal_planning_msgs::msg::Scenario::EMPTY;
  }

  route_ = msg;
}

void ScenarioSelectorNode::onOdom(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
  current_pose_ = msg;
  auto twist = std::make_shared<geometry_msgs::msg::TwistStamped>();
  twist->header = msg->header;
  twist->twist = msg->twist.twist;

  twist_ = twist;
  twist_buffer_.push_back(twist);

  // Delete old data in buffer
  while (true) {
    const auto time_diff =
      rclcpp::Time(msg->header.stamp) - rclcpp::Time(twist_buffer_.front()->header.stamp);

    if (time_diff.seconds() < th_stopped_time_sec_) {
      break;
    }

    twist_buffer_.pop_front();
  }
}

bool ScenarioSelectorNode::isDataReady()
{
  if (!current_pose_) {
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for current pose.");
    return false;
  }

  if (!route_handler_) {
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000, "Waiting for route handler.");
    return false;
  }

  if (!route_) {
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for route.");
    return false;
  }

  if (!twist_) {
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for twist.");
    return false;
  }

  if (!operation_mode_state_) {
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000, "Waiting for operation mode state.");
    return false;
  }

  // Check route handler is ready
  route_handler_->setRoute(*route_);
  if (!route_handler_->isHandlerReady()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000, "Waiting for route handler.");
    return false;
  }

  return true;
}

void ScenarioSelectorNode::updateData()
{
  {
    stop_watch.tic();
  }
  {
    auto msg = sub_parking_state_->take_data();
    is_parking_completed_ = msg ? msg->data : is_parking_completed_;
  }

  {
    auto msgs = sub_odom_->take_data();
    for (const auto & msg : msgs) {
      onOdom(msg);
    }
  }

  {
    auto msg = sub_operation_mode_state_->take_data();
    if (msg) operation_mode_state_ = msg;
  }
}

void ScenarioSelectorNode::onTimer()
{
  updateData();

  if (!isDataReady()) {
    return;
  }

  // Initialize Scenario
  if (current_scenario_ == autoware_internal_planning_msgs::msg::Scenario::EMPTY) {
    current_scenario_ = selectScenarioByPosition();
  }

  updateCurrentScenario();
  autoware_internal_planning_msgs::msg::Scenario scenario;
  scenario.current_scenario = current_scenario_;

  if (current_scenario_ == autoware_internal_planning_msgs::msg::Scenario::PARKING) {
    scenario.activating_scenarios.push_back(current_scenario_);
  }

  pub_scenario_->publish(scenario);

  // Publish ProcessingTime
  autoware_internal_debug_msgs::msg::Float64Stamped processing_time_msg;
  processing_time_msg.stamp = get_clock()->now();
  processing_time_msg.data = stop_watch.toc();
  pub_processing_time_->publish(processing_time_msg);
}

void ScenarioSelectorNode::onLaneDrivingTrajectory(
  const autoware_planning_msgs::msg::Trajectory::ConstSharedPtr msg)
{
  lane_driving_trajectory_ = msg;

  if (current_scenario_ != autoware_internal_planning_msgs::msg::Scenario::LANEDRIVING) {
    return;
  }

  publishTrajectory(far_stop_.on ? shapeFarStop(msg) : msg);
}

void ScenarioSelectorNode::onParkingTrajectory(
  const autoware_planning_msgs::msg::Trajectory::ConstSharedPtr msg)
{
  parking_trajectory_ = msg;

  if (current_scenario_ != autoware_internal_planning_msgs::msg::Scenario::PARKING) {
    return;
  }

  publishTrajectory(msg);
}

autoware_planning_msgs::msg::Trajectory::ConstSharedPtr ScenarioSelectorNode::shapeFarStop(
  const autoware_planning_msgs::msg::Trajectory::ConstSharedPtr & msg)
{
  auto & f = far_stop_;
  const auto & pts = msg->points;
  if (!current_pose_ || pts.size() < 2) return msg;
  const double now_s = this->now().seconds();
  const auto & ego = current_pose_->pose.pose;
  const double ego_yaw = tf2::getYaw(ego.orientation);
  const double v_odom = current_pose_->twist.twist.linear.x;

  // ego on the trajectory: the nearest segment heading the same way
  std::vector<double> s(pts.size(), 0.0);
  for (size_t i = 1; i < pts.size(); ++i) {
    s[i] = s[i - 1] + autoware_utils::calc_distance2d(pts[i - 1], pts[i]);
  }
  size_t k = 0;
  double s_ego = 0.0, best = std::numeric_limits<double>::max();
  for (size_t i = 0; i + 1 < pts.size(); ++i) {
    const auto & p = pts[i].pose.position;
    const auto & q = pts[i + 1].pose.position;
    const double dx = q.x - p.x, dy = q.y - p.y, l2 = dx * dx + dy * dy;
    if (l2 < 1e-9 || std::cos(std::atan2(dy, dx) - ego_yaw) < 0.0) continue;
    const double t =
      std::clamp(((ego.position.x - p.x) * dx + (ego.position.y - p.y) * dy) / l2, 0.0, 1.0);
    const double e = std::hypot(p.x + t * dx - ego.position.x, p.y + t * dy - ego.position.y);
    if (e < best) {
      best = e;
      k = i;
      s_ego = s[i] + t * (s[i + 1] - s[i]);
    }
  }
  if (best > 3.0) return msg;

  // the first stop ahead of ego
  size_t j = k + 1;
  while (j < pts.size() && pts[j].longitudinal_velocity_mps >= 0.01) ++j;
  if (j >= pts.size()) {
    if (f.has_stop && now_s - f.seen_s > 1.5) f.has_stop = false;
    return msg;
  }
  const double d = s[j] - s_ego;
  const auto & sp = pts[j].pose.position;

  // the same stop as the last cycle (within 2 m) keeps its decision; ego resting 1 s re-decides
  if (!f.has_stop || std::hypot(sp.x - f.sx, sp.y - f.sy) > 2.0 || now_s - f.seen_s > 1.5) {
    f.has_stop = true;
    f.car = false;
    f.a_use = 0.0;
  }
  f.sx = sp.x;
  f.sy = sp.y;
  f.seen_s = now_s;
  if (std::abs(v_odom) < 0.05) {
    if (f.rest_s < 0.0) f.rest_s = now_s;
    if (now_s - f.rest_s > 1.0) {
      f.car = false;
      f.a_use = 0.0;
      f.rest_s = std::numeric_limits<double>::max();  // once per rest
    }
  } else {
    f.rest_s = -1.0;
  }
  if (f.car) return msg;

  // the plan's own state at ego (the smoother plans open loop from it); the robot's speed if there is none
  double v_e = v_odom, a_e = 0.0;
  if (
    far_stop_plan_ && !far_stop_plan_->points.empty() &&
    (this->now() - rclcpp::Time(far_stop_plan_->header.stamp)).seconds() < 1.0) {
    double e_min = std::numeric_limits<double>::max();
    for (const auto & p : far_stop_plan_->points) {
      const double e = autoware_utils::calc_distance2d(p.pose.position, ego.position);
      if (e < e_min) {
        e_min = e;
        v_e = p.longitudinal_velocity_mps;
        a_e = p.acceleration_mps2;
      }
    }
    if (e_min > 3.0) {
      v_e = v_odom;
      a_e = 0.0;
    }
  }
  v_e = std::max(v_e, 0.0);

  // the comfort plateau grows with the plan's speed (a_c at v_ref: about the car's own stopping time at every speed)
  const double a_lo =
    f.v_ref > 0.0 ? std::clamp(f.a_c * std::min(v_e, f.v_ref) / f.v_ref, f.a_end, f.a_c) : f.a_c;
  // the gentlest plateau in [lo0, a_max] that makes the stop from the plan's state; none = the car's braking
  const auto need = [&](const double lo0) {
    const auto dist = [&](const double a) {
      return farStopNeed(v_e, a_e, a, f.v_taper, f.a_end, f.j_ramp);
    };
    if (dist(lo0) <= d) return lo0;
    if (dist(f.a_max) > d) return f.a_max + 1.0;
    double lo = lo0, hi = f.a_max;
    for (int i = 0; i < 20; ++i) {
      const double mid = 0.5 * (lo + hi);
      (dist(mid) <= d ? hi : lo) = mid;
    }
    return hi;
  };
  const bool decide = f.a_use <= 0.0;
  if (decide || v_e > farStopCeiling(d, f.a_use, f.v_taper, f.a_end) + 0.15) {
    // a stop that is already close when it is first decided (an object appearing, not a stop line known from far):
    // at least k_need times its constant-deceleration need, up to a_close (the car's own plan is firm and quick there;
    // a plateau that follows only the speed would cost more time than the car's stop, a higher one would brake later)
    const double n0 = v_e * v_e / (2.0 * std::max(d, 0.1));
    const double a_first = decide && f.k_need > 0.0 ? std::min(f.k_need * n0, f.a_close) : 0.0;
    // a stop first decided with a constant-deceleration need above n_hand (at v_ref; the time a gentler profile costs
    // grows with the speed, so the limit is n_hand * v_ref / v): no profile within the comfort limits stops there in
    // about the car's time (the car's own stop is firm and quick) -> the car's braking, unchanged
    const double n_lim = f.v_ref > 0.0 ? f.n_hand * f.v_ref / std::max(v_e, 0.1) : f.n_hand;
    const double a = decide && f.n_hand > 0.0 && n0 > n_lim ? f.a_max + 1.0 : need(std::max(a_lo, a_first));
    if (a > f.a_max) {
      f.car = true;
      RCLCPP_INFO(
        get_logger(), "[nero far-stop] stop %.1f m ahead (%.1f, %.1f), plan v %.2f a %.2f: the car's braking",
        d, sp.x, sp.y, v_e, a_e);
      return msg;
    }
    f.a_use = std::max(f.a_use, a);
    f.a_log = f.a_use;
    RCLCPP_INFO(
      get_logger(), "[nero far-stop] stop %.1f m ahead (%.1f, %.1f), plan v %.2f a %.2f: %s %.2f", d, sp.x,
      sp.y, v_e, a_e, decide ? "comfort" : "raised to", f.a_use);
  } else if (a_lo > f.a_use) {
    // the plan sped up toward the stop (a launch): its plateau follows (never lower for the same stop)
    f.a_use = a_lo;
    if (f.a_use > f.a_log + 0.05) {
      f.a_log = f.a_use;
      RCLCPP_INFO(
        get_logger(), "[nero far-stop] stop %.1f m ahead, plan v %.2f: comfort %.2f", d, v_e, f.a_use);
    }
  }

  auto out = std::make_shared<autoware_planning_msgs::msg::Trajectory>(*msg);
  for (size_t i = 0; i < j; ++i) {
    auto & v = out->points[i].longitudinal_velocity_mps;
    if (v <= 0.0) continue;
    v = std::min(
      v, static_cast<float>(farStopCeiling(s[j] - s[i], f.a_use, f.v_taper, f.a_end)));
  }
  return out;
}

void ScenarioSelectorNode::publishTrajectory(
  const autoware_planning_msgs::msg::Trajectory::ConstSharedPtr msg)
{
  const auto now = this->now();
  const auto delay_sec = (now - msg->header.stamp).seconds();
  if (delay_sec <= th_max_message_delay_sec_) {
    pub_trajectory_->publish(*msg);
    published_time_publisher_->publish_if_subscribed(pub_trajectory_, msg->header.stamp);
  } else {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), std::chrono::milliseconds(1000).count(),
      "trajectory is delayed: scenario = %s, delay = %f, th_max_message_delay = %f",
      current_scenario_.c_str(), delay_sec, th_max_message_delay_sec_);
  }
}

ScenarioSelectorNode::ScenarioSelectorNode(const rclcpp::NodeOptions & node_options)
: Node("scenario_selector", node_options),
  current_scenario_(autoware_internal_planning_msgs::msg::Scenario::EMPTY),
  update_rate_(this->declare_parameter<double>("update_rate")),
  th_max_message_delay_sec_(this->declare_parameter<double>("th_max_message_delay_sec")),
  th_arrived_distance_m_(this->declare_parameter<double>("th_arrived_distance_m")),
  th_stopped_time_sec_(this->declare_parameter<double>("th_stopped_time_sec")),
  th_stopped_velocity_mps_(this->declare_parameter<double>("th_stopped_velocity_mps")),
  enable_mode_switching_(this->declare_parameter<bool>("enable_mode_switching")),
  is_parking_completed_(false)
{
  lane_driving_stop_time_ = {};
  empty_parking_trajectory_time_ = {};

  // [nero] far-stop comfort ceiling:
  //   NERO_SCSEL_FAR_STOP=<a_c>,<a_max>,<v_taper>,<a_end>[,<j_ramp>[,<v_ref>[,<k_need>[,<n_hand>[,<a_close>]]]]]
  if (const char * e = std::getenv("NERO_SCSEL_FAR_STOP"); e && *e) {
    auto & f = far_stop_;
    f.a_close = -1.0;
    const int n = std::sscanf(
      e, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &f.a_c, &f.a_max, &f.v_taper, &f.a_end, &f.j_ramp, &f.v_ref,
      &f.k_need, &f.n_hand, &f.a_close);
    if (f.a_close <= 0.0 || f.a_close > f.a_c) f.a_close = f.a_c;
    f.on = n >= 1 && f.a_c > 0.0 && f.a_max >= f.a_c && f.v_taper > 0.0 && f.a_end > 0.0 &&
           f.a_end <= f.a_c && f.j_ramp > 0.0;
    if (f.on) {
      sub_far_stop_plan_ = this->create_subscription<autoware_planning_msgs::msg::Trajectory>(
        "/planning/scenario_planning/trajectory", rclcpp::QoS{1},
        [this](const autoware_planning_msgs::msg::Trajectory::ConstSharedPtr msg) {
          far_stop_plan_ = msg;
        });
    }
    RCLCPP_INFO(
      get_logger(),
      "[nero far-stop] %s: a_c %.2f a_max %.2f v_taper %.2f a_end %.2f j_ramp %.2f v_ref %.2f k_need %.2f "
      "n_hand %.2f a_close %.2f",
      f.on ? "on" : "off (bad value)", f.a_c, f.a_max, f.v_taper, f.a_end, f.j_ramp, f.v_ref, f.k_need, f.n_hand,
      f.a_close);
  }

  // Input
  sub_lane_driving_trajectory_ = this->create_subscription<autoware_planning_msgs::msg::Trajectory>(
    "input/lane_driving/trajectory", rclcpp::QoS{1},
    std::bind(&ScenarioSelectorNode::onLaneDrivingTrajectory, this, std::placeholders::_1));

  sub_parking_trajectory_ = this->create_subscription<autoware_planning_msgs::msg::Trajectory>(
    "input/parking/trajectory", rclcpp::QoS{1},
    std::bind(&ScenarioSelectorNode::onParkingTrajectory, this, std::placeholders::_1));

  sub_lanelet_map_ = this->create_subscription<autoware_map_msgs::msg::LaneletMapBin>(
    "input/lanelet_map", rclcpp::QoS{1}.transient_local(),
    std::bind(&ScenarioSelectorNode::onMap, this, std::placeholders::_1));

  sub_route_ = this->create_subscription<autoware_planning_msgs::msg::LaneletRoute>(
    "input/route", rclcpp::QoS{1}.transient_local(),
    std::bind(&ScenarioSelectorNode::onRoute, this, std::placeholders::_1));

  sub_odom_ = decltype(sub_odom_)::element_type::create_subscription(
    this, "input/odometry", rclcpp::QoS{100});

  sub_parking_state_ = decltype(sub_parking_state_)::element_type::create_subscription(
    this, "is_parking_completed", rclcpp::QoS{1});

  sub_operation_mode_state_ =
    decltype(sub_operation_mode_state_)::element_type::create_subscription(
      this, "input/operation_mode_state", rclcpp::QoS{1});

  // Output
  pub_scenario_ = this->create_publisher<autoware_internal_planning_msgs::msg::Scenario>(
    "output/scenario", rclcpp::QoS{1});
  pub_trajectory_ = this->create_publisher<autoware_planning_msgs::msg::Trajectory>(
    "output/trajectory", rclcpp::QoS{1});

  // Timer Callback
  const auto period_ns = rclcpp::Rate(static_cast<double>(update_rate_)).period();

  timer_ = rclcpp::create_timer(
    this, get_clock(), period_ns, std::bind(&ScenarioSelectorNode::onTimer, this));
  published_time_publisher_ = std::make_unique<autoware_utils::PublishedTimePublisher>(this);
  pub_processing_time_ = this->create_publisher<autoware_internal_debug_msgs::msg::Float64Stamped>(
    "~/debug/processing_time_ms", 1);
}
}  // namespace autoware::scenario_selector

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::scenario_selector::ScenarioSelectorNode)
