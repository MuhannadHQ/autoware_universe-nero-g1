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
 * Copyright 2015-2019 Autoware Foundation. All rights reserved.
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

#include "autoware/freespace_planner/freespace_planner_node.hpp"

#include "autoware/freespace_planner/utils.hpp"
#include "autoware/freespace_planning_algorithms/abstract_algorithm.hpp"

#include <autoware/motion_utils/trajectory/trajectory.hpp>
#include <autoware_utils/geometry/geometry.hpp>
#include <autoware_utils/geometry/pose_deviation.hpp>
#include <autoware_utils/system/stop_watch.hpp>

#include <tf2/utils.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace autoware::freespace_planner
{
namespace
{
// NERO_FREESPACE_STRAIGHT_IN=<m> (nero, 2026-10-24): drive the last metres into the goal straight
// along the goal's heading. The controller stops 0.2-0.3 m short of a stop point (pid smooth stop
// from 0.5 m) and A* may reach the goal on a curve or an S (its goal tolerance is +-3 deg / +-0.25
// m); in the parking sims that left the vehicle 5-21 deg crooked in the spot. With the switch, A*
// plans to the pose d m out of the spot (in front of it for a back-in, behind it for a nose-in: the
// side the upstream plan arrives from) for d = <m>, 0.75 <m>, 0.5 <m>, 0.33 <m> (a short street
// leaves room only for a short straight piece), and a straight piece on the goal's axis is appended
// (collision-checked). The first d whose plan keeps the final direction and has no more direction
// changes than the upstream plan is used; if none, the upstream plan. Unset = upstream.
double nero_straight_in_m()
{
  const char * v = std::getenv("NERO_FREESPACE_STRAIGHT_IN");
  return v ? std::max(0.0, std::atof(v)) : 0.0;
}

size_t nero_direction_changes(const autoware::freespace_planning_algorithms::PlannerWaypoints & w)
{
  size_t n = 0;
  for (size_t i = 1; i < w.waypoints.size(); ++i) {
    n += w.waypoints.at(i).is_back != w.waypoints.at(i - 1).is_back;
  }
  return n;
}

// NERO_FREESPACE_STRAIGHT_IN_STOP=1 (nero, 2026-10-24, with NERO_FREESPACE_STRAIGHT_IN): end the
// segment before the straight piece just inside it, so the vehicle stops where the straight piece
// begins and then drives a partial trajectory that is only straight. The MPC computes the
// feed-forward curvature over +-1.5 m (curvature_smoothing_num_ref_steer 15 at 0.1 m), longer than
// the straight piece: on one segment it kept steering through the whole straight piece and the
// vehicle stopped 2.4-5.7 deg off the spot's heading (parking sims at the real place, 2026-10-24,
// lateral diagnostic). On a straight-only segment the feed-forward is zero. Unset = one segment.
bool nero_straight_in_stop()
{
  static const bool on = [] {
    const char * v = std::getenv("NERO_FREESPACE_STRAIGHT_IN_STOP");
    return v && std::atoi(v) == 1;
  }();
  return on;
}

// NERO_FREESPACE_STRAIGHT_IN_WAIT_S=<s> (nero, 2026-10-24, with NERO_FREESPACE_STRAIGHT_IN): when
// the straight approach is only blocked (a pose before the goal or the straight piece is occupied,
// e.g. by a person still walking off) and not impossible, wait and re-plan up to <s> seconds
// instead of taking the A* plan without the straight piece (after a wait for a person the vehicle
// ended 4-12 deg crooked, parking sims 2026-10-24). Unset = no wait.
double nero_straight_in_wait_s()
{
  const char * v = std::getenv("NERO_FREESPACE_STRAIGHT_IN_WAIT_S");
  return v ? std::clamp(std::atof(v), 0.0, 60.0) : 0.0;
}

// NERO_FREESPACE_REPARK_DEG=<deg> (nero, 2026-10-24; upstream autoware_universe #11170's
// re-parking idea): when the vehicle has finished but stands more than <deg> off the heading of the
// pose it planned to, plan once more from there: to the pose before the goal in any direction, then
// the straight piece (NERO_FREESPACE_STRAIGHT_IN) into the goal. One attempt. Unset = upstream.
double nero_repark_deg()
{
  const char * v = std::getenv("NERO_FREESPACE_REPARK_DEG");
  return v ? std::clamp(std::atof(v), 0.0, 45.0) : 0.0;
}

// NERO_FREESPACE_PREDICTED_HOLD_S=<s> (nero, 2026-10-24): in Parking only things already on the
// path stop the vehicle (costmap). Before it starts a partial trajectory (standing), the vehicle
// now also waits while a moving road user (>= 0.5 m/s, any class) on its predicted path within <s>
// seconds (footprint + 0.3 m, as costmap_generator expands objects) would overlap the partial
// trajectory's footprints (+ half of vehicle_shape_margin_m) and the vehicle's current footprint is
// clear of it: it never starts a move into an approaching car's or person's way; when it is already
// in the way, it goes on (moving out). Once moving it finishes the segment; the costmap check still
// stops it for anything on the path. Parking sims 2026-10-24: the nose swung into lane 42942 in
// front of an oncoming car, whose driver had to stop. Unset = upstream (no subscription).
double nero_predicted_hold_s()
{
  const char * v = std::getenv("NERO_FREESPACE_PREDICTED_HOLD_S");
  return v ? std::clamp(std::atof(v), 0.0, 10.0) : 0.0;
}

using NeroQuad = std::array<std::array<double, 2>, 4>;

NeroQuad nero_quad(
  const geometry_msgs::msg::Pose & p, const double front, const double rear, const double half_w)
{
  const double yaw = tf2::getYaw(p.orientation);
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  NeroQuad q;
  const double corners[4][2] = {{front, half_w}, {-rear, half_w}, {-rear, -half_w}, {front, -half_w}};
  for (int i = 0; i < 4; ++i) {
    q[i] = {
      p.position.x + c * corners[i][0] - s * corners[i][1],
      p.position.y + s * corners[i][0] + c * corners[i][1]};
  }
  return q;
}

bool nero_overlap(const NeroQuad & a, const NeroQuad & b)
{
  for (const NeroQuad * q : {&a, &b}) {
    for (int i = 0; i < 4; ++i) {
      const auto & p0 = (*q)[i];
      const auto & p1 = (*q)[(i + 1) % 4];
      const double nx = p1[1] - p0[1];
      const double ny = p0[0] - p1[0];
      double a_min = 1e18, a_max = -1e18, b_min = 1e18, b_max = -1e18;
      for (int k = 0; k < 4; ++k) {
        const double pa = a[k][0] * nx + a[k][1] * ny;
        const double pb = b[k][0] * nx + b[k][1] * ny;
        a_min = std::min(a_min, pa);
        a_max = std::max(a_max, pa);
        b_min = std::min(b_min, pb);
        b_max = std::max(b_max, pb);
      }
      if (a_max < b_min || b_max < a_min) return false;
    }
  }
  return true;
}

// NERO_FREESPACE_HALF_CELL=1 (nero, 2026-10-24): A* takes cell i of its map as centred at i * res
// (pose2index rounds, index2pose = i * res), an OccupancyGrid cell i spans [i * res, (i + 1) * res)
// from the origin. So A* saw every occupied cell half a cell (0.15 m) towards -x and -y: backing
// towards -x its footprint with margin was accepted 0.075 m inside an occupied cell, driving
// forwards towards +x it stopped 0.175 m short of one (tools/plan_check.py, 2026-10-24). With the
// switch, A* gets the grid with its origin moved half a cell along the grid's axes, so its rounding
// picks the cell a point lies in. Unset = upstream.
bool nero_half_cell()
{
  static const bool on = [] {
    const char * v = std::getenv("NERO_FREESPACE_HALF_CELL");
    return v && std::atoi(v) == 1;
  }();
  return on;
}

nav_msgs::msg::OccupancyGrid nero_half_cell_grid(const nav_msgs::msg::OccupancyGrid & grid)
{
  auto out = grid;
  const double h = grid.info.resolution / 2.0;
  const double yaw = tf2::getYaw(grid.info.origin.orientation);
  out.info.origin.position.x += h * std::cos(yaw) - h * std::sin(yaw);
  out.info.origin.position.y += h * std::sin(yaw) + h * std::cos(yaw);
  return out;
}

// NERO_FREESPACE_GOAL_SLIDE_M=<m> (nero, 2026-10-24): when the goal itself is blocked (A* would
// throw "Invalid start or goal pose" and the vehicle would stand on the lane, retrying), plan to
// the deepest free pose on the goal's axis, moved in 5 cm steps up to <m> towards the side the
// vehicle is on (for a parking spot: out of the spot, towards the street). Things behind a spot
// (bins, a step, a parked bicycle) then leave the vehicle parked as deep as the planner's margin
// allows instead of not parking. The route goal (arrival check, 1 m) stays as it is. Unset = upstream.
double nero_goal_slide_m()
{
  const char * v = std::getenv("NERO_FREESPACE_GOAL_SLIDE_M");
  return v ? std::clamp(std::atof(v), 0.0, 1.0) : 0.0;
}
}  // namespace

FreespacePlannerNode::FreespacePlannerNode(const rclcpp::NodeOptions & node_options)
: Node("freespace_planner", node_options)
{
  using std::placeholders::_1;

  // NodeParam
  {
    auto & p = node_param_;
    p.planning_algorithm = declare_parameter<std::string>("planning_algorithm");
    p.waypoints_velocity = declare_parameter<double>("waypoints_velocity");
    p.update_rate = declare_parameter<double>("update_rate");
    p.th_arrived_distance_m = declare_parameter<double>("th_arrived_distance_m");
    p.th_stopped_time_sec = declare_parameter<double>("th_stopped_time_sec");
    p.th_stopped_velocity_mps = declare_parameter<double>("th_stopped_velocity_mps");
    p.th_course_out_distance_m = declare_parameter<double>("th_course_out_distance_m");
    p.th_obstacle_time_sec = declare_parameter<double>("th_obstacle_time_sec");
    p.vehicle_shape_margin_m = declare_parameter<double>("vehicle_shape_margin_m");
    p.replan_when_obstacle_found = declare_parameter<bool>("replan_when_obstacle_found");
    p.replan_when_course_out = declare_parameter<bool>("replan_when_course_out");
  }

  // set vehicle_info
  {
    const auto vehicle_info =
      autoware::vehicle_info_utils::VehicleInfoUtils(*this).getVehicleInfo();
    vehicle_shape_.length = vehicle_info.vehicle_length_m;
    vehicle_shape_.width = vehicle_info.vehicle_width_m;
    vehicle_shape_.base_length = vehicle_info.wheel_base_m;
    vehicle_shape_.max_steering = vehicle_info.max_steer_angle_rad;
    vehicle_shape_.base2back = vehicle_info.rear_overhang_m;
  }

  // Planning
  initializePlanningAlgorithm();

  // Subscribers
  if (nero_predicted_hold_s() > 0.0) {
    nero_objects_sub_ = create_subscription<autoware_perception_msgs::msg::PredictedObjects>(
      "/perception/object_recognition/objects", rclcpp::QoS{1},
      [this](const autoware_perception_msgs::msg::PredictedObjects::ConstSharedPtr msg) {
        nero_objects_ = msg;
      });
  }
  route_sub_ = create_subscription<LaneletRoute>(
    "~/input/route", rclcpp::QoS{1}.transient_local(),
    std::bind(&FreespacePlannerNode::onRoute, this, _1));

  // Publishers
  {
    rclcpp::QoS qos{1};
    qos.transient_local();  // latch
    trajectory_pub_ = create_publisher<Trajectory>("~/output/trajectory", qos);
    debug_pose_array_pub_ = create_publisher<PoseArray>("~/debug/pose_array", qos);
    debug_partial_pose_array_pub_ = create_publisher<PoseArray>("~/debug/partial_pose_array", qos);
    parking_state_pub_ = create_publisher<std_msgs::msg::Bool>("is_completed", qos);
    processing_time_pub_ = create_publisher<autoware_internal_debug_msgs::msg::Float64Stamped>(
      "~/debug/processing_time_ms", 1);
  }

  // TF
  {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  }

  // Timer
  {
    const auto period_ns = rclcpp::Rate(node_param_.update_rate).period();
    timer_ = rclcpp::create_timer(
      this, get_clock(), period_ns, std::bind(&FreespacePlannerNode::onTimer, this));
  }

  logger_configure_ = std::make_unique<autoware_utils::LoggerLevelConfigure>(this);
}

PlannerCommonParam FreespacePlannerNode::getPlannerCommonParam()
{
  PlannerCommonParam p;

  // search configs
  p.time_limit = declare_parameter<double>("time_limit");

  p.theta_size = declare_parameter<int>("theta_size");
  p.angle_goal_range = declare_parameter<double>("angle_goal_range");
  p.curve_weight = declare_parameter<double>("curve_weight");
  p.reverse_weight = declare_parameter<double>("reverse_weight");
  p.direction_change_weight = declare_parameter<double>("direction_change_weight");
  p.lateral_goal_range = declare_parameter<double>("lateral_goal_range");
  p.longitudinal_goal_range = declare_parameter<double>("longitudinal_goal_range");
  p.max_turning_ratio = declare_parameter<double>("max_turning_ratio");
  p.turning_steps = declare_parameter<int>("turning_steps");

  // costmap configs
  p.obstacle_threshold = declare_parameter<int>("obstacle_threshold");

  return p;
}

bool FreespacePlannerNode::isPlanRequired()
{
  if (trajectory_.points.empty()) {
    return true;
  }

  if (node_param_.replan_when_obstacle_found && checkCurrentTrajectoryCollision()) {
    RCLCPP_DEBUG(get_logger(), "Found obstacle");
    return true;
  }

  if (node_param_.replan_when_course_out) {
    const bool is_course_out = utils::calc_distance_2d(trajectory_, current_pose_.pose) >
                               node_param_.th_course_out_distance_m;
    if (is_course_out) {
      RCLCPP_INFO(get_logger(), "Course out");
      return true;
    }
  }

  return false;
}

bool FreespacePlannerNode::checkCurrentTrajectoryCollision()
{
  algo_->setMap(nero_half_cell() ? nero_half_cell_grid(*occupancy_grid_) : *occupancy_grid_);

  const size_t nearest_index_partial = autoware::motion_utils::findNearestIndex(
    partial_trajectory_.points, current_pose_.pose.position);
  const size_t end_index_partial = partial_trajectory_.points.size() - 1;
  const auto forward_trajectory = utils::get_partial_trajectory(
    partial_trajectory_, nearest_index_partial, end_index_partial, get_clock());

  const bool is_obs_found =
    algo_->hasObstacleOnTrajectory(utils::trajectory_to_pose_array(forward_trajectory));

  if (!is_obs_found) {
    obs_found_time_ = {};
    return false;
  }

  if (!obs_found_time_) obs_found_time_ = get_clock()->now();

  return (get_clock()->now() - obs_found_time_.get()).seconds() > node_param_.th_obstacle_time_sec;
}

void FreespacePlannerNode::updateTargetIndex()
{
  if (!utils::is_stopped(odom_buffer_, node_param_.th_stopped_velocity_mps)) {
    return;
  }

  const auto is_near_target = utils::is_near_target(
    trajectory_.points.at(target_index_).pose, current_pose_.pose,
    node_param_.th_arrived_distance_m);

  if (!is_near_target) return;

  const auto new_target_index =
    utils::get_next_target_index(trajectory_.points.size(), reversing_indices_, target_index_);

  if (new_target_index == target_index_) {
    if (nero_repark_deg() > 0.0 && nero_straight_in_m() > 0.0 && nero_reparks_ < 1) {
      const auto & end_pose = trajectory_.points.back().pose;
      const double dyaw = std::abs(autoware_utils::normalize_radian(
        tf2::getYaw(end_pose.orientation) - tf2::getYaw(current_pose_.pose.orientation)));
      if (dyaw > nero_repark_deg() * M_PI / 180.0) {
        ++nero_reparks_;
        nero_repark_back_ = trajectory_.points.back().longitudinal_velocity_mps < 0.0;
        RCLCPP_INFO(
          get_logger(), "[NERO_FREESPACE_REPARK_DEG] %.1f deg off the planned heading: parking once more",
          dyaw * 180.0 / M_PI);
        nero_repark_now_ = true;
        planTrajectory();
        nero_repark_now_ = false;
        if (trajectory_.points.size() > 1) return;
        RCLCPP_INFO(get_logger(), "[NERO_FREESPACE_REPARK_DEG] no plan: staying");
      }
    }
    // Finished publishing all partial trajectories
    is_completed_ = true;
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Freespace planning completed");
    std_msgs::msg::Bool is_completed_msg;
    is_completed_msg.data = is_completed_;
    parking_state_pub_->publish(is_completed_msg);
  } else {
    // Switch to next partial trajectory
    prev_target_index_ = target_index_;
    target_index_ = new_target_index;
  }
}

bool FreespacePlannerNode::neroPredictedHold()
{
  const auto release = [this]() {
    if (nero_holding_) {
      RCLCPP_INFO(get_logger(), "[NERO_FREESPACE_PREDICTED_HOLD_S] clear: going on");
    }
    nero_holding_ = false;
    return false;
  };
  if (!nero_objects_ || partial_trajectory_.points.size() < 2) return release();
  // only before a partial trajectory starts (standing); once moving the vehicle finishes it
  if (!utils::is_stopped(odom_buffer_, node_param_.th_stopped_velocity_mps)) return release();
  if (nero_objects_->header.frame_id != partial_trajectory_.header.frame_id) return release();

  const double horizon = nero_predicted_hold_s();
  const double m = node_param_.vehicle_shape_margin_m / 2.0;
  const double front = vehicle_shape_.length - vehicle_shape_.base2back + m;
  const double rear = vehicle_shape_.base2back + m;
  const double half_w = vehicle_shape_.width / 2.0 + m;
  const auto now_fp = nero_quad(current_pose_.pose, front, rear, half_w);
  std::vector<NeroQuad> path_fp;
  const size_t nearest =
    autoware::motion_utils::findNearestIndex(partial_trajectory_.points, current_pose_.pose.position);
  for (size_t i = nearest; i < partial_trajectory_.points.size(); ++i) {
    path_fp.push_back(nero_quad(partial_trajectory_.points.at(i).pose, front, rear, half_w));
  }

  for (const auto & o : nero_objects_->objects) {
    const auto & k = o.kinematics;
    const double v = std::hypot(
      k.initial_twist_with_covariance.twist.linear.x, k.initial_twist_with_covariance.twist.linear.y);
    if (v < 0.5) continue;  // standing / creeping: the costmap check handles it
    const double ol = o.shape.dimensions.x / 2.0 + 0.3;
    const double ow = o.shape.dimensions.y / 2.0 + 0.3;
    std::vector<std::pair<double, geometry_msgs::msg::Pose>> poses{
      {0.0, k.initial_pose_with_covariance.pose}};
    const autoware_perception_msgs::msg::PredictedPath * best = nullptr;
    for (const auto & p : k.predicted_paths) {
      if (!best || p.confidence > best->confidence) best = &p;
    }
    if (best) {
      const double dt = rclcpp::Duration(best->time_step).seconds();
      for (size_t i = 0; i < best->path.size() && dt > 0.0 && i * dt <= horizon; ++i) {
        poses.emplace_back(i * dt, best->path.at(i));
      }
    }
    bool in_path = false;
    bool in_now = false;
    double t_hit = 0.0;
    for (const auto & [t, pose] : poses) {
      const auto oq = nero_quad(pose, ol, ol, ow);
      if (nero_overlap(oq, now_fp)) in_now = true;
      if (!in_path) {
        for (const auto & q : path_fp) {
          if (nero_overlap(oq, q)) {
            in_path = true;
            t_hit = t;
            break;
          }
        }
      }
    }
    if (in_path && !in_now) {
      if (!nero_holding_) {
        RCLCPP_INFO(
          get_logger(),
          "[NERO_FREESPACE_PREDICTED_HOLD_S] holding: a road user (label %d, %.1f m/s, %.1f m away) "
          "crosses the next move within %.1f s",
          o.classification.empty() ? -1 : static_cast<int>(o.classification.front().label), v,
          autoware_utils::calc_distance2d(k.initial_pose_with_covariance.pose, current_pose_.pose),
          t_hit);
      }
      nero_holding_ = true;
      return true;
    }
  }
  return release();
}

void FreespacePlannerNode::onRoute(const LaneletRoute::ConstSharedPtr msg)
{
  route_ = msg;

  goal_pose_.header = msg->header;
  goal_pose_.pose = msg->goal_pose;

  is_new_parking_cycle_ = true;
  nero_reparks_ = 0;
  nero_straight_wait_since_ = -1.0;

  reset();
}

void FreespacePlannerNode::onOdometry(const Odometry::ConstSharedPtr msg)
{
  odom_ = msg;

  odom_buffer_.push_back(msg);

  // Delete old data in buffer
  while (true) {
    const auto time_diff =
      rclcpp::Time(msg->header.stamp) - rclcpp::Time(odom_buffer_.front()->header.stamp);

    if (time_diff.seconds() < node_param_.th_stopped_time_sec) {
      break;
    }

    odom_buffer_.pop_front();
  }
}

void FreespacePlannerNode::updateData()
{
  occupancy_grid_ = occupancy_grid_sub_.take_data();

  {
    auto msgs = odom_sub_.take_data();
    for (const auto & msg : msgs) {
      onOdometry(msg);
    }
  }
}

bool FreespacePlannerNode::isDataReady()
{
  bool is_ready = true;

  if (!route_) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000, "Waiting for route data.");
    is_ready = false;
  }

  if (!occupancy_grid_) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000, "Waiting for occupancy grid.");
    is_ready = false;
  }

  if (!odom_) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000, "Waiting for odometry.");
    is_ready = false;
  }

  return is_ready;
}

void FreespacePlannerNode::onTimer()
{
  autoware_utils::StopWatch<std::chrono::milliseconds> stop_watch;

  scenario_ = scenario_sub_.take_data();
  if (!utils::is_active(scenario_)) {
    reset();
    return;
  }

  updateData();

  if (!isDataReady()) {
    return;
  }

  if (is_completed_) {
    partial_trajectory_.header = odom_->header;
    const auto stop_trajectory = utils::create_stop_trajectory(partial_trajectory_);
    trajectory_pub_->publish(stop_trajectory);
    return;
  }

  // Get current pose
  current_pose_.pose = odom_->pose.pose;
  current_pose_.header = odom_->header;

  if (current_pose_.header.frame_id == "") {
    return;
  }

  // Must stop before replanning any new trajectory
  const bool is_reset_required = !reset_in_progress_ && isPlanRequired();
  if (is_reset_required) {
    // Stop before planning new trajectory, except in a new parking cycle as the vehicle already
    // stops.
    if (!is_new_parking_cycle_) {
      const auto stop_trajectory = partial_trajectory_.points.empty()
                                     ? utils::create_stop_trajectory(current_pose_, get_clock())
                                     : utils::create_stop_trajectory(partial_trajectory_);
      trajectory_pub_->publish(stop_trajectory);
      debug_pose_array_pub_->publish(utils::trajectory_to_pose_array(stop_trajectory));
      debug_partial_pose_array_pub_->publish(utils::trajectory_to_pose_array(stop_trajectory));
    }

    reset();

    reset_in_progress_ = true;
  }

  if (reset_in_progress_) {
    const auto is_ego_stopped =
      utils::is_stopped(odom_buffer_, node_param_.th_stopped_velocity_mps);
    if (is_ego_stopped) {
      // Plan new trajectory
      planTrajectory();
      reset_in_progress_ = false;
    } else {
      // Will keep current stop trajectory
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "Waiting for the vehicle to stop before generating a new trajectory.");
    }
  }

  // StopTrajectory
  if (trajectory_.points.size() <= 1) {
    // NERO: no plan yet (waiting to stop, replanning, A* failed, new parking cycle).
    // Hold in place instead of going silent: a silent output trips
    // topic_state_monitor (timeout 1 s) and the MRM emergency stop.
    const auto hold = utils::create_stop_trajectory(current_pose_, get_clock());
    trajectory_pub_->publish(hold);
    debug_partial_pose_array_pub_->publish(utils::trajectory_to_pose_array(hold));
    is_new_parking_cycle_ = false;
    return;
  }

  // Update partial trajectory
  updateTargetIndex();
  partial_trajectory_ =
    utils::get_partial_trajectory(trajectory_, prev_target_index_, target_index_, get_clock());

  if (nero_predicted_hold_s() > 0.0 && neroPredictedHold()) {
    const auto hold = utils::create_stop_trajectory(partial_trajectory_);
    trajectory_pub_->publish(hold);
    debug_pose_array_pub_->publish(utils::trajectory_to_pose_array(trajectory_));
    debug_partial_pose_array_pub_->publish(utils::trajectory_to_pose_array(hold));
    is_new_parking_cycle_ = false;
    return;
  }

  // Publish messages
  trajectory_pub_->publish(partial_trajectory_);
  debug_pose_array_pub_->publish(utils::trajectory_to_pose_array(trajectory_));
  debug_partial_pose_array_pub_->publish(utils::trajectory_to_pose_array(partial_trajectory_));

  is_new_parking_cycle_ = false;

  // Publish ProcessingTime
  autoware_internal_debug_msgs::msg::Float64Stamped processing_time_msg;
  processing_time_msg.stamp = get_clock()->now();
  processing_time_msg.data = stop_watch.toc();
  processing_time_pub_->publish(processing_time_msg);
}

void FreespacePlannerNode::planTrajectory()
{
  if (occupancy_grid_ == nullptr) {
    return;
  }

  if (nero_straight_wait_since_ >= 0.0) {
    // NERO_FREESPACE_STRAIGHT_IN_WAIT_S: while waiting for a blocked straight approach, re-plan
    // every 0.5 s (a full A* search each cycle would keep a CPU core busy)
    const double now = get_clock()->now().seconds();
    if (now - nero_straight_wait_try_ < 0.5) {
      reset();
      return;
    }
    nero_straight_wait_try_ = now;
  }

  // Provide robot shape and map for the planner
  algo_->setMap(nero_half_cell() ? nero_half_cell_grid(*occupancy_grid_) : *occupancy_grid_);

  // Calculate poses in costmap frame
  const auto current_pose_in_costmap_frame = utils::transform_pose(
    current_pose_.pose,
    getTransform(occupancy_grid_->header.frame_id, current_pose_.header.frame_id));

  auto goal_pose_in_costmap_frame = utils::transform_pose(
    goal_pose_.pose, getTransform(occupancy_grid_->header.frame_id, goal_pose_.header.frame_id));

  if (const double slide = nero_goal_slide_m(); slide > 0.0) {
    const auto blocked = [this](const geometry_msgs::msg::Pose & p) {
      geometry_msgs::msg::PoseArray a;
      a.poses.push_back(p);
      return algo_->hasObstacleOnTrajectory(a);
    };
    if (blocked(goal_pose_in_costmap_frame)) {
      const auto & g = goal_pose_in_costmap_frame;
      const double yaw = tf2::getYaw(g.orientation);
      const double ux = std::cos(yaw);
      const double uy = std::sin(yaw);
      const auto & c = current_pose_in_costmap_frame.position;
      const double sgn =
        (c.x - g.position.x) * ux + (c.y - g.position.y) * uy >= 0.0 ? 1.0 : -1.0;
      bool found = false;
      for (int k = 1; k * 0.05 <= slide + 1e-6 && !found; ++k) {
        auto p = g;
        p.position.x += sgn * k * 0.05 * ux;
        p.position.y += sgn * k * 0.05 * uy;
        if (!blocked(p)) {
          RCLCPP_INFO(
            get_logger(),
            "[NERO_FREESPACE_GOAL_SLIDE_M] goal blocked: planning to the pose %.2f m out of it",
            k * 0.05);
          goal_pose_in_costmap_frame = p;
          found = true;
        }
      }
      if (!found) {
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "[NERO_FREESPACE_GOAL_SLIDE_M] goal blocked up to %.2f m out of it", slide);
      }
    }
  }

  // execute planning
  const rclcpp::Time start = get_clock()->now();
  std::string error_msg;
  bool result = false;
  if (nero_repark_now_) {
    // NERO_FREESPACE_REPARK_DEG: only the pose before the goal + the straight piece (below); A* to
    // the goal itself fails when the vehicle already stands on it a few degrees off
    result = true;
  } else {
    try {
      result = algo_->makePlan(current_pose_in_costmap_frame, goal_pose_in_costmap_frame);
    } catch (const std::exception & e) {
      error_msg = e.what();
    }
  }
  const rclcpp::Time end = get_clock()->now();

  RCLCPP_DEBUG(get_logger(), "Freespace planning: %f [s]", (end - start).seconds());

  if (result) {
    RCLCPP_DEBUG(get_logger(), "Found goal!");
    autoware::freespace_planning_algorithms::PlannerWaypoints waypoints;
    if (!nero_repark_now_) waypoints = algo_->getWaypoints();
    size_t straight_from = 0;  // index of the first point of the straight piece (0 = none)
    if (const double d = nero_straight_in_m();
        d > 0.0 && (nero_repark_now_ || waypoints.waypoints.size() > 1)) {
      const bool back = nero_repark_now_ ? nero_repark_back_ : waypoints.waypoints.back().is_back;
      const auto & goal = goal_pose_in_costmap_frame;
      const double yaw = tf2::getYaw(goal.orientation);
      const double ux = std::cos(yaw);
      const double uy = std::sin(yaw);
      const double sgn = back ? 1.0 : -1.0;  // the side of the goal the vehicle comes from
      std::string why;
      bool only_blocked = true;  // every failure was an occupied pose (not geometry)
      for (const double f : {1.0, 0.75, 0.5, 0.33}) {
        // a time budget for the extra A* calls: the timer must publish at least every second
        // (topic_state_monitor on the trajectory: 1 s -> MRM). Over budget = try again later.
        if ((get_clock()->now() - start).seconds() > 0.4) {
          why += "time budget (0.4 s) used up; ";
          break;
        }
        if (nero_straight_in_stop() && f * d < 0.8) {
          // with its own segment the straight piece must be long enough to cut (0.3 m) and still
          // drive straight; shorter ones keep the +-1.5 m curvature feed-forward: skip, then wait
          // (blocked) or the plan without it and NERO_FREESPACE_REPARK_DEG
          continue;
        }
        auto pre = goal;
        pre.position.x += sgn * f * d * ux;
        pre.position.y += sgn * f * d * uy;
        bool pre_found = false;
        std::string astar_err;
        try {
          // A* throws when the pose is blocked or unreachable: that is "no plan" here, never an
          // exception out of the timer (it would stop the planner)
          pre_found = algo_->makePlan(current_pose_in_costmap_frame, pre);
        } catch (const std::exception & e) {
          pre_found = false;
          astar_err = e.what();
        }
        const std::string dtxt = std::to_string(f * d).substr(0, 4) + " m: ";
        if (!pre_found) {
          why += dtxt + "no plan to the pose before the goal (" + astar_err + "); ";
          only_blocked = only_blocked && astar_err.find("Invalid") != std::string::npos;
        } else {
          auto w2 = algo_->getWaypoints();
          if (w2.waypoints.empty()) {
            why += dtxt + "empty plan; ";
            only_blocked = false;
          } else if (!nero_repark_now_ && w2.waypoints.back().is_back != back) {
            why += dtxt + "it arrives in the other direction; ";
            only_blocked = false;
          } else if (
            !nero_repark_now_ && nero_direction_changes(w2) > nero_direction_changes(waypoints)) {
            only_blocked = false;
            why += dtxt + "more direction changes (" + std::to_string(nero_direction_changes(w2)) +
                   " vs " + std::to_string(nero_direction_changes(waypoints)) + "); ";
          } else {
            // the straight piece on the goal's axis, from where that plan ends (along the axis) to
            // the goal
            const auto & last = w2.waypoints.back().pose.pose.position;
            const double s0 = std::max(
              0.3, sgn * ((last.x - goal.position.x) * ux + (last.y - goal.position.y) * uy));
            const int n = std::max(1, static_cast<int>(std::ceil(s0 / 0.1)));
            geometry_msgs::msg::PoseArray straight;
            straight.header = w2.header;
            for (int i = n - 1; i >= 0; --i) {
              auto p = goal;
              p.position.x += sgn * s0 * i / n * ux;
              p.position.y += sgn * s0 * i / n * uy;
              straight.poses.push_back(p);
            }
            if (algo_->hasObstacleOnTrajectory(straight)) {
              why += dtxt + "the straight piece is blocked; ";
            } else {
              straight_from = w2.waypoints.size();
              nero_straight_wait_since_ = -1.0;
              for (const auto & p : straight.poses) {
                autoware::freespace_planning_algorithms::PlannerWaypoint wp;
                wp.pose.header = w2.header;
                wp.pose.pose = p;
                wp.is_back = back;
                w2.waypoints.push_back(wp);
              }
              RCLCPP_INFO(
                get_logger(), "[NERO_FREESPACE_STRAIGHT_IN] last %.2f m %s straight into the goal",
                s0, back ? "reversing" : "forward");
              waypoints = w2;
              why.clear();
              break;
            }
          }
        }
      }
      if (!why.empty() && only_blocked && nero_straight_in_wait_s() > 0.0 && !nero_repark_now_) {
        const double now = get_clock()->now().seconds();
        if (nero_straight_wait_since_ < 0.0) {
          nero_straight_wait_since_ = now;
          nero_straight_wait_try_ = now;
        }
        if (now - nero_straight_wait_since_ < nero_straight_in_wait_s()) {
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "[NERO_FREESPACE_STRAIGHT_IN_WAIT_S] the straight approach is blocked: waiting (%.1f s)",
            now - nero_straight_wait_since_);
          reset();
          return;
        }
      }
      if (nero_repark_now_ && straight_from == 0) {
        RCLCPP_INFO(get_logger(), "[NERO_FREESPACE_REPARK_DEG] no plan: %s", why.c_str());
        reset();
        return;
      }
      if (!why.empty()) {
        nero_straight_wait_since_ = -1.0;
        RCLCPP_INFO(
          get_logger(), "[NERO_FREESPACE_STRAIGHT_IN] upstream plan kept: %s", why.c_str());
      }
    }
    trajectory_ =
      utils::create_trajectory(current_pose_, waypoints, node_param_.waypoints_velocity);
    reversing_indices_ = utils::get_reversing_indices(trajectory_);
    // NERO_FREESPACE_STRAIGHT_IN_STOP=1: the straight piece as its own partial trajectory, cut 0.3 m
    // into it (the vehicle stops 0.2-0.3 m short of a segment end, i.e. where the straight begins).
    const bool straight_own_segment =
      straight_from > 0 && std::find(
                             reversing_indices_.begin(), reversing_indices_.end(),
                             straight_from - 1) != reversing_indices_.end();
    if (const size_t cut = straight_from + 2;
        nero_straight_in_stop() && straight_from > 0 && !straight_own_segment &&
        cut + 5 < trajectory_.points.size() &&
        (reversing_indices_.empty() || cut > reversing_indices_.back())) {
      reversing_indices_.push_back(cut);
      RCLCPP_INFO(
        get_logger(), "[NERO_FREESPACE_STRAIGHT_IN_STOP] stop where the straight piece begins, then %.2f m straight",
        autoware_utils::calc_distance2d(trajectory_.points.at(cut), trajectory_.points.back()));
    }
    prev_target_index_ = 0;
    target_index_ = utils::get_next_target_index(
      trajectory_.points.size(), reversing_indices_, prev_target_index_);
  } else {
    RCLCPP_INFO(get_logger(), "Can't find goal: %s", error_msg.c_str());
    reset();
  }
}

void FreespacePlannerNode::reset()
{
  trajectory_ = Trajectory();
  partial_trajectory_ = Trajectory();
  is_completed_ = false;
  std_msgs::msg::Bool is_completed_msg;
  is_completed_msg.data = is_completed_;
  parking_state_pub_->publish(is_completed_msg);
  obs_found_time_ = {};
}

TransformStamped FreespacePlannerNode::getTransform(
  const std::string & from, const std::string & to)
{
  TransformStamped tf;
  try {
    tf =
      tf_buffer_->lookupTransform(from, to, rclcpp::Time(0), rclcpp::Duration::from_seconds(1.0));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR(get_logger(), "%s", ex.what());
  }
  return tf;
}

void FreespacePlannerNode::initializePlanningAlgorithm()
{
  // Extend robot shape
  autoware::freespace_planning_algorithms::VehicleShape extended_vehicle_shape = vehicle_shape_;
  const double margin = node_param_.vehicle_shape_margin_m;
  extended_vehicle_shape.length += margin;
  extended_vehicle_shape.width += margin;
  extended_vehicle_shape.base2back += margin / 2;
  extended_vehicle_shape.setMinMaxDimension();

  const auto planner_common_param = getPlannerCommonParam();

  const auto algo_name = node_param_.planning_algorithm;

  // initialize specified algorithm
  if (algo_name == "astar") {
    algo_ = std::make_unique<AstarSearch>(planner_common_param, extended_vehicle_shape, *this);
  } else if (algo_name == "rrtstar") {
    algo_ = std::make_unique<RRTStar>(planner_common_param, extended_vehicle_shape, *this);
  } else {
    throw std::runtime_error("No such algorithm named " + algo_name + " exists.");
  }
  RCLCPP_INFO_STREAM(get_logger(), "initialize planning algorithm: " << algo_name);
}
}  // namespace autoware::freespace_planner

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::freespace_planner::FreespacePlannerNode)
