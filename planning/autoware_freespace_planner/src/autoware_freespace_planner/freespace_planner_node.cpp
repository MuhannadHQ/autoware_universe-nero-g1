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
#include <cstring>
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

// NERO_FREESPACE_STRAIGHT_FWD_MIN_M=<m> (nero, pull-out 2026-10-24, with NERO_FREESPACE_STRAIGHT_IN_STOP): a
// forward straight segment starts early enough to be at least <m> long. The MPC treats a standing vehicle with a
// zero target speed within 1 m ahead as stopped and then never reports the steering converged, and the PID waits
// for that before it departs: a forward straight segment of <= 1 m never started (pull-out sim at the real place:
// 0.92 m, the robot stood 1.2 m short of the step-out pose for good; upstream main has the same MPC check). A
// reversing segment is not affected (the MPC takes the minimum of its negative speeds). Unset = the cut 2 points
// into the straight piece in both directions.
double nero_straight_fwd_min_m()
{
  static const double m = [] {
    const char * v = std::getenv("NERO_FREESPACE_STRAIGHT_FWD_MIN_M");
    return v ? std::max(0.0, std::atof(v)) : 0.0;
  }();
  return m;
}

// where the straight piece's own segment begins: 2 points into it, earlier for a short forward one (above)
size_t nero_straight_cut(const autoware_planning_msgs::msg::Trajectory & traj, const size_t straight_from)
{
  size_t cut = straight_from + 2;
  if (nero_straight_fwd_min_m() <= 0.0 || cut >= traj.points.size()) return cut;
  if (traj.points.at(straight_from).longitudinal_velocity_mps <= 0.0) return cut;
  while (cut > 1 && autoware::motion_utils::calcSignedArcLength(traj.points, cut, traj.points.size() - 1) <
                      nero_straight_fwd_min_m()) {
    --cut;
  }
  return cut;
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

// NERO_FREESPACE_PREDICTED_HOLD_MOVING=1 (nero, pull-out 2026-10-24, with NERO_FREESPACE_PREDICTED_HOLD_S): the hold
// also while moving: the vehicle stops before its footprint enters a moving road user's way (not in it yet) and waits
// as when standing; once in that way it goes on. Pulling out of the parking spot the view up the lane opens only
// after the move started (a parked car beside the spot): a car on the lane first seen 29 m away at 3.4 s, the robot
// went on into its way (sim: a never-slowing driver hit it at 9.2 s). Unset = the check only before a segment.
bool nero_predicted_hold_moving()
{
  static const bool on = [] {
    const char * v = std::getenv("NERO_FREESPACE_PREDICTED_HOLD_MOVING");
    return v && std::atoi(v) == 1;
  }();
  return on;
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

bool nero_inside(const NeroQuad & q, const double x, const double y)
{
  bool pos = false;
  bool neg = false;
  for (int i = 0; i < 4; ++i) {
    const auto & a = q[i];
    const auto & b = q[(i + 1) % 4];
    const double cross = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
    pos = pos || cross > 0.0;
    neg = neg || cross < 0.0;
  }
  return !(pos && neg);
}

// how deep the vehicle at p is in a road user's way: its footprint's sample points (9 x 5) inside any of the quads
int nero_depth(
  const geometry_msgs::msg::Pose & p, const double front, const double rear, const double half_w,
  const std::vector<NeroQuad> & way)
{
  const double yaw = tf2::getYaw(p.orientation);
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  int n = 0;
  for (int i = 0; i < 9; ++i) {
    const double u = -rear + (front + rear) * i / 8.0;
    for (int j = 0; j < 5; ++j) {
      const double v = -half_w + 2.0 * half_w * j / 4.0;
      const double x = p.position.x + c * u - s * v;
      const double y = p.position.y + s * u + c * v;
      n += std::any_of(way.begin(), way.end(), [&](const NeroQuad & q) { return nero_inside(q, x, y); });
    }
  }
  return n;
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

// NERO_FREESPACE_MIN_SEGMENT=<fwd>,<rev>[,<extra>,<penalty>] (nero, parkfix 2026-10-25): A* (its library reads the
// same switch) plans no first segment the controller cannot start. Here, before a partial trajectory starts from
// standstill: one too short to start from where the vehicle stands (it stops 0.2-0.4 m before a reversing point) -
// forward < 1.2 m (mpc_lateral_controller's stopped state looks 1 m ahead), reversing < 0.6 m (the PID's and the
// velocity smoother's 0.5 m) - is not published: the last one ends the parking when the vehicle is parked
// (NERO_FREESPACE_PARKED_BOX), any other is planned again from here (then with a straight move it can start first,
// NERO_FREESPACE_STALL_S's lead-in). Car 2026-10-25: 0.50 m and 1.0 m forward moves, 18 s and 23 s standing. Replaces
// NERO_FREESPACE_STRAIGHT_FWD_MIN_M's 3 re-plans (A* is deterministic: the same plan 3 times). Unset = upstream.
bool nero_min_segment_on()
{
  static const bool on = [] {
    const char * v = std::getenv("NERO_FREESPACE_MIN_SEGMENT");
    return v && *v;
  }();
  return on;
}
constexpr double kNeroStartFwdM = 1.2;
constexpr double kNeroStartRevM = 0.6;

// NERO_FREESPACE_STALL_S=<s> (nero, parkfix 2026-10-25): in Parking, standing (autonomous, the controller commanding
// 0 m/s) for <s> seconds while the planner publishes a partial trajectory that says move always recovers: parked
// (NERO_FREESPACE_PARKED_BOX) = done; else plan again from here, starting with a straight move the vehicle can start
// (forward >= 1.5 m or reversing >= 1.0 m, its swept area and end pose free). Holds for traffic, a blocked plan, the
// operator's stop or a downstream stop (AEB, gate) never count: the planner then publishes a stop, or the controller
// commands motion. Car 2026-10-25: both stalls ended only when a passing road user forced a re-plan. Unset = upstream.
double nero_stall_s()
{
  static const double s = [] {
    const char * v = std::getenv("NERO_FREESPACE_STALL_S");
    return v ? std::clamp(std::atof(v), 0.0, 60.0) : 0.0;
  }();
  return s;
}

// NERO_FREESPACE_PARKED_BOX=<front_m>,<side_m>,<deg> (nero, parkfix 2026-10-25): the vehicle counts as parked when its
// footprint reaches at most <front_m> further out (or deeper) than at the planned end pose, no corner is more than
// <side_m> to either side of the planned axis, and it stands within <deg> of the planned heading. NERO_FREESPACE_REPARK_DEG
// then parks once more only outside this box (car 2026-10-25: 0.34 m short, 0.20 m to the side, 9.4 deg - "this
// position is good enough" - re-parked at 3 deg and stood 23 s). Unset = NERO_FREESPACE_REPARK_DEG's degrees alone.
struct NeroBox
{
  bool on = false;
  double front = 0.6;
  double side = 0.9;
  double deg = 15.0;
};

const NeroBox & nero_parked_box()
{
  static const NeroBox b = [] {
    NeroBox r;
    const char * v = std::getenv("NERO_FREESPACE_PARKED_BOX");
    if (!v || !*v) return r;
    double a[3] = {r.front, r.side, r.deg};
    if (std::sscanf(v, "%lf,%lf,%lf", &a[0], &a[1], &a[2]) < 3) return r;
    r.on = true;
    r.front = std::max(0.0, a[0]);
    r.side = std::max(0.0, a[1]);
    r.deg = std::clamp(a[2], 0.0, 90.0);
    return r;
  }();
  return b;
}

// NERO_FREESPACE_START_ESCAPE=1 (nero, parkfix 2026-10-25): a start pose whose footprint + vehicle_shape_margin_m
// touches an occupied cell (A* "Invalid start or goal pose" every 0.1 s) begins with a straight move away from it the
// vehicle can start: forward >= 1.5 m or reversing >= 1.0 m whose swept area is free (the real footprint, and the
// margin ahead of it) and whose end pose is free with the margin; A* plans on from there. Car 2026-10-25 12:23: parked by
// hand 0.43 m deeper than the spot, its rear 0.28 m from the step / wall behind; it never left. Unset = upstream.
// NERO_FREESPACE_START_ESCAPE=1,<s> (pull-out e2e 2026-10-25): only once the start has been inside the margin <s>
// seconds in a row; a shorter block is waited out as upstream does ("Invalid start or goal pose", plan again next
// cycle). The e2e home run with the field's perception replayed: a blob made the end of the forward swing invalid for
// ~0.2 s; escaping at once reversed 1.0 m straight and the park ended 10 deg crooked after 3 gear changes. Plain 1 = 0 s.
bool nero_start_escape()
{
  static const bool on = [] {
    const char * v = std::getenv("NERO_FREESPACE_START_ESCAPE");
    return v && std::atoi(v) == 1;
  }();
  return on;
}

double nero_start_escape_hold_s()
{
  static const double s = [] {
    const char * v = std::getenv("NERO_FREESPACE_START_ESCAPE");
    const char * c = v ? std::strchr(v, ',') : nullptr;
    return c ? std::max(0.0, std::atof(c + 1)) : 0.0;
  }();
  return s;
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
  if (nero_stall_s() > 0.0) {
    nero_op_sub_ = create_subscription<autoware_adapi_v1_msgs::msg::OperationModeState>(
      "/system/operation_mode/state", rclcpp::QoS{1},
      [this](const autoware_adapi_v1_msgs::msg::OperationModeState::ConstSharedPtr msg) { nero_op_ = msg; });
    nero_cmd_sub_ = create_subscription<autoware_control_msgs::msg::Control>(
      "/control/trajectory_follower/control_cmd", rclcpp::QoS{1},
      [this](const autoware_control_msgs::msg::Control::ConstSharedPtr msg) { nero_cmd_ = msg; });
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

  // NERO_FREESPACE_PREDICTED_HOLD_MOVING: no re-plan while holding (standing) for a moving road user crossing the
  // path: the plan is still valid when it has passed (sim: a re-plan from the held pose, 2 gear changes, +13 s)
  if (
    node_param_.replan_when_obstacle_found && !(nero_predicted_hold_moving() && nero_holding_) &&
    checkCurrentTrajectoryCollision()) {
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
  // NERO_FREESPACE_MIN_SEGMENT: no partial trajectory (a plan of one point) is no collision; findNearestIndex() would
  // throw out of the timer and the parking container dies (sim 2026-10-25 182513 -> MRM)
  if (nero_min_segment_on() && partial_trajectory_.points.empty()) return false;

  algo_->setMap(nero_half_cell() ? nero_half_cell_grid(*occupancy_grid_) : *occupancy_grid_);

  const size_t nearest_index_partial = autoware::motion_utils::findNearestIndex(
    partial_trajectory_.points, current_pose_.pose.position);
  const size_t end_index_partial = partial_trajectory_.points.size() - 1;
  const auto forward_trajectory = utils::get_partial_trajectory(
    partial_trajectory_, nearest_index_partial, end_index_partial, get_clock());

  bool is_obs_found = false;
  const size_t lead_end = nero_lead_end_ < 0 ? 0 : static_cast<size_t>(nero_lead_end_);
  if (nero_lead_end_ >= 0 && prev_target_index_ + nearest_index_partial <= lead_end) {
    // NERO_FREESPACE_START_ESCAPE / NERO_FREESPACE_STALL_S: the straight move first is checked as it was planned (what
    // it moves away from never counts, anything new on its way does); the rest of this partial trajectory as usual
    const auto & e = trajectory_.points.at(lead_end).pose;
    const double yaw = tf2::getYaw(current_pose_.pose.orientation);
    const double left = std::abs(
      (e.position.x - current_pose_.pose.position.x) * std::cos(yaw) +
      (e.position.y - current_pose_.pose.position.y) * std::sin(yaw));
    is_obs_found = !neroStraightFree(current_pose_.pose, nero_lead_back_, left);
    PoseArray rest;
    rest.header = forward_trajectory.header;
    for (size_t k = lead_end + 1; k <= prev_target_index_ + end_index_partial; ++k) {
      rest.poses.push_back(trajectory_.points.at(k).pose);
    }
    if (!is_obs_found && !rest.poses.empty()) is_obs_found = algo_->hasObstacleOnTrajectory(rest);
  } else {
    is_obs_found = algo_->hasObstacleOnTrajectory(utils::trajectory_to_pose_array(forward_trajectory));
  }

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
      // NERO_FREESPACE_PARKED_BOX: once more only when the footprint is outside the box, not for degrees alone
      std::string box_why;
      const bool parked = nero_parked_box().on && neroParkedEnough(current_pose_.pose, end_pose, box_why);
      if (nero_parked_box().on) {
        RCLCPP_INFO(
          get_logger(), "[NERO_FREESPACE_PARKED_BOX] finished %.1f deg off the planned heading: %s: %s",
          dyaw * 180.0 / M_PI, box_why.c_str(), parked ? "parked" : "parking once more");
      }
      if (nero_parked_box().on ? !parked : dyaw > nero_repark_deg() * M_PI / 180.0) {
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
  // (NERO_FREESPACE_PREDICTED_HOLD_MOVING: also while moving)
  if (
    !nero_predicted_hold_moving() &&
    !utils::is_stopped(odom_buffer_, node_param_.th_stopped_velocity_mps)) {
    return release();
  }
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
    std::vector<NeroQuad> way;
    for (const auto & [t, pose] : poses) {
      const auto oq = nero_quad(pose, ol, ol, ow);
      way.push_back(oq);
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
    // NERO_FREESPACE_PREDICTED_HOLD_MOVING: already in its way, the vehicle goes on only when that does not take it
    // deeper in (along its lane in front of a car behind: the same depth); nosing further into the way of a car it saw
    // late, it stops (sim, cars parked on both sides: a car first seen 16 m away, the robot went on 0.4 m deeper)
    bool deeper = false;
    if (in_path && in_now && nero_predicted_hold_moving()) {
      double ahead = 0.0;
      size_t k = nearest;
      while (k + 1 < partial_trajectory_.points.size() && ahead < 0.5) {
        ahead += autoware_utils::calc_distance2d(
          partial_trajectory_.points.at(k).pose, partial_trajectory_.points.at(k + 1).pose);
        ++k;
      }
      const int d_now = nero_depth(current_pose_.pose, front, rear, half_w, way);
      const int d_next = nero_depth(partial_trajectory_.points.at(k).pose, front, rear, half_w, way);
      deeper = d_next > d_now;
      // nor while it is beside or in front of the vehicle: its body (now) on the next 2 m of the path
      double s_ahead = 0.0;
      for (size_t i = nearest; i < partial_trajectory_.points.size() && s_ahead <= 2.0 && !deeper; ++i) {
        if (i > nearest) {
          s_ahead += autoware_utils::calc_distance2d(
            partial_trajectory_.points.at(i - 1).pose, partial_trajectory_.points.at(i).pose);
        }
        deeper = nero_overlap(way.front(), path_fp.at(i - nearest));
      }
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "[NERO_FREESPACE_PREDICTED_HOLD_MOVING] in the way of a road user (%.1f m/s): depth %d now, %d %.1f m on "
        "(%zu -> %zu of %zu): %s",
        v, d_now, d_next, ahead, nearest, k, partial_trajectory_.points.size(), deeper ? "holding" : "going on");
    }
    if (in_path && (!in_now || deeper)) {
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

bool FreespacePlannerNode::neroParkedEnough(
  const geometry_msgs::msg::Pose & pose, const geometry_msgs::msg::Pose & ref, std::string & why)
{
  const auto & b = nero_parked_box();
  const double front = vehicle_shape_.length - vehicle_shape_.base2back;
  const double rear = vehicle_shape_.base2back;
  const double half_w = vehicle_shape_.width / 2.0;
  const double ryaw = tf2::getYaw(ref.orientation);
  const double dyaw = autoware_utils::normalize_radian(tf2::getYaw(pose.orientation) - ryaw);
  // the real footprint's corners in the planned end pose's frame (u along its heading, v to its left)
  const auto q = nero_quad(pose, front, rear, half_w);
  double u_max = -1e9, u_min = 1e9, v_abs = 0.0;
  for (const auto & c : q) {
    const double dx = c[0] - ref.position.x;
    const double dy = c[1] - ref.position.y;
    const double u = std::cos(ryaw) * dx + std::sin(ryaw) * dy;
    const double v = -std::sin(ryaw) * dx + std::cos(ryaw) * dy;
    u_max = std::max(u_max, u);
    u_min = std::min(u_min, u);
    v_abs = std::max(v_abs, std::abs(v));
  }
  const double out = u_max - front;    // > 0: reaches further out than planned
  const double deep = -rear - u_min;   // > 0: deeper than planned
  char buf[200];
  std::snprintf(
    buf, sizeof(buf), "footprint %.2f m out / %.2f m deep of the planned one, corners within %.2f m of its axis, %.1f deg",
    out, deep, v_abs, std::abs(dyaw) * 180.0 / M_PI);
  why = buf;
  return out <= b.front && deep <= b.front && v_abs <= b.side && std::abs(dyaw) <= b.deg * M_PI / 180.0;
}

bool FreespacePlannerNode::neroStraightFree(
  const geometry_msgs::msg::Pose & start, const bool back, const double len)
{
  if (!occupancy_grid_) return false;
  const auto & g = *occupancy_grid_;
  const double res = g.info.resolution;
  const int threshold = static_cast<int>(get_parameter("obstacle_threshold").as_int());
  const double m = node_param_.vehicle_shape_margin_m / 2.0;
  const double front = vehicle_shape_.length - vehicle_shape_.base2back;
  const double rear = vehicle_shape_.base2back;
  const double half_w = vehicle_shape_.width / 2.0;
  const double yaw = tf2::getYaw(start.orientation);
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  const double gyaw = tf2::getYaw(g.info.origin.orientation);
  const double gc = std::cos(gyaw);
  const double gs = std::sin(gyaw);
  // the area the move adds to the footprint (forward: ahead of the front; reversing: behind the rear), for the real
  // footprint (cell centres inside) and its margin (cells reaching into it); what the start already stands next to
  // (the wall the vehicle moves away from) is left behind and never counts
  const double r0 = back ? -rear - len : front;
  const double r1 = back ? -rear : front + len;
  const double m0 = back ? -rear - len - m : front;
  const double m1 = back ? -rear : front + len + m;
  for (unsigned j = 0; j < g.info.height; ++j) {
    for (unsigned i = 0; i < g.info.width; ++i) {
      const int8_t val = g.data[j * g.info.width + i];
      if (val >= 0 && val < threshold) continue;
      const double lx = (i + 0.5) * res;
      const double ly = (j + 0.5) * res;
      const double x = g.info.origin.position.x + gc * lx - gs * ly - start.position.x;
      const double y = g.info.origin.position.y + gs * lx + gc * ly - start.position.y;
      const double u = c * x + s * y;
      const double v = -s * x + c * y;
      if (u >= r0 && u <= r1 && std::abs(v) <= half_w) return false;
      const double h = res / 2.0;
      if (u >= m0 - h && u <= m1 + h && std::abs(v) <= half_w + m + h) return false;
    }
  }
  return true;
}

bool FreespacePlannerNode::neroLeadIn(
  const geometry_msgs::msg::Pose & start, const bool start_blocked,
  std::vector<autoware::freespace_planning_algorithms::PlannerWaypoint> & lead, std::string & why)
{
  lead.clear();
  if (!occupancy_grid_) return false;
  const double yaw = tf2::getYaw(start.orientation);
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  // forward >= 1.5 m / reversing >= 1.0 m first (MPC 1 m / PID 0.5 m from standstill), longer ones after
  const std::array<std::pair<bool, double>, 6> cands{
    {{false, 1.5}, {true, 1.0}, {false, 2.0}, {true, 1.5}, {false, 2.5}, {true, 2.0}}};
  for (const auto & [back, len] : cands) {
    if (!neroStraightFree(start, back, len)) {
      why += std::string(back ? "rev " : "fwd ") + std::to_string(len).substr(0, 3) + " m blocked; ";
      continue;
    }
    geometry_msgs::msg::Pose end = start;
    const double d = back ? -len : len;
    end.position.x += c * d;
    end.position.y += s * d;
    geometry_msgs::msg::PoseArray a;
    a.poses.push_back(end);
    if (algo_->hasObstacleOnTrajectory(a)) {
      why += std::string(back ? "rev " : "fwd ") + std::to_string(len).substr(0, 3) + " m ends inside the margin; ";
      continue;
    }
    const int n = static_cast<int>(std::ceil(len / 0.25));
    for (int k = 0; k <= n; ++k) {
      autoware::freespace_planning_algorithms::PlannerWaypoint wp;
      wp.pose.header.frame_id = occupancy_grid_->header.frame_id;
      wp.pose.pose = start;
      wp.pose.pose.position.x += c * d * k / n;
      wp.pose.pose.position.y += s * d * k / n;
      wp.is_back = back;
      lead.push_back(wp);
    }
    why = std::string(back ? "reversing " : "forward ") + std::to_string(len).substr(0, 3) + " m straight" +
          (start_blocked ? " away from it" : "");
    return true;
  }
  return false;
}

bool FreespacePlannerNode::neroStallGuard()
{
  const double now = get_clock()->now().seconds();
  // only consecutive cycles of a published moving partial trajectory count (a hold in between restarts the clock)
  const bool gap = nero_stall_last_t_ < 0.0 || now - nero_stall_last_t_ > 0.5;
  nero_stall_last_t_ = now;
  const bool autonomous =
    nero_op_ && nero_op_->mode == autoware_adapi_v1_msgs::msg::OperationModeState::AUTONOMOUS &&
    nero_op_->is_autoware_control_enabled;
  // the controller itself commands 0 m/s (its STOPPED state): a downstream stop (AEB, the gate) leaves it commanding
  // motion and never counts
  const bool cmd_zero = nero_cmd_ && std::abs(nero_cmd_->longitudinal.velocity) < 1e-3;
  const bool stopped = utils::is_stopped(odom_buffer_, node_param_.th_stopped_velocity_mps);
  const bool standing = autonomous && cmd_zero && stopped;
  if (gap || !standing) {
    nero_stall_since_ = standing ? now : -1.0;
    return false;
  }
  if (nero_stall_since_ < 0.0) nero_stall_since_ = now;
  if (now - nero_stall_since_ < nero_stall_s()) return false;
  const double stood = now - nero_stall_since_;
  nero_stall_since_ = -1.0;
  ++nero_stall_recoveries_;
  const size_t i0 =
    autoware::motion_utils::findNearestIndex(partial_trajectory_.points, current_pose_.pose.position);
  const double len = std::abs(autoware::motion_utils::calcSignedArcLength(
    partial_trajectory_.points, i0, partial_trajectory_.points.size() - 1));
  std::string why;
  if (nero_parked_box().on && neroParkedEnough(current_pose_.pose, trajectory_.points.back().pose, why)) {
    RCLCPP_INFO(
      get_logger(), "[NERO_FREESPACE_STALL_S] standing %.1f s with a plan that says move (%.2f m left); %s: parked",
      stood, len, why.c_str());
    is_completed_ = true;
    std_msgs::msg::Bool is_completed_msg;
    is_completed_msg.data = true;
    parking_state_pub_->publish(is_completed_msg);
    return true;
  }
  RCLCPP_INFO(
    get_logger(),
    "[NERO_FREESPACE_STALL_S] standing %.1f s with a plan that says move (%.2f m left): planning again from here, a "
    "straight move first (%d)",
    stood, len, nero_stall_recoveries_);
  reset();
  reset_in_progress_ = true;
  nero_lead_in_now_ = true;
  return true;
}

void FreespacePlannerNode::onRoute(const LaneletRoute::ConstSharedPtr msg)
{
  route_ = msg;

  goal_pose_.header = msg->header;
  goal_pose_.pose = msg->goal_pose;

  is_new_parking_cycle_ = true;
  nero_reparks_ = 0;
  nero_short_replans_ = 0;
  nero_straight_wait_since_ = -1.0;
  nero_stall_since_ = -1.0;
  nero_stall_recoveries_ = 0;
  nero_unstartable_replans_ = 0;
  nero_lead_in_now_ = false;

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
      if (is_completed_) {  // NERO_FREESPACE_PARKED_BOX: already parked at the goal
        trajectory_pub_->publish(utils::create_stop_trajectory(partial_trajectory_));
        return;
      }
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

  // NERO_FREESPACE_STRAIGHT_FWD_MIN_M: a segment (not the last) the vehicle cannot start from standstill - forward
  // shorter than the MPC/PID need, reversing under the velocity smoother's 0.5 m engage distance - is planned again
  // from where the vehicle stands (sim: a 0.5 m reverse after a re-plan round a passing car: the robot stood for
  // good), at most 3 times per goal
  const bool nero_stopped = utils::is_stopped(odom_buffer_, node_param_.th_stopped_velocity_mps);
  if (!nero_stopped) {
    nero_unstartable_replans_ = 0;
    nero_stall_recoveries_ = 0;
  }
  if (
    !nero_min_segment_on() && nero_straight_fwd_min_m() > 0.0 && nero_short_replans_ < 3 &&
    target_index_ + 1 < trajectory_.points.size() && partial_trajectory_.points.size() > 1 && nero_stopped) {
    const size_t i0 =
      autoware::motion_utils::findNearestIndex(partial_trajectory_.points, current_pose_.pose.position);
    const double len = std::abs(autoware::motion_utils::calcSignedArcLength(
      partial_trajectory_.points, i0, partial_trajectory_.points.size() - 1));
    const bool fwd = partial_trajectory_.points.front().longitudinal_velocity_mps > 0.0;
    if (len < (fwd ? nero_straight_fwd_min_m() : 0.6)) {
      ++nero_short_replans_;
      RCLCPP_INFO(
        get_logger(),
        "[NERO_FREESPACE_STRAIGHT_FWD_MIN_M] the next %s segment is %.2f m from here, too short to start: "
        "planning again from here (%d/3)",
        fwd ? "forward" : "reversing", len, nero_short_replans_);
      trajectory_pub_->publish(utils::create_stop_trajectory(partial_trajectory_));
      reset();
      reset_in_progress_ = true;
      return;
    }
  }

  // NERO_FREESPACE_MIN_SEGMENT: never publish a move the vehicle cannot start from where it stands (not once the
  // planner has finished: updateTargetIndex() just completed it)
  if (nero_min_segment_on() && !is_completed_ && nero_stopped && partial_trajectory_.points.size() > 1) {
    const size_t i0 =
      autoware::motion_utils::findNearestIndex(partial_trajectory_.points, current_pose_.pose.position);
    const double len = std::abs(autoware::motion_utils::calcSignedArcLength(
      partial_trajectory_.points, i0, partial_trajectory_.points.size() - 1));
    const bool fwd = partial_trajectory_.points.front().longitudinal_velocity_mps > 0.0;
    if (len < (fwd ? kNeroStartFwdM : kNeroStartRevM)) {
      const bool last = target_index_ + 1 >= trajectory_.points.size();
      std::string why;
      if (last && nero_parked_box().on && neroParkedEnough(current_pose_.pose, trajectory_.points.back().pose, why)) {
        RCLCPP_INFO(
          get_logger(), "[NERO_FREESPACE_MIN_SEGMENT] the last %s segment is %.2f m from here, too short to start; %s: parked",
          fwd ? "forward" : "reversing", len, why.c_str());
        is_completed_ = true;
        std_msgs::msg::Bool is_completed_msg;
        is_completed_msg.data = true;
        parking_state_pub_->publish(is_completed_msg);
        trajectory_pub_->publish(utils::create_stop_trajectory(partial_trajectory_));
        return;
      }
      ++nero_unstartable_replans_;
      // the 1st time a plain re-plan (A*'s first segment can be started), then with a straight move first
      nero_lead_in_now_ = nero_unstartable_replans_ > 1;
      RCLCPP_INFO(
        get_logger(),
        "[NERO_FREESPACE_MIN_SEGMENT] the next %s segment is %.2f m from here, too short to start: planning again from "
        "here%s (%d)",
        fwd ? "forward" : "reversing", len, nero_lead_in_now_ ? ", a straight move first" : "",
        nero_unstartable_replans_);
      trajectory_pub_->publish(utils::create_stop_trajectory(partial_trajectory_));
      reset();
      reset_in_progress_ = true;
      return;
    }
  }

  if (nero_predicted_hold_s() > 0.0 && neroPredictedHold()) {
    const auto hold = utils::create_stop_trajectory(partial_trajectory_);
    trajectory_pub_->publish(hold);
    debug_pose_array_pub_->publish(utils::trajectory_to_pose_array(trajectory_));
    debug_partial_pose_array_pub_->publish(utils::trajectory_to_pose_array(hold));
    is_new_parking_cycle_ = false;
    return;
  }

  // NERO_FREESPACE_STALL_S: standing too long with this plan -> recovered (done or planning again): stop for now
  if (nero_stall_s() > 0.0 && !is_completed_ && neroStallGuard()) {
    trajectory_pub_->publish(utils::create_stop_trajectory(partial_trajectory_));
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

  if (nero_predicted_hold_moving() && get_clock()->now().seconds() - nero_failed_search_t_ < 1.0) {
    // NERO_FREESPACE_PREDICTED_HOLD_MOVING: a full A* search that found nothing used its whole time limit inside this
    // timer; the next one waits 1 s so the hold keeps being published (sim: after a hold for a person standing on the
    // path, 9 failing searches back to back dropped the trajectory rate -> emergency stop)
    reset();
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

  // NERO_FREESPACE_PARKED_BOX: a vehicle already standing parked at the goal (a route sent again at home) is done
  if (nero_parked_box().on && !nero_repark_now_) {
    std::string why;
    if (neroParkedEnough(current_pose_.pose, goal_pose_.pose, why)) {
      RCLCPP_INFO(get_logger(), "[NERO_FREESPACE_PARKED_BOX] already parked at the goal (%s): done", why.c_str());
      trajectory_ = utils::create_stop_trajectory(current_pose_, get_clock());
      partial_trajectory_ = trajectory_;
      reversing_indices_.clear();
      prev_target_index_ = 0;
      target_index_ = trajectory_.points.size() - 1;
      is_completed_ = true;
      std_msgs::msg::Bool is_completed_msg;
      is_completed_msg.data = true;
      parking_state_pub_->publish(is_completed_msg);
      return;
    }
  }

  // NERO_FREESPACE_START_ESCAPE (a start inside the margin) / NERO_FREESPACE_STALL_S, NERO_FREESPACE_MIN_SEGMENT (stood
  // with a plan that says move): begin with a straight move the vehicle can start; A* plans on from its end
  std::vector<autoware::freespace_planning_algorithms::PlannerWaypoint> nero_lead;
  auto plan_start = current_pose_in_costmap_frame;
  {
    bool start_blocked = false;
    if (nero_start_escape()) {
      geometry_msgs::msg::PoseArray a;
      a.poses.push_back(current_pose_in_costmap_frame);
      start_blocked = algo_->hasObstacleOnTrajectory(a);
      const double now_s = get_clock()->now().seconds();
      if (!start_blocked) {
        nero_start_blocked_since_ = -1.0;
      } else {
        if (nero_start_blocked_since_ < 0.0) nero_start_blocked_since_ = now_s;
        const double held = now_s - nero_start_blocked_since_;
        if (held < nero_start_escape_hold_s()) {
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 1000,
            "[NERO_FREESPACE_START_ESCAPE] the start is inside the margin since %.1f s: waiting %.1f s before "
            "escaping",
            held, nero_start_escape_hold_s());
          start_blocked = false;  // A* below says "Invalid start or goal pose" as upstream; planned again next cycle
        }
      }
    }
    if (start_blocked || nero_lead_in_now_) {
      std::string why;
      if (neroLeadIn(current_pose_in_costmap_frame, start_blocked, nero_lead, why)) {
        plan_start = nero_lead.back().pose.pose;
        RCLCPP_INFO(
          get_logger(), "[%s] %s: %s, then A* from there",
          start_blocked ? "NERO_FREESPACE_START_ESCAPE" : "NERO_FREESPACE_STALL_S",
          start_blocked ? "the start is inside the margin" : "planning again", why.c_str());
      } else {
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 2000, "[%s] no straight move it can start from here: %s",
          start_blocked ? "NERO_FREESPACE_START_ESCAPE" : "NERO_FREESPACE_STALL_S", why.c_str());
      }
    }
    nero_lead_in_now_ = false;
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
      result = algo_->makePlan(plan_start, goal_pose_in_costmap_frame);
    } catch (const std::exception & e) {
      error_msg = e.what();
    }
  }
  const rclcpp::Time end = get_clock()->now();
  if (!result && !nero_repark_now_ && (end - start).seconds() > 0.1) {
    nero_failed_search_t_ = end.seconds();
  }

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
      // NERO_FREESPACE_STRAIGHT_FWD_MIN_M: a forward straight piece long enough to start even after the vehicle
      // stopped 0.2-0.3 m early at the end of a reverse before it (the MPC/PID deadlock, see there)
      const double fwd_min = back ? 0.0 : nero_straight_fwd_min_m();
      const double dd = fwd_min > 0.0 ? std::max(d, fwd_min + 0.35) : d;
      for (const double f : {1.0, 0.75, 0.5, 0.33}) {
        // a time budget for the extra A* calls: the timer must publish at least every second
        // (topic_state_monitor on the trajectory: 1 s -> MRM). Over budget = try again later.
        if ((get_clock()->now() - start).seconds() > 0.4) {
          why += "time budget (0.4 s) used up; ";
          break;
        }
        if (fwd_min > 0.0 && f * dd < fwd_min + 0.35) continue;
        if (nero_straight_in_stop() && f * dd < 0.8) {
          // with its own segment the straight piece must be long enough to cut (0.3 m) and still
          // drive straight; shorter ones keep the +-1.5 m curvature feed-forward: skip, then wait
          // (blocked) or the plan without it and NERO_FREESPACE_REPARK_DEG
          continue;
        }
        auto pre = goal;
        pre.position.x += sgn * f * dd * ux;
        pre.position.y += sgn * f * dd * uy;
        bool pre_found = false;
        std::string astar_err;
        try {
          // A* throws when the pose is blocked or unreachable: that is "no plan" here, never an
          // exception out of the timer (it would stop the planner)
          pre_found = algo_->makePlan(plan_start, pre);
        } catch (const std::exception & e) {
          pre_found = false;
          astar_err = e.what();
        }
        const std::string dtxt = std::to_string(f * dd).substr(0, 4) + " m: ";
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
    if (!nero_lead.empty() && !waypoints.waypoints.empty()) {
      // the straight move first: its last point is A*'s start (dropped), a reversing point when A* starts the other way
      waypoints.waypoints.erase(waypoints.waypoints.begin());
      waypoints.waypoints.insert(waypoints.waypoints.begin(), nero_lead.begin(), nero_lead.end());
      if (straight_from > 0) straight_from += nero_lead.size() - 1;
    }
    nero_lead_end_ = nero_lead.empty() ? -1 : static_cast<int>(nero_lead.size()) - 1;
    nero_lead_back_ = !nero_lead.empty() && nero_lead.front().is_back;
    trajectory_ =
      utils::create_trajectory(current_pose_, waypoints, node_param_.waypoints_velocity);
    if (nero_min_segment_on() && trajectory_.points.size() <= 1) {
      // NERO_FREESPACE_MIN_SEGMENT: A*'s plan is its start alone: the vehicle already stands at the goal (within
      // A*'s goal ranges); done (upstream holds with an empty partial trajectory and the next collision check throws)
      RCLCPP_INFO(get_logger(), "[NERO_FREESPACE_MIN_SEGMENT] already at the goal (A* plans no move): done");
      trajectory_ = utils::create_stop_trajectory(current_pose_, get_clock());
      partial_trajectory_ = trajectory_;
      reversing_indices_.clear();
      prev_target_index_ = 0;
      target_index_ = trajectory_.points.size() - 1;
      nero_lead_end_ = -1;
      is_completed_ = true;
      std_msgs::msg::Bool is_completed_msg;
      is_completed_msg.data = true;
      parking_state_pub_->publish(is_completed_msg);
      return;
    }
    reversing_indices_ = utils::get_reversing_indices(trajectory_);
    // NERO_FREESPACE_STRAIGHT_IN_STOP=1: the straight piece as its own partial trajectory, cut 0.3 m
    // into it (the vehicle stops 0.2-0.3 m short of a segment end, i.e. where the straight begins).
    const bool straight_own_segment =
      straight_from > 0 && std::find(
                             reversing_indices_.begin(), reversing_indices_.end(),
                             straight_from - 1) != reversing_indices_.end();
    if (const size_t cut = nero_straight_cut(trajectory_, straight_from);
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
  nero_lead_end_ = -1;
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
