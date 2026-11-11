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

#include "remaining_distance_time_calculator_node.hpp"

#include <autoware_lanelet2_extension/utility/message_conversion.hpp>
#include <autoware_lanelet2_extension/utility/query.hpp>
#include <autoware_lanelet2_extension/utility/utilities.hpp>
#include <autoware_utils/geometry/geometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/timer.hpp>

#include <autoware_internal_planning_msgs/msg/velocity_limit.hpp>

#include <boost/geometry/algorithms/comparable_distance.hpp>

#include <lanelet2_core/geometry/Polygon.h>
#include <tf2/utils.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
#include <functional>
#include <memory>
#include <string>
#include <utility>


// NERO_RDTC_FAST (trjrate 2026-11-09, QUEUE TRJ3): the same remaining distance with less CPU. Upstream runs
// lanelet::utils::query::getClosestLanelet over EVERY road lanelet of the map twice per cycle (ego and the goal), building each
// lanelet's polygon2d().basicPolygon() on every call (10 Hz, ~0.15 of a NeroStation core, a planning process on the car's cores).
//   unset / 0 = the car. 1 = fast: (a) the goal's lanelet is found once per route (goal pose and map fixed until the next route /
//   map); (b) ego's lanelet with the same algorithm (same order, same ties, same angle step) on polygons built once per map and a
//   box lower bound that skips a lanelet which can be neither a new minimum nor a tie (bound 1e-9 relative above the running
//   minimum: rounding cannot flip it); (c) the shortest path re-used while (ego lanelet, goal lanelet) are unchanged (the routing
//   graph is fixed). 2 = check: both computed every cycle and compared (lanelet ids, path ids); upstream's used; totals every
//   10 s and at exit as [NERO_RDTC].
namespace
{
int nero_rdtc_mode()
{
  static const int m = []() {
    const char * e = std::getenv("NERO_RDTC_FAST");
    if (!e || !*e) return 0;
    const int v = std::atoi(e);
    return (v == 1 || v == 2) ? v : 0;
  }();
  return m;
}
struct NeroRdtc
{
  struct Poly
  {
    lanelet::BasicPolygon2d poly;
    double minx, miny, maxx, maxy;
  };
  const void * map_key{nullptr};
  size_t n_lanelets{0};
  std::vector<Poly> polys;
  bool goal_valid{false};
  bool goal_found{false};
  lanelet::ConstLanelet goal_lanelet;
  lanelet::Id path_from{lanelet::InvalId}, path_to{lanelet::InvalId};
  bool path_ok{false};
  lanelet::routing::LaneletPath path;
  std::atomic<uint64_t> cycles{0}, skipped{0}, scanned{0}, mismatch{0};
  int64_t last_print_ns{0};
  void print(const char * when)
  {
    std::fprintf(
      stderr,
      "[NERO_RDTC] %s mode %d: cycles %llu, lanelet distances %llu (box-skipped %llu), MISMATCH %llu\n", when,
      nero_rdtc_mode(), (unsigned long long)cycles.load(), (unsigned long long)scanned.load(),
      (unsigned long long)skipped.load(), (unsigned long long)mismatch.load());
    std::fflush(stderr);
  }
  ~NeroRdtc()
  {
    if (nero_rdtc_mode() != 0) print("exit");
  }
};
NeroRdtc & nero_rdtc()
{
  static NeroRdtc s;
  return s;
}
void nero_rdtc_polys(const lanelet::ConstLanelets & lanelets, const void * map_key)
{
  auto & st = nero_rdtc();
  if (st.map_key == map_key && st.n_lanelets == lanelets.size()) return;
  st.map_key = map_key;
  st.n_lanelets = lanelets.size();
  st.polys.clear();
  st.polys.reserve(lanelets.size());
  for (const auto & llt : lanelets) {
    NeroRdtc::Poly p{llt.polygon2d().basicPolygon(), std::numeric_limits<double>::max(),
                     std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
                     std::numeric_limits<double>::lowest()};
    for (const auto & q : p.poly) {
      p.minx = std::min(p.minx, q.x());
      p.miny = std::min(p.miny, q.y());
      p.maxx = std::max(p.maxx, q.x());
      p.maxy = std::max(p.maxy, q.y());
    }
    st.polys.push_back(std::move(p));
  }
  st.goal_valid = false;
  st.path_ok = false;
  st.path_from = st.path_to = lanelet::InvalId;
}
double nero_normalize_radian(const double rad)  // = lanelet::utils::impl::normalize_radian (lanelet2_extension, private)
{
  constexpr double pi = 3.14159265358979323846;
  constexpr double min_rad = -pi;
  const auto max_rad = min_rad + 2 * pi;
  const auto value = std::fmod(rad, 2 * pi);
  if (min_rad <= value && value < max_rad) {
    return value;
  }
  return value - std::copysign(2 * pi, value);
}
// = lanelet::utils::query::getClosestLanelet(lanelets, pose, out) on the kept polygons (same order, same ties, same result)
bool nero_closest(
  const lanelet::ConstLanelets & lanelets, const geometry_msgs::msg::Pose & search_pose,
  lanelet::ConstLanelet * closest_lanelet_ptr)
{
  auto & st = nero_rdtc();
  if (lanelets.empty()) {
    return false;
  }
  bool found = false;
  lanelet::BasicPoint2d search_point(search_pose.position.x, search_pose.position.y);
  std::vector<size_t> candidates;
  {
    double min_distance = std::numeric_limits<double>::max();
    for (size_t i = 0; i < lanelets.size(); ++i) {
      const auto & p = st.polys[i];
      const double dx = std::max({p.minx - search_point.x(), 0.0, search_point.x() - p.maxx});
      const double dy = std::max({p.miny - search_point.y(), 0.0, search_point.y() - p.maxy});
      const double lb = dx * dx + dy * dy;  // <= the squared (comparable) distance to the polygon
      if (min_distance < std::numeric_limits<double>::max() && lb > min_distance * (1.0 + 1e-9) + 1e-12) {
        st.skipped++;
        continue;
      }
      st.scanned++;
      const double distance = boost::geometry::comparable_distance(p.poly, search_point);
      if (std::abs(distance - min_distance) <= std::numeric_limits<double>::epsilon()) {
        candidates.push_back(i);
      } else if (distance < min_distance) {
        found = true;
        candidates.clear();
        candidates.push_back(i);
        min_distance = distance;
      }
    }
  }
  if (candidates.size() == 1) {
    *closest_lanelet_ptr = lanelets.at(candidates.at(0));
    return found;
  }
  {
    double min_angle = std::numeric_limits<double>::max();
    const double pose_yaw = tf2::getYaw(search_pose.orientation);
    for (const auto i : candidates) {
      const auto & llt = lanelets.at(i);
      lanelet::ConstLineString3d segment = lanelet::utils::getClosestSegment(search_point, llt.centerline());
      double angle_diff = M_PI;
      if (!segment.empty()) {
        const double segment_angle = std::atan2(
          segment.back().y() - segment.front().y(), segment.back().x() - segment.front().x());
        angle_diff = std::abs(nero_normalize_radian(segment_angle - pose_yaw));
      }
      if (angle_diff < min_angle) {
        min_angle = angle_diff;
        *closest_lanelet_ptr = llt;
      }
    }
  }
  return found;
}
}  // namespace

namespace autoware::remaining_distance_time_calculator
{

RemainingDistanceTimeCalculatorNode::RemainingDistanceTimeCalculatorNode(
  const rclcpp::NodeOptions & options)
: Node("remaining_distance_time_calculator", options),
  is_graph_ready_{false},
  has_received_route_{false},
  has_received_scenario_{false},
  velocity_limit_{99.99},
  remaining_distance_{0.0},
  remaining_time_{0.0}
{
  using std::placeholders::_1;

  sub_odometry_ = create_subscription<Odometry>(
    "~/input/odometry", 1, std::bind(&RemainingDistanceTimeCalculatorNode::on_odometry, this, _1));

  const auto qos_transient_local = rclcpp::QoS{1}.transient_local();

  sub_map_ = create_subscription<HADMapBin>(
    "~/input/map", qos_transient_local,
    std::bind(&RemainingDistanceTimeCalculatorNode::on_map, this, _1));
  sub_route_ = create_subscription<LaneletRoute>(
    "~/input/route", qos_transient_local,
    std::bind(&RemainingDistanceTimeCalculatorNode::on_route, this, _1));
  sub_planning_velocity_ = create_subscription<autoware_internal_planning_msgs::msg::VelocityLimit>(
    "/planning/scenario_planning/current_max_velocity", qos_transient_local,
    std::bind(&RemainingDistanceTimeCalculatorNode::on_velocity_limit, this, _1));
  sub_scenario_ = this->create_subscription<autoware_internal_planning_msgs::msg::Scenario>(
    "~/input/scenario", 1, std::bind(&RemainingDistanceTimeCalculatorNode::on_scenario, this, _1));

  pub_mission_remaining_distance_time_ = create_publisher<MissionRemainingDistanceTime>(
    "~/output/mission_remaining_distance_time",
    rclcpp::QoS(rclcpp::KeepLast(10)).durability_volatile().reliable());

  param_listener_ = std::make_shared<::remaining_distance_time_calculator::ParamListener>(
    this->get_node_parameters_interface());
  const auto param = param_listener_->get_params();

  const auto period_ns = rclcpp::Rate(param.update_rate).period();
  timer_ = rclcpp::create_timer(
    this, get_clock(), period_ns, std::bind(&RemainingDistanceTimeCalculatorNode::on_timer, this));
}

void RemainingDistanceTimeCalculatorNode::on_map(const HADMapBin::ConstSharedPtr & msg)
{
  route_handler_.setMap(*msg);
  lanelet_map_ptr_ = std::make_shared<lanelet::LaneletMap>();
  lanelet::utils::conversion::fromBinMsg(
    *msg, lanelet_map_ptr_, &traffic_rules_ptr_, &routing_graph_ptr_);
  lanelet::ConstLanelets all_lanelets = lanelet::utils::query::laneletLayer(lanelet_map_ptr_);
  road_lanelets_ = lanelet::utils::query::roadLanelets(all_lanelets);
  nero_rdtc().map_key = nullptr;  // NERO_RDTC_FAST: polygons, goal lanelet and path are built again for the new map
  is_graph_ready_ = true;
}

void RemainingDistanceTimeCalculatorNode::on_odometry(const Odometry::ConstSharedPtr & msg)
{
  current_vehicle_pose_ = msg->pose.pose;
  current_vehicle_velocity_ = msg->twist.twist.linear;
}

void RemainingDistanceTimeCalculatorNode::on_route(const LaneletRoute::ConstSharedPtr & msg)
{
  goal_pose_ = msg->goal_pose;
  has_received_route_ = true;
  nero_rdtc().goal_valid = false;  // NERO_RDTC_FAST: the goal's lanelet is found again for the new route
}

void RemainingDistanceTimeCalculatorNode::on_velocity_limit(
  const VelocityLimit::ConstSharedPtr & msg)
{
  if (msg->max_velocity > 1e-5) {
    velocity_limit_ = msg->max_velocity;
  }
}

void RemainingDistanceTimeCalculatorNode::on_scenario(
  const autoware_internal_planning_msgs::msg::Scenario::ConstSharedPtr & msg)
{
  scenario_ = msg;
  has_received_scenario_ = true;
}

void RemainingDistanceTimeCalculatorNode::on_timer()
{
  if (!has_received_scenario_) {
    return;
  }

  // check if the scenario is parking or not
  if (scenario_->current_scenario == autoware_internal_planning_msgs::msg::Scenario::PARKING) {
    remaining_distance_ = 0.0;
    remaining_time_ = 0.0;
    publish_mission_remaining_distance_time();
  } else if (is_graph_ready_ && has_received_route_) {
    calculate_remaining_distance();
    calculate_remaining_time();
    publish_mission_remaining_distance_time();
  }
}

void RemainingDistanceTimeCalculatorNode::calculate_remaining_distance()
{
  if (nero_rdtc_mode() != 0) {
    calculate_remaining_distance_nero();
    return;
  }
  calculate_remaining_distance_upstream();
}

// NERO_RDTC_FAST: the same result as calculate_remaining_distance_upstream (mode 1), checked against it (mode 2)
void RemainingDistanceTimeCalculatorNode::calculate_remaining_distance_nero()
{
  auto & st = nero_rdtc();
  const int mode = nero_rdtc_mode();
  st.cycles++;
  const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
  if (now - st.last_print_ns > 10'000'000'000LL) {
    if (st.last_print_ns != 0) st.print("10s");
    st.last_print_ns = now;
  }
  nero_rdtc_polys(road_lanelets_, lanelet_map_ptr_.get());
  lanelet::ConstLanelet current_lanelet;
  const bool cur_found = nero_closest(road_lanelets_, current_vehicle_pose_, &current_lanelet);
  if (!st.goal_valid) {
    st.goal_found = nero_closest(road_lanelets_, goal_pose_, &st.goal_lanelet);
    st.goal_valid = true;
  }
  if (mode == 2) {
    const double before = remaining_distance_;
    calculate_remaining_distance_upstream();
    const double ref = remaining_distance_;
    remaining_distance_ = before;
    lanelet::ConstLanelet ref_cur;
    lanelet::ConstLanelet ref_goal;
    const bool rc = lanelet::utils::query::getClosestLanelet(road_lanelets_, current_vehicle_pose_, &ref_cur);
    const bool rg = lanelet::utils::query::getClosestLanelet(road_lanelets_, goal_pose_, &ref_goal);
    bool same = rc == cur_found && rg == st.goal_found && (!rc || ref_cur.id() == current_lanelet.id()) &&
                (!rg || ref_goal.id() == st.goal_lanelet.id());
    calculate_remaining_distance_fast(cur_found, current_lanelet);
    same = same && remaining_distance_ == ref;
    if (!same) st.mismatch++;
    remaining_distance_ = ref;
    return;
  }
  calculate_remaining_distance_fast(cur_found, current_lanelet);
}

void RemainingDistanceTimeCalculatorNode::calculate_remaining_distance_fast(
  const bool cur_found, const lanelet::ConstLanelet & current_lanelet)
{
  auto & st = nero_rdtc();
  if (!cur_found) {
    RCLCPP_WARN_STREAM_THROTTLE(
      this->get_logger(), *get_clock(), 3000, "Failed to find current lanelet.");
    return;
  }
  if (!st.goal_found) {
    RCLCPP_WARN_STREAM_THROTTLE(
      this->get_logger(), *get_clock(), 3000, "Failed to find goal lanelet.");
    return;
  }
  const auto & goal_lanelet = st.goal_lanelet;
  if (!(st.path_from == current_lanelet.id() && st.path_to == goal_lanelet.id())) {
    const lanelet::Optional<lanelet::routing::Route> optional_route =
      routing_graph_ptr_->getRoute(current_lanelet, goal_lanelet, 0);
    st.path_from = current_lanelet.id();
    st.path_to = goal_lanelet.id();
    st.path_ok = static_cast<bool>(optional_route);
    st.path = st.path_ok ? optional_route->shortestPath() : lanelet::routing::LaneletPath{};
  }
  if (!st.path_ok) {
    RCLCPP_WARN_STREAM_THROTTLE(
      this->get_logger(), *get_clock(), 3000, "Failed to find proper route.");
    return;
  }
  const auto & remaining_shortest_path = st.path;
  size_t index = 0;
  remaining_distance_ = 0.0;

  for (auto & llt : remaining_shortest_path) {
    if (remaining_shortest_path.size() == 1) {
      auto arc_coord_cur = lanelet::utils::getArcCoordinates({llt}, current_vehicle_pose_);
      auto arc_coord_goal = lanelet::utils::getArcCoordinates({llt}, goal_pose_);
      remaining_distance_ += arc_coord_goal.length - arc_coord_cur.length;
      break;
    }

    if (index == 0) {
      lanelet::ArcCoordinates arc_coord =
        lanelet::utils::getArcCoordinates({llt}, current_vehicle_pose_);
      double this_lanelet_length = lanelet::utils::getLaneletLength2d(llt);

      remaining_distance_ += this_lanelet_length - arc_coord.length;
    } else if (index == (remaining_shortest_path.size() - 1)) {
      lanelet::ArcCoordinates arc_coord = lanelet::utils::getArcCoordinates({llt}, goal_pose_);
      remaining_distance_ += arc_coord.length;
    } else {
      remaining_distance_ += lanelet::utils::getLaneletLength2d(llt);
    }

    index++;
  }
}

void RemainingDistanceTimeCalculatorNode::calculate_remaining_distance_upstream()
{
  size_t index = 0;

  lanelet::ConstLanelet current_lanelet;
  if (!lanelet::utils::query::getClosestLanelet(
        road_lanelets_, current_vehicle_pose_, &current_lanelet)) {
    RCLCPP_WARN_STREAM_THROTTLE(
      this->get_logger(), *get_clock(), 3000, "Failed to find current lanelet.");

    return;
  }

  lanelet::ConstLanelet goal_lanelet;
  if (!lanelet::utils::query::getClosestLanelet(road_lanelets_, goal_pose_, &goal_lanelet)) {
    RCLCPP_WARN_STREAM_THROTTLE(
      this->get_logger(), *get_clock(), 3000, "Failed to find goal lanelet.");

    return;
  }

  const lanelet::Optional<lanelet::routing::Route> optional_route =
    routing_graph_ptr_->getRoute(current_lanelet, goal_lanelet, 0);
  if (!optional_route) {
    RCLCPP_WARN_STREAM_THROTTLE(
      this->get_logger(), *get_clock(), 3000, "Failed to find proper route.");

    return;
  }

  lanelet::routing::LaneletPath remaining_shortest_path;
  remaining_shortest_path = optional_route->shortestPath();

  remaining_distance_ = 0.0;

  for (auto & llt : remaining_shortest_path) {
    if (remaining_shortest_path.size() == 1) {
      auto arc_coord_cur = lanelet::utils::getArcCoordinates({llt}, current_vehicle_pose_);
      auto arc_coord_goal = lanelet::utils::getArcCoordinates({llt}, goal_pose_);
      remaining_distance_ += arc_coord_goal.length - arc_coord_cur.length;
      break;
    }

    if (index == 0) {
      lanelet::ArcCoordinates arc_coord =
        lanelet::utils::getArcCoordinates({llt}, current_vehicle_pose_);
      double this_lanelet_length = lanelet::utils::getLaneletLength2d(llt);

      remaining_distance_ += this_lanelet_length - arc_coord.length;
    } else if (index == (remaining_shortest_path.size() - 1)) {
      lanelet::ArcCoordinates arc_coord = lanelet::utils::getArcCoordinates({llt}, goal_pose_);
      remaining_distance_ += arc_coord.length;
    } else {
      remaining_distance_ += lanelet::utils::getLaneletLength2d(llt);
    }

    index++;
  }
}

void RemainingDistanceTimeCalculatorNode::calculate_remaining_time()
{
  if (velocity_limit_ > 0.0) {
    remaining_time_ = remaining_distance_ / velocity_limit_;
  }
}

void RemainingDistanceTimeCalculatorNode::publish_mission_remaining_distance_time()
{
  MissionRemainingDistanceTime mission_remaining_distance_time;

  mission_remaining_distance_time.remaining_distance = remaining_distance_;
  mission_remaining_distance_time.remaining_time = remaining_time_;
  pub_mission_remaining_distance_time_->publish(mission_remaining_distance_time);
}

}  // namespace autoware::remaining_distance_time_calculator

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(
  autoware::remaining_distance_time_calculator::RemainingDistanceTimeCalculatorNode)
