// Copyright 2019 Autoware Foundation
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

#ifndef LANELET2_PLUGINS__DEFAULT_PLANNER_HPP_
#define LANELET2_PLUGINS__DEFAULT_PLANNER_HPP_

#include <autoware/mission_planner_universe/mission_planner_plugin.hpp>
#include <autoware/route_handler/route_handler.hpp>
#include <autoware_utils/geometry/geometry.hpp>
#include <autoware_vehicle_info_utils/vehicle_info_utils.hpp>
#include <rclcpp/rclcpp.hpp>

#include <autoware_map_msgs/msg/lanelet_map_bin.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <lanelet2_routing/RoutingGraph.h>
#include <lanelet2_traffic_rules/TrafficRulesFactory.h>

#include <optional>
#include <vector>

namespace autoware::mission_planner_universe::lanelet2
{

struct DefaultPlannerParameters
{
  double goal_angle_threshold_deg;
  bool enable_correct_goal_pose;
  bool consider_no_drivable_lanes;
  bool check_footprint_inside_lanes;
};

class DefaultPlanner : public mission_planner_universe::PlannerPlugin
{
public:
  DefaultPlanner() : vehicle_info_(), is_graph_ready_(false), param_(), node_(nullptr) {}

  void initialize(rclcpp::Node * node) override;
  void initialize(rclcpp::Node * node, const LaneletMapBin::ConstSharedPtr msg) override;
  [[nodiscard]] bool ready() const override;
  LaneletRoute plan(const RoutePoints & points) override;
  void updateRoute(const PlannerPlugin::LaneletRoute & route) override;
  void clearRoute() override;
  [[nodiscard]] MarkerArray visualize(const LaneletRoute & route) const override;
  [[nodiscard]] static MarkerArray visualize_debug_footprint(
    autoware_utils::LinearRing2d goal_footprint);
  autoware::vehicle_info_utils::VehicleInfo vehicle_info_;

protected:
  using RouteSections = std::vector<autoware_planning_msgs::msg::LaneletSegment>;
  using Pose = geometry_msgs::msg::Pose;
  bool is_graph_ready_;
  autoware::route_handler::RouteHandler route_handler_;

  DefaultPlannerParameters param_;

  rclcpp::Node * node_;
  rclcpp::Subscription<LaneletMapBin>::SharedPtr map_subscriber_;
  rclcpp::Publisher<MarkerArray>::SharedPtr pub_goal_footprint_marker_;

  void initialize_common(rclcpp::Node * node);
  // NERO_KEEP_RIGHT: the goal moved to the rightmost same-direction lane (nullopt = keep it)
  std::optional<Pose> nero_keep_right_goal(const Pose & goal) const;
  // NERO_KEEP_RIGHT_EARLY: every lane change to the right in the lanelet path moved to the earliest lanelet from which
  // the right lane runs parallel up to it (after the U-turn: on 18952 -> 18956, not on 93606 -> 28793)
  // fallback: (right-lane lanelet, inner lanelet beside it) for every lanelet the change was moved past — the inner
  // ones are put back into the route as non-preferred lanes, so ego that cannot change early stays in the inner lane
  // and changes at the original place (the change is attempted early, not forced early)
  lanelet::ConstLanelets nero_early_right_changes(
    const lanelet::ConstLanelets & path,
    std::vector<std::pair<lanelet::Id, lanelet::Id>> * fallback) const;
  // NERO_KEEP_RIGHT_LANES=<D0>,<D1>: every route section prefers its rightmost lane that continues the route; only within
  // D0 + D1 x (lane changes) metres before a place that needs an inner lane (a left turn / fork / the goal lane) the
  // route's own lane stays preferred. At a junction the right lane's parallel piece (same turn) is added to the route
  // and preferred; the route's own piece stays in the route (non-preferred), so ego that cannot get right keeps going
  void nero_keep_right_lanes(RouteSections & sections, double d0, double d1) const;
  void map_callback(const LaneletMapBin::ConstSharedPtr msg);

  /**
   * @brief check if the goal_footprint is within the route lanelets plus the
   * succeeding lanelets around the goal
   * @attention this function will terminate when the accumulated search length from the initial
   * current_lanelet exceeds max_longitudinal_offset_m + search_margin, so under normal assumptions
   * (i.e. the map is composed of finite elements of practically normal sized lanelets), it is
   * assured to terminate
   * @param closest_lanelet_to_goal the route lanelet closest to the goal
   * @param path_lanelets route lanelets
   * @param goal_footprint footprint of the ego vehicle at the goal pose
   */
  [[nodiscard]] bool check_goal_footprint_inside_lanes(
    const lanelet::ConstLanelet & closest_lanelet_to_goal,
    const lanelet::ConstLanelets & path_lanelets,
    const autoware_utils::Polygon2d & goal_footprint) const;

  /**
   * @brief return true if (1)the goal is in parking area or (2)the goal is on the lanes and the
   * footprint around the goal does not overlap the lanes
   */
  bool is_goal_valid(
    const geometry_msgs::msg::Pose & goal, const lanelet::ConstLanelets & path_lanelets);

  /**
   * @brief project the specified goal pose onto the goal lanelet(the last preferred lanelet of
   * route_sections) and return the z-aligned goal position
   */
  Pose refine_goal_height(const Pose & goal, const RouteSections & route_sections);
};

}  // namespace autoware::mission_planner_universe::lanelet2

#endif  // LANELET2_PLUGINS__DEFAULT_PLANNER_HPP_
