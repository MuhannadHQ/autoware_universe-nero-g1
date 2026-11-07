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

#ifndef AUTOWARE__BEHAVIOR_VELOCITY_BLIND_SPOT_MODULE__SCENE_HPP_
#define AUTOWARE__BEHAVIOR_VELOCITY_BLIND_SPOT_MODULE__SCENE_HPP_

#include <autoware/behavior_velocity_blind_spot_module/parameter.hpp>
#include <autoware/behavior_velocity_blind_spot_module/util.hpp>
#include <autoware/behavior_velocity_planner_common/utilization/state_machine.hpp>
#include <autoware/behavior_velocity_rtc_interface/scene_module_interface_with_rtc.hpp>
#include <rclcpp/rclcpp.hpp>

#include <autoware_perception_msgs/msg/predicted_objects.hpp>

#include <lanelet2_routing/RoutingGraph.h>

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace autoware::behavior_velocity_planner
{

/**
 * @brief represent action
 */
struct InternalError
{
  const std::string error;
};

struct OverPassJudge
{
  const std::string report;
};

struct Unsafe
{
  const size_t stop_line_idx;
  const std::optional<autoware_perception_msgs::msg::PredictedObject> collision_obstacle;
};

struct Safe
{
  const size_t stop_line_idx;
};

using BlindSpotDecision = std::variant<InternalError, OverPassJudge, Unsafe, Safe>;

class BlindSpotModule : public SceneModuleInterfaceWithRTC
{
public:
  struct DebugData
  {
    std::optional<geometry_msgs::msg::Pose> virtual_wall_pose{std::nullopt};
    std::optional<lanelet::CompoundPolygon3d> detection_area;
    autoware_perception_msgs::msg::PredictedObjects conflicting_targets;
    // NERO ixlpass (NERO_BS_DETECT_NON_VRU): the vehicle check's attention area and conflict area
    std::optional<lanelet::BasicPolygon2d> nero_veh_attention_area;
    std::optional<lanelet::BasicPolygon2d> nero_veh_conflict_area;
  };

public:
  BlindSpotModule(
    const int64_t module_id, const int64_t lane_id, const TurnDirection turn_direction,
    const std::shared_ptr<const PlannerData> planner_data, const PlannerParam & planner_param,
    const rclcpp::Logger logger, const rclcpp::Clock::SharedPtr clock,
    const std::shared_ptr<autoware_utils::TimeKeeper> time_keeper,
    const std::shared_ptr<planning_factor_interface::PlanningFactorInterface>
      planning_factor_interface);

  /**
   * @brief plan go-stop velocity at traffic crossing with collision check between reference path
   * and object predicted path
   */
  bool modifyPathVelocity(PathWithLaneId * path) override;

  visualization_msgs::msg::MarkerArray createDebugMarkerArray() override;
  std::vector<autoware::motion_utils::VirtualWall> createVirtualWalls() override;

private:
  // (semi) const variables
  const int64_t lane_id_;
  const PlannerParam planner_param_;
  const TurnDirection turn_direction_;
  std::optional<lanelet::ConstLanelet> sibling_straight_lanelet_{std::nullopt};
  std::optional<lanelet::ConstLanelets> blind_spot_lanelets_{std::nullopt};

  // state variables
  bool is_over_pass_judge_line_{false};

  // Parameter

  void initializeRTCStatus();
  BlindSpotDecision modifyPathVelocityDetail(PathWithLaneId * path);
  // setSafe(), setDistance()
  void setRTCStatus(
    const BlindSpotDecision & decision,
    const autoware_internal_planning_msgs::msg::PathWithLaneId & path);
  template <typename Decision>
  void setRTCStatusByDecision(
    const Decision & decision, const autoware_internal_planning_msgs::msg::PathWithLaneId & path);
  // stop/GO
  void reactRTCApproval(const BlindSpotDecision & decision, PathWithLaneId * path);
  template <typename Decision>
  void reactRTCApprovalByDecision(
    const Decision & decision, autoware_internal_planning_msgs::msg::PathWithLaneId * path);

  /**
   * @brief Generate a stop line and insert it into the path.
   * A stop line is at an intersection point of straight path with vehicle path
   * @param detection_areas used to generate stop line
   * @param path            ego-car lane
   * @param stop_line_idx   generated stop line index
   * @param pass_judge_line_idx  generated pass judge line index
   * @return false when generation failed
   */
  std::optional<std::pair<size_t, size_t>> generateStopLine(
    const InterpolatedPathInfo & interpolated_path_info,
    autoware_internal_planning_msgs::msg::PathWithLaneId * path) const;

  std::optional<OverPassJudge> isOverPassJudge(
    const autoware_internal_planning_msgs::msg::PathWithLaneId & input_path,
    const geometry_msgs::msg::Pose & stop_point_pose) const;

  double computeTimeToPassStopLine(
    const lanelet::ConstLanelets & blind_spot_lanelets,
    const geometry_msgs::msg::Pose & stop_line_pose) const;

  /**
   * @brief Check obstacle is in blind spot areas.
   * Condition1: Object's position is in broad blind spot area.
   * Condition2: Object's predicted position is in narrow blind spot area.
   * If both conditions are met, return true
   * @param path path information associated with lane id
   * @param objects_ptr dynamic objects
   * @param closest_idx closest path point index from ego car in path points
   * @return true when an object is detected in blind spot
   */
  std::optional<autoware_perception_msgs::msg::PredictedObject> isCollisionDetected(
    const lanelet::ConstLanelets & blind_spot_lanelets,
    const geometry_msgs::msg::Pose & stop_line_pose, const lanelet::CompoundPolygon3d & area,
    const double ego_time_to_reach_stop_line);

  /**
   * @brief Check if object is belong to targeted classes
   * @param object Dynamic object
   * @return True when object belong to targeted classes
   */
  bool isTargetObjectType(const autoware_perception_msgs::msg::PredictedObject & object) const;

  /**
   * @brief Modify objects predicted path. remove path point if the time exceeds timer_thr.
   * @param objects_ptr target objects
   * @param time_thr    time threshold to cut path
   */
  autoware_perception_msgs::msg::PredictedObject cutPredictPathWithDuration(
    const std_msgs::msg::Header & header,
    const autoware_perception_msgs::msg::PredictedObject & object_original,
    const double time_thr) const;

  StateMachine state_machine_;  //! for state

  // NERO ixlpass (NERO_BS_DETECT_NON_VRU, LEFT turns only; src/nero_passer.cpp): vehicles (car /
  // truck / bus / trailer) passing on the turning side, judged as upstream's re-designed blind_spot
  // (#11079) judges them: an attention area from ego's turning-side edge to the blind side's far
  // edge (here the OPPOSITE lane's far edge: the car's #5635 strip, which the redesign dropped),
  // carried straight on past the junction entry, and the time intervals in which ego and each
  // object pass ego's turn path (#11265's lateral clearance filter; the overlap test with both
  // margins)
  struct NeroVehGeometry
  {
    lanelet::LineString3d
      far_before;                     //!< far edge of the blind side before turning (ego direction)
    lanelet::LineString3d far_after;  //!< its straight extension past the junction entry (clipped)
    lanelet::LineString3d road_centerline;  //!< the approach lanes' centreline (ego direction)
    lanelet::ConstLanelets road_lanelets;
    std::string kind;
  };
  struct NeroVehDecision
  {
    bool active{false};      //!< the check ran
    bool unsafe_now{false};  //!< an object conflicts this cycle
    bool hold{false};        //!< unsafe now or within the debounce time of the last unsafe cycle
    std::optional<size_t> stop_line_idx{std::nullopt};  //!< where to hold (inserted into the path)
    bool inserted{false};  //!< a new point went into the path at stop_line_idx
    std::optional<autoware_perception_msgs::msg::PredictedObject> object{std::nullopt};
    std::string note;
  };
  std::optional<NeroVehGeometry> nero_veh_geometry_{std::nullopt};
  bool nero_veh_geometry_failed_{false};
  std::optional<rclcpp::Time> nero_veh_last_unsafe_{std::nullopt};
  std::optional<autoware_perception_msgs::msg::PredictedObject> nero_veh_last_object_{std::nullopt};
  bool nero_veh_over_pass_judge_{false};
  std::optional<geometry_msgs::msg::Pose> nero_veh_stop_pose_{};  //!< the hold's latched stop point
  std::optional<lanelet::BasicPoint2d>
    nero_veh_entry_pt_{};  //!< ego's path on the turn lanelet's entry line
  //! yield memory per track id: (seen moving >= 1 m/s, stood after that and not faster than
  //! resume_v since)
  std::map<std::string, std::pair<bool, bool>> nero_veh_stood_;
  std::map<std::string, std::deque<std::pair<double, double>>>
    nero_veh_vhist_;  //!< (t, v) per track, 1.5 s
  std::optional<NeroVehGeometry> neroVehBuildGeometry(
    const std::vector<lanelet::Id> & lane_ids_upto_intersection) const;
  NeroVehDecision neroVehicleCheck(
    const InterpolatedPathInfo & interpolated_path_info,
    autoware_internal_planning_msgs::msg::PathWithLaneId * path);
  void neroDebugOldArea(
    const lanelet::ConstLanelets & blind_spot_lanelets,
    const geometry_msgs::msg::Pose & stop_line_pose, const lanelet::CompoundPolygon3d & area,
    const double ego_time_to_reach_stop_line, const bool vru_collision) const;

  // Debug
  mutable DebugData debug_data_;
};
}  // namespace autoware::behavior_velocity_planner

#endif  // AUTOWARE__BEHAVIOR_VELOCITY_BLIND_SPOT_MODULE__SCENE_HPP_
