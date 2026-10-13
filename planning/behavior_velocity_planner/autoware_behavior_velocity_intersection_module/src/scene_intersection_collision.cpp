// Copyright 2024 Tier IV, Inc.
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

#include "scene_intersection.hpp"
#include "util.hpp"

#include <autoware/behavior_velocity_planner_common/utilization/boost_geometry_helper.hpp>  // for toGeomPoly
#include <autoware/behavior_velocity_planner_common/utilization/trajectory_utils.hpp>  // for smoothPath
#include <autoware/motion_utils/trajectory/trajectory.hpp>
#include <autoware_lanelet2_extension/utility/utilities.hpp>
#include <autoware_utils/geometry/boost_polygon_utils.hpp>  // for toPolygon2d
#include <autoware_utils/geometry/geometry.hpp>
#include <magic_enum.hpp>

#include <boost/geometry/algorithms/correct.hpp>
#include <boost/geometry/algorithms/intersects.hpp>
#include <boost/geometry/algorithms/within.hpp>

#include <fmt/format.h>
#include <lanelet2_core/geometry/Polygon.h>
#include <lanelet2_routing/RoutingGraph.h>  // Nero c3: following() of the assigned lanelet

#include <algorithm>
#include <cmath>
#include <limits>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace autoware::behavior_velocity_planner
{
namespace bg = boost::geometry;

bool IntersectionModule::isTargetCollisionVehicleType(
  const autoware_perception_msgs::msg::PredictedObject & object) const
{
  const auto label = object.classification.at(0).label;
  const auto & p = planner_param_.collision_detection.target_type;

  if (label == autoware_perception_msgs::msg::ObjectClassification::CAR && p.car) {
    return true;
  }
  if (label == autoware_perception_msgs::msg::ObjectClassification::BUS && p.bus) {
    return true;
  }
  if (label == autoware_perception_msgs::msg::ObjectClassification::TRUCK && p.truck) {
    return true;
  }
  if (label == autoware_perception_msgs::msg::ObjectClassification::TRAILER && p.trailer) {
    return true;
  }
  if (label == autoware_perception_msgs::msg::ObjectClassification::MOTORCYCLE && p.motorcycle) {
    return true;
  }
  if (label == autoware_perception_msgs::msg::ObjectClassification::BICYCLE && p.bicycle) {
    return true;
  }
  if (label == autoware_perception_msgs::msg::ObjectClassification::UNKNOWN && p.unknown) {
    // Nero c3 (unknown_filter): with the filter on, an UNKNOWN is a target only if it is big or
    // has been moving fast for a while (small/static UNKNOWN blobs such as curbs, bushes or ghost
    // points no longer make ego yield). Filter off = stock behaviour.
    if (planner_param_.collision_detection.unknown_filter.enable) {
      return isUnknownObjectCollisionTarget(object);
    }
    return true;
  }
  return false;
}

// ==========================================================================================
// Nero c3 (unknown_filter)
// ==========================================================================================
namespace
{
double footprintArea(const autoware_perception_msgs::msg::Shape & shape)
{
  using autoware_perception_msgs::msg::Shape;
  if (shape.type == Shape::BOUNDING_BOX) {
    return shape.dimensions.x * shape.dimensions.y;
  }
  if (shape.type == Shape::CYLINDER) {
    const double r = shape.dimensions.x / 2.0;  // dimensions.x is the diameter
    return M_PI * r * r;
  }
  // POLYGON: shoelace formula on the footprint
  const auto & pts = shape.footprint.points;
  double twice_area = 0.0;
  for (size_t i = 0; i < pts.size(); ++i) {
    const auto & a = pts.at(i);
    const auto & b = pts.at((i + 1) % pts.size());
    twice_area += static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
  }
  return std::fabs(twice_area) / 2.0;
}

double objectSpeed(const autoware_perception_msgs::msg::PredictedObject & object)
{
  const auto & v = object.kinematics.initial_twist_with_covariance.twist.linear;
  return std::hypot(v.x, v.y);
}
}  // namespace

void IntersectionModule::updateUnknownSpeedHistory()
{
  const auto & uf = planner_param_.collision_detection.unknown_filter;
  if (!uf.enable || !planner_param_.collision_detection.target_type.unknown) {
    unknown_speed_history_.clear();
    return;
  }
  const auto now = clock_->now();
  for (const auto & object : planner_data_->predicted_objects->objects) {
    if (
      object.classification.empty() || object.classification.at(0).label !=
                                          autoware_perception_msgs::msg::ObjectClassification::UNKNOWN) {
      continue;
    }
    auto & history = unknown_speed_history_[uuidKey(object.object_id)];
    history.last_seen = now;
    if (objectSpeed(object) >= uf.min_speed) {
      if (!history.fast_since) {
        history.fast_since = now;  // start of a new "fast" streak
      }
    } else {
      history.fast_since = std::nullopt;  // streak broken
    }
  }
  // forget objects that have not been seen for a while (track lost / uuid changed)
  static constexpr double forget_after_sec = 1.0;
  for (auto it = unknown_speed_history_.begin(); it != unknown_speed_history_.end();) {
    if ((now - it->second.last_seen).seconds() > forget_after_sec) {
      it = unknown_speed_history_.erase(it);
    } else {
      ++it;
    }
  }
}

bool IntersectionModule::isUnknownObjectCollisionTarget(
  const autoware_perception_msgs::msg::PredictedObject & object) const
{
  const auto & uf = planner_param_.collision_detection.unknown_filter;
  // (a) big enough footprint
  if (footprintArea(object.shape) >= uf.min_area) {
    return true;
  }
  // (b) speed >= min_speed held continuously for >= min_duration
  const auto it = unknown_speed_history_.find(uuidKey(object.object_id));
  if (it == unknown_speed_history_.end() || !it->second.fast_since) {
    return false;
  }
  return (clock_->now() - it->second.fast_since.value()).seconds() >= uf.min_duration;
}

// ==========================================================================================
// Nero c3 (wrong_way_watch)
// ==========================================================================================
bool IntersectionModule::isWrongWayOnLanelets(
  const autoware_perception_msgs::msg::PredictedObject & object,
  const lanelet::ConstLanelets & ego_lanelets) const
{
  const auto & ww = planner_param_.collision_detection.wrong_way_watch;
  // slow objects are excluded: the tracker often flips the heading of slow/stopped vehicles, and
  // a flipped lead car must not look like an oncoming wrong-way car
  if (objectSpeed(object) < ww.min_speed) {  // planar speed (UNKNOWN hulls move in vy)
    return false;
  }
  // heading of motion (flipped if the object reports negative longitudinal velocity)
  const auto pose = util::getObjectPoseWithVelocityDirection(object.kinematics);
  const double pose_angle = tf2::getYaw(pose.orientation);
  for (const auto & ll : ego_lanelets) {
    // object centre must be inside the lanelet itself (no margin): an oncoming car on the
    // neighbouring (opposite) lane is not "on ego's lane"
    if (!lanelet::utils::isInLanelet(pose, ll, 0.0)) {
      continue;
    }
    const double ll_angle = lanelet::utils::getLaneletAngle(ll, pose.position);
    const double angle_diff = autoware_utils::normalize_radian(ll_angle - pose_angle, -M_PI);
    if (std::fabs(angle_diff) > M_PI / 2.0) {
      return true;
    }
  }
  return false;
}

void IntersectionModule::updateObjectInfoManagerArea()
{
  const auto & intersection_lanelets = intersection_lanelets_.value();
  const auto & attention_lanelets = intersection_lanelets.attention();
  const auto & attention_lanelet_stoplines = intersection_lanelets.attention_stoplines();
  const auto & adjacent_lanelets = intersection_lanelets.adjacent();
  const auto lanelet_map_ptr = planner_data_->route_handler_->getLaneletMapPtr();
  const auto & assigned_lanelet = lanelet_map_ptr->laneletLayer.get(lane_id_);
  const auto intersection_area = util::getIntersectionArea(assigned_lanelet, lanelet_map_ptr);

  // Nero c3 (unknown_filter): keep the per-object speed streaks up to date
  updateUnknownSpeedHistory();

  // Nero c3 (wrong_way_watch): a vehicle driving against ego's exit lane (following lanelets of
  // the assigned lanelet) is registered as an attention object even if it is not on an attention
  // lanelet, and is never dropped as an "adjacent" (same-lane) object. Once flagged, it stays
  // flagged while it drives against the assigned lanelet itself (it came out of the exit lane into
  // the junction towards ego). The assigned lanelet alone is NOT enough to flag an object, since
  // crossing/turning traffic inside the junction often lies on it with a heading > pi/2 off.
  const bool wrong_way_watch = planner_param_.collision_detection.wrong_way_watch.enable;
  const auto prev_wrong_way_uuids = wrong_way_ego_lane_object_uuids_;
  wrong_way_ego_lane_object_uuids_.clear();
  lanelet::ConstLanelets ego_exit_lanelets{};
  if (wrong_way_watch) {
    for (const auto & ll :
         planner_data_->route_handler_->getRoutingGraphPtr()->following(assigned_lanelet)) {
      ego_exit_lanelets.push_back(ll);
    }
  }

  // ==========================================================================================
  // entries that are not observed in this iteration need to be cleared
  //
  // NOTE: old_map is not referenced because internal data of object_info_manager is cleared
  // ==========================================================================================
  const auto old_map = object_info_manager_.getObjectsMap();
  object_info_manager_.clearObjects();

  for (const auto & predicted_object : planner_data_->predicted_objects->objects) {
    if (!isTargetCollisionVehicleType(predicted_object)) {
      continue;
    }

    // ==========================================================================================
    // NOTE: is_parked_vehicle is used because sometimes slow vehicle direction is
    // incorrect/reversed/flipped due to tracking. if is_parked_vehicle is true, object direction
    // is not checked
    // ==========================================================================================
    const auto object_direction =
      util::getObjectPoseWithVelocityDirection(predicted_object.kinematics);
    // Nero c3: use the full planar speed (UNKNOWN hull objects move mostly in vy)
    const auto is_parked_vehicle =
      std::hypot(
        predicted_object.kinematics.initial_twist_with_covariance.twist.linear.x,
        predicted_object.kinematics.initial_twist_with_covariance.twist.linear.y) <
      planner_param_.occlusion.ignore_parked_vehicle_speed_threshold;

    const bool is_wrong_way_on_ego_lane =
      wrong_way_watch &&
      (isWrongWayOnLanelets(predicted_object, ego_exit_lanelets) ||
       (prev_wrong_way_uuids.count(uuidKey(predicted_object.object_id)) > 0 &&
        isWrongWayOnLanelets(predicted_object, lanelet::ConstLanelets{assigned_lanelet})));
    if (is_wrong_way_on_ego_lane) {
      wrong_way_ego_lane_object_uuids_.insert(uuidKey(predicted_object.object_id));
    }

    const auto belong_adjacent_lanelet_id =
      checkAngleForTargetLanelets(object_direction, adjacent_lanelets, false);
    if (belong_adjacent_lanelet_id && !is_wrong_way_on_ego_lane) {
      continue;
    }
    const auto belong_attention_lanelet_id =
      checkAngleForTargetLanelets(object_direction, attention_lanelets, is_parked_vehicle);
    const auto & obj_pos = predicted_object.kinematics.initial_pose_with_covariance.pose.position;
    const bool in_intersection_area = [&]() {
      if (!intersection_area) {
        return false;
      }
      return bg::within(autoware_utils::Point2d{obj_pos.x, obj_pos.y}, intersection_area.value());
    }();
    std::optional<lanelet::ConstLanelet> attention_lanelet{std::nullopt};
    std::optional<lanelet::ConstLineString3d> stopline{std::nullopt};
    if (!belong_attention_lanelet_id && !in_intersection_area && !is_wrong_way_on_ego_lane) {
      continue;
    } else if (belong_attention_lanelet_id) {
      const auto idx = belong_attention_lanelet_id.value();
      attention_lanelet = attention_lanelets.at(idx);
      stopline = attention_lanelet_stoplines.at(idx);
    }
    // Nero c3 (wrong_way_watch): a wrong-way vehicle on ego's own lane ahead is an attention object
    const bool belong_attention_area =
      belong_attention_lanelet_id.has_value() || is_wrong_way_on_ego_lane;

    const auto object_it = old_map.find(predicted_object.object_id);
    if (object_it != old_map.end()) {
      auto object_info = object_it->second;
      object_info_manager_.registerExistingObject(
        predicted_object.object_id, belong_attention_area, in_intersection_area, is_parked_vehicle,
        object_info);
      object_info->initialize(predicted_object, attention_lanelet, stopline);
    } else {
      auto object_info = object_info_manager_.registerObject(
        predicted_object.object_id, belong_attention_area, in_intersection_area,
        is_parked_vehicle);
      object_info->initialize(predicted_object, attention_lanelet, stopline);
    }
  }
}

void IntersectionModule::updateObjectInfoManagerCollision(
  const PathLanelets & path_lanelets,
  const IntersectionModule::TimeDistanceArray & time_distance_array,
  const IntersectionModule::TrafficPrioritizedLevel & traffic_prioritized_level,
  const bool passed_1st_judge_line_first_time, const bool passed_2nd_judge_line_first_time,
  autoware_internal_debug_msgs::msg::Float64MultiArrayStamped * object_ttc_time_array)
{
  const auto & intersection_lanelets = intersection_lanelets_.value();

  if (passed_1st_judge_line_first_time) {
    object_info_manager_.setPassed1stPassJudgeLineFirstTime(clock_->now());
  }
  if (passed_2nd_judge_line_first_time) {
    object_info_manager_.setPassed2ndPassJudgeLineFirstTime(clock_->now());
  }

  const double passing_time = time_distance_array.back().first;
  const auto & concat_lanelets = path_lanelets.all;
  const auto closest_arc_coords =
    lanelet::utils::getArcCoordinates(concat_lanelets, planner_data_->current_odometry->pose);
  const auto & ego_lane = path_lanelets.ego_or_entry2exit;
  debug_data_.ego_lane = ego_lane.polygon3d();
  const auto ego_poly = ego_lane.polygon2d().basicPolygon();

  // Nero c3 (wrong_way_watch): for wrong-way objects on ego's own lane the collision area is ego's
  // whole remaining path from its current position: ego_or_entry2exit plus path_lanelets.next
  // (the exit lane), because a head-on conflict with such a car happens on the exit lane, which
  // the stock ego_poly (assigned lane part only) does not cover.
  const auto & ww_param = planner_param_.collision_detection.wrong_way_watch;
  lanelet::BasicPolygon2d ego_forward_poly{};
  if (ww_param.enable && !wrong_way_ego_lane_object_uuids_.empty()) {
    const auto forward_poly3d = lanelet::utils::getPolygonFromArcLength(
      concat_lanelets, std::max(0.0, closest_arc_coords.length),
      lanelet::utils::getLaneletLength2d(concat_lanelets));
    for (const auto & p : forward_poly3d) {
      ego_forward_poly.emplace_back(p.x(), p.y());
    }
  }

  // ==========================================================================================
  // dynamically change TTC margin according to traffic light color to gradually relax from green to
  // red
  // ==========================================================================================
  const auto [collision_start_margin_time, collision_end_margin_time] = [&]() {
    if (traffic_prioritized_level == TrafficPrioritizedLevel::FULLY_PRIORITIZED) {
      return std::make_pair(
        planner_param_.collision_detection.fully_prioritized.collision_start_margin_time,
        planner_param_.collision_detection.fully_prioritized.collision_end_margin_time);
    }
    if (traffic_prioritized_level == TrafficPrioritizedLevel::PARTIALLY_PRIORITIZED) {
      return std::make_pair(
        planner_param_.collision_detection.partially_prioritized.collision_start_margin_time,
        planner_param_.collision_detection.partially_prioritized.collision_end_margin_time);
    }
    return std::make_pair(
      planner_param_.collision_detection.not_prioritized.collision_start_margin_time,
      planner_param_.collision_detection.not_prioritized.collision_end_margin_time);
  }();

  constexpr size_t object_debug_size = 57;
  {
    object_ttc_time_array->stamp = clock_->now();
    object_ttc_time_array->layout.dim.resize(3);
    object_ttc_time_array->layout.dim.at(0).label = "objects";
    object_ttc_time_array->layout.dim.at(0).size =
      1;  // incremented in the loop, first row is lane_id
    object_ttc_time_array->layout.dim.at(1).label =
      "[x, y, th, length, width, speed, dangerous, ref_obj_enter_time, ref_obj_exit_time, "
      "start_time, start_dist, "
      "end_time, end_dist, first_collision_x, first_collision_y, last_collision_x, "
      "last_collision_y, "
      "prd_x[0], ... pred_x[19], pred_y[0], ... pred_y[19]]";
    object_ttc_time_array->layout.dim.at(1).size = object_debug_size;
    for (unsigned i = 0; i < object_debug_size; ++i) {
      object_ttc_time_array->data.push_back(lane_id_);
    }
  }

  for (auto & object_info : object_info_manager_.attentionObjects()) {
    const auto & predicted_object = object_info->predicted_object();
    // Nero c3 (wrong_way_watch)
    const bool is_wrong_way_on_ego_lane =
      ww_param.enable && ego_forward_poly.size() >= 3 &&
      wrong_way_ego_lane_object_uuids_.count(uuidKey(predicted_object.object_id)) > 0;
    bool safe_under_traffic_control = false;
    const auto label = predicted_object.classification.at(0).label;
    const auto expected_deceleration =
      (label == autoware_perception_msgs::msg::ObjectClassification::MOTORCYCLE ||
       label == autoware_perception_msgs::msg::ObjectClassification::BICYCLE)
        ? planner_param_.collision_detection.ignore_on_amber_traffic_light
            .object_expected_deceleration.bike
        : planner_param_.collision_detection.ignore_on_amber_traffic_light
            .object_expected_deceleration.car;
    if (
      traffic_prioritized_level == TrafficPrioritizedLevel::PARTIALLY_PRIORITIZED &&
      object_info->can_stop_before_stopline(expected_deceleration)) {
      safe_under_traffic_control = true;
    }
    if (
      traffic_prioritized_level == TrafficPrioritizedLevel::FULLY_PRIORITIZED &&
      object_info->can_stop_before_ego_lane(
        expected_deceleration,
        planner_param_.collision_detection.ignore_on_red_traffic_light.object_margin_to_path,
        ego_lane)) {
      safe_under_traffic_control = true;
    }
    if (is_wrong_way_on_ego_lane) {
      // Nero c3: a wrong-way driver is not expected to obey the traffic control of its lane
      safe_under_traffic_control = false;
    }

    // ==========================================================================================
    // check the PredictedPath in the ascending order of its confidence to save the safe/unsafe
    // CollisionKnowledge for most probable path
    // ==========================================================================================
    std::list<const autoware_perception_msgs::msg::PredictedPath *> sorted_predicted_paths;
    for (unsigned i = 0; i < predicted_object.kinematics.predicted_paths.size(); ++i) {
      sorted_predicted_paths.push_back(&predicted_object.kinematics.predicted_paths.at(i));
    }
    sorted_predicted_paths.sort(
      [](const auto path1, const auto path2) { return path1->confidence > path2->confidence; });

    // ==========================================================================================
    // if all of the predicted path is lower confidence/geometrically does not intersect with ego
    // path, both will be null, which is interpreted as SAFE. if any of the path is "normal", either
    // of them has value, not both
    // ==========================================================================================
    std::optional<CollisionInterval> unsafe_interval{std::nullopt};
    std::optional<CollisionInterval> safe_interval{std::nullopt};
    std::optional<std::vector<double>> object_debug_info{std::nullopt};

    for (const auto & predicted_path_ptr : sorted_predicted_paths) {
      auto predicted_path = *predicted_path_ptr;
      if (
        predicted_path.confidence <
        planner_param_.collision_detection.min_predicted_path_confidence) {
        continue;
      }
      if (is_wrong_way_on_ego_lane) {
        // ==========================================================================================
        // Nero c3 (wrong_way_watch): the object drives against ego's lane, so any overlap of its
        // predicted footprint with ego's remaining path (from ego to the end of the exit lane) is a
        // head-on conflict, whatever the timing. ego's time/distance profile only spans the
        // intersection, so the stock TTC window is not used here. The horizon is at least
        // prediction_horizon so that a conflict further down the exit lane is still seen.
        // ==========================================================================================
        cutPredictPathWithinDuration(
          planner_data_->predicted_objects->header.stamp,
          std::max(passing_time, ww_param.prediction_horizon), &predicted_path);
        const auto ww_interval_opt = findPassageInterval(
          predicted_path, predicted_object.shape, ego_forward_poly,
          intersection_lanelets.first_attention_lane(),
          intersection_lanelets.second_attention_lane());
        if (ww_interval_opt) {
          safe_interval = std::nullopt;
          unsafe_interval = ww_interval_opt;
          break;
        }
        continue;
      }
      // junction-yield 10-11 (autoware_universe#11774): with the flag the horizon covers the start margin, so an
      // object entering up to collision_start_margin_time after the ego's (modelled) exit is still checked
      const double prediction_horizon =
        planner_param_.collision_detection.cut_predicted_path_with_start_margin
          ? passing_time + collision_start_margin_time
          : passing_time;
      cutPredictPathWithinDuration(
        planner_data_->predicted_objects->header.stamp, prediction_horizon, &predicted_path);
      const auto object_passage_interval_opt = findPassageInterval(
        predicted_path, predicted_object.shape, ego_poly,
        intersection_lanelets.first_attention_lane(),
        intersection_lanelets.second_attention_lane());
      if (!object_passage_interval_opt) {
        // there is no chance of geometric collision for the entire prediction horizon
        continue;
      }
      const auto & object_passage_interval = object_passage_interval_opt.value();
      const auto object_enter_exit_time = object_passage_interval.interval_time;
      const auto object_enter_time = std::get<0>(object_enter_exit_time);
      const auto object_exit_time = std::get<1>(object_enter_exit_time);
      const auto ego_start_itr = std::lower_bound(
        time_distance_array.begin(), time_distance_array.end(),
        object_enter_time - collision_start_margin_time,
        [](const auto & a, const double b) { return a.first < b; });
      if (ego_start_itr == time_distance_array.end()) {
        // ==========================================================================================
        // this is the case where at time "object_enter_time - collision_start_margin_time", ego is
        // arriving at the exit of the intersection, which means even if we assume that the object
        // accelerates and the first collision happens faster by the TTC margin, ego will be already
        // arriving at the exist of the intersection.
        // ==========================================================================================
        continue;
      }
      auto ego_end_itr = std::lower_bound(
        time_distance_array.begin(), time_distance_array.end(),
        object_exit_time + collision_end_margin_time,
        [](const auto & a, const double b) { return a.first < b; });
      if (ego_end_itr == time_distance_array.end()) {
        ego_end_itr = time_distance_array.end() - 1;
      }
      const double ego_start_arc_length = std::max(
        0.0, closest_arc_coords.length + ego_start_itr->second -
               planner_data_->vehicle_info_.rear_overhang_m);
      const double ego_end_arc_length = std::min(
        closest_arc_coords.length + ego_end_itr->second +
          planner_data_->vehicle_info_.max_longitudinal_offset_m,
        lanelet::utils::getLaneletLength2d(concat_lanelets));
      const auto trimmed_ego_polygon = lanelet::utils::getPolygonFromArcLength(
        concat_lanelets, ego_start_arc_length, ego_end_arc_length);
      if (trimmed_ego_polygon.empty()) {
        continue;
      }
      Polygon2d polygon{};
      for (const auto & p : trimmed_ego_polygon) {
        polygon.outer().emplace_back(p.x(), p.y());
      }
      bg::correct(polygon);
      debug_data_.candidate_collision_ego_lane_polygon = toGeomPoly(polygon);

      const auto & object_path = object_passage_interval.path;
      const auto [begin, end] = object_passage_interval.interval_position;
      bool collision_detected = false;
      for (auto i = begin; i <= end; ++i) {
        if (bg::intersects(
              polygon, autoware_utils::to_polygon2d(object_path.at(i), predicted_object.shape))) {
          collision_detected = true;
          break;
        }
      }
      auto get_object_info = [&]() {
        // debug info
        const auto & pose = predicted_object.kinematics.initial_pose_with_covariance.pose;
        const auto & shape = predicted_object.shape;
        const auto [object_start_itr, object_end_itr] = object_passage_interval.interval_position;
        const auto & object_start_pos = object_passage_interval.path.at(object_start_itr).position;
        const auto & object_end_pos = object_passage_interval.path.at(object_end_itr).position;
        std::vector<double> debug;
        debug.reserve(object_debug_size);

        debug.insert(
          debug.end(),
          {pose.position.x, pose.position.y, tf2::getYaw(pose.orientation), shape.dimensions.x,
           shape.dimensions.y,
           predicted_object.kinematics.initial_twist_with_covariance.twist.linear.x,
           1.0 * static_cast<int>(collision_detected), object_enter_time, object_exit_time,
           ego_start_itr->first, ego_start_itr->second, ego_end_itr->first, ego_end_itr->second,
           object_start_pos.x, object_start_pos.y, object_end_pos.x, object_end_pos.y});
        for (unsigned i = 0; i < 20; ++i) {
          const auto & pos = object_passage_interval.path
                               .at(std::min<size_t>(i, object_passage_interval.path.size() - 1))
                               .position;
          debug.push_back(pos.x);
          debug.push_back(pos.y);
        }
        return debug;
      };

      if (collision_detected) {
        // if judged as UNSAFE, return
        safe_interval = std::nullopt;
        unsafe_interval = object_passage_interval;
        object_debug_info = get_object_info();
        break;
      }
      if (!safe_interval) {
        // ==========================================================================================
        // save the safe_decision_knowledge for the most probable path. this value is nullified if
        // judged UNSAFE during the iteration
        // ==========================================================================================
        safe_interval = object_passage_interval;
        object_debug_info = get_object_info();
      }
    }
    object_info->update_safety(unsafe_interval, safe_interval, safe_under_traffic_control);
    if (passed_1st_judge_line_first_time) {
      object_info->setDecisionAt1stPassJudgeLinePassage(CollisionKnowledge{
        clock_->now(),  // stamp
        unsafe_interval
          ? CollisionKnowledge::SafeType::UNSAFE
          : (safe_under_traffic_control ? CollisionKnowledge::SafeType::SAFE_UNDER_TRAFFIC_CONTROL
                                        : CollisionKnowledge::SafeType::SAFE),  // safe
        unsafe_interval ? unsafe_interval : safe_interval,                      // interval
        predicted_object.kinematics.initial_twist_with_covariance.twist.linear
          .x  // observed_velocity
      });
    }
    if (passed_2nd_judge_line_first_time) {
      object_info->setDecisionAt2ndPassJudgeLinePassage(CollisionKnowledge{
        clock_->now(),  // stamp
        unsafe_interval
          ? CollisionKnowledge::SafeType::UNSAFE
          : (safe_under_traffic_control ? CollisionKnowledge::SafeType::SAFE_UNDER_TRAFFIC_CONTROL
                                        : CollisionKnowledge::SafeType::SAFE),  // safe
        unsafe_interval ? unsafe_interval : safe_interval,                      // interval
        predicted_object.kinematics.initial_twist_with_covariance.twist.linear
          .x  // observed_velocity
      });
    }

    // debug
    if (object_debug_info) {
      const auto & data = object_debug_info.value();
      object_ttc_time_array->layout.dim.at(0).size++;
      std::copy(data.begin(), data.end(), std::back_inserter(object_ttc_time_array->data));
    }
  }
}

void IntersectionModule::cutPredictPathWithinDuration(
  const builtin_interfaces::msg::Time & object_stamp, const double time_thr,
  autoware_perception_msgs::msg::PredictedPath * path) const
{
  const rclcpp::Time current_time = clock_->now();
  const auto original_path = path->path;
  path->path.clear();

  for (size_t k = 0; k < original_path.size(); ++k) {  // each path points
    const auto & predicted_pose = original_path.at(k);
    const auto predicted_time =
      rclcpp::Time(object_stamp) + rclcpp::Duration(path->time_step) * static_cast<double>(k);
    if ((predicted_time - current_time).seconds() < time_thr) {
      path->path.push_back(predicted_pose);
    }
  }
}

std::optional<NonOccludedCollisionStop> IntersectionModule::isGreenPseudoCollisionStatus(
  const size_t closest_idx, const size_t collision_stopline_idx,
  const IntersectionStopLines & intersection_stoplines) const
{
  // ==========================================================================================
  // if there are any vehicles on the attention area when ego entered the intersection on green
  // light, do pseudo collision detection because collision is likely to happen.
  // ==========================================================================================
  if (initial_green_light_observed_time_) {
    const auto now = clock_->now();
    const bool still_wait =
      (rclcpp::Duration((now - initial_green_light_observed_time_.value())).seconds() <
       planner_param_.collision_detection.yield_on_green_traffic_light.duration);
    if (!still_wait) {
      return std::nullopt;
    }
    const auto & attention_objects = object_info_manager_.attentionObjects();
    const bool exist_close_vehicles = std::any_of(
      attention_objects.begin(), attention_objects.end(), [&](const auto & object_info) {
        return object_info->before_stopline_by(
          planner_param_.collision_detection.yield_on_green_traffic_light.object_dist_to_stopline);
      });
    if (exist_close_vehicles) {
      const auto occlusion_stopline_idx = intersection_stoplines.occlusion_peeking_stopline.value();
      return NonOccludedCollisionStop{
        closest_idx, collision_stopline_idx, occlusion_stopline_idx, std::string("")};
    }
  }
  return std::nullopt;
}

std::string IntersectionModule::generateDetectionBlameDiagnosis(
  const std::vector<
    std::pair<IntersectionModule::CollisionStatus::BlameType, std::shared_ptr<ObjectInfo>>> &
    too_late_detect_objects,
  const std::vector<
    std::pair<IntersectionModule::CollisionStatus::BlameType, std::shared_ptr<ObjectInfo>>> &
    misjudge_objects) const
{
  std::string diag;
  if (!safely_passed_1st_judge_line_time_) {
    return diag;
  }
  const auto [passed_1st_judge_line_time, passed_1st_judge_line_pose] =
    safely_passed_1st_judge_line_time_.value();
  const auto passed_1st_judge_line_time_double =
    static_cast<double>(passed_1st_judge_line_time.nanoseconds()) / 1e+9;

  const auto now = clock_->now();
  const auto now_double = static_cast<double>(now.nanoseconds()) / 1e+9;

  // CAVEAT: format library causes runtime fault if the # of placeholders is more than the # of
  // vargs
  for (const auto & [blame_type, object_info] : too_late_detect_objects) {
    if (
      blame_type == CollisionStatus::BLAME_AT_FIRST_PASS_JUDGE && object_info->unsafe_interval()) {
      const auto & unsafe_interval = object_info->unsafe_interval().value();
      const double time_diff = now_double - passed_1st_judge_line_time_double;
      diag += fmt::format(
        "object {0} was not detected when ego passed the 1st pass judge line at {1}, but now at "
        "{2}, collision is detected after {3}~{4} seconds on the first attention lanelet of type "
        "{5}.\n",
        object_info->uuid_str,                                // 0
        passed_1st_judge_line_time_double,                    // 1
        now_double,                                           // 2
        unsafe_interval.interval_time.first,                  // 3
        unsafe_interval.interval_time.second,                 // 4
        magic_enum::enum_name(unsafe_interval.lane_position)  // 5
      );
      const auto past_position_opt = object_info->estimated_past_position(time_diff);
      if (past_position_opt) {
        const auto & past_position = past_position_opt.value();
        diag += fmt::format(
          "this object is estimated to have been at x = {0}, y = {1} when ego passed the 1st pass "
          "judge line({2} seconds before from now) given the estimated current velocity {3}[m/s]. "
          "ego was at x = {4}, y = {5} when it passed the 1st pass judge line so it is the fault "
          "of detection side that failed to detect around {6}[m] range at that time.\n",
          past_position.x,                                                            // 0
          past_position.y,                                                            // 1
          time_diff,                                                                  // 2
          object_info->observed_velocity(),                                           // 3
          passed_1st_judge_line_pose.position.x,                                      // 4
          passed_1st_judge_line_pose.position.y,                                      // 5
          autoware_utils::calc_distance2d(passed_1st_judge_line_pose, past_position)  // 6
        );
      }
    }
    if (
      safely_passed_2nd_judge_line_time_ &&
      blame_type == CollisionStatus::BLAME_AT_SECOND_PASS_JUDGE && object_info->unsafe_interval()) {
      const auto [passed_2nd_judge_line_time, passed_2nd_judge_line_pose] =
        safely_passed_2nd_judge_line_time_.value();
      const auto passed_2nd_judge_line_time_double =
        static_cast<double>(passed_2nd_judge_line_time.nanoseconds()) / 1e+9;

      const auto & unsafe_interval = object_info->unsafe_interval().value();
      const double time_diff = now_double - passed_2nd_judge_line_time_double;
      diag += fmt::format(
        "object {0} was not detected when ego passed the 2nd pass judge line at {1}, but now at "
        "{2}, collision is detected after {3}~{4} seconds on the lanelet of type {5}.\n",
        object_info->uuid_str,                                // 0
        passed_2nd_judge_line_time_double,                    // 1
        now_double,                                           // 2
        unsafe_interval.interval_time.first,                  // 3
        unsafe_interval.interval_time.second,                 // 4
        magic_enum::enum_name(unsafe_interval.lane_position)  // 5
      );
      const auto past_position_opt = object_info->estimated_past_position(time_diff);
      if (past_position_opt) {
        const auto & past_position = past_position_opt.value();
        diag += fmt::format(
          "this object is estimated to have been at x = {0}, y = {1} when ego passed the 2nd pass "
          "judge line({2} seconds before from now) given the estimated current velocity {3}[m/s]. "
          "ego was at x = {4}, y = {5} when it passed the 2nd pass judge line so it is the fault "
          "of detection side that failed to detect around {6}[m] range at that time.\n",
          past_position.x,                                                            // 0
          past_position.y,                                                            // 1
          time_diff,                                                                  // 2
          object_info->observed_velocity(),                                           // 3
          passed_2nd_judge_line_pose.position.x,                                      // 4
          passed_2nd_judge_line_pose.position.y,                                      // 5
          autoware_utils::calc_distance2d(passed_2nd_judge_line_pose, past_position)  // 6
        );
      }
    }
  }
  for (const auto & [blame_type, object_info] : misjudge_objects) {
    if (
      blame_type == CollisionStatus::BLAME_AT_FIRST_PASS_JUDGE && object_info->unsafe_interval() &&
      object_info->decision_at_1st_pass_judge_line_passage()) {
      const auto & decision_at_1st_pass_judge_line =
        object_info->decision_at_1st_pass_judge_line_passage().value();
      const auto decision_at_1st_pass_judge_line_time =
        static_cast<double>(decision_at_1st_pass_judge_line.stamp.nanoseconds()) / 1e+9;
      const auto & unsafe_interval = object_info->unsafe_interval().value();
      diag += fmt::format(
        "object {0} was judged as {1} when ego passed the 1st pass judge line at time {2} "
        "previously with the estimated velocity {3}[m/s], but now at {4} collision is detected "
        "after {5}~{6} seconds on the first attention lanelet of type {7} with the estimated "
        "current velocity {8}[m/s]\n",
        object_info->uuid_str,                                             // 0
        magic_enum::enum_name(decision_at_1st_pass_judge_line.safe_type),  // 1
        decision_at_1st_pass_judge_line_time,                              // 2
        decision_at_1st_pass_judge_line.observed_velocity,                 // 3
        now_double,                                                        // 4
        unsafe_interval.interval_time.first,                               // 5
        unsafe_interval.interval_time.second,                              // 6
        magic_enum::enum_name(unsafe_interval.lane_position),              // 7
        object_info->observed_velocity()                                   // 8
      );
    }
    if (
      blame_type == CollisionStatus::BLAME_AT_SECOND_PASS_JUDGE && object_info->unsafe_interval() &&
      object_info->decision_at_2nd_pass_judge_line_passage()) {
      const auto & decision_at_2nd_pass_judge_line =
        object_info->decision_at_2nd_pass_judge_line_passage().value();
      const auto decision_at_2nd_pass_judge_line_time =
        static_cast<double>(decision_at_2nd_pass_judge_line.stamp.nanoseconds()) / 1e+9;
      const auto & unsafe_interval = object_info->unsafe_interval().value();
      diag += fmt::format(
        "object {0} was judged as {1} when ego passed the 2nd pass judge line at time {2} "
        "previously with the estimated velocity {3}[m/s], but now at {4} collision is detected "
        "after {5}~{6} seconds on the lanelet of type {7} with the estimated current velocity "
        "{8}[m/s]\n",
        object_info->uuid_str,                                             // 0
        magic_enum::enum_name(decision_at_2nd_pass_judge_line.safe_type),  // 1
        decision_at_2nd_pass_judge_line_time,                              // 2
        decision_at_2nd_pass_judge_line.observed_velocity,                 // 3
        now_double,                                                        // 4
        unsafe_interval.interval_time.first,                               // 5
        unsafe_interval.interval_time.second,                              // 6
        magic_enum::enum_name(unsafe_interval.lane_position),              // 7
        object_info->observed_velocity()                                   // 8
      );
    }
  }
  return diag;
}

std::string IntersectionModule::generateEgoRiskEvasiveDiagnosis(
  const autoware_internal_planning_msgs::msg::PathWithLaneId & path, const size_t closest_idx,
  const IntersectionModule::TimeDistanceArray & ego_time_distance_array,
  const std::vector<
    std::pair<IntersectionModule::CollisionStatus::BlameType, std::shared_ptr<ObjectInfo>>> &
    too_late_detect_objects,
  [[maybe_unused]] const std::vector<
    std::pair<IntersectionModule::CollisionStatus::BlameType, std::shared_ptr<ObjectInfo>>> &
    misjudge_objects) const
{
  static constexpr double min_vel = 1e-2;
  std::string diag;
  const double ego_vel = planner_data_->current_velocity->twist.linear.x;
  for (const auto & [blame_type, object_info] : too_late_detect_objects) {
    if (!object_info->unsafe_interval()) {
      continue;
    }
    // object side
    const auto & unsafe_interval = object_info->unsafe_interval().value();
    const auto [begin, end] = unsafe_interval.interval_position;
    const auto &p1 = unsafe_interval.path.at(begin).position,
               p2 = unsafe_interval.path.at(end).position;
    const auto collision_pos =
      autoware_utils::create_point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2);
    const auto object_dist_to_margin_point =
      autoware_utils::calc_distance2d(
        object_info->predicted_object().kinematics.initial_pose_with_covariance.pose.position,
        collision_pos) -
      planner_param_.collision_detection.avoid_collision_by_acceleration
          .object_time_margin_to_collision_point *
        object_info->observed_velocity();
    if (object_dist_to_margin_point <= 0.0) {
      continue;
    }
    const auto object_eta_to_margin_point =
      object_dist_to_margin_point / std::max(min_vel, object_info->observed_velocity());
    // ego side
    const double ego_dist_to_collision_pos =
      autoware::motion_utils::calcSignedArcLength(path.points, closest_idx, collision_pos);
    const auto ego_eta_to_collision_pos_it = std::lower_bound(
      ego_time_distance_array.begin(), ego_time_distance_array.end(), ego_dist_to_collision_pos,
      [](const auto & a, const double b) { return a.second < b; });
    if (ego_eta_to_collision_pos_it == ego_time_distance_array.end()) {
      continue;
    }
    const double ego_eta_to_collision_pos = ego_eta_to_collision_pos_it->first;
    if (ego_eta_to_collision_pos < object_eta_to_margin_point) {
      // this case, ego will pass the collision point before this object get close to the collision
      // point within margin just by keeping current plan, so no need to accelerate
      continue;
    }
    diag += fmt::format(
      "the object is expected to approach the collision point by the TTC margin in {0} seconds, "
      "while ego will arrive there in {1} seconds, so ego needs to accelerate from current "
      "velocity {2}[m/s] to {3}[m/s]\n",
      object_eta_to_margin_point,                             // 0
      ego_eta_to_collision_pos,                               // 1
      ego_vel,                                                // 2
      ego_dist_to_collision_pos / object_eta_to_margin_point  // 3
    );
  }
  return diag;
}

IntersectionModule::CollisionStatus IntersectionModule::detectCollision(
  const bool is_over_1st_pass_judge_line,
  const std::optional<bool> is_over_2nd_pass_judge_line) const
{
  // ==========================================================================================
  // if collision is detected for multiple objects, we prioritize collision on the first
  // attention lanelet
  // ==========================================================================================
  bool collision_at_first_lane = false;
  bool collision_at_non_first_lane = false;

  // ==========================================================================================
  // find the objects which are judged as UNSAFE after ego passed pass judge lines.
  //
  // misjudge_objects are those that were once judged as safe when ego passed the pass judge line
  //
  // too_late_detect_objects are those that (1) were not detected when ego passed the pass judge
  // line (2) were judged as dangerous at the same time when ego passed the pass judge line, which
  // means they were expected to have been detected when ego passed the pass judge lines or in the
  // prior iteration, because ego could have judged them as UNSAFE if their information was
  // available at that time.
  //
  // that case is both "too late to stop" and "too late to go" for the planner. and basically
  // detection side is responsible for this fault
  // ==========================================================================================
  std::vector<std::pair<CollisionStatus::BlameType, std::shared_ptr<ObjectInfo>>> misjudge_objects;
  std::vector<std::pair<CollisionStatus::BlameType, std::shared_ptr<ObjectInfo>>>
    too_late_detect_objects;
  for (const auto & object_info : object_info_manager_.attentionObjects()) {
    if (object_info->is_safe_under_traffic_control()) {
      debug_data_.safe_under_traffic_control_targets.objects.push_back(
        object_info->predicted_object());
      continue;
    }
    const auto unsafe_info_opt = object_info->unsafe_info();
    if (!unsafe_info_opt) {
      continue;
    }
    const auto & unsafe_info = unsafe_info_opt.value();
    // ==========================================================================================
    // if ego is over the pass judge lines, then the visualization as "too_late_objects" or
    // "misjudge_objects" is more important than that for "unsafe"
    //
    // NOTE: consider a vehicle which was not detected at 1st_pass_judge_passage, and now collision
    // detected on the 1st lane, which is "too_late" for 1st lane passage, but once it decelerated
    // or yielded, so it turned safe, and ego passed the 2nd pass judge line, but at the same it
    // accelerated again, which is "misjudge" for 2nd lane passage. In this case this vehicle is
    // visualized as "misjudge"
    // ==========================================================================================
    auto * debug_container = &debug_data_.unsafe_targets.objects;
    if (unsafe_info.lane_position == CollisionInterval::LanePosition::FIRST) {
      collision_at_first_lane = true;
    } else {
      collision_at_non_first_lane = true;
    }
    if (
      is_over_1st_pass_judge_line &&
      unsafe_info.lane_position == CollisionInterval::LanePosition::FIRST) {
      const auto & decision_at_1st_pass_judge_opt =
        object_info->decision_at_1st_pass_judge_line_passage();
      if (!decision_at_1st_pass_judge_opt) {
        too_late_detect_objects.emplace_back(
          CollisionStatus::BlameType::BLAME_AT_FIRST_PASS_JUDGE, object_info);
        debug_container = &debug_data_.too_late_detect_targets.objects;
      } else {
        const auto & decision_at_1st_pass_judge = decision_at_1st_pass_judge_opt.value();
        if (decision_at_1st_pass_judge.safe_type != CollisionKnowledge::SafeType::UNSAFE) {
          misjudge_objects.emplace_back(
            CollisionStatus::BlameType::BLAME_AT_FIRST_PASS_JUDGE, object_info);
          debug_container = &debug_data_.misjudge_targets.objects;
        } else {
          too_late_detect_objects.emplace_back(
            CollisionStatus::BlameType::BLAME_AT_FIRST_PASS_JUDGE, object_info);
          debug_container = &debug_data_.too_late_detect_targets.objects;
        }
      }
    }
    if (is_over_2nd_pass_judge_line && is_over_2nd_pass_judge_line.value()) {
      const auto & decision_at_2nd_pass_judge_opt =
        object_info->decision_at_2nd_pass_judge_line_passage();
      if (!decision_at_2nd_pass_judge_opt) {
        too_late_detect_objects.emplace_back(
          CollisionStatus::BlameType::BLAME_AT_SECOND_PASS_JUDGE, object_info);
        debug_container = &debug_data_.too_late_detect_targets.objects;
      } else {
        const auto & decision_at_2nd_pass_judge = decision_at_2nd_pass_judge_opt.value();
        if (decision_at_2nd_pass_judge.safe_type != CollisionKnowledge::SafeType::UNSAFE) {
          misjudge_objects.emplace_back(
            CollisionStatus::BlameType::BLAME_AT_SECOND_PASS_JUDGE, object_info);
          debug_container = &debug_data_.misjudge_targets.objects;
        } else {
          too_late_detect_objects.emplace_back(
            CollisionStatus::BlameType::BLAME_AT_SECOND_PASS_JUDGE, object_info);
          debug_container = &debug_data_.too_late_detect_targets.objects;
        }
      }
    }
    debug_container->emplace_back(object_info->predicted_object());
  }
  if (collision_at_first_lane) {
    return {true, CollisionInterval::FIRST, too_late_detect_objects, misjudge_objects};
  } else if (collision_at_non_first_lane) {
    return {true, CollisionInterval::ELSE, too_late_detect_objects, misjudge_objects};
  }
  return {false, CollisionInterval::ELSE, too_late_detect_objects, misjudge_objects};
}

std::optional<size_t> IntersectionModule::checkAngleForTargetLanelets(

  const geometry_msgs::msg::Pose & pose, const lanelet::ConstLanelets & target_lanelets,
  const bool is_parked_vehicle) const
{
  const double detection_area_angle_thr = planner_param_.common.attention_area_angle_threshold;
  // Nero c3 fix: this used to read common.attention_area_angle_threshold (a non-zero double),
  // which made it always true regardless of collision_detection.consider_wrong_direction_vehicle
  const bool consider_wrong_direction_vehicle =
    planner_param_.collision_detection.consider_wrong_direction_vehicle;
  const double dist_margin = planner_param_.common.attention_area_margin;

  for (unsigned i = 0; i < target_lanelets.size(); ++i) {
    const auto & ll = target_lanelets.at(i);
    if (!lanelet::utils::isInLanelet(pose, ll, dist_margin)) {
      continue;
    }
    const double ll_angle = lanelet::utils::getLaneletAngle(ll, pose.position);
    const double pose_angle = tf2::getYaw(pose.orientation);
    const double angle_diff = autoware_utils::normalize_radian(ll_angle - pose_angle, -M_PI);
    if (consider_wrong_direction_vehicle) {
      if (std::fabs(angle_diff) > 1.57 || std::fabs(angle_diff) < detection_area_angle_thr) {
        return std::make_optional<size_t>(i);
      }
    } else {
      if (std::fabs(angle_diff) < detection_area_angle_thr) {
        return std::make_optional<size_t>(i);
      }
      // NOTE: sometimes parked vehicle direction is reversed even if its longitudinal velocity is
      // positive
      if (
        is_parked_vehicle && (std::fabs(angle_diff) < detection_area_angle_thr ||
                              (std::fabs(angle_diff + M_PI) < detection_area_angle_thr))) {
        return std::make_optional<size_t>(i);
      }
    }
  }
  return std::nullopt;
}

IntersectionModule::TimeDistanceArray IntersectionModule::calcIntersectionPassingTime(
  const autoware_internal_planning_msgs::msg::PathWithLaneId & path, const bool is_prioritized,
  const IntersectionStopLines & intersection_stoplines,
  autoware_internal_debug_msgs::msg::Float64MultiArrayStamped * ego_ttc_array) const
{
  const double intersection_velocity =
    planner_param_.collision_detection.velocity_profile.default_velocity;
  const double minimum_ego_velocity =
    planner_param_.collision_detection.velocity_profile.minimum_default_velocity;
  const bool use_upstream_velocity =
    planner_param_.collision_detection.velocity_profile.use_upstream;
  const double minimum_upstream_velocity =
    planner_param_.collision_detection.velocity_profile.minimum_upstream_velocity;
  const double current_velocity = planner_data_->current_velocity->twist.linear.x;

  // ==========================================================================================
  // if ego is waiting for collision detection, the entry time into the intersection
  // is a bit delayed for the chattering hold, so we need to "shift" the TimeDistanceArray by
  // this delay
  // ==========================================================================================
  const bool is_go_out = (activated_ && occlusion_activated_);
  const double time_delay = (is_go_out || is_prioritized)
                              ? 0.0
                              : (planner_param_.collision_detection.collision_detection_hold_time -
                                 collision_state_machine_.getDuration());

  // ==========================================================================================
  // to account for the stopline generated by upstream behavior_velocity modules (like walkway,
  // crosswalk), if use_upstream flag is true, the raw velocity of path points after
  // last_intersection_stopline_candidate_idx is used, which maybe almost-zero. at those almost-zero
  // velocity path points, ego future profile is almost "fixed" there.
  //
  // last_intersection_stopline_candidate_idx must be carefully decided especially when ego
  // velocity is almost zero, because if last_intersection_stopline_candidate_idx is at the
  // closest_idx for example, ego is almost "fixed" at current position for the entire
  // spatiotemporal profile, which is judged as SAFE because that profile does not collide
  // with the predicted paths of objects.
  //
  // if second_attention_lane exists, second_attention_stopline_idx is used. if not,
  // max(occlusion_stopline_idx, first_attention_stopline_idx) is used because
  // occlusion_stopline_idx varies depending on the peeking offset parameter
  // ==========================================================================================
  const auto second_attention_stopline_idx = intersection_stoplines.second_attention_stopline;
  const auto occlusion_stopline_idx = intersection_stoplines.occlusion_peeking_stopline.value();
  const auto first_attention_stopline_idx = intersection_stoplines.first_attention_stopline.value();
  const auto closest_idx = intersection_stoplines.closest_idx;
  const auto last_intersection_stopline_candidate_idx =
    second_attention_stopline_idx ? second_attention_stopline_idx.value()
                                  : std::max(occlusion_stopline_idx, first_attention_stopline_idx);

  bool assigned_lane_found = false;
  // crop intersection part of the path, and set the reference velocity to intersection_velocity
  // for ego's ttc
  PathWithLaneId reference_path;
  std::optional<size_t> upstream_stopline{std::nullopt};
  for (size_t i = 0; i + 1 < path.points.size(); ++i) {
    auto reference_point = path.points.at(i);
    // assume backward velocity is current ego velocity
    if (i < closest_idx) {
      reference_point.point.longitudinal_velocity_mps = current_velocity;
    }
    if (
      i > last_intersection_stopline_candidate_idx &&
      std::fabs(reference_point.point.longitudinal_velocity_mps) <
        std::numeric_limits<double>::epsilon() &&
      !upstream_stopline) {
      upstream_stopline = i;
    }
    if (!use_upstream_velocity) {
      reference_point.point.longitudinal_velocity_mps = intersection_velocity;
    }
    reference_path.points.push_back(reference_point);
    bool has_objective_lane_id = util::hasLaneIds(path.points.at(i), associative_ids_);
    if (assigned_lane_found && !has_objective_lane_id) {
      break;
    }
    assigned_lane_found = has_objective_lane_id;
  }
  if (!assigned_lane_found) {
    return {{0.0, 0.0}};  // has already passed the intersection.
  }

  // apply smoother to reference velocity
  PathWithLaneId smoothed_reference_path = reference_path;
  if (!smoothPath(reference_path, smoothed_reference_path, planner_data_)) {
    smoothed_reference_path = reference_path;
  }

  // calculate when ego is going to reach each (interpolated) points on the path
  TimeDistanceArray time_distance_array{};
  double dist_sum = 0.0;
  double passing_time = time_delay;
  time_distance_array.emplace_back(passing_time, dist_sum);

  // NOTE: `reference_path` is resampled in `reference_smoothed_path`, so
  // `last_intersection_stopline_candidate_idx` makes no sense
  const auto smoothed_path_closest_idx =
    autoware::motion_utils::findFirstNearestIndexWithSoftConstraints(
      smoothed_reference_path.points, path.points.at(closest_idx).point.pose,
      planner_data_->ego_nearest_dist_threshold, planner_data_->ego_nearest_yaw_threshold);

  const std::optional<size_t> upstream_stopline_idx_opt = [&]() -> std::optional<size_t> {
    if (upstream_stopline) {
      const auto upstream_stopline_point =
        reference_path.points.at(upstream_stopline.value()).point.pose;
      return autoware::motion_utils::findFirstNearestIndexWithSoftConstraints(
        smoothed_reference_path.points, upstream_stopline_point,
        planner_data_->ego_nearest_dist_threshold, planner_data_->ego_nearest_yaw_threshold);
    } else {
      return std::nullopt;
    }
  }();

  // junction-yield 10-11 (param-gated, standstill_start_acceleration 0 = stock): while (almost) standing the smoothed
  // reference velocity is ~0 and the profile collapses to minimum_default_velocity, a constant speed the robot never
  // drives (it waits, then accelerates). Model the pull-away instead: v(s) = min(sqrt(v0^2 + 2 a s), raw path velocity),
  // plus standstill_start_delay when standing.
  const double standstill_acc =
    planner_param_.collision_detection.velocity_profile.standstill_start_acceleration;
  const bool standstill_model = standstill_acc > 0.0 && current_velocity < 0.5;
  std::vector<double> raw_s;
  std::vector<double> raw_v;
  if (standstill_model) {
    double s = 0.0;
    for (size_t i = std::min(closest_idx, reference_path.points.size() - 1); i < reference_path.points.size();
         ++i) {
      if (!raw_s.empty()) {
        s += autoware_utils::calc_distance2d(reference_path.points.at(i - 1), reference_path.points.at(i));
      }
      raw_s.push_back(s);
      raw_v.push_back(path.points.at(i).point.longitudinal_velocity_mps);
    }
    if (current_velocity < 0.1) {
      passing_time += planner_param_.collision_detection.velocity_profile.standstill_start_delay;
      time_distance_array.back().first = passing_time;
    }
  }

  for (size_t i = smoothed_path_closest_idx; i + 1 < smoothed_reference_path.points.size(); ++i) {
    const auto & p1 = smoothed_reference_path.points.at(i);
    const auto & p2 = smoothed_reference_path.points.at(i + 1);

    const double dist = autoware_utils::calc_distance2d(p1, p2);
    dist_sum += dist;

    // use average velocity between p1 and p2
    const double average_velocity =
      (p1.point.longitudinal_velocity_mps + p2.point.longitudinal_velocity_mps) / 2.0;
    const double passing_velocity = [&]() {
      if (use_upstream_velocity) {
        if (upstream_stopline_idx_opt && i > upstream_stopline_idx_opt.value()) {
          return minimum_upstream_velocity;
        }
      }
      if (standstill_model && !raw_s.empty()) {
        const double s_mid = std::max(dist_sum - dist / 2.0, 0.0);
        const auto it = std::upper_bound(raw_s.begin(), raw_s.end(), s_mid);
        const size_t k = (it == raw_s.begin()) ? 0 : static_cast<size_t>(it - raw_s.begin() - 1);
        // a stop point in the raw path (the stop sign the ego stands at) is not a speed: use the intersection velocity
        const double cap = raw_v.at(k) < 0.1 ? intersection_velocity : raw_v.at(k);
        const double v_kin =
          std::sqrt(current_velocity * current_velocity + 2.0 * standstill_acc * s_mid);
        return std::max(std::min(v_kin, cap), 0.1);
      }
      return std::max<double>(average_velocity, minimum_ego_velocity);
    }();
    passing_time += (dist / passing_velocity);

    time_distance_array.emplace_back(passing_time, dist_sum);
  }
  ego_ttc_array->stamp = clock_->now();
  ego_ttc_array->layout.dim.resize(3);
  ego_ttc_array->layout.dim.at(0).label = "lane_id_@[0][0], ttc_time, ttc_dist, path_x, path_y";
  constexpr size_t ego_debug_size = 5;
  ego_ttc_array->layout.dim.at(0).size = ego_debug_size;
  ego_ttc_array->layout.dim.at(1).label = "values";
  ego_ttc_array->layout.dim.at(1).size = time_distance_array.size();
  ego_ttc_array->data.reserve(time_distance_array.size() * ego_debug_size);
  for (unsigned i = 0; i < time_distance_array.size(); ++i) {
    ego_ttc_array->data.push_back(lane_id_);
  }
  for (const auto & [t, d] : time_distance_array) {
    ego_ttc_array->data.push_back(t);
  }
  for (const auto & [t, d] : time_distance_array) {
    ego_ttc_array->data.push_back(d);
  }
  for (size_t i = smoothed_path_closest_idx; i < smoothed_reference_path.points.size(); ++i) {
    const auto & p = smoothed_reference_path.points.at(i).point.pose.position;
    ego_ttc_array->data.push_back(p.x);
  }
  for (size_t i = smoothed_path_closest_idx; i < smoothed_reference_path.points.size(); ++i) {
    const auto & p = smoothed_reference_path.points.at(i).point.pose.position;
    ego_ttc_array->data.push_back(p.y);
  }
  return time_distance_array;
}

}  // namespace autoware::behavior_velocity_planner
