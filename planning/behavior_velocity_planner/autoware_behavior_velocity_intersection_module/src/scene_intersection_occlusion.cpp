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

#include <autoware_utils/geometry/boost_polygon_utils.hpp>  // for toPolygon2d
#include <autoware_utils/geometry/geometry.hpp>
#include <opencv2/imgproc.hpp>

#include <boost/geometry/algorithms/convex_hull.hpp>
#include <boost/geometry/algorithms/correct.hpp>
#include <boost/geometry/algorithms/intersection.hpp>

#include <lanelet2_core/geometry/Polygon.h>
#include <tf2/utils.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <string>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace autoware::behavior_velocity_planner
{
namespace bg = boost::geometry;

std::tuple<
  IntersectionModule::OcclusionType, bool /* module detection with margin */,
  bool /* reconciled occlusion disapproval */>
IntersectionModule::getOcclusionStatus(
  const TrafficPrioritizedLevel & traffic_prioritized_level,
  const InterpolatedPathInfo & interpolated_path_info)
{
  const auto & intersection_lanelets = intersection_lanelets_.value();
  const auto & occlusion_attention_lanelets = intersection_lanelets.occlusion_attention();

  // ==========================================================================================
  // for the convenience of Psim user, this module ignores occlusion if there has not been any
  // information published for the associated traffic light even if occlusion.enable is true,
  // and only runs collision checking on that intersection lane.
  //
  // this is because Psim-users/scenario-files do not set traffic light information perfectly
  // most of the times, and they just set bare minimum traffic information only for traffic lights
  // they are interested in or want to test.
  //
  // no_tl_info_ever variable is defined for that purpose. if there has been any
  // information published for the associated traffic light in the real world through perception/V2I
  // or in the simulation, then it should be kept in last_tl_valid_observation_ and this variable
  // becomes false
  // ==========================================================================================
  const bool no_tl_info_ever = (has_traffic_light_ && !last_tl_valid_observation_.has_value());
  const bool is_amber_or_red_or_no_tl_info_ever =
    (traffic_prioritized_level == TrafficPrioritizedLevel::PARTIALLY_PRIORITIZED) ||
    (traffic_prioritized_level == TrafficPrioritizedLevel::FULLY_PRIORITIZED) || no_tl_info_ever;
  // check occlusion on detection lane
  auto occlusion_status =
    (planner_param_.occlusion.enable && !occlusion_attention_lanelets.empty() &&
     !is_amber_or_red_or_no_tl_info_ever)
      ? detectOcclusion(interpolated_path_info)
      : NotOccluded{};

  // ==========================================================================================
  // if the traffic light changed from green to yellow/red, hysteresis time for occlusion is
  // unnecessary
  // ==========================================================================================
  const auto transition_to_prioritized =
    (previous_prioritized_level_ == TrafficPrioritizedLevel::NOT_PRIORITIZED &&
     traffic_prioritized_level != TrafficPrioritizedLevel::NOT_PRIORITIZED);
  if (transition_to_prioritized) {
    occlusion_stop_state_machine_.setState(StateMachine::State::GO);
  } else {
    occlusion_stop_state_machine_.setStateWithMarginTime(
      std::holds_alternative<NotOccluded>(occlusion_status) ? StateMachine::State::GO
                                                            : StateMachine::STOP,
      logger_.get_child("occlusion_stop"), *clock_);
  }

  const bool is_occlusion_cleared_with_margin =
    (occlusion_stop_state_machine_.getState() == StateMachine::State::GO);  // module's detection
  // distinguish if ego detected occlusion or RTC detects occlusion
  const bool ext_occlusion_requested =
    (is_occlusion_cleared_with_margin && !occlusion_activated_);  // RTC's detection
  if (ext_occlusion_requested) {
    occlusion_status = RTCOccluded{};
  }
  const bool is_occlusion_state =
    (!is_occlusion_cleared_with_margin || ext_occlusion_requested);  // including approval
  if (is_occlusion_state && std::holds_alternative<NotOccluded>(occlusion_status)) {
    occlusion_status = prev_occlusion_status_;
  } else {
    prev_occlusion_status_ = occlusion_status;
  }
  return {occlusion_status, is_occlusion_cleared_with_margin, is_occlusion_state};
}

IntersectionModule::OcclusionType IntersectionModule::detectOcclusion(
  const InterpolatedPathInfo & interpolated_path_info) const
{
  const auto & intersection_lanelets = intersection_lanelets_.value();
  const auto & adjacent_lanelets = intersection_lanelets.adjacent();
  const auto & attention_areas = intersection_lanelets.occlusion_attention_area();
  const auto first_attention_area = intersection_lanelets.first_attention_area().value();
  const auto & lane_divisions = occlusion_attention_divisions_.value();

  const auto & occ_grid = *planner_data_->occupancy_grid;
  const auto & current_pose = planner_data_->current_odometry->pose;
  const double occlusion_dist_thr = planner_param_.occlusion.occlusion_required_clearance_distance;

  const auto & path_ip = interpolated_path_info.path;
  const auto & lane_interval_ip = interpolated_path_info.lane_id_interval.value();

  const auto first_attention_area_idx =
    util::getFirstPointInsidePolygon(path_ip, lane_interval_ip, first_attention_area);
  if (!first_attention_area_idx) {
    return NotOccluded{};
  }

  const auto first_inside_attention_idx_ip_opt =
    util::getFirstPointInsidePolygon(path_ip, lane_interval_ip, first_attention_area);
  const std::pair<size_t, size_t> lane_attention_interval_ip =
    first_inside_attention_idx_ip_opt
      ? std::make_pair(first_inside_attention_idx_ip_opt.value(), std::get<1>(lane_interval_ip))
      : lane_interval_ip;
  const auto [lane_start_idx, lane_end_idx] = lane_attention_interval_ip;

  const int width = occ_grid.info.width;
  const int height = occ_grid.info.height;
  const double resolution = occ_grid.info.resolution;
  const auto & origin = occ_grid.info.origin.position;
  auto coord2index = [&](const double x, const double y) {
    const int idx_x = (x - origin.x) / resolution;
    const int idx_y = (y - origin.y) / resolution;
    if (idx_x < 0 || idx_x >= width) return std::make_tuple(false, -1, -1);
    if (idx_y < 0 || idx_y >= height) return std::make_tuple(false, -1, -1);
    return std::make_tuple(true, idx_x, idx_y);
  };

  Polygon2d grid_poly;
  grid_poly.outer().emplace_back(origin.x, origin.y);
  grid_poly.outer().emplace_back(origin.x + (width - 1) * resolution, origin.y);
  grid_poly.outer().emplace_back(
    origin.x + (width - 1) * resolution, origin.y + (height - 1) * resolution);
  grid_poly.outer().emplace_back(origin.x, origin.y + (height - 1) * resolution);
  grid_poly.outer().emplace_back(origin.x, origin.y);
  bg::correct(grid_poly);

  auto findCommonCvPolygons =
    [&](const auto & area2d, std::vector<std::vector<cv::Point>> & cv_polygons) -> void {
    autoware_utils::Polygon2d area2d_poly;
    for (const auto & p : area2d) {
      area2d_poly.outer().emplace_back(p.x(), p.y());
    }
    area2d_poly.outer().push_back(area2d_poly.outer().front());
    bg::correct(area2d_poly);
    std::vector<Polygon2d> common_areas;
    bg::intersection(area2d_poly, grid_poly, common_areas);
    if (common_areas.empty()) {
      return;
    }
    for (size_t i = 0; i < common_areas.size(); ++i) {
      common_areas[i].outer().push_back(common_areas[i].outer().front());
      bg::correct(common_areas[i]);
    }
    for (const auto & common_area : common_areas) {
      std::vector<cv::Point> cv_polygon;
      for (const auto & p : common_area.outer()) {
        const int idx_x = static_cast<int>((p.x() - origin.x) / resolution);
        const int idx_y = static_cast<int>((p.y() - origin.y) / resolution);
        cv_polygon.emplace_back(idx_x, height - 1 - idx_y);
      }
      cv_polygons.push_back(cv_polygon);
    }
  };

  // (1) prepare detection area mask
  // attention: 255
  // non-attention: 0
  // NOTE: interesting area is set to 255 for later masking
  cv::Mat attention_mask(width, height, CV_8UC1, cv::Scalar(0));
  std::vector<std::vector<cv::Point>> attention_area_cv_polygons;
  for (const auto & attention_area : attention_areas) {
    const auto area2d = lanelet::utils::to2D(attention_area);
    findCommonCvPolygons(area2d, attention_area_cv_polygons);
  }
  for (const auto & poly : attention_area_cv_polygons) {
    cv::fillPoly(attention_mask, poly, cv::Scalar(255), cv::LINE_AA);
  }
  // (1.1)
  // reset adjacent_lanelets area to 0 on attention_mask
  std::vector<std::vector<cv::Point>> adjacent_lane_cv_polygons;
  for (const auto & adjacent_lanelet : adjacent_lanelets) {
    const auto area2d = adjacent_lanelet.polygon2d().basicPolygon();
    findCommonCvPolygons(area2d, adjacent_lane_cv_polygons);
  }
  for (const auto & poly : adjacent_lane_cv_polygons) {
    cv::fillPoly(attention_mask, poly, cv::Scalar(0), cv::LINE_AA);
  }

  // (2) prepare unknown mask
  // In OpenCV the pixel at (X=x, Y=y) (with left-upper origin) is accessed by img[y, x]
  // unknown: 255
  // not-unknown: 0
  cv::Mat unknown_mask_raw(width, height, CV_8UC1, cv::Scalar(0));
  cv::Mat unknown_mask(width, height, CV_8UC1, cv::Scalar(0));
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      const int idx = y * width + x;
      const unsigned char intensity = occ_grid.data.at(idx);
      if (
        planner_param_.occlusion.free_space_max <= intensity &&
        intensity < planner_param_.occlusion.occupied_min) {
        unknown_mask_raw.at<unsigned char>(height - 1 - y, x) = 255;
      }
    }
  }
  // (2.1) apply morphologyEx
  const int morph_size = static_cast<int>(planner_param_.occlusion.denoise_kernel / resolution);
  cv::morphologyEx(
    unknown_mask_raw, unknown_mask, cv::MORPH_OPEN,
    cv::getStructuringElement(cv::MORPH_RECT, cv::Size(morph_size, morph_size)));

  // (3) occlusion mask
  static constexpr unsigned char OCCLUDED = 255;
  static constexpr unsigned char BLOCKED = 127;
  cv::Mat occlusion_mask(width, height, CV_8UC1, cv::Scalar(0));
  cv::bitwise_and(attention_mask, unknown_mask, occlusion_mask);
  // re-use attention_mask
  attention_mask = cv::Mat(width, height, CV_8UC1, cv::Scalar(0));
  // (3.1) draw all cells on attention_mask behind blocking vehicles as not occluded
  const auto & blocking_attention_objects = object_info_manager_.parkedObjects();
  for (const auto & blocking_attention_object_info : blocking_attention_objects) {
    debug_data_.parked_targets.objects.push_back(
      blocking_attention_object_info->predicted_object());
  }
  std::vector<std::vector<cv::Point>> blocking_polygons;
  for (const auto & blocking_attention_object_info : blocking_attention_objects) {
    const Polygon2d obj_poly =
      autoware_utils::to_polygon2d(blocking_attention_object_info->predicted_object());
    findCommonCvPolygons(obj_poly.outer(), blocking_polygons);
  }
  for (const auto & blocking_polygon : blocking_polygons) {
    cv::fillPoly(attention_mask, blocking_polygon, cv::Scalar(BLOCKED), cv::LINE_AA);
  }
  for (const auto & division : lane_divisions) {
    bool blocking_vehicle_found = false;
    for (const auto & point_it : division) {
      const auto [valid, idx_x, idx_y] = coord2index(point_it.x(), point_it.y());
      if (!valid) continue;
      if (blocking_vehicle_found) {
        occlusion_mask.at<unsigned char>(height - 1 - idx_y, idx_x) = 0;
        continue;
      }
      if (attention_mask.at<unsigned char>(height - 1 - idx_y, idx_x) == BLOCKED) {
        blocking_vehicle_found = true;
        occlusion_mask.at<unsigned char>(height - 1 - idx_y, idx_x) = 0;
      }
    }
  }

  // (4) extract occlusion polygons
  const auto & possible_object_bbox = planner_param_.occlusion.possible_object_bbox;
  const double possible_object_bbox_x = possible_object_bbox.at(0) / resolution;
  const double possible_object_bbox_y = possible_object_bbox.at(1) / resolution;
  const double possible_object_area = possible_object_bbox_x * possible_object_bbox_y;
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(occlusion_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  std::vector<std::vector<cv::Point>> valid_contours;
  for (const auto & contour : contours) {
    if (contour.size() <= 2) {
      continue;
    }
    std::vector<cv::Point> approx_contour;
    cv::approxPolyDP(
      contour, approx_contour,
      std::round(std::min(possible_object_bbox_x, possible_object_bbox_y) / std::sqrt(2.0)), true);
    if (approx_contour.size() <= 2) continue;
    // check area
    const double poly_area = cv::contourArea(approx_contour);
    if (poly_area < possible_object_area) continue;
    // check bounding box size
    const auto bbox = cv::minAreaRect(approx_contour);
    if (const auto size = bbox.size; std::min(size.height, size.width) <
                                       std::min(possible_object_bbox_x, possible_object_bbox_y) ||
                                     std::max(size.height, size.width) <
                                       std::max(possible_object_bbox_x, possible_object_bbox_y)) {
      continue;
    }
    valid_contours.push_back(approx_contour);
    geometry_msgs::msg::Polygon polygon_msg;
    geometry_msgs::msg::Point32 point_msg;
    for (const auto & p : approx_contour) {
      const double glob_x = (p.x + 0.5) * resolution + origin.x;
      const double glob_y = (height - 0.5 - p.y) * resolution + origin.y;
      point_msg.x = glob_x;
      point_msg.y = glob_y;
      point_msg.z = origin.z;
      polygon_msg.points.push_back(point_msg);
    }
    debug_data_.occlusion_polygons.push_back(polygon_msg);
  }
  // (4.1) re-draw occluded cells using valid_contours
  occlusion_mask = cv::Mat(width, height, CV_8UC1, cv::Scalar(0));
  for (const auto & valid_contour : valid_contours) {
    // NOTE: drawContour does not work well
    cv::fillPoly(occlusion_mask, valid_contour, cv::Scalar(OCCLUDED), cv::LINE_AA);
  }

  // (5) find distance
  // (5.1) discretize path_ip with resolution for computational cost
  LineString2d path_linestring;
  path_linestring.emplace_back(
    path_ip.points.at(lane_start_idx).point.pose.position.x,
    path_ip.points.at(lane_start_idx).point.pose.position.y);
  {
    auto prev_path_linestring_point = path_ip.points.at(lane_start_idx).point.pose.position;
    for (auto i = lane_start_idx + 1; i <= lane_end_idx; i++) {
      const auto path_linestring_point = path_ip.points.at(i).point.pose.position;
      if (
        autoware_utils::calc_distance2d(prev_path_linestring_point, path_linestring_point) <
        1.0 /* rough tick for computational cost */) {
        continue;
      }
      path_linestring.emplace_back(path_linestring_point.x, path_linestring_point.y);
      prev_path_linestring_point = path_linestring_point;
    }
  }

  auto findNearestPointToProjection = [](
                                        const lanelet::ConstLineString3d & division,
                                        const Point2d & projection, const double dist_thresh) {
    double min_dist = std::numeric_limits<double>::infinity();
    auto nearest = division.end();
    for (auto it = division.begin(); it != division.end(); it++) {
      const double dist = std::hypot(it->x() - projection.x(), it->y() - projection.y());
      if (dist < min_dist) {
        min_dist = dist;
        nearest = it;
      }
      if (dist < dist_thresh) {
        break;
      }
    }
    return nearest;
  };
  struct NearestOcclusionInterval
  {
    int64 division_index{0};
    int64 point_index{0};
    geometry_msgs::msg::Point point;
    geometry_msgs::msg::Point projection;
    geometry_msgs::msg::Point visible_end;
  } nearest_occlusion_point;
  double min_dist = std::numeric_limits<double>::infinity();
  for (unsigned division_index = 0; division_index < lane_divisions.size(); ++division_index) {
    const auto & division = lane_divisions.at(division_index);
    LineString2d division_linestring;
    auto division_point_it = division.begin();
    division_linestring.emplace_back(division_point_it->x(), division_point_it->y());
    for (auto point_it = division.begin(); point_it != division.end(); point_it++) {
      if (
        std::hypot(point_it->x() - division_point_it->x(), point_it->y() - division_point_it->y()) <
        3.0 /* rough tick for computational cost */) {
        continue;
      }
      division_linestring.emplace_back(point_it->x(), point_it->y());
      division_point_it = point_it;
    }

    // find the intersection point of lane_line and path
    std::vector<Point2d> intersection_points;
    boost::geometry::intersection(division_linestring, path_linestring, intersection_points);
    if (intersection_points.empty()) {
      continue;
    }
    const auto & projection_point = intersection_points.at(0);
    const auto projection_it = findNearestPointToProjection(division, projection_point, resolution);
    if (projection_it == division.end()) {
      continue;
    }
    double acc_dist = 0.0;
    bool found_min_dist_for_this_division = false;
    bool is_prev_occluded = false;
    auto acc_dist_it = projection_it;
    for (auto point_it = projection_it; point_it != division.end(); point_it++) {
      const double dist =
        std::hypot(point_it->x() - acc_dist_it->x(), point_it->y() - acc_dist_it->y());
      acc_dist += dist;
      acc_dist_it = point_it;
      const auto [valid, idx_x, idx_y] = coord2index(point_it->x(), point_it->y());
      if (!valid) continue;
      const auto pixel = occlusion_mask.at<unsigned char>(height - 1 - idx_y, idx_x);
      if (pixel == BLOCKED) {
        break;
      }
      if (pixel == OCCLUDED) {
        if (acc_dist < min_dist) {
          min_dist = acc_dist;
          nearest_occlusion_point = {
            division_index, std::distance(division.begin(), point_it),
            autoware_utils::create_point(point_it->x(), point_it->y(), origin.z),
            autoware_utils::create_point(projection_it->x(), projection_it->y(), origin.z),
            autoware_utils::create_point(
              projection_it->x(), projection_it->y(),
              origin.z) /* initialize with projection point at first*/};
          found_min_dist_for_this_division = true;
        } else if (found_min_dist_for_this_division && is_prev_occluded) {
          // although this cell is not "nearest" cell, we have found the "nearest" cell on this
          // division previously in this iteration, and the iterated cells are still OCCLUDED since
          // then
          nearest_occlusion_point.visible_end =
            autoware_utils::create_point(point_it->x(), point_it->y(), origin.z);
        }
      }
      is_prev_occluded = (pixel == OCCLUDED);
    }
  }

  if (min_dist == std::numeric_limits<double>::infinity() || min_dist > occlusion_dist_thr) {
    return NotOccluded{min_dist};
  }

  debug_data_.nearest_occlusion_projection =
    std::make_pair(nearest_occlusion_point.point, nearest_occlusion_point.projection);
  debug_data_.nearest_occlusion_triangle = std::make_tuple(
    current_pose.position, nearest_occlusion_point.point, nearest_occlusion_point.visible_end);
  Polygon2d ego_occlusion_triangle;
  ego_occlusion_triangle.outer().emplace_back(current_pose.position.x, current_pose.position.y);
  ego_occlusion_triangle.outer().emplace_back(
    nearest_occlusion_point.point.x, nearest_occlusion_point.point.y);
  ego_occlusion_triangle.outer().emplace_back(
    nearest_occlusion_point.visible_end.x, nearest_occlusion_point.visible_end.y);
  bg::correct(ego_occlusion_triangle);
  for (const auto & attention_object_info : object_info_manager_.allObjects()) {
    const auto obj_poly = autoware_utils::to_polygon2d(attention_object_info->predicted_object());
    if (bg::intersects(obj_poly, ego_occlusion_triangle)) {
      debug_data_.static_occlusion = false;
      return DynamicallyOccluded{min_dist};
    }
  }
  debug_data_.static_occlusion = true;
  return StaticallyOccluded{min_dist};
}
}  // namespace autoware::behavior_velocity_planner

namespace autoware::behavior_velocity_planner
{
// NERO_IX_SIGHT=<D>,<v_min>,<margin_s>,<a_stop>,<t_delay>,<K>,<v_prior>,<a_go>,<go_margin_s> (leftsight, 2026-10-27;
// QUEUE LTP, 10-26e #23): at a stop-line approach without a traffic light whose first attention lane begins within K m
// past the default stop line (a turn across a priority lane right after the line), a vehicle the lidar sees but
// perception has not published yet is treated as a car (on the first attention lane only). #23: the 9.4 m/s oncoming
// car was first published 7.5 m away, 0.8 s before it reached the robot, which had left its line after the stop sign's
// 0.1 s hold.
// Upstream's occlusion check (#10868 / #10911 / #10947: stop at the peeking line until the attention lanes are seen
// clear) counts only UNKNOWN cells; the grid's binary Bayes filter keeps a cell seen free "free" for ~2 s after it is
// last seen (v_ratio 10: 21 frames to 43), so a car driving along a lane seen free casts no unknown shadow within its
// first ~19 m (9.4 m/s) and the check sees it only ~36 m out of a 55 m clearance: too late for a robot that needs ~5 s
// to cross. A car in the points is OCCUPIED within ~0.4 s. So: along the module's own attention lane divisions (the
// occlusion check's, from where each crosses ego's path upstream to D m), occupied cells that no predicted object
// explains (outside every object's box swept back 0.8 s along its velocity + 1 m, for 1 s after it was last
// published), that the lidar can see now (not behind a published object >= 1 m tall, seen from ego) and that span >= 2
// divisions are blobs; blobs are tracked cycle to cycle along the lane (a new one
// approaches at v_prior until its own motion shows otherwise); once ego is at its line, one seen in >= 3 cycles and
// approaching the conflict at >= v_min that passes it while ego is in the first attention lane (with margin_s after ego
// leaves it) is a collision (the module's normal stop / hold applies), while ego stands or can still stop before that
// lane (once it cannot, ego goes on as the car does today). Static occupied cells (kerbs, poles) never approach and are
// ignored; published vehicles are left to the collision check.
// With it, after ego has stood at its line clear of the first attention lane, a collision stop while it moves off whose
// docking stop at ego (the current acceleration for t_delay [s] while the command ramps down, then a_stop [m/s2] and
// the controller's -3.4 once 0.5 m past the point: QUEUE SMS) would end inside that lane becomes a go-through when ego
// leaves the lane before the colliding car or blob arrives (+go_margin_s; ego from its speed at a_go [m/s2] up to the
// path's speed: the field robot took off at ~0.55 m/s2, the module's own profile assumes <= 5 km/h through the
// junction) and every colliding car is on that lane; otherwise the stock stop stays. On the car's stack a car first
// published just after the take-off stopped the robot ~1 m inside the priority lane, where the car hit it (b01).
// Unset = the car exactly.
static const std::array<double, 9> & neroSightParam()
{
  static const std::array<double, 9> p = [] {
    std::array<double, 9> q{-1.0, 1.5, 2.0, 0.8, 0.4, 3.0, 12.0, 0.5, 0.5};
    const char * e = std::getenv("NERO_IX_SIGHT");
    if (e != nullptr) {
      std::sscanf(
        e, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &q[0], &q[1], &q[2], &q[3], &q[4], &q[5], &q[6], &q[7], &q[8]);
    }
    return q;
  }();
  return p;
}

static bool neroSightDebug()
{
  static const bool d = [] {
    const char * e = std::getenv("NERO_IX_SIGHT_DEBUG");
    return e != nullptr && std::atoi(e) > 0;
  }();
  return d;
}

void IntersectionModule::neroSightUpdate(
  const InterpolatedPathInfo & interpolated_path_info, const IntersectionStopLines & intersection_stoplines,
  const autoware_internal_planning_msgs::msg::PathWithLaneId & path,
  const TimeDistanceArray & time_distance_array)
{
  nero_sight_ = NeroSightStatus{};
  const auto & prm = neroSightParam();
  const double D = prm[0];
  if (D <= 0.0 || has_traffic_light_ || !intersection_lanelets_ || !occlusion_attention_divisions_) {
    return;
  }
  const auto & intersection_lanelets = intersection_lanelets_.value();
  if (!intersection_lanelets.first_attention_area() || !intersection_stoplines.default_stopline) {
    return;
  }
  const auto first_attention_area = intersection_lanelets.first_attention_area().value();
  const auto default_stopline_idx = intersection_stoplines.default_stopline.value();
  const auto first_attention_stopline_idx = intersection_stoplines.first_attention_stopline.value_or(0);
  // scope (once): a map stop line, the first attention lane within K m past the default stop line
  if (!nero_sight_scope_) {
    const auto & assigned_lanelet =
      planner_data_->route_handler_->getLaneletMapPtr()->laneletLayer.get(lane_id_);
    const bool map_stopline = getStopLineIndexFromMap(interpolated_path_info, assigned_lanelet).has_value();
    const double gap = autoware::motion_utils::calcSignedArcLength(
      path.points, default_stopline_idx, first_attention_stopline_idx);
    nero_sight_scope_ = map_stopline && gap <= prm[5];
    RCLCPP_INFO(
      logger_,
      "[NERO_IX] sight lane %ld: %s (map stop line %d, first attention lane %.2f m past the default stop line)",
      lane_id_, nero_sight_scope_.value() ? "in scope" : "not in scope", map_stopline, gap);
  }
  if (!nero_sight_scope_.value()) {
    return;
  }
  nero_sight_.in_scope = true;
  const auto & current_pose = planner_data_->current_odometry->pose;
  const double now = clock_->now().seconds();

  // ego stood at its line, footprint clear of the first attention area (latched for this module)
  if (!nero_sight_stood_clear_ && planner_data_->isVehicleStopped()) {
    const double past =
      autoware::motion_utils::calcSignedArcLength(path.points, default_stopline_idx, current_pose.position);
    const auto local_footprint = planner_data_->vehicle_info_.createFootprint(0.0, 0.0);
    const auto fp = autoware_utils::transform_vector(local_footprint, autoware_utils::pose2transform(current_pose));
    if (past > -1.5 && past < 1.5 && !bg::intersects(fp, lanelet::utils::to2D(first_attention_area).basicPolygon())) {
      nero_sight_stood_clear_ = true;
      RCLCPP_INFO(logger_, "[NERO_IX] sight lane %ld: ego stands %.2f m past the default stop line, clear of the "
                  "first attention lane", lane_id_, past);
    }
  }

  // ego's time to leave the first attention area (footprint), from the module's own ego profile
  {
    const auto local_footprint = planner_data_->vehicle_info_.createFootprint(0.0, 0.0);
    const auto area2d = lanelet::utils::to2D(first_attention_area).basicPolygon();
    std::optional<size_t> exit_idx;
    bool inside = false;
    for (size_t i = std::max<size_t>(intersection_stoplines.closest_idx, first_attention_stopline_idx);
         i < path.points.size(); ++i) {
      const auto fp = autoware_utils::transform_vector(
        local_footprint, autoware_utils::pose2transform(path.points.at(i).point.pose));
      const bool in = bg::intersects(fp, area2d);
      if (in) inside = true;
      if (inside && !in) {
        exit_idx = i;
        break;
      }
    }
    const double d_exit = exit_idx ? autoware::motion_utils::calcSignedArcLength(
                                       path.points, intersection_stoplines.closest_idx, exit_idx.value())
                                   : 0.0;
    nero_sight_.d_exit = d_exit;
    const double d_enter = std::max(
      0.0, autoware::motion_utils::calcSignedArcLength(
             path.points, intersection_stoplines.closest_idx, first_attention_stopline_idx));
    nero_sight_.ego_clear = time_distance_array.empty() ? 0.0 : time_distance_array.back().first;
    for (const auto & [t, d] : time_distance_array) {
      if (d >= d_exit) {
        nero_sight_.ego_clear = t;
        break;
      }
    }
    nero_sight_.ego_enter = 0.0;
    for (const auto & [t, d] : time_distance_array) {
      if (d >= d_enter) {
        nero_sight_.ego_enter = t;
        break;
      }
    }
  }

  // unexplained occupied cells along the attention lane divisions
  const auto & occ_grid = *planner_data_->occupancy_grid;
  const int width = occ_grid.info.width;
  const int height = occ_grid.info.height;
  const double resolution = occ_grid.info.resolution;
  const auto & origin = occ_grid.info.origin.position;
  // a published object explains the cells within 1 m of its box swept back along its velocity by 0.8 s (the grid's
  // filter lags a moving object by ~0.3-0.4 s: in the #1 "nice!" replay a 7-8 m/s car's cells trailed its box by ~3 m
  // and held the robot 3.5 s), and for 1 s after it was last published (its cells decay after the track ends)
  for (const auto & object : planner_data_->predicted_objects->objects) {
    const auto box = autoware_utils::to_polygon2d(object);
    const auto & tw = object.kinematics.initial_twist_with_covariance.twist.linear;
    const double yaw = tf2::getYaw(object.kinematics.initial_pose_with_covariance.pose.orientation);
    const double bx = -(tw.x * std::cos(yaw) - tw.y * std::sin(yaw)) * 0.8;
    const double by = -(tw.x * std::sin(yaw) + tw.y * std::cos(yaw)) * 0.8;
    std::vector<std::pair<double, double>> pts;
    for (const auto & q : box.outer()) {
      pts.emplace_back(q.x(), q.y());
      pts.emplace_back(q.x() + bx, q.y() + by);
    }
    std::string key(object.object_id.uuid.begin(), object.object_id.uuid.end());
    nero_sight_explained_[key] = {now, pts};
  }
  std::vector<Polygon2d> explained;
  for (auto it = nero_sight_explained_.begin(); it != nero_sight_explained_.end();) {
    if (now - it->second.first > 1.0) {
      it = nero_sight_explained_.erase(it);
      continue;
    }
    Polygon2d cloud;
    for (const auto & [x, y] : it->second.second) cloud.outer().emplace_back(x, y);
    Polygon2d hull;
    bg::convex_hull(cloud, hull);
    explained.push_back(hull);
    ++it;
  }
  // cells the lidar cannot see now (behind a published object from ego: its body blocks every ray behind it) are stale
  // and no evidence of a car: on the car's pointcloud grid a published car coming head-on leaves its own trail in its
  // shadow (never seen free, it decays only through the grid's filter, ~1.7 s); the trail's leading edge followed the
  // car ~8 m behind and read as a car following it (mr66508g stays: 62 UNSAFE cycles, the robot left 1.2 s later than
  // the car's stack at a must-go). A real car hidden there is hidden from the lidar too; out of the shadow it is a new
  // blob (v_prior). Low objects (a flat UNKNOWN) block no ray at the lidar's height.
  struct NeroShadow
  {
    double a0, lo, hi, rmin;
  };
  std::vector<NeroShadow> shadows;
  const Point2d ego2d(current_pose.position.x, current_pose.position.y);
  for (const auto & object : planner_data_->predicted_objects->objects) {
    if (object.shape.dimensions.z < 1.0) continue;
    const auto box = autoware_utils::to_polygon2d(object);
    const double rmin = bg::distance(ego2d, box);
    if (rmin < 0.5) continue;
    const auto & op = object.kinematics.initial_pose_with_covariance.pose.position;
    const double a0 = std::atan2(op.y - ego2d.y(), op.x - ego2d.x());
    double lo = 0.0;
    double hi = 0.0;
    for (const auto & q : box.outer()) {
      const double rel = autoware_utils::normalize_radian(std::atan2(q.y() - ego2d.y(), q.x() - ego2d.x()) - a0);
      lo = std::min(lo, rel);
      hi = std::max(hi, rel);
    }
    shadows.push_back({a0, lo, hi, rmin});
  }
  int n_shadowed = 0;
  const auto & path_ip = interpolated_path_info.path;
  const auto & lane_interval_ip = interpolated_path_info.lane_id_interval.value();
  const auto first_inside_ip = util::getFirstPointInsidePolygon(path_ip, lane_interval_ip, first_attention_area);
  const size_t lane_start_idx = first_inside_ip ? first_inside_ip.value() : std::get<0>(lane_interval_ip);
  const size_t lane_end_idx = std::get<1>(lane_interval_ip);
  LineString2d path_linestring;
  for (auto i = lane_start_idx; i <= lane_end_idx && i < path_ip.points.size(); ++i) {
    const auto & p = path_ip.points.at(i).point.pose.position;
    path_linestring.emplace_back(p.x, p.y);
  }
  const auto first_area2d = lanelet::utils::to2D(first_attention_area).basicPolygon();
  std::vector<double> starts;  // nearest distance of each occupied run along a division
  if (path_linestring.size() >= 2) {
    for (const auto & division : occlusion_attention_divisions_.value()) {
      if (division.size() < 2) continue;
      LineString2d division_linestring;
      for (const auto & p : division) division_linestring.emplace_back(p.x(), p.y());
      std::vector<Point2d> ips;
      bg::intersection(division_linestring, path_linestring, ips);
      if (ips.empty()) continue;
      // only the first attention lane (the priority lane right after the line): a blob on a later lane must not
      // make ego go through toward it (the move-off rule)
      if (!bg::within(ips.front(), first_area2d)) continue;
      // nearest division point to the crossing, then upstream (divisions run from the conflict upstream)
      auto it0 = division.begin();
      double best = std::numeric_limits<double>::infinity();
      for (auto it = division.begin(); it != division.end(); ++it) {
        const double dd = std::hypot(it->x() - ips.front().x(), it->y() - ips.front().y());
        if (dd < best) {
          best = dd;
          it0 = it;
        }
      }
      double acc = 0.0;
      bool prev_occ = false;
      auto prev_it = it0;
      for (auto it = it0; it != division.end() && acc <= D; ++it) {
        acc += std::hypot(it->x() - prev_it->x(), it->y() - prev_it->y());
        prev_it = it;
        const int ix = static_cast<int>((it->x() - origin.x) / resolution);
        const int iy = static_cast<int>((it->y() - origin.y) / resolution);
        bool occ = false;
        if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
          const int v = occ_grid.data.at(iy * width + ix);
          occ = v >= planner_param_.occlusion.occupied_min;
        }
        if (occ) {
          const Point2d pt(it->x(), it->y());
          for (const auto & poly : explained) {
            if (bg::distance(pt, poly) <= 1.0) {
              occ = false;
              break;
            }
          }
        }
        if (occ) {
          const double r = std::hypot(it->x() - ego2d.x(), it->y() - ego2d.y());
          const double a = std::atan2(it->y() - ego2d.y(), it->x() - ego2d.x());
          for (const auto & sh : shadows) {
            const double rel = autoware_utils::normalize_radian(a - sh.a0);
            if (r > sh.rmin + 0.5 && rel >= sh.lo && rel <= sh.hi) {
              occ = false;
              ++n_shadowed;
              break;
            }
          }
        }
        if (occ && !prev_occ) starts.push_back(acc);
        prev_occ = occ;
      }
    }
  }
  // blobs: run starts within 2 m of each other, seen on >= 2 divisions (a kerb / pole at the lane edge is on one)
  std::sort(starts.begin(), starts.end());
  std::vector<double> blobs;
  for (size_t i = 0; i < starts.size();) {
    size_t j = i;
    while (j + 1 < starts.size() && starts.at(j + 1) - starts.at(j) <= 2.0) ++j;
    if (j > i) blobs.push_back(starts.at(i));
    i = j + 1;
  }
  // track blobs along the lane: alpha-beta on (distance, approach speed); a new blob starts at the prior approach speed
  // v_prior (a vehicle until its own motion shows otherwise: the grid's leading edge jitters 1-3 m per cycle at
  // 50-60 m, b03 ls3: a regression over the first cycles kept restarting and reached 8.4 m/s of the true 9.4 only 2.3 s
  // after the car was in the grid); a blob that does not come closer slows its speed to 0 within ~1 s
  const double v_prior = prm[6];
  const double dt = nero_sight_last_t_ > 0.0 ? std::clamp(now - nero_sight_last_t_, 0.02, 0.5) : 0.1;
  nero_sight_last_t_ = now;
  constexpr double alpha = 0.5;
  constexpr double beta = 0.1;
  std::vector<bool> used(blobs.size(), false);
  for (auto & tr : nero_sight_tracks_) {
    const double pred = tr.d - tr.v * dt;
    int best = -1;
    double best_err = 2.0 + 0.3 * std::fabs(tr.v);
    for (size_t k = 0; k < blobs.size(); ++k) {
      if (used.at(k)) continue;
      const double err = std::fabs(blobs.at(k) - pred);
      if (err <= best_err) {
        best_err = err;
        best = static_cast<int>(k);
      }
    }
    if (best < 0) {
      ++tr.misses;
      tr.d = std::max(0.0, pred);
      continue;
    }
    used.at(best) = true;
    tr.misses = 0;
    const double r = blobs.at(best) - pred;
    tr.d = std::max(0.0, pred + alpha * r);
    tr.v = std::clamp(tr.v - beta * r / dt, -25.0, 25.0);
    tr.hist.emplace_back(now, tr.d);
    while (!tr.hist.empty() && now - tr.hist.front().first > 1.2) tr.hist.pop_front();
  }
  nero_sight_tracks_.erase(
    std::remove_if(nero_sight_tracks_.begin(), nero_sight_tracks_.end(), [](const auto & tr) { return tr.misses > 3; }),
    nero_sight_tracks_.end());
  for (size_t k = 0; k < blobs.size(); ++k) {
    if (!used.at(k)) {
      NeroSightTrack tr;
      tr.d = blobs.at(k);
      tr.v = v_prior;
      tr.hist.emplace_back(now, tr.d);
      nero_sight_tracks_.push_back(tr);
    }
  }
  // the earliest approaching blob; a conflict when it passes the conflict (front at arrival, its ~8 m of car + lane
  // after) while ego is in the first attention lane, with margin_s after ego leaves it and 1 s before ego enters.
  // Judged only once ego is at its line (within 2 m before the default stop line or past it): on the approach the
  // stop line holds ego anyway, and a car passing long before ego gets there is no conflict (ls3 held the robot 3.5 s
  // on the approach for 2df5)
  const double v_min = prm[1];
  const double margin = prm[2];
  double pass_end = -1.0;
  for (const auto & tr : nero_sight_tracks_) {
    if (tr.v < v_min || tr.hist.size() < 3) continue;  // a blob counts once it has been seen in 3 cycles
    const double a = tr.d / tr.v;
    if (a < nero_sight_.arrival) {
      nero_sight_.arrival = a;
      pass_end = a + 8.0 / tr.v;
    }
  }
  const double past_default =
    autoware::motion_utils::calcSignedArcLength(path.points, default_stopline_idx, current_pose.position);
  // only while ego stands or can still stop before the first attention lane (the car's real stop: the current
  // acceleration for t_delay while the command ramps down, then the docking stop): a blob seen once ego can no longer
  // stop short would only make it stop inside or at the edge of that lane (b07 vs b08: a 9.4 m/s car seen 0.6 s after
  // the take-off on its 12 m/s prior: the stop left the robot's corner at the lane edge, the car 0.07 m away; the car's
  // stack, blind to it, crossed 1.2 s ahead): then ego goes on as the car does today
  {
    const double v = std::max(0.0, planner_data_->current_velocity->twist.linear.x);
    const double a0 = std::max(0.0, planner_data_->current_acceleration->accel.accel.linear.x);
    const double td = prm[4];
    const double v1 = v + a0 * td;
    const double d_soft = v1 * v1 / (2.0 * prm[3]);
    const double d_need =
      v * td + 0.5 * a0 * td * td + (d_soft <= 0.5 ? d_soft : 0.5 + (v1 * v1 - 2.0 * prm[3] * 0.5) / (2.0 * 3.4));
    const double d_touch = autoware::motion_utils::calcSignedArcLength(
      path.points, current_pose.position, first_attention_stopline_idx);
    nero_sight_.can_stop_short = planner_data_->isVehicleStopped() || d_touch >= d_need;
  }
  nero_sight_.unsafe = nero_sight_.can_stop_short && past_default > -2.0 &&
                       nero_sight_.arrival < nero_sight_.ego_clear + margin &&
                       pass_end > nero_sight_.ego_enter - 1.0;
  if (neroSightDebug() || (nero_sight_.unsafe && nero_sight_.arrival < 1e8)) {
    std::string tracks;
    for (const auto & tr : nero_sight_tracks_) {
      tracks += " " + std::to_string(static_cast<int>(tr.d * 10) / 10.0).substr(0, 5) + "@" +
                std::to_string(static_cast<int>(tr.v * 10) / 10.0).substr(0, 4) + (tr.misses ? "m" : "");
    }
    RCLCPP_INFO(
      logger_,
      "[NERO_IX] sight lane %ld: %s blobs %zu tracks [%s ] arrival %.1f s, ego in %.1f..%.1f s (+%.1f), %.1f m past "
      "the default stop line%s, %d cells in shadows",
      lane_id_, nero_sight_.unsafe ? "UNSAFE" : "clear", blobs.size(), tracks.c_str(), nero_sight_.arrival,
      nero_sight_.ego_enter, nero_sight_.ego_clear, margin, past_default, nero_sight_stood_clear_ ? " stood" : "",
      n_shadowed);
  }
}
}  // namespace autoware::behavior_velocity_planner
