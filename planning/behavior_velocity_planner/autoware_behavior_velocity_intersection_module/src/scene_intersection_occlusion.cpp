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

// NERO_IX_SIGHT_EVIDENCE=<fresh_s>,<stand_win_s>,<stand_skip_s>,<stand_frac>,<moving_v>,<log>,<near_m>,<adv_frac>,<stand_mem_s>
// (intersections-2,
// 2026-11-15, QUEUE
// SIGHTPH; needs NERO_IX_SIGHT): what the sight rule takes as evidence of an unpublished car, from the grid's own history at every
// division point it samples. On the car's pointcloud grid every 11-08 sight hold (19:01 / 17:20 trips, 14:12, 15:13; 11-03 61556 the
// same places every pass) was a STANDING place read as a car: a new blob starts at the 12 m/s prior and the alpha-beta needs ~0.8 s to
// learn it does not move, the blob counts after 3 cycles at ~9 m/s -> a stop + the module's 1.5 s hold.
// (M) a cell is evidence only while the lidar hits it: the binary Bayes filter keeps a cell it no longer sees occupied for ~1.9 s
//     (no-information update v -> (v + 5) / 1.1 per frame: 99 -> 58 in 19 frames), and where no free ray reaches it (beyond ~40 m,
//     shadows) that memory is all there is: the trail of a published car driving AWAY beyond its swept-box explanation (61556 19:03:54,
//     c740 8 m/s), the cells of a standing car the tracker dropped (152697 19:03:21), a far car's last return after perception lost it
//     (61556 19:03:56). A cell counts while it was hit (value rose, or >= 98) within <fresh_s> s of the current grid message (a real
//     car 50 m out hits only some of the sampled cells per frame: 61556 17:22:13 hit them in frames 0, 1 and 3), and a track counts
//     once its blob was hit in two different grid messages (one return is no car); the history is updated once per grid message.
// (S) a NEW blob whose start points (+-2 points = 1 m along their divisions) were occupied (hit or remembered; not within 2 s after a
//     published object moving >= <moving_v> m/s lay over them: an approaching car meets the trail of one driving away on these two-way
//     streets, a hidden follower the footprint of its leader) in >= <stand_frac> of the grid messages in [now - stand_win_s,
//     now - stand_skip_s] starts at
//     0 m/s instead of the prior: it was already there (a car arriving at >= 3.3 m/s was >= 1 m upstream 0.3 s earlier); the
//     alpha-beta then follows any real motion. Kerbs / planters at road height (6337 14:12, 36280 19:15) and far standing clutter
//     (146634, 14119, 53079) were re-born at the prior each time their 2-division blob flickered. Occupancy under a moving published
//     object does not count: a hidden follower in a leader's footprint keeps the prior.
// (N) within <near_m> m along the lane the grid shows within 0.2 s whether a blob moves (cells 0.5 m, dense returns; the 1-3 m per
//     cycle jitter of the leading edge is a 50-60 m effect): when a track born at the prior first counts (3 cycles), a blob start
//     that advanced less than <adv_frac> x the prior x its age since its birth did not arrive at that speed: the track takes its
//     measured speed (advance minus one grid cell of quantisation, >= 0) and the alpha-beta goes on from there. Thin posts and parked cars that come into view as the robot rolls
//     up (140436 19:07:53, 152697 19:03:22, 11-03 61556 15:15:09) were read as 12 m/s cars for their first ~0.8 s.
// (A) blobs are matched to tracks best-first (all pairs by prediction error) instead of greedily in list order.
// (K) every cycle: a measurement nearer than predicted at a POINT that was already occupied (the (S) test on the front point itself)
//     re-anchors the track without raising its approach speed (the nearest edge of a standing row of planters flickering along it,
//     6337 14:12:01); a car pulling away from a stand moves its front onto points it never covered and speeds up as before.
// (E) a published object slower than <moving_v> explains its cells for <stand_mem_s> s after it was last published (others 1 s as
//     before): the tracker drops standing UNKNOWNs while a passing car hides them and re-confirms them only frames after they are seen
//     again (6337 14:12:01: kerb planters 7223 / ba2d behind car ec20); a standing thing that vanished is still there, a car that
//     drives off from it leaves its 1 m margin at once.
// Real cars the rule caught before perception published them (61556 17:22:13 oncoming, 36280 19:15:10, 14119 14:15:10 truck) arrived
// at places that were empty the second before: unchanged. Unset = the car exactly.
static const std::array<double, 9> & neroSightEvidenceParam()
{
  static const std::array<double, 9> p = [] {
    std::array<double, 9> q{-1.0, 1.0, 0.3, 0.5, 1.0, 1.0, 35.0, 0.3, 3.0};
    const char * e = std::getenv("NERO_IX_SIGHT_EVIDENCE");
    if (e != nullptr) {
      std::sscanf(
        e, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &q[0], &q[1], &q[2], &q[3], &q[4], &q[5], &q[6], &q[7], &q[8]);
    }
    return q;
  }();
  return p;
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
  const auto & ev_prm = neroSightEvidenceParam();
  const bool ev_on = ev_prm[0] > 0.0;  // NERO_IX_SIGHT_EVIDENCE

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
    if (ev_on) nero_sight_explained_v_[key] = std::hypot(tw.x, tw.y);
  }
  std::vector<Polygon2d> explained;
  std::vector<bool> explained_moving;  // NERO_IX_SIGHT_EVIDENCE
  for (auto it = nero_sight_explained_.begin(); it != nero_sight_explained_.end();) {
    double mem_s = 1.0;
    if (ev_on) {  // (E)
      const auto vit = nero_sight_explained_v_.find(it->first);
      if (vit != nero_sight_explained_v_.end() && vit->second < ev_prm[4]) mem_s = ev_prm[8];
    }
    if (now - it->second.first > mem_s) {
      if (ev_on) nero_sight_explained_v_.erase(it->first);
      it = nero_sight_explained_.erase(it);
      continue;
    }
    Polygon2d cloud;
    for (const auto & [x, y] : it->second.second) cloud.outer().emplace_back(x, y);
    Polygon2d hull;
    bg::convex_hull(cloud, hull);
    explained.push_back(hull);
    if (ev_on) {
      const auto vit = nero_sight_explained_v_.find(it->first);
      explained_moving.push_back(vit != nero_sight_explained_v_.end() && vit->second >= ev_prm[4]);
    }
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
  // NERO_IX_SIGHT_EVIDENCE: (division, point) of each run start; the grid history is updated once per grid message
  std::vector<std::pair<int, int>> start_pts;
  std::vector<double> start_hit;  // the newest grid message in which the lidar hit a cell of the run
  const double ev_stamp = rclcpp::Time(occ_grid.header.stamp).seconds();
  const bool ev_new_grid = ev_on && ev_stamp != nero_sight_ev_stamp_;
  if (ev_on) {
    nero_sight_ev_stamp_ = ev_stamp;
    if (nero_sight_ev_hist_.size() != occlusion_attention_divisions_.value().size()) {
      nero_sight_ev_hist_.assign(occlusion_attention_divisions_.value().size(), {});
    }
  }
  int ev_memory = 0;  // occupied cells left out as grid memory this cycle
  int div_idx = -1;
  if (path_linestring.size() >= 2) {
    for (const auto & division : occlusion_attention_divisions_.value()) {
      ++div_idx;
      if (division.size() < 2) continue;
      if (ev_on && nero_sight_ev_hist_.at(div_idx).size() != division.size()) {
        nero_sight_ev_hist_.at(div_idx).assign(division.size(), NeroSightPointHist{});
      }
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
        double cell_hit = -1e9;  // NERO_IX_SIGHT_EVIDENCE: the newest grid message that hit this cell
        int cell_v = -1;
        if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
          const int v = occ_grid.data.at(iy * width + ix);
          cell_v = v;
          occ = v >= planner_param_.occlusion.occupied_min;
        }
        const bool occ_raw = occ;
        bool expl_moving = false;
        if (occ) {
          const Point2d pt(it->x(), it->y());
          for (size_t k = 0; k < explained.size(); ++k) {
            if (bg::distance(pt, explained.at(k)) <= 1.0) {
              occ = false;
              if (!ev_on) break;
              if (explained_moving.at(k)) {
                expl_moving = true;
                break;
              }
            }
          }
        }
        if (ev_on) {
          auto & h = nero_sight_ev_hist_.at(div_idx).at(static_cast<size_t>(it - division.begin()));
          if (ev_new_grid) {
            // the lidar hit the cell in this grid message: occupied and not decaying like the filter's no-information update
            const bool hit = occ_raw && (h.last_v < 0 || cell_v > h.last_v || cell_v >= 98);
            if (hit) h.hit_t = ev_stamp;
            if (expl_moving) h.moving_t = ev_stamp;
            h.last_v = cell_v;
            h.t.at(h.head) = ev_stamp;
            // (S) occupied, hit or remembered, but not within the grid's memory span (2 s) after a moving published object lay
            // over it: its trail is no standing thing (an approaching car meets the trail of one driving away on two-way streets)
            h.occ.at(h.head) = (occ_raw && ev_stamp - h.moving_t > 2.0) ? 1 : 0;
            h.head = static_cast<uint8_t>((h.head + 1) % h.t.size());
            if (h.n < h.t.size()) ++h.n;
          }
          if (occ && ev_stamp - h.hit_t > ev_prm[0]) {
            occ = false;  // (M) only the filter's memory: no evidence of a car now
            ++ev_memory;
          }
          cell_hit = h.hit_t;
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
        if (occ && !prev_occ) {
          starts.push_back(acc);
          if (ev_on) {
            start_pts.emplace_back(div_idx, static_cast<int>(it - division.begin()));
            start_hit.push_back(cell_hit);
          }
        } else if (ev_on && occ) {
          start_hit.back() = std::max(start_hit.back(), cell_hit);
        }
        prev_occ = occ;
      }
    }
  }
  // blobs: run starts within 2 m of each other, seen on >= 2 divisions (a kerb / pole at the lane edge is on one)
  if (ev_on) {  // keep each start's (division, point) with it
    std::vector<size_t> order(starts.size());
    for (size_t i = 0; i < order.size(); ++i) order.at(i) = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return starts.at(a) < starts.at(b); });
    std::vector<double> s2;
    std::vector<std::pair<int, int>> p2;
    std::vector<double> n2;
    for (const auto i : order) {
      s2.push_back(starts.at(i));
      p2.push_back(start_pts.at(i));
      n2.push_back(start_hit.at(i));
    }
    starts.swap(s2);
    start_pts.swap(p2);
    start_hit.swap(n2);
  } else {
    std::sort(starts.begin(), starts.end());
  }
  std::vector<double> blobs;
  std::vector<std::pair<size_t, size_t>> blob_members;  // NERO_IX_SIGHT_EVIDENCE: [i, j] in starts
  std::vector<double> blob_hit;                         // NERO_IX_SIGHT_EVIDENCE: newest hit among its runs
  for (size_t i = 0; i < starts.size();) {
    size_t j = i;
    while (j + 1 < starts.size() && starts.at(j + 1) - starts.at(j) <= 2.0) ++j;
    if (j > i) {
      blobs.push_back(starts.at(i));
      blob_members.emplace_back(i, j);
      if (ev_on) {
        double nh = -1e9;
        for (size_t m = i; m <= j; ++m) nh = std::max(nh, start_hit.at(m));
        blob_hit.push_back(nh);
      }
    }
    i = j + 1;
  }
  if (ev_on && ev_memory > 0 && ev_prm[5] > 0.0 && now - nero_sight_ev_log_t_ >= 1.0) {
    nero_sight_ev_log_t_ = now;
    RCLCPP_INFO(
      logger_, "[NERO_IX] sight evidence: lane %ld left out %d occupied cells the lidar has not hit for %.1f s (grid memory)",
      lane_id_, ev_memory, ev_prm[0]);
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
  // NERO_IX_SIGHT_EVIDENCE (S): the share of grid messages in [now - stand_win_s, now - stand_skip_s] in which a blob's start points
  // (+-2 points = 1 m along their divisions) were occupied (hit or remembered; not within 2 s after a moving published object).
  // Only the blob's leading edge (run starts within 0.5 m of its nearest one): a standing parked car merged into the same blob
  // must not hide the front of a car coming up beside it.
  // nb = points either side along the division: 2 (1 m) for a birth, 0 for (K) every cycle (a car pulling away from a stand moves
  // its front onto points it never covered; +-1 m would still hold its old front and read it standing until it passes 3.3 m/s)
  const auto place_share = [&](size_t k, int nb) {
    double share_max = 0.0;
    const size_t m0 = blob_members.at(k).first;
    for (size_t m = m0; m <= blob_members.at(k).second && starts.at(m) <= starts.at(m0) + 0.5; ++m) {
      const auto [di, pi] = start_pts.at(m);
      const auto & hd = nero_sight_ev_hist_.at(di);
      for (int q = std::max(0, pi - nb); q <= std::min(static_cast<int>(hd.size()) - 1, pi + nb); ++q) {
        const auto & h = hd.at(q);
        int n_win = 0;
        int n_occ = 0;
        for (uint8_t e = 0; e < h.n; ++e) {
          const double te = h.t.at(e);
          if (te < ev_stamp - ev_prm[1] || te > ev_stamp - ev_prm[2]) continue;
          ++n_win;
          n_occ += h.occ.at(e);
        }
        if (n_win >= 3) share_max = std::max(share_max, static_cast<double>(n_occ) / n_win);
      }
    }
    return share_max;
  };
  // NERO_IX_SIGHT_EVIDENCE (A): each blob goes to the track that predicted it best (all pairs by error), not to the first track in
  // the list whose gate it falls in: a track coasting on a parked row took a standing neighbour's blob 2.5 m nearer and read it
  // as 5-6 m/s of approach (11-03 61556 15:15:19, #32 took #37's blob it had predicted to 0.0 m)
  std::vector<int> ev_assign;
  if (ev_on) {
    ev_assign.assign(nero_sight_tracks_.size(), -1);
    std::vector<std::tuple<double, size_t, size_t>> pairs;
    for (size_t ti = 0; ti < nero_sight_tracks_.size(); ++ti) {
      const auto & tr = nero_sight_tracks_.at(ti);
      const double pred = tr.d - tr.v * dt;
      for (size_t k = 0; k < blobs.size(); ++k) {
        const double err = std::fabs(blobs.at(k) - pred);
        if (err <= 2.0 + 0.3 * std::fabs(tr.v)) pairs.emplace_back(err, ti, k);
      }
    }
    std::sort(pairs.begin(), pairs.end());
    std::vector<bool> blob_taken(blobs.size(), false);
    for (const auto & [err, ti, k] : pairs) {
      if (ev_assign.at(ti) >= 0 || blob_taken.at(k)) continue;
      ev_assign.at(ti) = static_cast<int>(k);
      blob_taken.at(k) = true;
    }
  }
  int ev_known = 0;  // (K) track updates at a known place this cycle
  for (size_t ti = 0; ti < nero_sight_tracks_.size(); ++ti) {
    auto & tr = nero_sight_tracks_.at(ti);
    const double pred = tr.d - tr.v * dt;
    int best = -1;
    if (ev_on) {
      best = ev_assign.at(ti);
    } else {
      double best_err = 2.0 + 0.3 * std::fabs(tr.v);
      for (size_t k = 0; k < blobs.size(); ++k) {
        if (used.at(k)) continue;
        const double err = std::fabs(blobs.at(k) - pred);
        if (err <= best_err) {
          best_err = err;
          best = static_cast<int>(k);
        }
      }
    }
    if (best < 0) {
      ++tr.misses;
      tr.d = std::max(0.0, pred);
      continue;
    }
    used.at(best) = true;
    tr.misses = 0;
    if (ev_on && blob_hit.at(best) > tr.hit_last) {  // (M) a new lidar frame hit the blob
      tr.hit_last = blob_hit.at(best);
      ++tr.hit_msgs;
    }
    const double r = blobs.at(best) - pred;
    if (ev_on && r < 0.0 && place_share(static_cast<size_t>(best), 0) >= ev_prm[3]) {
      // (K) the blob's front is at a place that was already occupied most of the second before: no evidence of approach (a
      // car's front enters cells that were empty); the nearest edge of a standing row flickering along it (6337 14:12:01,
      // planters 9-13 m) re-anchors the track, it does not speed it up
      tr.d = blobs.at(best);
      ++ev_known;
    } else {
      tr.d = std::max(0.0, pred + alpha * r);
      tr.v = std::clamp(tr.v - beta * r / dt, -25.0, 25.0);
    }
    tr.hist.emplace_back(now, tr.d);
    while (!tr.hist.empty() && now - tr.hist.front().first > 1.2) tr.hist.pop_front();
    if (ev_on && !tr.adv_checked && tr.hist.size() >= 3) {
      // (N) at its first count: near blobs must have advanced like the prior says
      tr.adv_checked = true;
      const double age = std::max(0.05, now - tr.meas_t0);
      const double adv = tr.meas_d0 - blobs.at(best);
      if (tr.v_born > 0.0 && tr.meas_d0 <= ev_prm[6] && adv < ev_prm[7] * tr.v_born * age) {
        // one grid cell of the leading edge's quantisation is no motion (a standing kerb's edge stepping 0.5 m in 0.2 s read as 2.5
        // m/s at 6337 14:12:10.9)
        const double v_meas = std::max(0.0, (adv - resolution) / age);
        if (ev_prm[5] > 0.0) {
          RCLCPP_INFO(
            logger_,
            "[NERO_IX] sight evidence: lane %ld blob born at %.1f m advanced %.2f m in %.2f s (a %.1f m/s car: %.2f m): "
            "measured %.1f m/s, not %.1f",
            lane_id_, tr.meas_d0, adv, age, tr.v_born, tr.v_born * age, v_meas, tr.v);
        }
        tr.v = v_meas;
      }
    }
  }
  if (ev_on && ev_known > 0 && ev_prm[5] > 0.0 && now - nero_sight_ev_log_t_ >= 1.0) {
    nero_sight_ev_log_t_ = now;
    RCLCPP_INFO(
      logger_, "[NERO_IX] sight evidence: lane %ld %d track(s) re-anchored at a place occupied before (no approach from it)",
      lane_id_, ev_known);
  }
  nero_sight_tracks_.erase(
    std::remove_if(nero_sight_tracks_.begin(), nero_sight_tracks_.end(), [](const auto & tr) { return tr.misses > 3; }),
    nero_sight_tracks_.end());
  for (size_t k = 0; k < blobs.size(); ++k) {
    if (!used.at(k)) {
      NeroSightTrack tr;
      tr.d = blobs.at(k);
      tr.v = v_prior;
      if (ev_on) {
        // (S) a new blob at a place occupied most of the second before (no moving published object over it within 2 s) was
        // already there: it starts standing
        const double share_max = place_share(k, 2);
        if (share_max >= ev_prm[3]) {
          tr.v = 0.0;
          if (ev_prm[5] > 0.0) {
            RCLCPP_INFO(
              logger_,
              "[NERO_IX] sight evidence: lane %ld new blob at %.1f m was there before (occupied %.2f of the grid messages "
              "%.1f..%.1f s ago): starts at 0 m/s, not %.1f",
              lane_id_, tr.d, share_max, ev_prm[1], ev_prm[2], v_prior);
          }
        }
      }
      if (ev_on) {
        tr.hit_last = blob_hit.at(k);
        tr.hit_msgs = 1;
        tr.meas_d0 = blobs.at(k);
        tr.meas_t0 = now;
        tr.v_born = tr.v;
      }
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
    if (ev_on && tr.hit_msgs < 2) {
      // (M) and once the lidar hit it in two frames: a single return (the last one of a car perception lost at 70 m,
      // 61556 19:03:56.3) is no approaching car; a real far car keeps adding hits (61556 17:22:13: 3 frames in 0.3 s)
      if (ev_prm[5] > 0.0 && now - nero_sight_ev_log_t_ >= 1.0) {
        nero_sight_ev_log_t_ = now;
        RCLCPP_INFO(
          logger_, "[NERO_IX] sight evidence: lane %ld blob at %.1f m hit in one lidar frame only: not counted", lane_id_,
          tr.d);
      }
      continue;
    }
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
