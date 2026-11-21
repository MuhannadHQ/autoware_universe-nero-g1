// Copyright 2023 TIER IV, Inc.
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

#include "scene.hpp"

#include "autoware/behavior_path_planner_common/utils/drivable_area_expansion/static_drivable_area.hpp"
#include "autoware/behavior_path_planner_common/utils/path_safety_checker/objects_filtering.hpp"
#include "autoware/behavior_path_planner_common/utils/path_utils.hpp"
#include "autoware/behavior_path_planner_common/utils/utils.hpp"
#include "autoware/behavior_path_static_obstacle_avoidance_module/utils.hpp"

#include <autoware/behavior_path_lane_change_module/utils/calculation.hpp>
#include <autoware/behavior_path_lane_change_module/utils/utils.hpp>
#include <autoware/behavior_path_static_obstacle_avoidance_module/data_structs.hpp>
#include <autoware_lanelet2_extension/utility/utilities.hpp>
#include <autoware_utils/ros/uuid_helper.hpp>
#include <autoware_utils/geometry/boost_polygon_utils.hpp>
#include <autoware_lanelet2_extension/utility/query.hpp>
#include <rclcpp/logging.hpp>

#include <boost/geometry/algorithms/centroid.hpp>
#include <boost/geometry/strategies/cartesian/centroid_bashein_detmer.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace
{
// NERO_ABLC_DEBUG=1 (log only, mainlane 2026-11-11): why avoidance_by_lane_change does / does not request (once per 0.5 s):
// the objects in the current lanes (dropped as on the centre line / kept, avoid_required), the nearest target's distance vs
// the minimum lane change / avoidance lengths, the direction and the target lanes.
bool nero_ablc_dbg_every(const int period_ms)
{
  static const bool on = std::getenv("NERO_ABLC_DEBUG") != nullptr;
  if (!on) return false;
  static std::chrono::steady_clock::time_point last;
  const auto now = std::chrono::steady_clock::now();
  if (now - last < std::chrono::milliseconds(period_ms)) return false;
  last = now;
  return true;
}
int nero_ablc_n_in_lane = 0;    // objects overlapping the current lanes in the last fill
int nero_ablc_n_centre = 0;     // ... dropped for being within th_offset_from_centerline of the centre line
std::string nero_ablc_centre_ids;

// NERO_ABLC_NEED=<margin_m>,<max_dev_m>,<row_n>,<row_gap_m>,<start_m>,<start_s> (izparked-2 2026-11-22, Ahmed 11-21: "it should go to
// the other lane only if there is no space on its lane or it would keep hopping back and forth because of too many double parked
// cars"; "the lane change avoidance was way too early pluas the robot could have squeexed through"). For each avoid_required target
// ahead, the free width of the own lane beside it (its own box to the far bound, the side with more room) against what static
// avoidance needs to pass it inside the lane: robot width + margin_m (hard margin + its envelope buffer) - max_dev_m (its
// max_deviation_from_lane). The other lane is requested only for (a) a target that does not fit, or (b) the first of >= row_n
// targets each taking the body over the line (free < width + margin) with gaps < row_gap_m; and only once that target is within
// max(start_m, speed * start_s). Everything else is static avoidance's in-lane pass. Unset = as before (every avoid_required
// target, from as far as it is seen: 71 m on the 21 Nov route).
struct NeroNeed
{
  bool on{false};
  double margin{0.4};
  double max_dev{0.2};
  int row_n{3};
  double row_gap{20.0};
  double start_m{30.0};
  double start_s{6.0};
};
const NeroNeed & nero_need_cfg()
{
  static const NeroNeed c = [] {
    NeroNeed n;
    if (const char * e = std::getenv("NERO_ABLC_NEED")) {
      const int k = std::sscanf(
        e, "%lf,%lf,%d,%lf,%lf,%lf", &n.margin, &n.max_dev, &n.row_n, &n.row_gap, &n.start_m, &n.start_s);
      n.on = k >= 1;
      if (n.on) {
        RCLCPP_INFO(
          rclcpp::get_logger("avoidance_by_lane_change"),
          "[NERO_ABLC_NEED] on: margin %.2f max_dev %.2f row %d gap %.1f start max(%.1f m, %.1f s)", n.margin, n.max_dev,
          n.row_n, n.row_gap, n.start_m, n.start_s);
      }
    }
    return n;
  }();
  return c;
}
// signed distance of p from the polyline (> 0 left of its direction), nearest segment
double nero_signed_dist(const std::vector<lanelet::BasicPoint2d> & ls, const lanelet::BasicPoint2d & p)
{
  double best = std::numeric_limits<double>::max();
  double sd = 0.0;
  for (size_t i = 0; i + 1 < ls.size(); ++i) {
    const lanelet::BasicPoint2d ab = ls[i + 1] - ls[i];
    const double l2 = ab.squaredNorm();
    if (l2 < 1e-9) continue;
    const lanelet::BasicPoint2d ap = p - ls[i];
    const double t = std::clamp(ap.dot(ab) / l2, 0.0, 1.0);
    const double d = (ap - t * ab).norm();
    if (d < best) {
      best = d;
      sd = (ab.x() * ap.y() - ab.y() * ap.x()) >= 0.0 ? d : -d;
    }
  }
  return sd;
}

geometry_msgs::msg::Point32 create_point32(const geometry_msgs::msg::Pose & pose)
{
  geometry_msgs::msg::Point32 p;
  p.x = static_cast<float>(pose.position.x);
  p.y = static_cast<float>(pose.position.y);
  p.z = static_cast<float>(pose.position.z);
  return p;
};

geometry_msgs::msg::Polygon create_execution_area(
  const autoware::vehicle_info_utils::VehicleInfo & vehicle_info,
  const geometry_msgs::msg::Pose & pose, double additional_lon_offset, double additional_lat_offset)
{
  const double & base_to_front = vehicle_info.max_longitudinal_offset_m;
  const double & width = vehicle_info.vehicle_width_m;
  const double & base_to_rear = vehicle_info.rear_overhang_m;

  // if stationary object, extend forward and backward by the half of lon length
  const double forward_lon_offset = base_to_front + additional_lon_offset;
  const double backward_lon_offset = -base_to_rear;
  const double lat_offset = width / 2.0 + additional_lat_offset;

  const auto p1 = autoware_utils::calc_offset_pose(pose, forward_lon_offset, lat_offset, 0.0);
  const auto p2 = autoware_utils::calc_offset_pose(pose, forward_lon_offset, -lat_offset, 0.0);
  const auto p3 = autoware_utils::calc_offset_pose(pose, backward_lon_offset, -lat_offset, 0.0);
  const auto p4 = autoware_utils::calc_offset_pose(pose, backward_lon_offset, lat_offset, 0.0);
  geometry_msgs::msg::Polygon polygon;

  polygon.points.push_back(create_point32(p1));
  polygon.points.push_back(create_point32(p2));
  polygon.points.push_back(create_point32(p3));
  polygon.points.push_back(create_point32(p4));

  return polygon;
}
}  // namespace

namespace autoware::behavior_path_planner
{
using autoware::behavior_path_planner::Direction;
using autoware::behavior_path_planner::LaneChangeModuleType;
using autoware::behavior_path_planner::ObjectInfo;
using autoware::behavior_path_planner::Point2d;

AvoidanceByLaneChange::AvoidanceByLaneChange(
  const std::shared_ptr<LaneChangeParameters> & parameters,
  std::shared_ptr<AvoidanceByLCParameters> avoidance_parameters)
: NormalLaneChange(parameters, LaneChangeModuleType::AVOIDANCE_BY_LANE_CHANGE, Direction::NONE),
  avoidance_parameters_(std::move(avoidance_parameters)),
  avoidance_helper_{std::make_shared<AvoidanceHelper>(avoidance_parameters_)}
{
}

bool AvoidanceByLaneChange::specialRequiredCheck() const
{
  const auto & data = avoidance_data_;
  const bool nero_log = nero_ablc_dbg_every(500);
  const auto nero_lanes = [&]() {
    std::string out;
    if (common_data_ptr_ && common_data_ptr_->lanes_ptr) {
      out += "cur";
      for (const auto & l : common_data_ptr_->lanes_ptr->current) out += " " + std::to_string(l.id());
      out += " tgt";
      for (const auto & l : common_data_ptr_->lanes_ptr->target) out += " " + std::to_string(l.id());
    }
    return out;
  };

  if (data.target_objects.empty()) {
    RCLCPP_DEBUG(logger_, "no empty objects");
    if (nero_log) {
      RCLCPP_INFO(
        logger_, "[NERO_ABLC] no: no target | in_lane %d on_centre %d (%s) | dir %d | %s", nero_ablc_n_in_lane,
        nero_ablc_n_centre, nero_ablc_centre_ids.c_str(), static_cast<int>(direction_), nero_lanes().c_str());
    }
    return false;
  }

  const auto & object_parameters = avoidance_parameters_->object_parameters;

  const auto count_target_object = [&](const auto sum, const auto & p) {
    const auto & objects = avoidance_data_.target_objects;

    const auto is_avoidance_target = [&p](const auto & object) {
      const auto target_class = utils::getHighestProbLabel(object.object.classification) == p.first;
      return target_class && object.avoid_required;
    };

    return sum + std::count_if(objects.begin(), objects.end(), is_avoidance_target);
  };
  const auto num_of_avoidance_targets =
    std::accumulate(object_parameters.begin(), object_parameters.end(), 0UL, count_target_object);

  if (num_of_avoidance_targets < 1) {
    RCLCPP_DEBUG(logger_, "no avoidance target");
    if (nero_log) {
      const auto & o = data.target_objects.front();
      RCLCPP_INFO(
        logger_, "[NERO_ABLC] no: no avoid_required target | targets %zu nearest lon %.1f to_centre %.2f overhang %.2f | dir %d | %s",
        data.target_objects.size(), o.longitudinal, o.to_centerline, o.overhang_points.empty() ? 0.0 : o.overhang_points.front().first,
        static_cast<int>(direction_), nero_lanes().c_str());
    }
    return false;
  }

  // NERO_ABLC_NEED: only a target that does not fit in the own lane (or a row) asks for the other lane, and only near it
  if (nero_need_cfg().on) {
    if (nero_need_idx_ < 0 || nero_need_idx_ >= static_cast<int>(data.target_objects.size())) {
      if (nero_log) {
        RCLCPP_INFO(logger_, "[NERO_ABLC_NEED] no: %s | %s", nero_need_why_.c_str(), nero_lanes().c_str());
      }
      return false;
    }
    const auto & o = data.target_objects.at(nero_need_idx_);
    const double min_avoid = calcMinAvoidanceLength(o);
    const double min_lc = calc_minimum_dist_buffer();
    const bool ok = o.longitudinal > std::max(min_lc, min_avoid);
    if (nero_log) {
      RCLCPP_INFO(
        logger_, "[NERO_ABLC_NEED] %s | %s | min_lc %.1f min_avoid %.1f | dir %d | %s",
        ok ? "yes" : "no: nearer than the minimum lengths", nero_need_why_.c_str(), min_lc, min_avoid,
        static_cast<int>(direction_), nero_lanes().c_str());
    }
    return ok;
  }

  // NERO_ABLC_LANES (mainlane): measure the change's room to the nearest object that needs avoiding; upstream took the nearest
  // object of any kind, so a kerb-parked car needing no avoidance 16 m ahead ended the request while the wall it was for was
  // still 21 m ahead (sim S4 r31: "nearer than the minimum lengths" from 10.4 s, static avoidance took the wall)
  const auto & nearest_object = nero_nearest_required();
  const auto minimum_avoid_length = calcMinAvoidanceLength(nearest_object);
  const auto minimum_lane_change_length = calc_minimum_dist_buffer();

  lane_change_debug_.execution_area = create_execution_area(
    getCommonParam().vehicle_info, getEgoPose(),
    std::max(minimum_lane_change_length, minimum_avoid_length), calcLateralOffset());

  RCLCPP_DEBUG(
    logger_, "Conditions ? %f, %f, %f", nearest_object.longitudinal, minimum_lane_change_length,
    minimum_avoid_length);
  const bool nero_ok = nearest_object.longitudinal > std::max(minimum_lane_change_length, minimum_avoid_length);
  if (nero_log) {
    RCLCPP_INFO(
      logger_, "[NERO_ABLC] %s | targets %zu avoid_req %zu nearest lon %.1f to_centre %.2f | min_lc %.1f min_avoid %.1f | dir %d | %s",
      nero_ok ? "yes" : "no: nearer than the minimum lengths", data.target_objects.size(),
      static_cast<size_t>(num_of_avoidance_targets), nearest_object.longitudinal, nearest_object.to_centerline,
      minimum_lane_change_length, minimum_avoid_length, static_cast<int>(direction_), nero_lanes().c_str());
  }
  return nero_ok;
}

bool AvoidanceByLaneChange::specialExpiredCheck() const
{
  return !specialRequiredCheck();
}

void AvoidanceByLaneChange::updateSpecialData()
{
  const auto p = std::dynamic_pointer_cast<AvoidanceParameters>(avoidance_parameters_);

  avoidance_debug_data_ = DebugData();
  avoidance_data_ = calcAvoidancePlanningData(avoidance_debug_data_);

  if (avoidance_data_.target_objects.empty()) {
    direction_ = Direction::NONE;
  } else {
    direction_ = utils::static_obstacle_avoidance::isOnRight(avoidance_data_.target_objects.front())
                   ? Direction::LEFT
                   : Direction::RIGHT;
  }

  // NERO_ABLC_FIX_10669=1: port of upstream #10669's avoidance-by-lane-change half: compensate lost targets from the
  // stored objects without touching them, then refresh the stored objects from this cycle's detected targets (their
  // necessity already filled by calcAvoidancePlanningData). Unset = the car (one call that refreshes, then compensates).
  static const bool fix_10669 = [] {
    const char * v = std::getenv("NERO_ABLC_FIX_10669");
    return v != nullptr && std::atoi(v) > 0;
  }();
  if (fix_10669) {
    auto current_target_objects = avoidance_data_.target_objects;
    utils::static_obstacle_avoidance::compensateLostTargetObjects(
      avoidance_data_, registered_objects_, planner_data_);
    utils::static_obstacle_avoidance::updateStoredObjects(
      registered_objects_, current_target_objects, clock_.now(), p);
  } else {
    utils::static_obstacle_avoidance::compensateLostTargetObjects(
      registered_objects_, avoidance_data_, clock_.now(), planner_data_, p);
  }

  std::sort(
    avoidance_data_.target_objects.begin(), avoidance_data_.target_objects.end(),
    [](auto a, auto b) { return a.longitudinal < b.longitudinal; });
  // NERO_ABLC_LANES (mainlane): the direction from the nearest object that needs avoiding and is still ahead (upstream: the
  // first target in perception's order, before sorting: with barrier pieces on both sides of the 80325 work zone it flipped
  // left / right between cycles; field logsim w1108_1910). None ahead: as upstream.
  {
    static const bool nero_lanes_dir = [] {
      const char * v = std::getenv("NERO_ABLC_LANES");
      return v != nullptr && std::atoi(v) > 0;
    }();
    if (nero_lanes_dir && !avoidance_data_.target_objects.empty()) {
      const auto & objs = avoidance_data_.target_objects;
      const auto it = std::find_if(objs.begin(), objs.end(), [](const auto & o) {
        return o.avoid_required && o.longitudinal > 0.0;
      });
      if (it != objs.end()) {
        direction_ = utils::static_obstacle_avoidance::isOnRight(*it) ? Direction::LEFT : Direction::RIGHT;
      }
    }
  }
  // NERO_ABLC_NEED: which target (if any) really needs the other lane; the direction follows it
  nero_update_need();
  if (nero_need_idx_ >= 0) {
    direction_ = utils::static_obstacle_avoidance::isOnRight(avoidance_data_.target_objects.at(nero_need_idx_))
                   ? Direction::LEFT
                   : Direction::RIGHT;
  }
  // NERO_ABLC_DEBUG (log only): the module's own data each 0.5 s (the request chain: current lanes -> objects in them ->
  // targets -> direction -> target lanes -> isLaneChangeRequired)
  {
    static std::chrono::steady_clock::time_point nero_last;
    static const bool nero_on = std::getenv("NERO_ABLC_DEBUG") != nullptr;
    const auto now = std::chrono::steady_clock::now();
    if (nero_on && now - nero_last > std::chrono::milliseconds(500)) {
      nero_last = now;
      const auto & L = common_data_ptr_->lanes_ptr;
      const auto path_cur = utils::getCurrentLanesFromPath(prev_module_output_.path, planner_data_);
      const auto tl = path_cur.empty() ? lanelet::ConstLanelets{} : get_lane_change_lanes(path_cur);
      lanelet::ConstLanelet cl;
      const bool in_route = common_data_ptr_->route_handler_ptr &&
                            common_data_ptr_->route_handler_ptr->getClosestLaneletWithinRoute(getEgoPose(), &cl);
      RCLCPP_INFO(
        logger_,
        "[NERO_ABLC_DATA] cur %zu tgt %zu nbr %zu | objects %zu in_lane %d on_centre %d targets %zu | dir %d cdir %d | path_cur %zu "
        "(%ld) -> tgt %zu (%ld) | in_route %d",
        L ? L->current.size() : 0UL, L ? L->target.size() : 0UL, L ? L->target_neighbor.size() : 0UL,
        planner_data_->dynamic_object ? planner_data_->dynamic_object->objects.size() : 0UL, nero_ablc_n_in_lane,
        nero_ablc_n_centre, avoidance_data_.target_objects.size(), static_cast<int>(direction_),
        static_cast<int>(common_data_ptr_->direction), path_cur.size(), path_cur.empty() ? 0L : path_cur.front().id(),
        tl.size(), tl.empty() ? 0L : tl.front().id(), in_route ? 1 : 0);
    }
  }
}

AvoidancePlanningData AvoidanceByLaneChange::calcAvoidancePlanningData(
  AvoidanceDebugData & debug) const
{
  AvoidancePlanningData data;

  // reference pose
  data.reference_pose = getEgoPose();

  data.reference_path_rough = prev_module_output_.path;

  const auto resample_interval = avoidance_parameters_->resample_interval_for_planning;
  data.reference_path = utils::resamplePathWithSpline(data.reference_path_rough, resample_interval);

  data.current_lanelets = get_current_lanes();
  // NERO_ABLC_LANES=1 (mainlane 2026-11-11): the module's current lanes are set by update_lanes() only once it has target
  // lanes, the target lanes need a direction, the direction comes from the targets found here, and the targets are searched in
  // the current lanes: an idle avoidance_by_lane_change never has any of them (field: 0 lane changes in 1297 captures; sim:
  // "cur 0 tgt 0 | targets 0 | dir 0" every cycle with a parked car ahead). Seed the search with the lanes the previous
  // module's path runs in (what update_lanes() computes as current_lanes). Unset = upstream.
  static const bool nero_lanes = [] {
    const char * v = std::getenv("NERO_ABLC_LANES");
    return v != nullptr && std::atoi(v) > 0;
  }();
  if (nero_lanes && data.current_lanelets.empty()) {
    data.current_lanelets = utils::getCurrentLanesFromPath(prev_module_output_.path, planner_data_);
  }

  fillAvoidanceTargetObjects(data, debug);

  return data;
}

void AvoidanceByLaneChange::fillAvoidanceTargetObjects(
  AvoidancePlanningData & data, [[maybe_unused]] DebugData & debug) const
{
  const auto p = std::dynamic_pointer_cast<AvoidanceParameters>(avoidance_parameters_);

  const auto [object_within_target_lane, object_outside_target_lane] =
    utils::path_safety_checker::separateObjectsByLanelets(
      *planner_data_->dynamic_object, data.current_lanelets,
      [](const auto & obj, const auto & lane, const auto yaw_threshold) {
        return utils::path_safety_checker::isPolygonOverlapLanelet(obj, lane, yaw_threshold);
      });

  // Assume that the maximum allocation for data.other object is the sum of
  // objects_within_target_lane and object_outside_target_lane. The maximum allocation for
  // data.target_objects is equal to object_within_target_lane
  {
    const auto other_objects_size =
      object_within_target_lane.objects.size() + object_outside_target_lane.objects.size();
    data.other_objects.reserve(other_objects_size);
    data.target_objects.reserve(object_within_target_lane.objects.size());
  }

  {
    const auto & objects = object_outside_target_lane.objects;
    std::transform(
      objects.cbegin(), objects.cend(), std::back_inserter(data.other_objects),
      [](const auto & object) {
        ObjectData other_object;
        other_object.object = object;
        other_object.info = ObjectInfo::OUT_OF_TARGET_AREA;
        return other_object;
      });
  }

  ObjectDataArray target_lane_objects;
  target_lane_objects.reserve(object_within_target_lane.objects.size());
  nero_ablc_n_in_lane = static_cast<int>(object_within_target_lane.objects.size());
  nero_ablc_n_centre = 0;
  nero_ablc_centre_ids.clear();
  for (const auto & obj : object_within_target_lane.objects) {
    const auto target_lane_object = createObjectData(data, obj);
    if (!target_lane_object) {
      continue;
    }

    target_lane_objects.push_back(*target_lane_object);
  }

  data.target_objects = target_lane_objects;
}

std::optional<ObjectData> AvoidanceByLaneChange::createObjectData(
  const AvoidancePlanningData & data, const PredictedObject & object) const
{
  using autoware::motion_utils::findNearestIndex;
  using autoware_utils::calc_distance2d;
  using autoware_utils::calc_lateral_deviation;
  using boost::geometry::return_centroid;

  const auto p = std::dynamic_pointer_cast<AvoidanceParameters>(avoidance_parameters_);

  const auto & path_points = data.reference_path.points;
  const auto & object_pose = object.kinematics.initial_pose_with_covariance.pose;
  const auto object_closest_index = findNearestIndex(path_points, object_pose.position);
  const auto object_closest_pose = path_points.at(object_closest_index).point.pose;
  const auto t = utils::getHighestProbLabel(object.classification);
  const auto & object_parameter = avoidance_parameters_->object_parameters.at(t);

  ObjectData object_data{};
  // Calc lateral deviation from path to target object.
  object_data.to_centerline =
    lanelet::utils::getArcCoordinates(data.current_lanelets, object_pose).distance;

  // NERO_ABLC_CENTRE=<unknown_still_s>,<vehicle_still_s>,<queue_m> (mainlane 2026-11-11): upstream drops every object within
  // th_offset_from_centerline (0.7 m) of the lane centre as traffic, so a wall / barrier filling the lane (11-03 41680) or a
  // vehicle standing mid-lane is never a reason to change lanes. Keep such an object when it is a still UNKNOWN (>= the first
  // value), or a car / truck / bus / trailer that has stood >= the second value with no vehicle / UNKNOWN standing or crawling
  // (< 1 m/s) ahead of it in the lane within queue_m (a queue is traffic, not an obstacle). Decided below once its stop time is
  // known. Unset = upstream.
  struct NeroCentre
  {
    bool on{false};
    double unknown_still_s{1.0};
    double vehicle_still_s{3.0};
    double queue_m{20.0};
  };
  static const NeroCentre nero_centre = [] {
    NeroCentre c;
    if (const char * e = std::getenv("NERO_ABLC_CENTRE")) {
      c.on = std::sscanf(e, "%lf,%lf,%lf", &c.unknown_still_s, &c.vehicle_still_s, &c.queue_m) >= 1;
    }
    return c;
  }();
  const bool nero_on_centre =
    std::abs(object_data.to_centerline) < avoidance_parameters_->threshold_distance_object_is_on_center;
  const auto nero_drop_centre = [&](const char * why) {
    ++nero_ablc_n_centre;
    if (nero_ablc_centre_ids.size() < 90) {
      char b[64];
      std::snprintf(
        b, sizeof(b), "%02x%02x@%.2f:%s ", object.object_id.uuid[0], object.object_id.uuid[1],
        object_data.to_centerline, why);
      nero_ablc_centre_ids += b;
    }
  };
  if (nero_on_centre && !nero_centre.on) {
    nero_drop_centre("centre");
    return std::nullopt;
  }

  object_data.object = object;

  const auto lower = p->lower_distance_for_polygon_expansion;
  const auto upper = p->upper_distance_for_polygon_expansion;
  const auto clamp =
    std::clamp(calc_distance2d(getEgoPose(), object_pose) - lower, 0.0, upper) / upper;
  object_data.distance_factor = object_parameter.max_expand_ratio * clamp + 1.0;

  // Calc envelop polygon.
  utils::static_obstacle_avoidance::fillObjectEnvelopePolygon(
    object_data, registered_objects_, object_closest_pose, p);

  // calc object centroid.
  object_data.centroid = return_centroid<Point2d>(object_data.envelope_poly);

  // Calc moving time.
  utils::static_obstacle_avoidance::fillObjectMovingTime(object_data, stopped_objects_, p);

  if (nero_on_centre) {  // only reached with NERO_ABLC_CENTRE set
    const bool is_unknown = t == ObjectClassification::UNKNOWN;
    const bool is_vehicle = t == ObjectClassification::CAR || t == ObjectClassification::TRUCK ||
                            t == ObjectClassification::BUS || t == ObjectClassification::TRAILER;
    if (!is_unknown && !is_vehicle) {
      nero_drop_centre("class");
      return std::nullopt;
    }
    if (object_data.stop_time < (is_unknown ? nero_centre.unknown_still_s : nero_centre.vehicle_still_s)) {
      nero_drop_centre("moving");
      return std::nullopt;
    }
    if (is_vehicle) {
      const double s_obj = lanelet::utils::getArcCoordinates(data.current_lanelets, object_pose).length;
      for (const auto & o : planner_data_->dynamic_object->objects) {
        if (o.object_id == object.object_id) continue;
        const auto lo = utils::getHighestProbLabel(o.classification);
        if (lo == ObjectClassification::PEDESTRIAN || lo == ObjectClassification::BICYCLE) continue;
        const auto & tw = o.kinematics.initial_twist_with_covariance.twist.linear;
        if (std::hypot(tw.x, tw.y) >= 1.0) continue;
        const auto & op = o.kinematics.initial_pose_with_covariance.pose;
        if (std::hypot(op.position.x - object_pose.position.x, op.position.y - object_pose.position.y) > nero_centre.queue_m + 10.0) {
          continue;
        }
        const auto ac = lanelet::utils::getArcCoordinates(data.current_lanelets, op);
        if (std::abs(ac.distance) < 1.5 && ac.length > s_obj && ac.length - s_obj <= nero_centre.queue_m) {
          nero_drop_centre("queue");
          return std::nullopt;
        }
      }
    }
  }

  // NERO_ABLC_STILL=<unknown_s>,<vehicle_s> (ablcstill_1117, 10amimprovement 11-17): an object off the lane centre is a target of
  // avoidance by lane change only once it has stood still that long (the centre rule NERO_ABLC_CENTRE already asks it of centre objects).
  // 8 Nov 19:12:13.4: a CAR crossing and driving off 34-41 m ahead (2.8 -> 4.8 m/s) was taken as a target and a 0.5 s candidate (gap-ahead's
  // short prepare) was approved: a lane change LEFT away from the route's right lane. Standing objects (parked cars, walls) are unchanged.
  // Unset = as before.
  struct NeroStill
  {
    bool on{false};
    double unknown_s{1.0};
    double vehicle_s{3.0};
  };
  static const NeroStill nero_still = [] {
    NeroStill c;
    if (const char * e = std::getenv("NERO_ABLC_STILL")) {
      c.on = std::sscanf(e, "%lf,%lf", &c.unknown_s, &c.vehicle_s) >= 1;
      if (c.on) {
        RCLCPP_INFO(
          rclcpp::get_logger("avoidance_by_lane_change"),
          "[NERO_ABLC_STILL] on: an off-centre target must have stood still %.1f s (UNKNOWN) / %.1f s (other)", c.unknown_s,
          c.vehicle_s);
      }
    }
    return c;
  }();
  // NERO_ABLC_MOVER=<t_move s> (ablcmover_1117, 10amimprovement 11-17, the reworked ablcstill_1117): an object off the lane centre is NOT a target
  // of avoidance by lane change while it has been moving continuously for >= t_move (move_time: one fast sample - a parked car's tracker speed
  // blip - restarts NERO_ABLC_STILL's stand clock (window c25 17:46: an 11.2 s stand behind a standing car) but leaves move_time ~0). 8 Nov
  // 19:12:13: the car crossing and driving off ahead is dropped. Unset = as before.
  static const double nero_mover_t = [] {
    double t = -1.0;
    if (const char * e = std::getenv("NERO_ABLC_MOVER")) {
      if (std::sscanf(e, "%lf", &t) == 1 && t > 0.0) {
        RCLCPP_INFO(
          rclcpp::get_logger("avoidance_by_lane_change"), "[NERO_ABLC_MOVER] on: an off-centre object moving >= %.2f s is not a target", t);
      } else {
        t = -1.0;
      }
    }
    return t;
  }();
  if (nero_mover_t > 0.0 && !nero_on_centre && object_data.move_time >= nero_mover_t) {
    nero_drop_centre("mover");
    return std::nullopt;
  }
  if (nero_still.on && !nero_on_centre) {
    const double need = t == ObjectClassification::UNKNOWN ? nero_still.unknown_s : nero_still.vehicle_s;
    if (object_data.stop_time < need) {
      nero_drop_centre("still");
      return std::nullopt;
    }
  }

  object_data.direction = calc_lateral_deviation(object_closest_pose, object_pose.position) > 0.0
                            ? Direction::LEFT
                            : Direction::RIGHT;

  // Find the footprint point closest to the path, set to object_data.overhang_distance.
  object_data.overhang_points = utils::static_obstacle_avoidance::calcEnvelopeOverhangDistance(
    object_data, data.reference_path);

  // Check whether the the ego should avoid the object.
  const auto & vehicle_width = planner_data_->parameters.vehicle_width;
  utils::static_obstacle_avoidance::fillAvoidanceNecessity(
    object_data, registered_objects_, vehicle_width, p);
  // NERO_ABLC_CENTRE: a kept centre object blocks the lane; the necessity test takes |overhang| for objects on the right,
  // i.e. it reads an object reaching far over ego's line as "no need" (it assumes objects never cross the path)
  if (nero_on_centre) {
    object_data.avoid_required = true;
  }

  // NERO_ABLC_CROSS=<margin_m>,<over_m> (mainlane 2026-11-11): change lanes only for an object that cannot be passed inside
  // the lane: avoiding it with the robot <margin_m> clear of its envelope would put the body more than <over_m> over the
  // lane's line on the avoidance side (static avoidance's in-lane nudge stays the car's for everything else; avoidance by
  // lane change was dead until NERO_ABLC_LANES, so without this every small nudge on a multi-lane road would become a
  // full lane change). A kept centre object always qualifies. Unset = every avoid_required object (upstream).
  struct NeroCross
  {
    bool on{false};
    double margin{0.3};
    double over{0.0};
  };
  static const NeroCross nero_cross = [] {
    NeroCross c;
    if (const char * e = std::getenv("NERO_ABLC_CROSS")) {
      c.on = std::sscanf(e, "%lf,%lf", &c.margin, &c.over) >= 1;
    }
    return c;
  }();
  if (nero_cross.on && !nero_on_centre && object_data.avoid_required && !object_data.overhang_points.empty()) {
    const bool on_right = utils::static_obstacle_avoidance::isOnRight(object_data);
    const double half_w = 0.5 * vehicle_width;
    const double overhang = object_data.overhang_points.front().first;
    // the robot's centre after the shift (left +), with <margin> between the body and the envelope
    const double shift = on_right ? overhang + nero_cross.margin + half_w : overhang - nero_cross.margin - half_w;
    lanelet::ConstLanelet lane;
    double to_line = std::numeric_limits<double>::max();
    if (lanelet::utils::query::getClosestLanelet(data.current_lanelets, object_closest_pose, &lane)) {
      const lanelet::BasicPoint2d p(object_closest_pose.position.x, object_closest_pose.position.y);
      to_line = on_right ? boost::geometry::distance(lane.leftBound2d().basicLineString(), p)
                         : boost::geometry::distance(lane.rightBound2d().basicLineString(), p);
    }
    const double body_over = std::abs(shift) + half_w - to_line;  // > 0: over the line
    if ((on_right ? shift > 0.0 : shift < 0.0) && body_over <= nero_cross.over) {
      char b[48];
      std::snprintf(b, sizeof(b), "inlane %.2f", body_over);
      nero_drop_centre(b);
      return std::nullopt;
    }
  }

  utils::static_obstacle_avoidance::fillLongitudinalAndLengthByClosestEnvelopeFootprint(
    data.reference_path_rough, getEgoPosition(), object_data);
  return object_data;
}

const ObjectData & AvoidanceByLaneChange::nero_nearest_required() const
{
  // the nearest avoid_required target (targets are sorted by longitudinal distance); NERO_ABLC_LANES unset or none required:
  // the nearest target, as upstream
  static const bool on = [] {
    const char * v = std::getenv("NERO_ABLC_LANES");
    return v != nullptr && std::atoi(v) > 0;
  }();
  const auto & objs = avoidance_data_.target_objects;
  if (nero_need_idx_ >= 0 && nero_need_idx_ < static_cast<int>(objs.size())) {
    return objs.at(nero_need_idx_);  // NERO_ABLC_NEED: the target the change is for (the wait stop is placed before it)
  }
  if (on) {
    // ahead of ego only: a piece already beside / behind the robot (longitudinal <= 0) made every request "nearer than the
    // minimum lengths" (field logsim w1108_1910: nearest -6 .. -30 m)
    for (const auto & o : objs) {
      if (o.avoid_required && o.longitudinal > 0.0) return o;
    }
    for (const auto & o : objs) {
      if (o.avoid_required) return o;
    }
  }
  return objs.front();
}

void AvoidanceByLaneChange::nero_update_need()
{
  nero_need_idx_ = -1;
  nero_need_late_ = false;
  nero_need_why_.clear();
  const auto & cfg = nero_need_cfg();
  if (!cfg.on) return;
  const auto & objs = avoidance_data_.target_objects;
  const auto & lanes = avoidance_data_.current_lanelets;
  if (objs.empty() || lanes.empty()) {
    nero_need_why_ = objs.empty() ? "no target" : "no lanes";
    return;
  }
  // the own lane's bounds along the current lanes
  std::vector<lanelet::BasicPoint2d> lb, rb;
  const auto append = [](std::vector<lanelet::BasicPoint2d> & out, const auto & ls) {
    for (const auto & q : ls) {
      const lanelet::BasicPoint2d p(q.x(), q.y());
      if (out.empty() || (out.back() - p).norm() > 1e-3) out.push_back(p);
    }
  };
  for (const auto & l : lanes) {
    append(lb, l.leftBound2d().basicLineString());
    append(rb, l.rightBound2d().basicLineString());
  }
  const double w = planner_data_->parameters.vehicle_width;
  const double fit_need = w + cfg.margin - cfg.max_dev;  // free width static avoidance needs to stay in the lane
  const double line_need = w + cfg.margin;               // ... to keep the whole body inside the line
  int blocked = -1, row_first = -1, chain_first = -1, chain_n = 0;
  double chain_end = 0.0;
  std::string list;
  int listed = 0;
  for (size_t i = 0; i < objs.size(); ++i) {
    const auto & o = objs.at(i);
    if (!o.avoid_required || o.longitudinal <= 0.0) continue;
    const auto poly = autoware_utils::to_polygon2d(o.object);
    double max_l = -std::numeric_limits<double>::max(), min_r = std::numeric_limits<double>::max();
    for (const auto & q : poly.outer()) {
      const lanelet::BasicPoint2d p(q.x(), q.y());
      max_l = std::max(max_l, nero_signed_dist(lb, p));  // < 0: right of the left bound (inside)
      min_r = std::min(min_r, nero_signed_dist(rb, p));  // > 0: left of the right bound (inside)
    }
    const double free = std::max(-max_l, min_r);
    if (listed < 6) {
      char b[72];
      std::snprintf(
        b, sizeof(b), " %02x%02x@%.1f:free %.2f", o.object.object_id.uuid[0], o.object.object_id.uuid[1], o.longitudinal,
        free);
      list += b;
      ++listed;
    }
    if (blocked < 0 && free < fit_need) blocked = static_cast<int>(i);
    if (row_first < 0 && free < line_need) {
      if (chain_n > 0 && o.longitudinal - chain_end < cfg.row_gap) {
        ++chain_n;
      } else {
        chain_first = static_cast<int>(i);
        chain_n = 1;
      }
      chain_end = o.longitudinal + o.length;
      if (chain_n >= cfg.row_n) row_first = chain_first;
    }
  }
  int first = blocked;
  const char * kind = "blocked";
  if (row_first >= 0 && (first < 0 || objs.at(row_first).longitudinal < objs.at(first).longitudinal)) {
    first = row_first;
    kind = "row";
  }
  char head[160];
  if (first < 0) {
    std::snprintf(head, sizeof(head), "fits in the lane (need %.2f, line %.2f) |", fit_need, line_need);
    nero_need_why_ = head + list;
    return;
  }
  const double start = std::max(cfg.start_m, std::max(0.0, getEgoVelocity()) * cfg.start_s);
  const auto & f = objs.at(first);
  if (f.longitudinal > start) {
    nero_need_late_ = true;
    std::snprintf(head, sizeof(head), "%s %02x%02x at %.1f m > start %.1f m (need %.2f) |", kind,
      f.object.object_id.uuid[0], f.object.object_id.uuid[1], f.longitudinal, start, fit_need);
    nero_need_why_ = head + list;
    return;
  }
  nero_need_idx_ = first;
  std::snprintf(head, sizeof(head), "%s %02x%02x at %.1f m <= start %.1f m (need %.2f) |", kind,
    f.object.object_id.uuid[0], f.object.object_id.uuid[1], f.longitudinal, start, fit_need);
  nero_need_why_ = head + list;
}

std::vector<std::string> AvoidanceByLaneChange::nero_claim_ids() const
{
  std::vector<std::string> ids;
  for (const auto & o : avoidance_data_.target_objects) {
    if (o.avoid_required) {
      ids.push_back(autoware_utils::to_hex_string(o.object.object_id));
    }
  }
  return ids;
}

double AvoidanceByLaneChange::calcMinAvoidanceLength(const ObjectData & nearest_object) const
{
  const auto ego_width = getCommonParam().vehicle_width;
  const auto nearest_object_type = utils::getHighestProbLabel(nearest_object.object.classification);
  const auto nearest_object_parameter =
    avoidance_parameters_->object_parameters.at(nearest_object_type);
  const auto lateral_hard_margin = std::max(
    nearest_object_parameter.lateral_hard_margin,
    nearest_object_parameter.lateral_hard_margin_for_parked_vehicle);
  const auto avoid_margin = lateral_hard_margin * nearest_object.distance_factor +
                            nearest_object_parameter.lateral_soft_margin + 0.5 * ego_width;

  avoidance_helper_->setData(planner_data_);
  const auto shift_length = avoidance_helper_->getShiftLength(
    nearest_object, utils::static_obstacle_avoidance::isOnRight(nearest_object), avoid_margin);

  return avoidance_helper_->getMinAvoidanceDistance(shift_length);
}

void AvoidanceByLaneChange::insert_stop_point(
  const lanelet::ConstLanelets & lanelets, PathWithLaneId & path, const bool is_waiting_approval)
{
  // NERO_ABLC_WAIT=<buffer_m> (mainlane 2026-11-11): the base class inserts no stop at all while ego is in its preferred
  // lane (getNumLaneToPreferredLane == 0), i.e. always for this module: while it waited for a gap (a car beside / behind in
  // the target lane) the robot rolled up to obstacle_stop's stop 5 m before the object, from where the module can never
  // request again (object nearer than one minimum lane change): stuck for good (sim lc05 drv_yield 1111-0808, the field's
  // 41680 wall stands). Wait where the change stays possible: nearest target - max(min lane change, min avoid) - buffer.
  // An object first seen nearer than that gets no stop from here (as before). Unset = upstream.
  static const double nero_wait = [] {
    const char * v = std::getenv("NERO_ABLC_WAIT");
    return v != nullptr ? std::atof(v) : -1.0;
  }();
  NormalLaneChange::insert_stop_point(lanelets, path, is_waiting_approval);
  if (nero_wait < 0.0 || !is_waiting_approval || avoidance_data_.target_objects.empty() || path.points.empty()) {
    return;
  }
  if (getStopPose().has_value()) {
    static std::chrono::steady_clock::time_point nero_base_last;
    static const bool nero_base_dbg = std::getenv("NERO_ABLC_DEBUG") != nullptr;
    if (nero_base_dbg && std::chrono::steady_clock::now() - nero_base_last > std::chrono::milliseconds(500)) {
      nero_base_last = std::chrono::steady_clock::now();
      RCLCPP_INFO(logger_, "[NERO_ABLC_WAIT] base stop kept (%s)", getStopPose()->detail.c_str());
    }
    return;  // the base class placed a stop (a route change's terminal / a lead object): keep it
  }
  const auto & o = nero_nearest_required();
  const double need = std::max(calc_minimum_dist_buffer(), calcMinAvoidanceLength(o)) + nero_wait;
  const double d_stop = o.longitudinal - need;
  if (d_stop <= 0.0) {
    return;
  }
  // only a stop the robot can make comfortably (<= 1.0 m/s2 after 0.5 s): the work-zone block at 41680 was perceived ~22 m
  // ahead at 5 m/s and this stop came 4-7 m ahead = a hard brake (field logsim w1108_1747). Too close: no stop from here; the
  // request ends at the minimum lengths and static avoidance takes the object as the car does today.
  {
    const double v = std::max(0.0, getEgoVelocity());
    const double comfort = v * 0.5 + v * v / 2.0;
    if (d_stop < comfort) {
      static std::chrono::steady_clock::time_point nero_c_last;
      static const bool nero_c_dbg = std::getenv("NERO_ABLC_DEBUG") != nullptr;
      if (nero_c_dbg && std::chrono::steady_clock::now() - nero_c_last > std::chrono::milliseconds(500)) {
        nero_c_last = std::chrono::steady_clock::now();
        RCLCPP_INFO(
          logger_, "[NERO_ABLC_WAIT] no stop: %.1f m ahead < %.1f m comfortable at %.2f m/s", d_stop, comfort, v);
      }
      return;
    }
  }
  const double ego_arc =
    autoware::motion_utils::calcSignedArcLength(path.points, path.points.front().point.pose.position, getEgoPosition());
  set_stop_pose(ego_arc + d_stop, path, "avoidance lane change: wait for a gap");
  static std::chrono::steady_clock::time_point nero_wait_last;
  static const bool nero_wait_dbg = std::getenv("NERO_ABLC_DEBUG") != nullptr;
  if (nero_wait_dbg && std::chrono::steady_clock::now() - nero_wait_last > std::chrono::milliseconds(500)) {
    nero_wait_last = std::chrono::steady_clock::now();
    RCLCPP_INFO(
      logger_, "[NERO_ABLC_WAIT] stop %.1f m ahead (object %.1f m - need %.1f m)", d_stop, o.longitudinal, need);
  }
}

double AvoidanceByLaneChange::calc_minimum_dist_buffer() const
{
  const auto [_, dist_buffer] = utils::lane_change::calculation::calc_lc_length_and_dist_buffer(
    common_data_ptr_, get_current_lanes());
  return dist_buffer.min;
}

double AvoidanceByLaneChange::calcLateralOffset() const
{
  auto additional_lat_offset{0.0};
  for (const auto & [type, p] : avoidance_parameters_->object_parameters) {
    const auto lateral_hard_margin =
      std::max(p.lateral_hard_margin, p.lateral_hard_margin_for_parked_vehicle);
    const auto offset =
      2.0 * p.envelope_buffer_margin + lateral_hard_margin + p.lateral_soft_margin;
    additional_lat_offset = std::max(additional_lat_offset, offset);
  }
  return additional_lat_offset;
}
}  // namespace autoware::behavior_path_planner
