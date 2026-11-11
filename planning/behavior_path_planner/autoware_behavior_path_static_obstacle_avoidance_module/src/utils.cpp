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

#include "autoware/behavior_path_planner_common/utils/utils.hpp"
#include "autoware/behavior_path_planner_common/utils/nero_geom_fast.hpp"  // NERO_BPP_GEOM_FAST (trjrate-2)
#include "autoware/behavior_path_planner_common/utils/nero_mu.hpp"  // NERO_BPP_PROJ_FAST

#include "autoware/behavior_path_planner_common/utils/drivable_area_expansion/static_drivable_area.hpp"
#include "autoware/behavior_path_planner_common/utils/path_safety_checker/objects_filtering.hpp"
#include "autoware/behavior_path_planner_common/utils/path_utils.hpp"
#include "autoware/behavior_path_planner_common/utils/traffic_light_utils.hpp"
#include "autoware/behavior_path_static_obstacle_avoidance_module/data_structs.hpp"
#include "autoware/behavior_path_static_obstacle_avoidance_module/utils.hpp"

#include <mutex>
#include <Eigen/Dense>
#include <autoware_lanelet2_extension/regulatory_elements/road_marking.hpp>
#include <autoware_lanelet2_extension/utility/message_conversion.hpp>
#include <autoware_lanelet2_extension/utility/query.hpp>
#include <autoware_lanelet2_extension/utility/utilities.hpp>
#include <magic_enum.hpp>

#include <boost/geometry/algorithms/buffer.hpp>
#include <boost/geometry/algorithms/convex_hull.hpp>
#include <boost/geometry/algorithms/correct.hpp>
#include <boost/geometry/algorithms/distance.hpp>
#include <boost/geometry/algorithms/union.hpp>
#include <boost/geometry/geometries/point_xy.hpp>

#include <lanelet2_routing/RoutingGraphContainer.h>
#include <tf2/utils.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace autoware::behavior_path_planner::utils::static_obstacle_avoidance
{

using autoware::behavior_path_planner::utils::traffic_light::calcDistanceToRedTrafficLight;
using autoware::behavior_path_planner::utils::traffic_light::getDistanceToNextTrafficLight;
using autoware_perception_msgs::msg::TrafficLightElement;

namespace
{
geometry_msgs::msg::Point32 createPoint32(const double x, const double y, const double z)
{
  geometry_msgs::msg::Point32 p;
  p.x = x;
  p.y = y;
  p.z = z;
  return p;
}

geometry_msgs::msg::Polygon toMsg(const autoware_utils::Polygon2d & polygon, const double z)
{
  geometry_msgs::msg::Polygon ret;
  for (const auto & p : polygon.outer()) {
    ret.points.push_back(createPoint32(p.x(), p.y(), z));
  }
  return ret;
}

geometry_msgs::msg::Polygon createVehiclePolygon(
  const autoware::vehicle_info_utils::VehicleInfo & vehicle_info, const double offset)
{
  const auto & i = vehicle_info;
  const auto & front_m = i.max_longitudinal_offset_m;
  const auto & width_m = i.vehicle_width_m / 2.0 + offset;
  const auto & back_m = i.rear_overhang_m;

  geometry_msgs::msg::Polygon polygon{};

  polygon.points.push_back(createPoint32(front_m, -width_m, 0.0));
  polygon.points.push_back(createPoint32(front_m, width_m, 0.0));
  polygon.points.push_back(createPoint32(-back_m, width_m, 0.0));
  polygon.points.push_back(createPoint32(-back_m, -width_m, 0.0));

  return polygon;
}

Polygon2d createOneStepPolygon(
  const geometry_msgs::msg::Pose & p1, const geometry_msgs::msg::Pose & p2,
  const geometry_msgs::msg::Pose & p3, const geometry_msgs::msg::Pose & p4,
  const geometry_msgs::msg::Polygon & base_polygon)
{
  Polygon2d one_step_polygon{};

  {
    geometry_msgs::msg::Polygon out_polygon{};
    geometry_msgs::msg::TransformStamped geometry_tf{};
    geometry_tf.transform = pose2transform(p1);
    tf2::doTransform(base_polygon, out_polygon, geometry_tf);

    for (const auto & p : out_polygon.points) {
      one_step_polygon.outer().push_back(Point2d(p.x, p.y));
    }
  }

  {
    geometry_msgs::msg::Polygon out_polygon{};
    geometry_msgs::msg::TransformStamped geometry_tf{};
    geometry_tf.transform = pose2transform(p2);
    tf2::doTransform(base_polygon, out_polygon, geometry_tf);

    for (const auto & p : out_polygon.points) {
      one_step_polygon.outer().push_back(Point2d(p.x, p.y));
    }
  }

  {
    geometry_msgs::msg::Polygon out_polygon{};
    geometry_msgs::msg::TransformStamped geometry_tf{};
    geometry_tf.transform = pose2transform(p3);
    tf2::doTransform(base_polygon, out_polygon, geometry_tf);

    for (const auto & p : out_polygon.points) {
      one_step_polygon.outer().push_back(Point2d(p.x, p.y));
    }
  }

  {
    geometry_msgs::msg::Polygon out_polygon{};
    geometry_msgs::msg::TransformStamped geometry_tf{};
    geometry_tf.transform = pose2transform(p4);
    tf2::doTransform(base_polygon, out_polygon, geometry_tf);

    for (const auto & p : out_polygon.points) {
      one_step_polygon.outer().push_back(Point2d(p.x, p.y));
    }
  }

  Polygon2d hull_polygon{};
  boost::geometry::convex_hull(one_step_polygon, hull_polygon);
  boost::geometry::correct(hull_polygon);

  return hull_polygon;
}

bool isEndPointsConnected(
  const lanelet::ConstLanelet & left_lane, const lanelet::ConstLanelet & right_lane)
{
  const auto & left_back_point_2d = right_lane.leftBound2d().back().basicPoint();
  const auto & right_back_point_2d = left_lane.rightBound2d().back().basicPoint();

  constexpr double epsilon = 1e-5;
  return (right_back_point_2d - left_back_point_2d).norm() < epsilon;
}

template <typename T>
void pushUniqueVector(T & base_vector, const T & additional_vector)
{
  base_vector.insert(base_vector.end(), additional_vector.begin(), additional_vector.end());
}

}  // namespace

// ---- nero (2026-10-22, overtake job): the car decides the overtake of a stopped car itself (no operator approval,
// avoidance_for_ambiguous_vehicle.policy "auto"); these switches give it the judgement the operator gave.
// Every switch is read once from the environment; unset = the module as it ran on 10-21.
//  NERO_SOA_AMBIGUOUS_IN_LANE=1     a stopped car on ego's lane, on the side of the oncoming lane, is an ambiguous
//                                   vehicle (stopped-time / moved-distance / wait-and-see conditions) instead of
//                                   "never avoid" (IS_NOT_PARKING_OBJECT): Al Falah's two-way streets have no
//                                   shoulder, cars stop anywhere across the lane (capture #8)
//  NERO_SOA_AMBIGUOUS_STOP_LINE_M=d a stopped car that is not parked, on ego's lane, less than d m before a stop line
//                                   (stop sign / stop-line marking) or before the next junction lanelet is waiting
//                                   there: never avoided (the stock ignore_area only knows traffic lights and
//                                   crosswalks; Al Falah has 880 stop signs and no traffic light)
//  NERO_SOA_AMBIGUOUS_QUEUE_M=q     ... less than q m behind another vehicle in ego's lane that is stopped or slower
//                                   than 2 m/s is in a queue: never avoided
//  NERO_SOA_SIDE_BY_SPACE=1         a stopped vehicle target on ego's lane is passed on the side that has room: when
//                                   the side away from its centroid has no avoid margin (INSUFFICIENT_DRIVABLE_SPACE),
//                                   the other side is tried (stock: always the side away from the centroid; a car
//                                   stopped 0.45 m left of centre could only be passed on the right, into the parked
//                                   cars - capture #8); also for still UNKNOWN targets (debris, a car's rear fragment)
//  NERO_SOA_UNKNOWN_STABLE_S=s      an UNKNOWN counts as standing still once UNKNOWN objects have been seen at its
//                                   place (within 1 m) for s seconds, whatever their track ids (stock: the hard-coded
//                                   3 s of stop time per track id; the tracker re-created the nose clusters of
//                                   nose-in parked cars every 0.5-1.5 s, so avoidance never planned around them while
//                                   obstacle_stop stopped for them - capture #1)
namespace nero_filter
{
struct Env
{
  bool ambiguous_in_lane{false};
  double stop_line_m{-1.0};
  double queue_m{-1.0};
  bool side_by_space{false};
  double unknown_stable_s{-1.0};
};

const Env & env()
{
  static const Env e = [] {
    Env r;
    if (const char * v = std::getenv("NERO_SOA_AMBIGUOUS_IN_LANE"); v) r.ambiguous_in_lane = std::atoi(v) > 0;
    if (const char * v = std::getenv("NERO_SOA_AMBIGUOUS_STOP_LINE_M"); v) r.stop_line_m = std::atof(v);
    if (const char * v = std::getenv("NERO_SOA_AMBIGUOUS_QUEUE_M"); v) r.queue_m = std::atof(v);
    if (const char * v = std::getenv("NERO_SOA_SIDE_BY_SPACE"); v) r.side_by_space = std::atoi(v) > 0;
    if (const char * v = std::getenv("NERO_SOA_UNKNOWN_STABLE_S"); v) r.unknown_stable_s = std::atof(v);
    std::fprintf(
      stderr,
      "[NERO_SOA] ambiguous_in_lane=%d ambiguous_stop_line_m=%.1f ambiguous_queue_m=%.1f side_by_space=%d "
      "unknown_stable_s=%.1f\n",
      r.ambiguous_in_lane, r.stop_line_m, r.queue_m, r.side_by_space, r.unknown_stable_s);
    return r;
  }();
  return e;
}

// UNKNOWN objects seen at (about) the same place: true once that place has been occupied for `stable_s` without a gap
// longer than 1 s, whatever the track ids
bool isStableByPosition(const geometry_msgs::msg::Point & p, const double now, const double stable_s)
{
  struct Spot
  {
    double x, y, first, last;
  };
  static std::vector<Spot> spots;
  spots.erase(
    std::remove_if(spots.begin(), spots.end(), [&](const Spot & s) { return now - s.last > 5.0; }),
    spots.end());
  for (auto & s : spots) {
    if (std::hypot(s.x - p.x, s.y - p.y) > 1.0) continue;
    if (now - s.last > 1.0) s.first = now;
    s.last = now;
    s.x = 0.8 * s.x + 0.2 * p.x;
    s.y = 0.8 * s.y + 0.2 * p.y;
    return now - s.first >= stable_s;
  }
  spots.push_back(Spot{p.x, p.y, now, now});
  return false;
}

// stop lines (stop signs' ref lines, stop-line markings) of a lanelet
std::vector<lanelet::ConstLineString3d> stopLines(const lanelet::ConstLanelet & llt)
{
  std::vector<lanelet::ConstLineString3d> out;
  for (const auto & ts : llt.regulatoryElementsAs<lanelet::TrafficSign>()) {
    for (const auto & l : ts->refLines()) out.push_back(l);
  }
  for (const auto & rm : llt.regulatoryElementsAs<lanelet::autoware::RoadMarking>()) {
    const auto l = rm->roadMarking();
    if (l.attributeOr(lanelet::AttributeName::Type, "") == std::string("stop_line")) out.push_back(l);
  }
  return out;
}

// arc length from the object along the route lanelets to the next stop line or junction lanelet (inf: none)
double distanceToStopLineOrJunction(const ObjectData & object, const lanelet::ConstLanelets & lanelets)
{
  lanelet::ConstLanelet current;
  if (!nero_geom_fast::getClosestLanelet(lanelets, object.getPose(), &current)) {
    return std::numeric_limits<double>::infinity();
  }
  const auto centerline = lanelet::utils::to2D(current.centerline());
  const auto p = lanelet::utils::to2D(lanelet::utils::conversion::toLaneletPoint(object.getPose().position));
  const double s_obj = lanelet::geometry::toArcCoordinates(centerline, p.basicPoint()).length;
  for (const auto & l : stopLines(current)) {
    const double s_line =
      lanelet::geometry::toArcCoordinates(centerline, lanelet::utils::to2D(l).front().basicPoint()).length;
    if (s_line > s_obj) return s_line - s_obj;
  }
  double dist = lanelet::utils::getLaneletLength2d(current) - s_obj;
  bool found = false;
  for (const auto & llt : lanelets) {
    if (llt.id() == current.id()) {
      found = true;
      continue;
    }
    if (!found) continue;
    if (llt.hasAttribute("turn_direction")) return dist;
    const auto c = lanelet::utils::to2D(llt.centerline());
    for (const auto & l : stopLines(llt)) {
      return dist + lanelet::geometry::toArcCoordinates(c, lanelet::utils::to2D(l).front().basicPoint()).length;
    }
    dist += lanelet::utils::getLaneletLength2d(llt);
  }
  return std::numeric_limits<double>::infinity();
}

// another vehicle stands (or creeps) in front of the object, in ego's lane, closer than queue_m
bool isQueued(
  const ObjectData & object, const AvoidancePlanningData & data,
  const std::shared_ptr<const PlannerData> & planner_data, const double queue_m)
{
  using autoware_perception_msgs::msg::ObjectClassification;
  const auto & path = data.reference_path_rough.points;
  if (path.size() < 2 || !planner_data->dynamic_object) return false;
  const auto & me = object.getPosition();
  const double my_len = object.object.shape.dimensions.x;
  for (const auto & o : planner_data->dynamic_object->objects) {
    if (o.object_id == object.object.object_id) continue;
    const auto label = utils::getHighestProbLabel(o.classification);
    if (
      label != ObjectClassification::CAR && label != ObjectClassification::TRUCK &&
      label != ObjectClassification::BUS && label != ObjectClassification::TRAILER &&
      label != ObjectClassification::MOTORCYCLE) {
      continue;
    }
    if (std::hypot(
          o.kinematics.initial_twist_with_covariance.twist.linear.x,
          o.kinematics.initial_twist_with_covariance.twist.linear.y) > 2.0) {
      continue;
    }
    const auto & op = o.kinematics.initial_pose_with_covariance.pose.position;
    const double ds = nero_mu::calcSignedArcLength(path, me, op);
    const double gap = ds - 0.5 * my_len - 0.5 * o.shape.dimensions.x;
    if (ds <= 0.0 || gap > queue_m) continue;
    if (std::abs(nero_mu::calcLateralOffset(path, op)) > 1.5) continue;
    return true;
  }
  return false;
}
}  // namespace nero_filter

// ---- nero (2026-10-30, retaa job, QUEUE RETAA): a standing vehicle first reported as an UNKNOWN fragment (one end of
// the car) and later as a vehicle went down the ambiguous-vehicle tests (avoidance_for_ambiguous_vehicle.condition
// th_stopped_time / th_moving_distance) with the fragment's history broken: on the same track id the moved distance was
// measured from the FRAGMENT's first place, 1.9-2.4 m from the car's centre = "moved", never a target while it stood
// (field 10-29 17:06 #4, passed at 0.25-0.29 m at 13 km/h); at a new track id the stop clock restarted = 3 s of "not a
// target" with the robot coming up beside it (sim 0.17 m).
//  NERO_SOA_AMBIGUOUS_FRAGMENT=m,g  a vehicle that stands (below its moving speed) where an UNKNOWN stood keeps that
//                                   UNKNOWN's standing history when it becomes a vehicle: the UNKNOWN seen <= g s ago,
//                                   its last place inside the vehicle's box + m, and not creeping (its mean velocity
//                                   over th_stopped_time below 0.2 m/s and th_moving_distance / th_stopped_time). Then the stop
//                                   clock runs from when the UNKNOWN started to stand and the moved distance counts
//                                   from the vehicle's first place: on the same id (the registry entry of the fragment,
//                                   or a new entry when the module instance was replaced) and for a NEW vehicle track
//                                   never seen before (the longest-standing UNKNOWN inside its box). The history is the
//                                   switch's own table, not the module instance's registry. A vehicle faster than its
//                                   moving speed is unchanged (stop time 0); one that creeps > th_moving_distance from
//                                   its first place is "moved" as before. Log: [NERO_SOA_FRAG] at each carry.
//                                   ,t (3rd field, default 0.1 m): for such a vehicle the shift line generator's
//                                   "avoidance path already approved" shortcut (any approved shift on the object's side
//                                   = avoidable with the full desired shift, no feasibility check) holds only when the
//                                   approved shift at the object keeps the hard margin + t (the robot's body hard +
//                                   envelope buffer + t from the box): the fragment's line (0.90 m) was taken as approved
//                                   for the car that needed 1.40 m (1.04 for the hard margin), the bigger line never
//                                   became comfortable and the robot passed on the fragment's line (sim 0.17 m). Else
//                                   the module's own checks decide (shift under the jerk limit, the best-effort shift
//                                   that keeps the hard margin, or its stop: scene.cpp stops before such a vehicle at
//                                   the module's stop line for it - or, when that is nearer than its max_deceleration
//                                   stop distance, at the -2.0 m/s2 / 2.0 m/s3 one (firm; a hard brake is <= -2.5) -
//                                   until a line that keeps the hard margin is registered (not for a vehicle with no
//                                   room: INSUFFICIENT_DRIVABLE_SPACE stays as stock), and kept while that vehicle
//                                   leaves the target list by coming toward the robot (> 0.2 m/s), then also obstacle_
//                                   stop's 5 m stop margin to its near end; a line in progress across the
//                                   new line's start hands over to it there at the path's actual shift, so the new line
//                                   is valid while the robot is part-way through the fragment's line).
//  NERO_SOA_AMBIGUOUS_DEBUG=1       logging only: one [NERO_SOA_AMB] line per vehicle and cycle on the vehicle path (stop
//                                   time, moved distance, parked, margin, verdict); [NERO_SOA_VALID] why a new shift
//                                   line is invalid
namespace nero_frag
{
struct Env
{
  bool on{false};
  double margin_m{0.5};
  double gap_s{1.0};
  double approved_tol_m{0.1};
  bool debug{false};
};

const Env & env()
{
  static const Env e = [] {
    Env r;
    if (const char * v = std::getenv("NERO_SOA_AMBIGUOUS_FRAGMENT"); v && *v) {
      r.on = std::sscanf(v, "%lf,%lf,%lf", &r.margin_m, &r.gap_s, &r.approved_tol_m) >= 1;
    }
    if (const char * v = std::getenv("NERO_SOA_AMBIGUOUS_DEBUG"); v) r.debug = std::atoi(v) > 0;
    std::fprintf(
      stderr, "[NERO_SOA] ambiguous_fragment=%d margin_m=%.2f gap_s=%.2f approved_tol_m=%.2f ambiguous_debug=%d\n",
      r.on, r.margin_m, r.gap_s, r.approved_tol_m, r.debug);
    return r;
  }();
  return e;
}

std::string hexId(const unique_identifier_msgs::msg::UUID & id)
{
  char s[9];
  std::snprintf(s, sizeof(s), "%02x%02x%02x%02x", id.uuid[0], id.uuid[1], id.uuid[2], id.uuid[3]);
  return s;
}

// what the module saw of each track id in its target area: its last place, and while it was an UNKNOWN its last place
// and its mean velocity
struct Track
{
  double t{0.0};
  double ux{0.0}, uy{0.0}, ut{-1.0};  // last place / time as an UNKNOWN
  double uvx{0.0}, uvy{0.0};          // velocity as an UNKNOWN in the map frame, averaged over ~th_stopped_time
  double ustand_t{-1.0};              // the UNKNOWN stands (below its moving speed) since then (-1: moving)
  double ufast_t{-1.0};               // the UNKNOWN is faster than its moving speed since then (-1: not)
  bool carried{false};                // a vehicle that carries an UNKNOWN's standing history
};
using Key = std::array<uint8_t, 16>;

std::map<Key, Track> & tracks()
{
  static std::map<Key, Track> m;
  return m;
}

// the fragment's last place lies inside the vehicle's box (+ margin)
bool insideBox(const PredictedObject & vehicle, const double x, const double y, const double margin)
{
  const auto & p = vehicle.kinematics.initial_pose_with_covariance.pose;
  const double yaw = tf2::getYaw(p.orientation);
  const double dx = x - p.position.x;
  const double dy = y - p.position.y;
  const double lon = std::cos(yaw) * dx + std::sin(yaw) * dy;
  const double lat = -std::sin(yaw) * dx + std::cos(yaw) * dy;
  if (vehicle.shape.type == autoware_perception_msgs::msg::Shape::BOUNDING_BOX) {
    return std::abs(lon) <= 0.5 * vehicle.shape.dimensions.x + margin &&
           std::abs(lat) <= 0.5 * vehicle.shape.dimensions.y + margin;
  }
  const auto poly = autoware_utils::to_polygon2d(vehicle);
  return boost::geometry::distance(autoware_utils::Point2d(x, y), poly) <= margin;
}

bool on()
{
  return env().on;
}

bool debug()
{
  return env().debug;
}

// a vehicle that carries an UNKNOWN fragment's standing history
bool carried(const ObjectData & o)
{
  if (!env().on) return false;
  Key k;
  std::copy(o.object.object_id.uuid.begin(), o.object.object_id.uuid.end(), k.begin());
  const auto it = tracks().find(k);
  return it != tracks().end() && it->second.carried;
}

// keep the "already approved" shortcut for this object? No for a carried vehicle whose approved shift at it does not
// keep the hard margin + approved_tol_m (or lies on the other side)
bool keepApprovedShortcut(
  const ObjectData & o, const double approved_shift, const double vehicle_width,
  const std::shared_ptr<AvoidanceParameters> & parameters)
{
  const auto & e = env();
  if (!e.on || o.overhang_points.empty()) return true;
  Key k;
  std::copy(o.object.object_id.uuid.begin(), o.object.object_id.uuid.end(), k.begin());
  const auto it = tracks().find(k);
  if (it == tracks().end() || !it->second.carried) return true;
  const auto & op = parameters->object_parameters.at(utils::getHighestProbLabel(o.object.classification));
  const double hard = o.is_parked ? op.lateral_hard_margin_for_parked_vehicle : op.lateral_hard_margin;
  const double need =
    calcShiftLength(isOnRight(o), o.overhang_points.front().first, 0.5 * vehicle_width + hard + e.approved_tol_m);
  if (approved_shift * need > 0.0 && std::abs(approved_shift) >= std::abs(need)) return true;
  static rclcpp::Clock steady_clock(RCL_STEADY_TIME);
  RCLCPP_INFO_THROTTLE(
    rclcpp::get_logger(logger_namespace), steady_clock, 1000,
    "[NERO_SOA_FRAG] %s: the approved shift at it %.2f m does not keep the hard margin (needs %.2f m): its "
    "feasibility is checked",
    hexId(o.object.object_id).c_str(), approved_shift, need);
  return false;
}

// update the track table with this cycle's object (after the history decision); an UNKNOWN stops standing only after
// moving_time of being faster than moving_v (as the module's registry)
void see(
  const ObjectData & o, const double now, const double avg_s, const double moving_v, const double moving_time)
{
  auto & m = tracks();
  Key k;
  std::copy(o.object.object_id.uuid.begin(), o.object.object_id.uuid.end(), k.begin());
  auto & tr = m[k];
  tr.t = now;  // first: an entry the history decision just made has t 0 and would be pruned below
  static double last_prune = 0.0;
  if (now - last_prune > 1.0) {  // ids unseen for 5 s, once a second (not per object)
    last_prune = now;
    for (auto it = m.begin(); it != m.end();) {
      it = now - it->second.t > 5.0 ? m.erase(it) : std::next(it);
    }
  }
  if (
    utils::getHighestProbLabel(o.object.classification) !=
    autoware_perception_msgs::msg::ObjectClassification::UNKNOWN) {
    return;
  }
  const auto & p = o.object.kinematics.initial_pose_with_covariance.pose;
  const auto & tw = o.object.kinematics.initial_twist_with_covariance.twist;
  const double yaw = tf2::getYaw(p.orientation);
  const double vx = std::cos(yaw) * tw.linear.x - std::sin(yaw) * tw.linear.y;
  const double vy = std::sin(yaw) * tw.linear.x + std::cos(yaw) * tw.linear.y;
  const double a = tr.ut < 0.0 ? 1.0 : std::clamp((now - tr.ut) / std::max(avg_s, 0.1), 0.0, 1.0);
  tr.uvx += a * (vx - tr.uvx);
  tr.uvy += a * (vy - tr.uvy);
  if (std::hypot(tw.linear.x, tw.linear.y) > moving_v) {
    if (tr.ufast_t < 0.0) tr.ufast_t = now;
    if (now - tr.ufast_t > moving_time) tr.ustand_t = -1.0;
  } else {
    tr.ufast_t = -1.0;
    if (tr.ustand_t < 0.0) tr.ustand_t = now;
  }
  tr.ux = p.position.x;
  tr.uy = p.position.y;
  tr.ut = now;
}
}  // namespace nero_frag

namespace filtering_utils
{
/**
 * @brief check whether the object is avoidance target object type.
 * @param object data.
 * @param parameters.
 * @return if the object is avoidance target object type, return true.
 */
bool isAvoidanceTargetObjectType(
  const PredictedObject & object, const std::shared_ptr<AvoidanceParameters> & parameters)
{
  const auto object_type = utils::getHighestProbLabel(object.classification);

  if (parameters->object_parameters.count(object_type) == 0) {
    return false;
  }

  return parameters->object_parameters.at(object_type).is_avoidance_target;
}

/**
 * @brief check whether the object is safety check target object type.
 * @param object data.
 * @param parameters.
 * @return if the object is safety check target object type, return true.
 */
bool isSafetyCheckTargetObjectType(
  const PredictedObject & object, const std::shared_ptr<AvoidanceParameters> & parameters)
{
  const auto object_type = utils::getHighestProbLabel(object.classification);

  if (parameters->object_parameters.count(object_type) == 0) {
    return false;
  }

  return parameters->object_parameters.at(object_type).is_safety_check_target;
}

/**
 * @brief check whether the object type is ObjectClassification::UNKNOWN.
 * @param object data.
 * @return if the object label whose probability is the highest among the candidate is UNKNOWN,
 * return true.
 */
bool isUnknownTypeObject(const ObjectData & object)
{
  const auto object_type = utils::getHighestProbLabel(object.object.classification);
  return object_type == ObjectClassification::UNKNOWN;
}

/**
 * @brief classify object by whether it's vehicle or not.
 * @param object data.
 * @return if the object type is vehicle, return true.
 */
bool isVehicleTypeObject(const ObjectData & object)
{
  const auto object_type = utils::getHighestProbLabel(object.object.classification);

  if (object_type == ObjectClassification::PEDESTRIAN) {
    return false;
  }

  if (object_type == ObjectClassification::BICYCLE) {
    return false;
  }

  return true;
}

/**
 * @brief check whether the object is moving or not.
 * @param object data.
 * @param parameters.
 * @return if the object keeps moving more than threshold time duration, return true. if the object
 * hasn't been moving for more than threshold time, this function return false even if the object
 * speed is NOT zero.
 */
bool isMovingObject(
  const ObjectData & object, const std::shared_ptr<AvoidanceParameters> & parameters)
{
  const auto object_type = utils::getHighestProbLabel(object.object.classification);
  const auto object_parameter = parameters->object_parameters.at(object_type);
  return object.move_time > object_parameter.moving_time_threshold;
}

bool isWithinCrosswalk(
  const ObjectData & object,
  const std::shared_ptr<const lanelet::routing::RoutingGraphContainer> & overall_graphs)
{
  using Point = boost::geometry::model::d2::point_xy<double>;

  const Point p_object{object.getPosition().x, object.getPosition().y};

  // get conflicting crosswalk crosswalk
  constexpr int PEDESTRIAN_GRAPH_ID = 1;
  const auto conflicts =
    overall_graphs->conflictingInGraph(object.overhang_lanelet, PEDESTRIAN_GRAPH_ID);

  constexpr double THRESHOLD = 2.0;
  for (const auto & crosswalk : conflicts) {
    auto polygon = crosswalk.polygon2d().basicPolygon();

    boost::geometry::correct(polygon);

    // ignore objects around the crosswalk
    if (boost::geometry::distance(p_object, polygon) < THRESHOLD) {
      return true;
    }
  }

  return false;
}

bool isWithinIntersection(
  const ObjectData & object, const std::shared_ptr<RouteHandler> & route_handler)
{
  const std::string area_id = object.overhang_lanelet.attributeOr("intersection_area", "else");
  if (area_id == "else") {
    return false;
  }

  if (!std::atoi(area_id.c_str())) {
    return false;
  }

  const std::string location = object.overhang_lanelet.attributeOr("location", "else");
  if (location == "private") {
    return false;
  }

  const auto polygon_opt =
    route_handler->getLaneletMapPtr()->polygonLayer.find(std::atoi(area_id.c_str()));
  if (polygon_opt == route_handler->getLaneletMapPtr()->polygonLayer.end()) {
    return false;
  }
  const auto & polygon = *polygon_opt;

  return boost::geometry::within(
    lanelet::utils::to2D(lanelet::utils::conversion::toLaneletPoint(object.getPosition()))
      .basicPoint(),
    lanelet::utils::to2D(polygon.basicPolygon()));
}

bool isWithinFreespace(
  const ObjectData & object, const std::shared_ptr<RouteHandler> & route_handler)
{
  auto polygons = lanelet::utils::query::getAllParkingLots(route_handler->getLaneletMapPtr());
  if (polygons.empty()) {
    return false;
  }

  std::sort(polygons.begin(), polygons.end(), [&object](const auto & a, const auto & b) {
    const double a_distance = boost::geometry::distance(
      lanelet::utils::to2D(a).basicPolygon(),
      lanelet::utils::to2D(lanelet::utils::conversion::toLaneletPoint(object.getPosition()))
        .basicPoint());
    const double b_distance = boost::geometry::distance(
      lanelet::utils::to2D(b).basicPolygon(),
      lanelet::utils::to2D(lanelet::utils::conversion::toLaneletPoint(object.getPosition()))
        .basicPoint());
    return a_distance < b_distance;
  });

  return boost::geometry::within(
    lanelet::utils::to2D(lanelet::utils::conversion::toLaneletPoint(object.getPosition()))
      .basicPoint(),
    lanelet::utils::to2D(polygons.front().basicPolygon()));
}

/**
 * @brief check whether the object is on ego driving lane.
 * @param object data.
 * @param route handler.
 * @return if the object is on ego lane, return true.
 */
bool isOnEgoLane(const ObjectData & object, const std::shared_ptr<RouteHandler> & route_handler)
{
  if (boost::geometry::within(
        lanelet::utils::to2D(lanelet::utils::conversion::toLaneletPoint(object.getPosition()))
          .basicPoint(),
        object.overhang_lanelet.polygon2d().basicPolygon())) {
    return true;
  }

  // push previous lanelet
  lanelet::ConstLanelets prev_lanelet;
  if (route_handler->getPreviousLaneletsWithinRoute(object.overhang_lanelet, &prev_lanelet)) {
    if (boost::geometry::within(
          lanelet::utils::to2D(lanelet::utils::conversion::toLaneletPoint(object.getPosition()))
            .basicPoint(),
          prev_lanelet.front().polygon2d().basicPolygon())) {
      return true;
    }
  }

  // push next lanelet
  lanelet::ConstLanelet next_lanelet;
  if (route_handler->getNextLaneletWithinRoute(object.overhang_lanelet, &next_lanelet)) {
    if (boost::geometry::within(
          lanelet::utils::to2D(lanelet::utils::conversion::toLaneletPoint(object.getPosition()))
            .basicPoint(),
          next_lanelet.polygon2d().basicPolygon())) {
      return true;
    }
  } else {
    for (const auto & lane : route_handler->getNextLanelets(object.overhang_lanelet)) {
      if (boost::geometry::within(
            lanelet::utils::to2D(lanelet::utils::conversion::toLaneletPoint(object.getPosition()))
              .basicPoint(),
            lane.polygon2d().basicPolygon())) {
        return true;
      }
    }
  }

  return false;
}

bool isParallelToEgoLane(const ObjectData & object, const double threshold)
{
  const auto closest_pose =
    nero_geom_fast::getClosestCenterPose(object.overhang_lanelet, object.getPosition());
  const auto yaw_deviation = std::abs(calc_yaw_deviation(closest_pose, object.getPose()));

  return yaw_deviation < threshold || yaw_deviation > M_PI - threshold;
}

bool isMergingToEgoLane(const ObjectData & object)
{
  const auto closest_pose =
    nero_geom_fast::getClosestCenterPose(object.overhang_lanelet, object.getPosition());
  const auto yaw_deviation = calc_yaw_deviation(closest_pose, object.getPose());

  if (isOnRight(object)) {
    if (yaw_deviation < 0.0 && -1.0 * M_PI_2 < yaw_deviation) {
      return false;
    }

    if (yaw_deviation > M_PI_2) {
      return false;
    }
  } else {
    if (yaw_deviation > 0.0 && M_PI_2 > yaw_deviation) {
      return false;
    }

    if (yaw_deviation < -1.0 * M_PI_2) {
      return false;
    }
  }

  return true;
}

/**
 * @brief check whether the object is parking on road shoulder.
 * @param object polygon.
 * @param avoidance module data.
 * @param route handler.
 * @param parameters.
 * @return if the object is close to road shoulder of the lane, return true.
 */
bool isParkedVehicle(
  ObjectData & object, const AvoidancePlanningData & data,
  const std::shared_ptr<RouteHandler> & route_handler,
  const std::shared_ptr<AvoidanceParameters> & parameters)
{
  using lanelet::geometry::distance2d;
  using lanelet::geometry::toArcCoordinates;
  using lanelet::utils::to2D;
  using lanelet::utils::conversion::toLaneletPoint;

  if (object.is_within_intersection) {
    return false;
  }

  const auto centerline_pos =
    nero_geom_fast::getClosestCenterPose(object.overhang_lanelet, object.getPosition()).position;

  bool is_left_side_parked_vehicle = false;
  if (!isOnRight(object)) {
    const auto most_left_lanelet = [&]() {
      auto same_direction_lane =
        route_handler->getMostLeftLanelet(object.overhang_lanelet, true, true);
      const lanelet::Attribute & sub_type =
        same_direction_lane.attribute(lanelet::AttributeName::Subtype);
      if (sub_type == "road_shoulder") {
        return same_direction_lane;
      }

      const auto opposite_lanes = route_handler->getLeftOppositeLanelets(same_direction_lane);
      if (opposite_lanes.empty()) {
        return same_direction_lane;
      }

      return static_cast<lanelet::ConstLanelet>(
        route_handler->getMostRightLanelet(opposite_lanes.front()).invert());
    }();

    const auto center_to_left_boundary = distance2d(
      to2D(most_left_lanelet.leftBound().basicLineString()),
      to2D(toLaneletPoint(centerline_pos)).basicPoint());

    double object_shiftable_distance =
      center_to_left_boundary - 0.5 * object.object.shape.dimensions.y;

    const lanelet::Attribute & sub_type =
      most_left_lanelet.attribute(lanelet::AttributeName::Subtype);
    if (sub_type == "road_shoulder") {
      // assuming it's parked vehicle if its CoG is within road shoulder lanelet.
      if (boost::geometry::within(
            to2D(toLaneletPoint(object.getPosition())).basicPoint(),
            most_left_lanelet.polygon2d().basicPolygon())) {
        return true;
      }
    } else {
      // assuming there is 0.5m road shoulder even if it's not defined explicitly in HDMap.
      object_shiftable_distance += parameters->object_check_min_road_shoulder_width;
    }

    const auto arc_coordinates = toArcCoordinates(
      to2D(object.overhang_lanelet.centerline().basicLineString()),
      to2D(toLaneletPoint(object.getPosition())).basicPoint());
    object.shiftable_ratio = arc_coordinates.distance / object_shiftable_distance;

    is_left_side_parked_vehicle = object.shiftable_ratio > parameters->object_check_shiftable_ratio;
  }

  bool is_right_side_parked_vehicle = false;
  if (isOnRight(object)) {
    const auto most_right_lanelet = [&]() {
      auto same_direction_lane =
        route_handler->getMostRightLanelet(object.overhang_lanelet, true, true);
      const lanelet::Attribute & sub_type =
        same_direction_lane.attribute(lanelet::AttributeName::Subtype);
      if (sub_type == "road_shoulder") {
        return same_direction_lane;
      }

      const auto opposite_lanes = route_handler->getRightOppositeLanelets(same_direction_lane);
      if (opposite_lanes.empty()) {
        return same_direction_lane;
      }

      return static_cast<lanelet::ConstLanelet>(
        route_handler->getMostLeftLanelet(opposite_lanes.front()).invert());
    }();

    const auto center_to_right_boundary = distance2d(
      to2D(most_right_lanelet.rightBound().basicLineString()),
      to2D(toLaneletPoint(centerline_pos)).basicPoint());

    double object_shiftable_distance =
      center_to_right_boundary - 0.5 * object.object.shape.dimensions.y;

    const lanelet::Attribute & sub_type =
      most_right_lanelet.attribute(lanelet::AttributeName::Subtype);
    if (sub_type == "road_shoulder") {
      // assuming it's parked vehicle if its CoG is within road shoulder lanelet.
      if (boost::geometry::within(
            to2D(toLaneletPoint(object.getPosition())).basicPoint(),
            most_right_lanelet.polygon2d().basicPolygon())) {
        return true;
      }
    } else {
      // assuming there is 0.5m road shoulder even if it's not defined explicitly in HDMap.
      object_shiftable_distance += parameters->object_check_min_road_shoulder_width;
    }

    const auto arc_coordinates = toArcCoordinates(
      to2D(object.overhang_lanelet.centerline().basicLineString()),
      to2D(toLaneletPoint(object.getPosition())).basicPoint());
    object.shiftable_ratio = -1.0 * arc_coordinates.distance / object_shiftable_distance;

    is_right_side_parked_vehicle =
      object.shiftable_ratio > parameters->object_check_shiftable_ratio;
  }

  if (!is_left_side_parked_vehicle && !is_right_side_parked_vehicle) {
    return false;
  }

  object.to_centerline =
    lanelet::utils::getArcCoordinates(data.current_lanelets, object.getPose()).distance;
  return std::abs(object.to_centerline) >= parameters->threshold_distance_object_is_on_center;
}

bool isCloseToStopFactor(
  ObjectData & object, const AvoidancePlanningData & data,
  const std::shared_ptr<const PlannerData> & planner_data,
  const std::shared_ptr<AvoidanceParameters> & parameters)
{
  const auto & rh = planner_data->route_handler;
  const auto & ego_pose = planner_data->self_odometry->pose.pose;

  // force avoidance for stopped vehicle
  bool is_close_to_stop_factor = false;

  // check traffic light
  const auto to_traffic_light =
    getDistanceToNextTrafficLight(object.getPose(), data.extend_lanelets);
  {
    is_close_to_stop_factor =
      to_traffic_light < parameters->object_ignore_section_traffic_light_in_front_distance;
  }

  // check crosswalk
  const auto to_crosswalk =
    utils::getDistanceToCrosswalk(ego_pose, data.extend_lanelets, *rh->getOverallGraphPtr()) -
    object.longitudinal;
  {
    const auto stop_for_crosswalk =
      to_crosswalk < parameters->object_ignore_section_crosswalk_in_front_distance &&
      to_crosswalk > -1.0 * parameters->object_ignore_section_crosswalk_behind_distance;
    is_close_to_stop_factor = is_close_to_stop_factor || stop_for_crosswalk;
  }

  object.to_stop_factor_distance = std::min(to_traffic_light, to_crosswalk);

  return is_close_to_stop_factor;
}

bool isNeverAvoidanceTarget(
  ObjectData & object, const AvoidancePlanningData & data,
  const std::shared_ptr<const PlannerData> & planner_data,
  const std::shared_ptr<AvoidanceParameters> & parameters)
{
  if (object.is_within_intersection) {
    if (object.behavior == ObjectData::Behavior::NONE) {
      object.info = ObjectInfo::PARALLEL_TO_EGO_LANE;
      RCLCPP_DEBUG(
        rclcpp::get_logger(logger_namespace), "object belongs to ego lane. never avoid it.");
      return true;
    }

    if (object.behavior == ObjectData::Behavior::MERGING) {
      object.info = ObjectInfo::MERGING_TO_EGO_LANE;
      RCLCPP_DEBUG(
        rclcpp::get_logger(logger_namespace), "object belongs to ego lane. never avoid it.");
      return true;
    }
  }

  if (object.behavior == ObjectData::Behavior::MERGING) {
    object.info = ObjectInfo::MERGING_TO_EGO_LANE;
    if (
      isOnRight(object) && !object.is_parked &&
      object.overhang_points.front().first > parameters->th_overhang_distance) {
      RCLCPP_DEBUG(
        rclcpp::get_logger(logger_namespace),
        "merging vehicle. but overhang distance is larger than threshold.");
      return true;
    }
    if (
      !isOnRight(object) && !object.is_parked &&
      object.overhang_points.front().first < -1.0 * parameters->th_overhang_distance) {
      RCLCPP_DEBUG(
        rclcpp::get_logger(logger_namespace),
        "merging vehicle. but overhang distance is larger than threshold.");
      return true;
    }
  }

  if (object.behavior == ObjectData::Behavior::DEVIATING) {
    object.info = ObjectInfo::DEVIATING_FROM_EGO_LANE;
    if (
      isOnRight(object) && !object.is_parked &&
      object.overhang_points.front().first > parameters->th_overhang_distance) {
      RCLCPP_DEBUG(
        rclcpp::get_logger(logger_namespace),
        "deviating vehicle. but overhang distance is larger than threshold.");
      return true;
    }
    if (
      !isOnRight(object) && !object.is_parked &&
      object.overhang_points.front().first < -1.0 * parameters->th_overhang_distance) {
      RCLCPP_DEBUG(
        rclcpp::get_logger(logger_namespace),
        "deviating vehicle. but overhang distance is larger than threshold.");
      return true;
    }
  }

  if (object.is_on_ego_lane) {
    const auto right_lane =
      planner_data->route_handler->getRightLanelet(object.overhang_lanelet, true, true);
    if (right_lane.has_value() && isOnRight(object)) {
      const lanelet::Attribute & right_lane_sub_type =
        right_lane.value().attribute(lanelet::AttributeName::Subtype);
      if (right_lane_sub_type != "road_shoulder") {
        object.info = ObjectInfo::IS_NOT_PARKING_OBJECT;
        RCLCPP_DEBUG(
          rclcpp::get_logger(logger_namespace), "object isn't on the edge lane. never avoid it.");
        return true;
      }

      const auto object_polygon = autoware_utils::to_polygon2d(object.object);
      const auto is_disjoint_right_lane =
        boost::geometry::disjoint(object_polygon, right_lane.value().polygon2d().basicPolygon());
      if (is_disjoint_right_lane) {
        object.info = ObjectInfo::IS_NOT_PARKING_OBJECT;
        RCLCPP_DEBUG(
          rclcpp::get_logger(logger_namespace), "object isn't on the edge lane. never avoid it.");
        return true;
      }
    }

    const auto right_opposite_lanes =
      planner_data->route_handler->getRightOppositeLanelets(object.overhang_lanelet);
    if (!right_opposite_lanes.empty() && isOnRight(object) && !nero_filter::env().ambiguous_in_lane) {
      object.info = ObjectInfo::IS_NOT_PARKING_OBJECT;
      RCLCPP_DEBUG(
        rclcpp::get_logger(logger_namespace), "object isn't on the edge lane. never avoid it.");
      return true;
    }

    const auto left_lane =
      planner_data->route_handler->getLeftLanelet(object.overhang_lanelet, true, true);
    if (left_lane.has_value() && !isOnRight(object)) {
      const lanelet::Attribute & left_lane_sub_type =
        left_lane.value().attribute(lanelet::AttributeName::Subtype);
      if (left_lane_sub_type != "road_shoulder") {
        object.info = ObjectInfo::IS_NOT_PARKING_OBJECT;
        RCLCPP_DEBUG(
          rclcpp::get_logger(logger_namespace), "object isn't on the edge lane. never avoid it.");
        return true;
      }

      const auto object_polygon = autoware_utils::to_polygon2d(object.object);
      const auto is_disjoint_left_lane =
        boost::geometry::disjoint(object_polygon, left_lane.value().polygon2d().basicPolygon());
      if (is_disjoint_left_lane) {
        object.info = ObjectInfo::IS_NOT_PARKING_OBJECT;
        RCLCPP_DEBUG(
          rclcpp::get_logger(logger_namespace), "object isn't on the edge lane. never avoid it.");
        return true;
      }
    }

    const auto left_opposite_lanes =
      planner_data->route_handler->getLeftOppositeLanelets(object.overhang_lanelet);
    if (!left_opposite_lanes.empty() && !isOnRight(object) && !nero_filter::env().ambiguous_in_lane) {
      object.info = ObjectInfo::IS_NOT_PARKING_OBJECT;
      RCLCPP_DEBUG(
        rclcpp::get_logger(logger_namespace), "object isn't on the edge lane. never avoid it.");
      return true;
    }
  }

  if (isCloseToStopFactor(object, data, planner_data, parameters)) {
    if (object.is_on_ego_lane && !object.is_parked) {
      object.info = ObjectInfo::IS_NOT_PARKING_OBJECT;
      RCLCPP_DEBUG(
        rclcpp::get_logger(logger_namespace), "object is close to stop factor. never avoid it.");
      return true;
    }
  }

  if (object.is_on_ego_lane && !object.is_parked) {
    const auto & ne = nero_filter::env();
    if (ne.stop_line_m > 0.0) {
      const double d = nero_filter::distanceToStopLineOrJunction(object, data.extend_lanelets);
      object.to_stop_factor_distance = std::min(object.to_stop_factor_distance, d);
      if (d < ne.stop_line_m) {
        object.info = ObjectInfo::IS_NOT_PARKING_OBJECT;
        RCLCPP_DEBUG(
          rclcpp::get_logger(logger_namespace),
          "[NERO_SOA] waiting before a stop line / junction (%.1f m). never avoid it.", d);
        return true;
      }
    }
    if (ne.queue_m > 0.0 && nero_filter::isQueued(object, data, planner_data, ne.queue_m)) {
      object.info = ObjectInfo::IS_NOT_PARKING_OBJECT;
      RCLCPP_DEBUG(rclcpp::get_logger(logger_namespace), "[NERO_SOA] queued behind a vehicle. never avoid it.");
      return true;
    }
  }

  return false;
}

bool isObviousAvoidanceTarget(
  ObjectData & object, [[maybe_unused]] const AvoidancePlanningData & data,
  [[maybe_unused]] const std::shared_ptr<const PlannerData> & planner_data,
  [[maybe_unused]] const std::shared_ptr<AvoidanceParameters> & parameters)
{
  if (isWithinFreespace(object, planner_data->route_handler)) {
    if (!object.is_on_ego_lane) {
      if (object.stop_time > parameters->freespace_condition_th_stopped_time) {
        return true;
      }
    }
  }

  if (!object.is_within_intersection) {
    if (object.is_parked && object.behavior == ObjectData::Behavior::NONE) {
      RCLCPP_DEBUG(rclcpp::get_logger(logger_namespace), "object is obvious parked vehicle.");
      return true;
    }

    if (!object.is_on_ego_lane && object.behavior == ObjectData::Behavior::NONE) {
      RCLCPP_DEBUG(rclcpp::get_logger(logger_namespace), "object is adjacent vehicle.");
      return true;
    }
  }

  if (object.behavior == ObjectData::Behavior::MERGING) {
    object.info = ObjectInfo::MERGING_TO_EGO_LANE;
    if (
      isOnRight(object) && !object.is_on_ego_lane &&
      object.overhang_points.front().first < parameters->th_overhang_distance) {
      RCLCPP_DEBUG(
        rclcpp::get_logger(logger_namespace),
        "merging vehicle. but overhang distance is less than threshold.");
      return true;
    }
    if (
      !isOnRight(object) && !object.is_on_ego_lane &&
      object.overhang_points.front().first > -1.0 * parameters->th_overhang_distance) {
      RCLCPP_DEBUG(
        rclcpp::get_logger(logger_namespace),
        "merging vehicle. but overhang distance is less than threshold.");
      return true;
    }
  }

  if (object.behavior == ObjectData::Behavior::DEVIATING) {
    object.info = ObjectInfo::DEVIATING_FROM_EGO_LANE;
    if (
      isOnRight(object) && !object.is_on_ego_lane &&
      object.overhang_points.front().first < parameters->th_overhang_distance) {
      RCLCPP_DEBUG(
        rclcpp::get_logger(logger_namespace),
        "deviating vehicle. but overhang distance is less than threshold.");
      return true;
    }
    if (
      !isOnRight(object) && !object.is_on_ego_lane &&
      object.overhang_points.front().first > -1.0 * parameters->th_overhang_distance) {
      RCLCPP_DEBUG(
        rclcpp::get_logger(logger_namespace),
        "deviating vehicle. but overhang distance is less than threshold.");
      return true;
    }
  }

  if (!object.is_parked) {
    object.info = ObjectInfo::IS_NOT_PARKING_OBJECT;
  }

  if (object.behavior == ObjectData::Behavior::MERGING) {
    object.info = ObjectInfo::MERGING_TO_EGO_LANE;
  }

  return false;
}

/**
 * @brief this function includes some conditions which apply to both vehicle and non-vehicle object.
 * @param object data.
 * @param current reference path.
 * @param object detection range.
 * @param distance between object and goal point.
 * @param ego position.
 * @param if the goal point can be moved by external module when there is obstacle around the goal,
 * this flag will be true.
 * @param parameters.
 * @return if the object is potentially target object, return true.
 */
bool isSatisfiedWithCommonCondition(
  ObjectData & object, const PathWithLaneId & path, const double forward_detection_range,
  const double to_goal_distance, const Point & ego_pos, const bool is_allowed_goal_modification,
  const std::shared_ptr<AvoidanceParameters> & parameters)
{
  // Step1. filtered by target object type.
  if (!isAvoidanceTargetObjectType(object.object, parameters)) {
    object.info = ObjectInfo::IS_NOT_TARGET_OBJECT;
    return false;
  }

  // Step2. filtered stopped objects.
  if (filtering_utils::isMovingObject(object, parameters)) {
    object.info = ObjectInfo::MOVING_OBJECT;
    return false;
  }

  // Step3. filtered by longitudinal distance.
  fillLongitudinalAndLengthByClosestEnvelopeFootprint(path, ego_pos, object);

  if (object.longitudinal < -parameters->object_check_backward_distance) {
    object.info = ObjectInfo::FURTHER_THAN_THRESHOLD;
    return false;
  }

  if (object.longitudinal > forward_detection_range) {
    object.info = ObjectInfo::FURTHER_THAN_THRESHOLD;
    return false;
  }

  // Step4. filtered by distance between object and goal position.
  // TODO(Satoshi OTA): remove following two conditions after it can execute avoidance and goal
  // planner module simultaneously.
  if (object.longitudinal > to_goal_distance) {
    object.info = ObjectInfo::FURTHER_THAN_GOAL;
    return false;
  }

  if (!is_allowed_goal_modification) {
    if (
      object.longitudinal + object.length / 2 + parameters->object_check_goal_distance >
      to_goal_distance) {
      object.info = ObjectInfo::TOO_NEAR_TO_GOAL;
      return false;
    }
  }

  return true;
}

bool isSatisfiedWithNonVehicleCondition(
  ObjectData & object, [[maybe_unused]] const AvoidancePlanningData & data,
  const std::shared_ptr<const PlannerData> & planner_data,
  [[maybe_unused]] const std::shared_ptr<AvoidanceParameters> & parameters)
{
  // avoidance module ignore pedestrian and bicycle around crosswalk
  if (isWithinCrosswalk(object, planner_data->route_handler->getOverallGraphPtr())) {
    object.info = ObjectInfo::CROSSWALK_USER;
    return false;
  }

  // Object is on center line -> ignore.
  object.to_centerline =
    lanelet::utils::getArcCoordinates(data.current_lanelets, object.getPose()).distance;
  if (std::abs(object.to_centerline) < parameters->threshold_distance_object_is_on_center) {
    object.info = ObjectInfo::TOO_NEAR_TO_CENTERLINE;
    return false;
  }

  if (object.is_within_intersection) {
    RCLCPP_DEBUG(
      rclcpp::get_logger(logger_namespace),
      "object is within intersection. don't have to avoid it.");
    return false;
  }

  const auto right_lane =
    planner_data->route_handler->getRightLanelet(object.overhang_lanelet, true, true);
  if (right_lane.has_value() && isOnRight(object)) {
    RCLCPP_DEBUG(
      rclcpp::get_logger(logger_namespace), "object isn't on the edge lane. never avoid it.");
    return false;
  }

  const auto left_lane =
    planner_data->route_handler->getLeftLanelet(object.overhang_lanelet, true, true);
  if (left_lane.has_value() && !isOnRight(object)) {
    RCLCPP_DEBUG(
      rclcpp::get_logger(logger_namespace), "object isn't on the edge lane. never avoid it.");
    return false;
  }

  const auto right_opposite_lanes =
    planner_data->route_handler->getRightOppositeLanelets(object.overhang_lanelet);
  if (!right_opposite_lanes.empty() && isOnRight(object)) {
    RCLCPP_DEBUG(
      rclcpp::get_logger(logger_namespace), "object isn't on the edge lane. never avoid it.");
    return false;
  }

  const auto left_opposite_lanes =
    planner_data->route_handler->getLeftOppositeLanelets(object.overhang_lanelet);
  if (!left_opposite_lanes.empty() && !isOnRight(object)) {
    RCLCPP_DEBUG(
      rclcpp::get_logger(logger_namespace), "object isn't on the edge lane. never avoid it.");
    return false;
  }

  return true;
}

/**
 * @brief estimate object's behavior based on its relative yaw angle to lane.
 * @param object data.
 * @param parameters.
 * @return return DEVIATING, MERGING and NONE. NONE means the object has no intent to leave or merge
 * the ego lane.
 */
ObjectData::Behavior getObjectBehavior(
  const ObjectData & object, const std::shared_ptr<AvoidanceParameters> & parameters)
{
  if (isParallelToEgoLane(object, parameters->object_check_yaw_deviation)) {
    return ObjectData::Behavior::NONE;
  }

  return isMergingToEgoLane(object) ? ObjectData::Behavior::MERGING
                                    : ObjectData::Behavior::DEVIATING;
}

bool isSatisfiedWithVehicleCondition(
  ObjectData & object, const AvoidancePlanningData & data,
  const std::shared_ptr<const PlannerData> & planner_data,
  const std::shared_ptr<AvoidanceParameters> & parameters)
{
  object.behavior = getObjectBehavior(object, parameters);
  object.is_on_ego_lane = isOnEgoLane(object, planner_data->route_handler);

  if (isNeverAvoidanceTarget(object, data, planner_data, parameters)) {
    return false;
  }

  if (isObviousAvoidanceTarget(object, data, planner_data, parameters)) {
    return true;
  }

  // from here, filtering for ambiguous vehicle.

  if (parameters->policy_ambiguous_vehicle == "ignore") {
    object.info = ObjectInfo::AMBIGUOUS_STOPPED_VEHICLE;
    return false;
  }

  const auto stop_time_longer_than_threshold =
    object.stop_time > parameters->time_threshold_for_ambiguous_vehicle;
  if (!stop_time_longer_than_threshold) {
    object.info = ObjectInfo::AMBIGUOUS_STOPPED_VEHICLE;
    object.is_ambiguous = false;
    return false;
  }

  const auto is_moving_distance_longer_than_threshold =
    calc_distance2d(object.init_pose, object.getPose()) >
    parameters->distance_threshold_for_ambiguous_vehicle;
  if (is_moving_distance_longer_than_threshold) {
    object.info = ObjectInfo::AMBIGUOUS_STOPPED_VEHICLE;
    object.is_ambiguous = false;
    return false;
  }

  if (object.is_within_intersection) {
    if (object.behavior == ObjectData::Behavior::DEVIATING) {
      object.info = ObjectInfo::AMBIGUOUS_STOPPED_VEHICLE;
      object.is_ambiguous = true;
      return true;
    }
  } else {
    object.info = ObjectInfo::AMBIGUOUS_STOPPED_VEHICLE;
    object.is_ambiguous = true;
    return true;
  }

  object.info = ObjectInfo::IS_NOT_PARKING_OBJECT;
  return false;
}

/**
 * @brief check the ego has to avoid the object for lateral margin.
 * @param object data.
 * @param parameters.
 * @return if the ego doesn't have to shift driving position to avoid object, return false. if the
 * shift length is less than threshold, this fuction returns false.
 */
bool isNoNeedAvoidanceBehavior(
  ObjectData & object, const std::shared_ptr<AvoidanceParameters> & parameters)
{
  if (!object.avoid_margin.has_value()) {
    return false;
  }

  const auto shift_length = calcShiftLength(
    isOnRight(object), object.overhang_points.front().first, object.avoid_margin.value());
  if (!isShiftNecessary(isOnRight(object), shift_length)) {
    object.info = ObjectInfo::ENOUGH_LATERAL_DISTANCE;
    return true;
  }

  if (std::abs(shift_length) < parameters->lateral_execution_threshold) {
    object.info = ObjectInfo::LESS_THAN_EXECUTION_THRESHOLD;
    return true;
  }

  return false;
}

/**
 * @brief get avoidance lateral margin based on road width.
 * @param object data.
 * @param planner data, which includes ego vehicle footprint info.
 * @param parameters.
 * @return if this function finds there is no enough space to avoid, return nullopt.
 */
std::optional<double> getAvoidMargin(
  const ObjectData & object, const std::shared_ptr<const PlannerData> & planner_data,
  const std::shared_ptr<AvoidanceParameters> & parameters)
{
  const auto & vehicle_width = planner_data->parameters.vehicle_width;
  const auto object_type = utils::getHighestProbLabel(object.object.classification);
  const auto object_parameter = parameters->object_parameters.at(object_type);
  const auto lateral_hard_margin = object.is_parked
                                     ? object_parameter.lateral_hard_margin_for_parked_vehicle
                                     : object_parameter.lateral_hard_margin;

  const auto max_avoid_margin = lateral_hard_margin * object.distance_factor +
                                object_parameter.lateral_soft_margin + 0.5 * vehicle_width;
  const auto min_avoid_margin = lateral_hard_margin + 0.5 * vehicle_width;
  const auto soft_lateral_distance_limit =
    object.to_road_shoulder_distance - parameters->soft_drivable_bound_margin - 0.5 * vehicle_width;
  const auto hard_lateral_distance_limit =
    object.to_road_shoulder_distance - parameters->hard_drivable_bound_margin - 0.5 * vehicle_width;

  // Step1. check avoidable or not.
  if (hard_lateral_distance_limit < min_avoid_margin) {
    return std::nullopt;
  }

  // Step2. check if it should expand road shoulder margin.
  if (soft_lateral_distance_limit < min_avoid_margin) {
    return min_avoid_margin;
  }

  // Step3. nominal case. avoid margin is limited by soft constraint.
  return std::min(soft_lateral_distance_limit, max_avoid_margin);
}

/**
 * @brief get avoidance lateral margin based on road width.
 * @param object data.
 * @param avoidance module data, which includes current reference path.
 * @param planner data, which includes ego vehicle footprint info.
 * @return if this function finds there is no enough space to avoid, return nullopt.
 */
double getRoadShoulderDistance(
  ObjectData & object, const AvoidancePlanningData & data,
  const std::shared_ptr<const PlannerData> & planner_data)
{
  using autoware_utils::Point2d;
  using lanelet::utils::to2D;

  const auto object_closest_index =
    autoware::motion_utils::findNearestIndex(data.reference_path.points, object.getPosition());
  const auto object_closest_pose = data.reference_path.points.at(object_closest_index).point.pose;

  const auto rh = planner_data->route_handler;
  if (!nero_mu::getClosestLaneletWithinRoute(*rh, object_closest_pose, &object.overhang_lanelet)) {  // NERO_BPP_PROJ_FAST: memo
    return 0.0;
  }

  const auto centerline_pose =
    nero_geom_fast::getClosestCenterPose(object.overhang_lanelet, object.getPosition());
  // TODO(Satoshi OTA): check if the basic point is on right or left of bound.
  const auto bound = isOnRight(object) ? data.left_bound : data.right_bound;
  const auto envelope_polygon_width = boost::geometry::area(object.envelope_poly) /
                                      std::max(object.length, 1e-3);  // prevent division by zero

  std::vector<std::tuple<double, Point, Point>> intersects;
  for (const auto & p1 : object.overhang_points) {
    const auto p_tmp =
      geometry_msgs::build<Pose>().position(p1.second).orientation(centerline_pose.orientation);
    for (size_t i = 1; i < bound.size(); i++) {
      {
        const auto p2 =
          calc_offset_pose(p_tmp, 0.0, (isOnRight(object) ? 100.0 : -100.0), 0.0).position;
        const auto opt_intersect =
          autoware_utils::intersect(p1.second, p2, bound.at(i - 1), bound.at(i));

        if (opt_intersect.has_value()) {
          intersects.emplace_back(
            calc_distance2d(p1.second, opt_intersect.value()), p1.second, opt_intersect.value());
          break;
        }
      }
      {
        const auto p2 =
          calc_offset_pose(
            p_tmp, 0.0, (isOnRight(object) ? -0.5 : 0.5) * envelope_polygon_width, 0.0)
            .position;
        const auto opt_intersect =
          autoware_utils::intersect(p1.second, p2, bound.at(i - 1), bound.at(i));

        if (opt_intersect.has_value()) {
          intersects.emplace_back(
            -1.0 * calc_distance2d(p1.second, opt_intersect.value()), p1.second,
            opt_intersect.value());
          break;
        }
      }
    }
  }

  std::sort(intersects.begin(), intersects.end(), [](const auto & a, const auto & b) {
    return std::get<0>(a) < std::get<0>(b);
  });

  if (intersects.empty()) {
    return 0.0;
  }

  object.narrowest_place =
    std::make_pair(std::get<1>(intersects.front()), std::get<2>(intersects.front()));

  return std::get<0>(intersects.front());
}
}  // namespace filtering_utils

bool isOnRight(const ObjectData & obj)
{
  if (obj.direction == Direction::NONE) {
    throw std::logic_error("object direction is not initialized. something wrong.");
  }

  return obj.direction == Direction::RIGHT;
}

double calcShiftLength(
  const bool & is_object_on_right, const double & overhang_dist, const double & avoid_margin)
{
  const auto shift_length =
    is_object_on_right ? (overhang_dist + avoid_margin) : (overhang_dist - avoid_margin);
  return std::fabs(shift_length) > 1e-3 ? shift_length : 0.0;
}

bool isWithinLanes(
  const std::optional<lanelet::ConstLanelet> & closest_lanelet,
  const std::shared_ptr<const PlannerData> & planner_data)
{
  const auto & rh = planner_data->route_handler;
  const auto & ego_pose = planner_data->self_odometry->pose.pose;
  const auto transform = autoware_utils::pose2transform(ego_pose);
  const auto footprint = autoware_utils::transform_vector(
    planner_data->parameters.vehicle_info.createFootprint(), transform);

  if (!closest_lanelet.has_value()) {
    return true;
  }

  lanelet::ConstLanelets concat_lanelets{};

  // push previous lanelet
  lanelet::ConstLanelets prev_lanelet;
  if (rh->getPreviousLaneletsWithinRoute(closest_lanelet.value(), &prev_lanelet)) {
    concat_lanelets.push_back(prev_lanelet.front());
  }

  // push nearest lanelet
  {
    concat_lanelets.push_back(closest_lanelet.value());
  }

  // push next lanelet
  lanelet::ConstLanelet next_lanelet;
  if (rh->getNextLaneletWithinRoute(closest_lanelet.value(), &next_lanelet)) {
    concat_lanelets.push_back(next_lanelet);
  }

  const auto combine_lanelet = lanelet::utils::combineLaneletsShape(concat_lanelets);

  return boost::geometry::within(footprint, combine_lanelet.polygon2d().basicPolygon());
}

bool isShiftNecessary(const bool & is_object_on_right, const double & shift_length)
{
  /**
   *   ^
   *   |
   * --+----x-------------------------------x--->
   *   |                 x     x
   *   |                 ==obj==
   */
  if (is_object_on_right && shift_length < 0.0) {
    return false;
  }

  /**
   *   ^                 ==obj==
   *   |                 x     x
   * --+----x-------------------------------x--->
   *   |
   */
  if (!is_object_on_right && shift_length > 0.0) {
    return false;
  }

  return true;
}

bool isSameDirectionShift(const bool & is_object_on_right, const double & shift_length)
{
  return (is_object_on_right == std::signbit(shift_length));
}

ShiftedPath toShiftedPath(const PathWithLaneId & path)
{
  ShiftedPath out;
  out.path = path;
  out.shift_length.resize(path.points.size());
  std::fill(out.shift_length.begin(), out.shift_length.end(), 0.0);
  return out;
}

ShiftLineArray toShiftLineArray(const AvoidLineArray & avoid_points)
{
  ShiftLineArray shift_lines;
  for (const auto & ap : avoid_points) {
    shift_lines.push_back(ap);
  }
  return shift_lines;
}

size_t findPathIndexFromArclength(
  const std::vector<double> & path_arclength_arr, const double target_arc)
{
  if (path_arclength_arr.empty()) {
    return 0;
  }

  for (size_t i = 0; i < path_arclength_arr.size(); ++i) {
    if (path_arclength_arr.at(i) > target_arc) {
      return i;
    }
  }
  return path_arclength_arr.size() - 1;
}

std::vector<UUID> concatParentIds(const std::vector<UUID> & ids1, const std::vector<UUID> & ids2)
{
  std::vector<UUID> ret;

  for (const auto & id : ids1) {
    if (std::any_of(
          ret.begin(), ret.end(), [&id](const auto & exist_id) { return exist_id == id; })) {
      continue;
    }
    ret.push_back(id);
  }

  for (const auto & id : ids2) {
    if (std::any_of(
          ret.begin(), ret.end(), [&id](const auto & exist_id) { return exist_id == id; })) {
      continue;
    }
    ret.push_back(id);
  }

  return ret;
}

std::vector<UUID> calcParentIds(const AvoidLineArray & lines1, const AvoidLine & lines2)
{
  // Get the ID of the original AP whose transition area overlaps with the given AP,
  // and set it to the parent id.
  std::vector<UUID> ret;
  for (const auto & al : lines1) {
    const auto p_s = al.start_longitudinal;
    const auto p_e = al.end_longitudinal;
    const auto has_overlap = p_e >= lines2.start_longitudinal && lines2.end_longitudinal >= p_s;

    if (!has_overlap) {
      continue;
    }

    ret.push_back(al.id);
  }
  return ret;
}

double lerpShiftLengthOnArc(double arc, const AvoidLine & ap)
{
  if (ap.start_longitudinal <= arc && arc < ap.end_longitudinal) {
    const auto relative_longitudinal = ap.getRelativeLongitudinal();
    if (std::abs(relative_longitudinal) < 1.0e-5) {
      return ap.end_shift_length;
    }
    const auto start_weight = (ap.end_longitudinal - arc) / relative_longitudinal;
    return start_weight * ap.start_shift_length + (1.0 - start_weight) * ap.end_shift_length;
  }
  return 0.0;
}

void fillLongitudinalAndLengthByClosestEnvelopeFootprint(
  const PathWithLaneId & path, const Point & ego_pos, ObjectData & obj)
{
  double min_distance = std::numeric_limits<double>::max();
  double max_distance = std::numeric_limits<double>::lowest();
  for (const auto & p : obj.envelope_poly.outer()) {
    const auto point = autoware_utils::create_point(p.x(), p.y(), 0.0);
    // TODO(someone): search around first position where the ego should avoid the object.
    const double arc_length =
      nero_mu::calcSignedArcLength(path.points, ego_pos, point);
    min_distance = std::min(min_distance, arc_length);
    max_distance = std::max(max_distance, arc_length);
  }
  obj.longitudinal = min_distance;
  obj.length = max_distance - min_distance;
}

std::vector<std::pair<double, Point>> calcEnvelopeOverhangDistance(
  const ObjectData & object_data, const PathWithLaneId & path)
{
  std::vector<std::pair<double, Point>> overhang_points{};

  for (const auto & p : object_data.envelope_poly.outer()) {
    const auto point = autoware_utils::create_point(p.x(), p.y(), 0.0);
    // TODO(someone): search around first position where the ego should avoid the object.
    const auto idx = autoware::motion_utils::findNearestIndex(path.points, point);
    const auto lateral =
      calc_lateral_deviation(autoware_utils::get_pose(path.points.at(idx)), point);
    overhang_points.emplace_back(lateral, point);
  }
  std::sort(overhang_points.begin(), overhang_points.end(), [&](const auto & a, const auto & b) {
    return isOnRight(object_data) ? b.first < a.first : a.first < b.first;
  });
  return overhang_points;
}

void setEndData(
  AvoidLine & ap, const double length, const geometry_msgs::msg::Pose & end, const size_t end_idx,
  const double end_dist)
{
  ap.end_shift_length = length;
  ap.end = end;
  ap.end_idx = end_idx;
  ap.end_longitudinal = end_dist;
}

void setStartData(
  AvoidLine & ap, const double start_shift_length, const geometry_msgs::msg::Pose & start,
  const size_t start_idx, const double start_dist)
{
  ap.start_shift_length = start_shift_length;
  ap.start = start;
  ap.start_idx = start_idx;
  ap.start_longitudinal = start_dist;
}

Polygon2d createEnvelopePolygon(
  const Polygon2d & object_polygon, const Pose & closest_pose, const double envelope_buffer)
{
  namespace bg = boost::geometry;
  using autoware_utils::expand_polygon;
  using autoware_utils::Point2d;
  using autoware_utils::Polygon2d;
  using Box = bg::model::box<Point2d>;

  const auto toPolygon2d = [](const geometry_msgs::msg::Polygon & polygon) {
    Polygon2d ret{};

    for (const auto & p : polygon.points) {
      ret.outer().push_back(Point2d(p.x, p.y));
    }

    return ret;
  };

  Pose pose_2d = closest_pose;
  pose_2d.orientation = create_quaternion_from_rpy(0.0, 0.0, tf2::getYaw(closest_pose.orientation));

  TransformStamped geometry_tf{};
  geometry_tf.transform = pose2transform(pose_2d);

  tf2::Transform tf;
  tf2::fromMsg(geometry_tf.transform, tf);
  TransformStamped inverse_geometry_tf{};
  inverse_geometry_tf.transform = tf2::toMsg(tf.inverse());

  geometry_msgs::msg::Polygon out_ros_polygon{};
  tf2::doTransform(
    toMsg(object_polygon, closest_pose.position.z), out_ros_polygon, inverse_geometry_tf);

  const auto envelope_box = bg::return_envelope<Box>(toPolygon2d(out_ros_polygon));

  Polygon2d envelope_poly{};
  bg::convert(envelope_box, envelope_poly);

  geometry_msgs::msg::Polygon envelope_ros_polygon{};
  tf2::doTransform(
    toMsg(envelope_poly, closest_pose.position.z), envelope_ros_polygon, geometry_tf);

  const auto expanded_polygon = expand_polygon(toPolygon2d(envelope_ros_polygon), envelope_buffer);
  return expanded_polygon;
}

Polygon2d createEnvelopePolygon(
  const ObjectData & object_data, const Pose & closest_pose, const double envelope_buffer)
{
  const auto object_polygon = autoware_utils::to_polygon2d(object_data.object);
  return createEnvelopePolygon(object_polygon, closest_pose, envelope_buffer);
}

std::vector<DrivableAreaInfo::Obstacle> generateObstaclePolygonsForDrivableArea(
  const ObjectDataArray & objects, const std::shared_ptr<AvoidanceParameters> & parameters,
  const double vehicle_width)
{
  std::vector<DrivableAreaInfo::Obstacle> obstacles_for_drivable_area;

  if (objects.empty()) {
    return obstacles_for_drivable_area;
  }

  for (const auto & object : objects) {
    // check if avoid marin is calculated
    if (!object.avoid_margin.has_value()) {
      continue;
    }

    // check original polygon
    if (object.envelope_poly.outer().empty()) {
      continue;
    }

    const auto object_type = utils::getHighestProbLabel(object.object.classification);
    const auto object_parameter = parameters->object_parameters.at(object_type);

    // generate obstacle polygon
    const double diff_poly_buffer =
      object.avoid_margin.value() - object_parameter.envelope_buffer_margin - vehicle_width / 2.0;
    const auto obj_poly = autoware_utils::expand_polygon(object.envelope_poly, diff_poly_buffer);
    obstacles_for_drivable_area.push_back({object.getPose(), obj_poly, !isOnRight(object)});
  }
  return obstacles_for_drivable_area;
}

lanelet::ConstLanelets getCurrentLanesFromPath(
  const PathWithLaneId & path, const std::shared_ptr<const PlannerData> & planner_data)
{
  if (path.points.empty()) {
    throw std::logic_error("empty path.");
  }

  const auto idx = planner_data->findEgoIndex(path.points);

  if (path.points.at(idx).lane_ids.empty()) {
    throw std::logic_error("empty lane ids.");
  }

  const auto start_id = path.points.at(idx).lane_ids.front();
  const auto start_lane = planner_data->route_handler->getLaneletsFromId(start_id);
  const auto & p = planner_data->parameters;

  return planner_data->route_handler->getLaneletSequence(
    start_lane, p.backward_path_length, p.forward_path_length);
}

lanelet::ConstLanelets getExtendLanes(
  const lanelet::ConstLanelets & lanelets, const Pose & ego_pose,
  const std::shared_ptr<const PlannerData> & planner_data)
{
  if (lanelets.empty()) {
    RCLCPP_WARN(
      rclcpp::get_logger("static_obstacle_avoidance"), "Empty lanelets provided to getExtendLanes");
    return lanelets;
  }

  lanelet::ConstLanelets extend_lanelets = lanelets;

  while (rclcpp::ok()) {
    const double lane_length = lanelet::utils::getLaneletLength2d(extend_lanelets);
    const auto arc_coordinates = lanelet::utils::getArcCoordinates(extend_lanelets, ego_pose);
    const auto forward_length = lane_length - arc_coordinates.length;

    if (forward_length > planner_data->parameters.forward_path_length) {
      break;
    }

    const auto next_lanelets = planner_data->route_handler->getNextLanelets(extend_lanelets.back());

    if (next_lanelets.empty()) {
      break;
    }

    extend_lanelets.push_back(next_lanelets.front());
  }

  return extend_lanelets;
}

void insertDecelPoint(
  const Point & p_src, const double offset, const double velocity, PathWithLaneId & path,
  PoseWithDetailOpt & p_out)
{
  const auto decel_point =
    nero_mu::calcLongitudinalOffsetPoint(path.points, p_src, offset);

  if (!decel_point) {
    // TODO(Satoshi OTA)  Think later the process in the case of no decel point found.
    return;
  }

  const auto seg_idx =
    nero_mu::findNearestSegmentIndex(path.points, decel_point.value());
  const auto insert_idx =
    autoware::motion_utils::insertTargetPoint(seg_idx, decel_point.value(), path.points);

  if (!insert_idx) {
    // TODO(Satoshi OTA)  Think later the process in the case of no decel point found.
    return;
  }

  const auto insertVelocity = [&insert_idx](PathWithLaneId & path, const float v) {
    for (size_t i = insert_idx.value(); i < path.points.size(); ++i) {
      const auto & original_velocity = path.points.at(i).point.longitudinal_velocity_mps;
      path.points.at(i).point.longitudinal_velocity_mps = std::min(original_velocity, v);
    }
  };

  insertVelocity(path, velocity);

  p_out = PoseWithDetail(autoware_utils::get_pose(path.points.at(insert_idx.value())));
}

void fillObjectEnvelopePolygon(
  ObjectData & object_data, const ObjectDataArray & registered_objects, const Pose & closest_pose,
  const std::shared_ptr<AvoidanceParameters> & parameters)
{
  const auto object_type = utils::getHighestProbLabel(object_data.object.classification);
  const auto object_parameter = parameters->object_parameters.at(object_type);

  const auto & envelope_buffer_margin =
    object_parameter.envelope_buffer_margin * object_data.distance_factor;

  const auto id = object_data.object.object_id;
  const auto same_id_obj = std::find_if(
    registered_objects.begin(), registered_objects.end(),
    [&id](const auto & o) { return o.object.object_id == id; });

  if (same_id_obj == registered_objects.end()) {
    object_data.envelope_poly =
      createEnvelopePolygon(object_data, closest_pose, envelope_buffer_margin);
    object_data.error_eclipse_max =
      calcErrorEclipseLongRadius(object_data.object.kinematics.initial_pose_with_covariance);
    return;
  }

  const auto one_shot_envelope_poly =
    createEnvelopePolygon(object_data, closest_pose, envelope_buffer_margin);
  const double error_eclipse_long_radius =
    calcErrorEclipseLongRadius(object_data.object.kinematics.initial_pose_with_covariance);

  if (error_eclipse_long_radius > object_parameter.th_error_eclipse_long_radius) {
    if (error_eclipse_long_radius < same_id_obj->error_eclipse_max) {
      object_data.error_eclipse_max = error_eclipse_long_radius;
      object_data.envelope_poly = one_shot_envelope_poly;
      return;
    }
    object_data.envelope_poly = same_id_obj->envelope_poly;
    return;
  }

  object_data.error_eclipse_max = error_eclipse_long_radius;

  // If the one_shot_envelope_poly is within the registered envelope, use the registered one
  if (boost::geometry::within(one_shot_envelope_poly, same_id_obj->envelope_poly)) {
    object_data.envelope_poly = same_id_obj->envelope_poly;
    return;
  }

  std::vector<Polygon2d> unions;
  boost::geometry::union_(one_shot_envelope_poly, same_id_obj->envelope_poly, unions);

  // If union fails, use the current envelope
  if (unions.empty()) {
    object_data.envelope_poly = one_shot_envelope_poly;
    return;
  }

  boost::geometry::correct(unions.front());

  const auto multi_step_envelope_poly = createEnvelopePolygon(unions.front(), closest_pose, 0.0);

  const auto object_polygon = autoware_utils::to_polygon2d(object_data.object);
  const auto object_polygon_area = boost::geometry::area(object_polygon);
  const auto envelope_polygon_area = boost::geometry::area(multi_step_envelope_poly);

  // keep multi-step envelope polygon.
  constexpr double THRESHOLD = 5.0;
  if (envelope_polygon_area < object_polygon_area * THRESHOLD) {
    object_data.envelope_poly = multi_step_envelope_poly;
    return;
  }

  // use latest one-shot envelope polygon.
  object_data.envelope_poly = one_shot_envelope_poly;
}

// nero camwall: NERO_SOA_UNKNOWN_STILL_CLOCK=<cap_v>,<keep_s> (QUEUE WALL130): a STILL UNKNOWN keeps its stop clock (1) across
// module instances: the registry (stopped_objects_) belongs to the instance and BPP replaces the instance, so a wall became a "new
// object" whose first fast twist sample (a wall track's false speed 0.5-0.9 m/s) gave move_time = inf -> MOVING, else UNSTABLE for 3 s;
// here the clock of an UNKNOWN registered as still is kept per object id process-wide and restored into a new instance (forgotten
// keep_s after it was last seen); (2) against single fast samples: a registered still UNKNOWN reading faster than the moving speed
// but slower than cap_v counts as still until it has been fast for moving_time_threshold IN A ROW (the module's own moving time).
// Faster than cap_v (a rider, a walker, a car) -> stock at once. Unset = the car.
namespace nero_still_clock
{
struct Cfg
{
  bool on{false};
  double cap_v{1.2};
  double keep_s{5.0};
};
const Cfg & cfg()
{
  static const Cfg c = [] {
    Cfg o;
    const char * e = std::getenv("NERO_SOA_UNKNOWN_STILL_CLOCK");
    if (e == nullptr || *e == '\0') return o;
    double a = 0.0, b = 0.0;
    const int n = std::sscanf(e, "%lf,%lf", &a, &b);
    if (n < 1 || !(a > 0.0)) return o;
    o.on = true;
    o.cap_v = a;
    if (n >= 2 && b > 0.0) o.keep_s = b;
    RCLCPP_INFO(
      rclcpp::get_logger("static_obstacle_avoidance"), "[NERO_SOA_UNKNOWN_STILL_CLOCK] on: cap %.2f m/s, kept %.1f s", o.cap_v,
      o.keep_s);
    return o;
  }();
  return c;
}
struct Entry
{
  rclcpp::Time last_move;
  rclcpp::Time last_stop;
  geometry_msgs::msg::Pose init_pose;
  double seen{0.0};
  double fast_since{-1.0};
};
std::mutex g_mtx;
std::map<std::array<std::uint8_t, 16>, Entry> g_table;
}  // namespace nero_still_clock

void fillObjectMovingTime(
  ObjectData & object_data, ObjectDataArray & stopped_objects,
  const std::shared_ptr<AvoidanceParameters> & parameters)
{
  const auto object_type = utils::getHighestProbLabel(object_data.object.classification);
  const auto object_parameter = parameters->object_parameters.at(object_type);

  const auto & object_twist = object_data.object.kinematics.initial_twist_with_covariance.twist;
  const auto object_vel_norm = std::hypot(object_twist.linear.x, object_twist.linear.y);
  const auto raw_faster_than_threshold = object_vel_norm > object_parameter.moving_speed_threshold;

  const auto id = object_data.object.object_id;
  const rclcpp::Time now = rclcpp::Clock(RCL_ROS_TIME).now();

  // nero camwall NERO_SOA_UNKNOWN_STILL_CLOCK: restore a still UNKNOWN's registry entry kept from an earlier instance
  const auto & ncl = nero_still_clock::cfg();
  const bool ncl_on = ncl.on && object_type == ObjectClassification::UNKNOWN;
  std::array<std::uint8_t, 16> ncl_key{};
  std::copy(id.uuid.begin(), id.uuid.end(), ncl_key.begin());
  double ncl_fast_since = -1.0;
  if (ncl_on) {
    std::lock_guard<std::mutex> lk(nero_still_clock::g_mtx);
    auto & T = nero_still_clock::g_table;
    for (auto it = T.begin(); it != T.end();) {
      it = (now.seconds() - it->second.seen > ncl.keep_s) ? T.erase(it) : std::next(it);
    }
    const auto it = T.find(ncl_key);
    if (it != T.end()) {
      ncl_fast_since = it->second.fast_since;
      const bool in_registry = std::any_of(
        stopped_objects.begin(), stopped_objects.end(), [&id](const auto & o) { return o.object.object_id == id; });
      if (!in_registry) {
        ObjectData r = object_data;
        r.last_move = it->second.last_move;
        r.last_stop = it->second.last_stop;
        r.init_pose = it->second.init_pose;
        r.move_time = 0.0;
        r.stop_time = (now - it->second.last_move).seconds();
        stopped_objects.push_back(r);
      }
    }
  }

  const auto same_id_obj = std::find_if(
    stopped_objects.begin(), stopped_objects.end(),
    [&id](const auto & o) { return o.object.object_id == id; });

  const auto is_new_object = same_id_obj == stopped_objects.end();

  // nero camwall NERO_SOA_UNKNOWN_STILL_CLOCK: a registered still UNKNOWN's fast sample below cap_v is noise until it has been fast
  // for the moving time in a row
  if (ncl_on) {
    ncl_fast_since = raw_faster_than_threshold ? (ncl_fast_since < 0.0 ? now.seconds() : ncl_fast_since) : -1.0;
  }
  const auto is_faster_than_threshold =
    raw_faster_than_threshold &&
    !(ncl_on && !is_new_object && object_vel_norm < ncl.cap_v &&
      now.seconds() - ncl_fast_since < object_parameter.moving_time_threshold);
  struct NclOnExit
  {
    bool on;
    const std::array<std::uint8_t, 16> & key;
    const unique_identifier_msgs::msg::UUID & id;
    const ObjectDataArray & reg;
    double now, fast_since;
    ~NclOnExit()
    {
      if (!on) return;
      std::lock_guard<std::mutex> lk(nero_still_clock::g_mtx);
      const auto it = std::find_if(reg.begin(), reg.end(), [this](const auto & o) { return o.object.object_id == id; });
      if (it == reg.end()) {
        nero_still_clock::g_table.erase(key);  // moving (or never still): stock next time
        return;
      }
      auto & e = nero_still_clock::g_table[key];
      e.last_move = it->last_move;
      e.last_stop = it->last_stop;
      e.init_pose = it->init_pose;
      e.seen = now;
      e.fast_since = fast_since;
    }
  } ncl_on_exit{ncl_on, ncl_key, id, stopped_objects, now.seconds(), ncl_fast_since};

  // nero retaa: NERO_SOA_AMBIGUOUS_FRAGMENT (see nero_frag); the table is updated after the decision
  const auto & nf = nero_frag::env();
  struct SeeOnExit
  {
    const ObjectData & o;
    double now, avg_s, moving_v, moving_time;
    bool on;
    ~SeeOnExit()
    {
      if (on) nero_frag::see(o, now, avg_s, moving_v, moving_time);
    }
  } see_on_exit{
    object_data,
    now.seconds(),
    parameters->time_threshold_for_ambiguous_vehicle,
    object_parameter.moving_speed_threshold,
    object_parameter.moving_time_threshold,
    nf.on};
  const bool frag_check = nf.on && !is_faster_than_threshold &&
                          object_type != ObjectClassification::UNKNOWN &&
                          object_type != ObjectClassification::PEDESTRIAN &&
                          object_type != ObjectClassification::BICYCLE;
  // an UNKNOWN that stood where the vehicle stands now: its last place inside the vehicle's box (+ margin), seen within
  // gap_s, standing in the registry (stop clock) and not creeping
  // a creeping car (0.3-1.5 m/s) stays the give-way's / obstacle_stop's: below the module's own 1 m / 3 s and below
  // 0.2 m/s (field standing fragments average < 0.05 m/s; a 0.3 m/s creeper was carried at 1 m / 3 s)
  const double creep_v = std::min(
    0.2, parameters->distance_threshold_for_ambiguous_vehicle /
           std::max(parameters->time_threshold_for_ambiguous_vehicle, 0.1));
  const auto fragmentOk = [&](const nero_frag::Track & tr) {
    return tr.ut >= 0.0 && tr.ustand_t >= 0.0 && now.seconds() - tr.ut <= nf.gap_s &&
           std::hypot(tr.uvx, tr.uvy) < creep_v &&
           nero_frag::insideBox(object_data.object, tr.ux, tr.uy, nf.margin_m);
  };
  const auto pos = object_data.getPosition();
  const auto since = [&](const double t) { return now - rclcpp::Duration::from_seconds(now.seconds() - t); };
  nero_frag::Key key;
  std::copy(id.uuid.begin(), id.uuid.end(), key.begin());

  if (!is_faster_than_threshold) {
    object_data.last_stop = now;
    object_data.move_time = 0.0;
    if (is_new_object) {
      object_data.init_pose = object_data.getPose();
      object_data.stop_time = 0.0;
      object_data.last_move = now;
      if (frag_check) {
        // the same id as a standing UNKNOWN (the module instance's registry is new), else a vehicle track never seen
        // before: the longest-standing UNKNOWN inside its box
        auto & m = nero_frag::tracks();
        const auto self = m.find(key);
        const nero_frag::Track * from = nullptr;
        nero_frag::Key from_key = key;
        if (self != m.end()) {
          if (fragmentOk(self->second)) from = &self->second;
        } else {
          for (const auto & [k, tr] : m) {
            if (fragmentOk(tr) && (from == nullptr || tr.ustand_t < from->ustand_t)) {
              from = &tr;
              from_key = k;
            }
          }
        }
        if (from != nullptr) {
          m[key].carried = true;
          object_data.last_move = since(from->ustand_t);
          object_data.stop_time = (now - object_data.last_move).seconds();
          unique_identifier_msgs::msg::UUID from_id;
          std::copy(from_key.begin(), from_key.end(), from_id.uuid.begin());
          RCLCPP_INFO(
            rclcpp::get_logger(logger_namespace),
            "[NERO_SOA_FRAG] %s %s (label %u) carries the standing history of UNKNOWN %s: its last place %.2f m "
            "from the vehicle's centre, standing %.1f s, moved distance from here",
            self != m.end() ? "same id" : "new id", nero_frag::hexId(id).c_str(), object_type,
            nero_frag::hexId(from_id).c_str(), std::hypot(from->ux - pos.x, from->uy - pos.y),
            object_data.stop_time);
        }
      }
      stopped_objects.push_back(object_data);
    } else {
      if (
        frag_check && utils::getHighestProbLabel(same_id_obj->object.classification) ==
                        ObjectClassification::UNKNOWN) {
        // registered as an UNKNOWN: the moved distance from the vehicle's first place
        const auto self = nero_frag::tracks().find(key);
        if (self != nero_frag::tracks().end() && fragmentOk(self->second)) {
          const auto & tr = self->second;
          if (since(tr.ustand_t) < same_id_obj->last_move) same_id_obj->last_move = since(tr.ustand_t);
          RCLCPP_INFO(
            rclcpp::get_logger(logger_namespace),
            "[NERO_SOA_FRAG] same id %s UNKNOWN -> label %u carries its standing history: moved distance from the "
            "vehicle's place (the fragment's first place was %.2f m away, its last %.2f m), standing %.1f s",
            nero_frag::hexId(id).c_str(), object_type,
            calc_distance2d(same_id_obj->init_pose, object_data.getPose()),
            std::hypot(tr.ux - pos.x, tr.uy - pos.y), (now - same_id_obj->last_move).seconds());
          same_id_obj->init_pose = object_data.getPose();
          nero_frag::tracks()[key].carried = true;
        }
        same_id_obj->object.classification = object_data.object.classification;
      }
      same_id_obj->stop_time = (now - same_id_obj->last_move).seconds();
      same_id_obj->last_stop = now;
      same_id_obj->move_time = 0.0;
      object_data.stop_time = same_id_obj->stop_time;
      object_data.init_pose = same_id_obj->init_pose;
    }
    return;
  }

  if (is_new_object) {
    object_data.init_pose = object_data.getPose();
    object_data.move_time = std::numeric_limits<double>::infinity();
    object_data.stop_time = 0.0;
    object_data.last_move = now;
    return;
  }

  object_data.last_stop = same_id_obj->last_stop;
  object_data.move_time = (now - same_id_obj->last_stop).seconds();
  object_data.stop_time = 0.0;
  object_data.init_pose = object_data.getPose();

  if (object_data.move_time > object_parameter.moving_time_threshold) {
    stopped_objects.erase(same_id_obj);
  }
}

void fillAvoidanceNecessity(
  ObjectData & object_data, const ObjectDataArray & registered_objects, const double vehicle_width,
  const std::shared_ptr<AvoidanceParameters> & parameters)
{
  const auto object_type = utils::getHighestProbLabel(object_data.object.classification);
  const auto object_parameter = parameters->object_parameters.at(object_type);
  const auto lateral_hard_margin = object_data.is_parked
                                     ? object_parameter.lateral_hard_margin_for_parked_vehicle
                                     : object_parameter.lateral_hard_margin;
  const auto safety_margin =
    0.5 * vehicle_width + lateral_hard_margin * object_data.distance_factor;

  const auto check_necessity = [&](const auto hysteresis_factor) {
    return (isOnRight(object_data) && std::abs(object_data.overhang_points.front().first) <
                                        safety_margin * hysteresis_factor) ||
           (!isOnRight(object_data) &&
            object_data.overhang_points.front().first < safety_margin * hysteresis_factor);
  };

  const auto id = object_data.object.object_id;
  const auto same_id_obj = std::find_if(
    registered_objects.begin(), registered_objects.end(),
    [&id](const auto & o) { return o.object.object_id == id; });

  // First time
  if (same_id_obj == registered_objects.end()) {
    object_data.avoid_required = check_necessity(1.0);
    return;
  }

  // FALSE -> FALSE or FALSE -> TRUE
  if (!same_id_obj->avoid_required) {
    object_data.avoid_required = check_necessity(1.0);
    return;
  }

  // TRUE -> ? (check with hysteresis factor)
  object_data.avoid_required = check_necessity(parameters->hysteresis_factor_expand_rate);
}

void fillObjectStoppableJudge(
  ObjectData & object_data, const ObjectDataArray & registered_objects,
  const double feasible_stop_distance, const std::shared_ptr<AvoidanceParameters> & parameters)
{
  if (parameters->policy_deceleration == "reliable") {
    object_data.is_stoppable = true;
    return;
  }

  if (!object_data.avoid_required) {
    object_data.is_stoppable = false;
    return;
  }

  const auto id = object_data.object.object_id;
  const auto same_id_obj = std::find_if(
    registered_objects.begin(), registered_objects.end(),
    [&id](const auto & o) { return o.object.object_id == id; });

  const auto is_stoppable = object_data.to_stop_line > feasible_stop_distance;
  if (is_stoppable) {
    object_data.is_stoppable = true;
    return;
  }

  if (same_id_obj == registered_objects.end()) {
    object_data.is_stoppable = false;
    return;
  }

  object_data.is_stoppable = same_id_obj->is_stoppable;
}

void compensateLostTargetObjects(
  ObjectDataArray & stored_objects, AvoidancePlanningData & data, const rclcpp::Time & now,
  const std::shared_ptr<const PlannerData> & planner_data,
  const std::shared_ptr<AvoidanceParameters> & parameters)
{
  const auto include = [](const auto & objects, const auto & search_id) {
    return std::any_of(objects.begin(), objects.end(), [&search_id](const auto & o) {
      return o.object.object_id == search_id;
    });
  };

  // STEP.1 UPDATE STORED OBJECTS.
  const auto match = [&data](auto & object) {
    const auto & search_id = object.object.object_id;
    const auto same_id_object = std::find_if(
      data.target_objects.begin(), data.target_objects.end(),
      [&search_id](const auto & o) { return o.object.object_id == search_id; });

    // same id object is detected. update registered.
    if (same_id_object != data.target_objects.end()) {
      object = *same_id_object;
      return true;
    }

    const auto similar_pos_obj = std::find_if(
      data.target_objects.begin(), data.target_objects.end(), [&object](const auto & o) {
        constexpr auto POS_THR = 1.5;
        return calc_distance2d(object.getPose(), o.getPose()) < POS_THR;
      });

    // same id object is not detected, but object is found around registered. update registered.
    if (similar_pos_obj != data.target_objects.end()) {
      object = *similar_pos_obj;
      return true;
    }

    // Same ID nor similar position object does not found.
    return false;
  };

  // STEP1-1: REMOVE EXPIRED OBJECTS.
  const auto itr = std::remove_if(
    stored_objects.begin(), stored_objects.end(), [&now, &match, &parameters](auto & o) {
      if (!match(o)) {
        o.lost_time = (now - o.last_seen).seconds();
      } else {
        o.last_seen = now;
        o.lost_time = 0.0;
      }

      return o.lost_time > parameters->object_last_seen_threshold;
    });

  stored_objects.erase(itr, stored_objects.end());

  // STEP1-2: UPDATE STORED OBJECTS IF THERE ARE NEW OBJECTS.
  for (const auto & current_object : data.target_objects) {
    if (!include(stored_objects, current_object.object.object_id)) {
      stored_objects.push_back(current_object);
    }
  }

  // STEP2: COMPENSATE CURRENT TARGET OBJECTS
  const auto is_detected = [&](const auto & object_id) {
    return std::any_of(
      data.target_objects.begin(), data.target_objects.end(),
      [&object_id](const auto & o) { return o.object.object_id == object_id; });
  };

  const auto is_ignored = [&](const auto & object_id) {
    return std::any_of(
      data.other_objects.begin(), data.other_objects.end(),
      [&object_id](const auto & o) { return o.object.object_id == object_id; });
  };

  for (auto & stored_object : stored_objects) {
    if (is_detected(stored_object.object.object_id)) {
      continue;
    }
    if (is_ignored(stored_object.object.object_id)) {
      continue;
    }

    const auto & ego_pos = planner_data->self_odometry->pose.pose.position;
    fillLongitudinalAndLengthByClosestEnvelopeFootprint(
      data.reference_path_rough, ego_pos, stored_object);

    data.target_objects.push_back(stored_object);
  }
}

void updateClipObject(ObjectDataArray & clip_objects, AvoidancePlanningData & data)
{
  std::for_each(data.target_objects.begin(), data.target_objects.end(), [](auto & o) {
    if (o.is_avoidable) {
      o.is_clip_target = true;
    }
  });

  const auto itr =
    std::remove_if(clip_objects.begin(), clip_objects.end(), [&data](const auto & clip_object) {
      const auto id = clip_object.object.object_id;

      // update target objects
      {
        const auto same_id_obj = std::find_if(
          data.target_objects.begin(), data.target_objects.end(),
          [&id](const auto & o) { return o.object.object_id == id; });
        if (same_id_obj != data.target_objects.end()) {
          same_id_obj->is_clip_target = true;
          return false;
        }
      }

      // update other objects
      {
        const auto same_id_obj = std::find_if(
          data.other_objects.begin(), data.other_objects.end(),
          [&id](const auto & o) { return o.object.object_id == id; });
        if (same_id_obj != data.other_objects.end()) {
          same_id_obj->is_clip_target = true;
          return false;
        }
      }

      return true;
    });

  clip_objects.erase(itr, clip_objects.end());

  for (const auto & object : data.target_objects) {
    const auto id = object.object.object_id;
    const auto same_id_obj = std::find_if(
      clip_objects.begin(), clip_objects.end(),
      [&id](const auto & o) { return o.object.object_id == id; });

    if (same_id_obj != clip_objects.end()) {
      continue;
    }

    clip_objects.push_back(object);
  }
}

void updateRoadShoulderDistance(
  AvoidancePlanningData & data, const std::shared_ptr<const PlannerData> & planner_data,
  const std::shared_ptr<AvoidanceParameters> & parameters)
{
  ObjectDataArray clip_objects;
  std::for_each(data.other_objects.begin(), data.other_objects.end(), [&](const auto & object) {
    if (!filtering_utils::isMovingObject(object, parameters)) {
      clip_objects.push_back(object);
    }
  });

  if (clip_objects.empty()) return;

  for (auto & o : clip_objects) {
    const auto & vehicle_width = planner_data->parameters.vehicle_width;
    const auto object_type = utils::getHighestProbLabel(o.object.classification);
    const auto object_parameter = parameters->object_parameters.at(object_type);
    const auto lateral_hard_margin = o.is_parked
                                       ? object_parameter.lateral_hard_margin_for_parked_vehicle
                                       : object_parameter.lateral_hard_margin;

    o.avoid_margin = lateral_hard_margin + 0.5 * vehicle_width;
  }
  const auto extract_obstacles = generateObstaclePolygonsForDrivableArea(
    clip_objects, parameters, planner_data->parameters.vehicle_width / 2.0);

  auto tmp_path = data.reference_path;
  tmp_path.left_bound = data.left_bound;
  tmp_path.right_bound = data.right_bound;
  utils::extractObstaclesFromDrivableArea(tmp_path, extract_obstacles);

  data.left_bound = tmp_path.left_bound;
  data.right_bound = tmp_path.right_bound;

  for (auto & o : data.target_objects) {
    o.to_road_shoulder_distance = filtering_utils::getRoadShoulderDistance(o, data, planner_data);
    o.avoid_margin = filtering_utils::getAvoidMargin(o, planner_data, parameters);
  }
}

void filterTargetObjects(
  ObjectDataArray & objects, AvoidancePlanningData & data, const double forward_detection_range,
  const std::shared_ptr<const PlannerData> & planner_data,
  const std::shared_ptr<AvoidanceParameters> & parameters)
{
  if (data.current_lanelets.empty()) {
    return;
  }

  const rclcpp::Time now = rclcpp::Clock(RCL_ROS_TIME).now();
  const auto push_target_object = [&data, &now](auto & object) {
    object.last_seen = now;
    data.target_objects.push_back(object);
  };

  const auto & rh = planner_data->route_handler;
  const auto ego_idx = planner_data->findEgoIndex(data.reference_path_rough.points);
  const auto to_goal_distance =
    rh->isInGoalRouteSection(data.current_lanelets.back())
      ? nero_mu::calcSignedArcLength(
          data.reference_path_rough.points, ego_idx, data.reference_path_rough.points.size() - 1)
      : std::numeric_limits<double>::max();

  for (auto & o : objects) {
    if (!filtering_utils::isSatisfiedWithCommonCondition(
          o, data.reference_path_rough, forward_detection_range, to_goal_distance,
          planner_data->self_odometry->pose.pose.position, data.is_allowed_goal_modification,
          parameters)) {
      data.other_objects.push_back(o);
      continue;
    }

    // Find the footprint point closest to the path, set to object_data.overhang_distance.
    o.overhang_points =
      utils::static_obstacle_avoidance::calcEnvelopeOverhangDistance(o, data.reference_path);
    o.to_road_shoulder_distance = filtering_utils::getRoadShoulderDistance(o, data, planner_data);

    if (filtering_utils::isUnknownTypeObject(o)) {
      // TARGET: UNKNOWN

      // TODO(Satoshi Ota) parametrize stop time threshold if need.
      constexpr double STOP_TIME_THRESHOLD = 3.0;  // [s]
      const bool stable_by_place =
        nero_filter::env().unknown_stable_s > 0.0 &&
        nero_filter::isStableByPosition(o.getPosition(), now.seconds(), nero_filter::env().unknown_stable_s);
      if (o.stop_time < STOP_TIME_THRESHOLD && !stable_by_place) {
        o.info = ObjectInfo::UNSTABLE_OBJECT;
        data.other_objects.push_back(o);
        continue;
      }
      o.avoid_margin = filtering_utils::getAvoidMargin(o, planner_data, parameters);
      if (nero_filter::env().side_by_space && !o.avoid_margin.has_value()) {
        o.is_on_ego_lane = filtering_utils::isOnEgoLane(o, planner_data->route_handler);
        if (o.is_on_ego_lane) {
          auto flipped = o;
          flipped.direction = isOnRight(o) ? Direction::LEFT : Direction::RIGHT;
          flipped.overhang_points =
            utils::static_obstacle_avoidance::calcEnvelopeOverhangDistance(flipped, data.reference_path);
          flipped.to_road_shoulder_distance =
            filtering_utils::getRoadShoulderDistance(flipped, data, planner_data);
          flipped.avoid_margin = filtering_utils::getAvoidMargin(flipped, planner_data, parameters);
          if (flipped.avoid_margin.has_value()) o = flipped;
        }
      }
    } else if (filtering_utils::isVehicleTypeObject(o)) {
      // TARGET: CAR, TRUCK, BUS, TRAILER, MOTORCYCLE

      o.is_within_intersection =
        filtering_utils::isWithinIntersection(o, planner_data->route_handler);
      o.is_parked =
        filtering_utils::isParkedVehicle(o, data, planner_data->route_handler, parameters);
      o.avoid_margin = filtering_utils::getAvoidMargin(o, planner_data, parameters);

      // nero retaa: NERO_SOA_AMBIGUOUS_DEBUG=1 (logging only)
      const auto amb_debug = [&](const bool target) {
        if (!nero_frag::env().debug) return;
        const auto & tw = o.object.kinematics.initial_twist_with_covariance.twist;
        RCLCPP_INFO(
          rclcpp::get_logger(logger_namespace),
          "[NERO_SOA_AMB] id %s label %u v %.2f stop %.2f moved %.2f parked %d ratio %.2f on_ego_lane %d "
          "behavior %d long %.1f overhang %.2f shoulder %.2f margin %.2f info %s target %d",
          nero_frag::hexId(o.object.object_id).c_str(),
          utils::getHighestProbLabel(o.object.classification), std::hypot(tw.linear.x, tw.linear.y),
          o.stop_time, calc_distance2d(o.init_pose, o.getPose()), o.is_parked, o.shiftable_ratio,
          o.is_on_ego_lane, static_cast<int>(o.behavior), o.longitudinal,
          o.overhang_points.empty() ? 0.0 : o.overhang_points.front().first, o.to_road_shoulder_distance,
          o.avoid_margin.value_or(-1.0), magic_enum::enum_name(o.info).data(), target);
      };

      if (filtering_utils::isNoNeedAvoidanceBehavior(o, parameters)) {
        amb_debug(false);
        data.other_objects.push_back(o);
        continue;
      }

      const bool vehicle_condition =
        filtering_utils::isSatisfiedWithVehicleCondition(o, data, planner_data, parameters);
      amb_debug(vehicle_condition);
      if (!vehicle_condition) {
        data.other_objects.push_back(o);
        continue;
      }

      if (nero_filter::env().side_by_space && !o.avoid_margin.has_value() && o.is_on_ego_lane) {
        auto flipped = o;
        flipped.direction = isOnRight(o) ? Direction::LEFT : Direction::RIGHT;
        flipped.overhang_points =
          utils::static_obstacle_avoidance::calcEnvelopeOverhangDistance(flipped, data.reference_path);
        flipped.to_road_shoulder_distance =
          filtering_utils::getRoadShoulderDistance(flipped, data, planner_data);
        flipped.avoid_margin = filtering_utils::getAvoidMargin(flipped, planner_data, parameters);
        if (flipped.avoid_margin.has_value()) {
          static rclcpp::Clock steady_clock(RCL_STEADY_TIME);
          RCLCPP_INFO_THROTTLE(
            rclcpp::get_logger(logger_namespace), steady_clock, 3000,
            "[NERO_SOA] no room on the %s of the stopped vehicle: passing it on the %s",
            isOnRight(o) ? "left" : "right", isOnRight(o) ? "right" : "left");
          o = flipped;
        }
      }
    } else {
      // TARGET: PEDESTRIAN, BICYCLE

      o.is_within_intersection =
        filtering_utils::isWithinIntersection(o, planner_data->route_handler);
      o.is_parked = false;
      o.avoid_margin = filtering_utils::getAvoidMargin(o, planner_data, parameters);

      if (filtering_utils::isNoNeedAvoidanceBehavior(o, parameters)) {
        data.other_objects.push_back(o);
        continue;
      }

      if (!filtering_utils::isSatisfiedWithNonVehicleCondition(o, data, planner_data, parameters)) {
        data.other_objects.push_back(o);
        continue;
      }
    }

    push_target_object(o);
  }
}

AvoidLine fillAdditionalInfo(const AvoidancePlanningData & data, const AvoidLine & line)
{
  AvoidLineArray ret{line};
  fillAdditionalInfoFromPoint(data, ret);
  return ret.front();
}

void fillAdditionalInfoFromPoint(const AvoidancePlanningData & data, AvoidLineArray & lines)
{
  if (lines.empty()) {
    return;
  }

  const auto & path = data.reference_path;
  const auto & arc = data.arclength_from_ego;

  // calc longitudinal
  for (auto & sl : lines) {
    sl.start_idx = autoware::motion_utils::findNearestIndex(path.points, sl.start.position);
    sl.start_longitudinal = arc.at(sl.start_idx);
    sl.end_idx = autoware::motion_utils::findNearestIndex(path.points, sl.end.position);
    sl.end_longitudinal = arc.at(sl.end_idx);
  }
}

void fillAdditionalInfoFromLongitudinal(const AvoidancePlanningData & data, AvoidLine & line)
{
  const auto & path = data.reference_path;
  const auto & arc = data.arclength_from_ego;

  line.start_idx = findPathIndexFromArclength(arc, line.start_longitudinal);
  line.start = path.points.at(line.start_idx).point.pose;
  line.end_idx = findPathIndexFromArclength(arc, line.end_longitudinal);
  line.end = path.points.at(line.end_idx).point.pose;
}

void fillAdditionalInfoFromLongitudinal(
  const AvoidancePlanningData & data, AvoidOutlines & outlines)
{
  for (auto & outline : outlines) {
    fillAdditionalInfoFromLongitudinal(data, outline.avoid_line);
    if (outline.return_line.has_value()) {
      fillAdditionalInfoFromLongitudinal(data, outline.return_line.value());
    }

    std::for_each(outline.middle_lines.begin(), outline.middle_lines.end(), [&](auto & line) {
      fillAdditionalInfoFromLongitudinal(data, line);
    });
  }
}

void fillAdditionalInfoFromLongitudinal(const AvoidancePlanningData & data, AvoidLineArray & lines)
{
  const auto & path = data.reference_path;
  const auto & arc = data.arclength_from_ego;

  for (auto & sl : lines) {
    sl.start_idx = findPathIndexFromArclength(arc, sl.start_longitudinal);
    sl.start = path.points.at(sl.start_idx).point.pose;
    sl.end_idx = findPathIndexFromArclength(arc, sl.end_longitudinal);
    sl.end = path.points.at(sl.end_idx).point.pose;
  }
}

AvoidLineArray combineRawShiftLinesWithUniqueCheck(
  const AvoidLineArray & base_lines, const AvoidLineArray & added_lines)
{
  // TODO(Horibe) parametrize
  const auto isSimilar = [](const AvoidLine & a, const AvoidLine & b) {
    using autoware_utils::calc_distance2d;
    if (calc_distance2d(a.start, b.start) > 1.0) {
      return false;
    }
    if (calc_distance2d(a.end, b.end) > 1.0) {
      return false;
    }
    if (std::abs(a.end_shift_length - b.end_shift_length) > 0.5) {
      return false;
    }
    return true;
  };
  const auto hasSameObjectId = [](const auto & a, const auto & b) {
    return a.object.object.object_id == b.object.object.object_id;
  };

  auto combined = base_lines;  // initialized
  for (const auto & added_line : added_lines) {
    bool skip = false;

    for (const auto & base_line : base_lines) {
      if (hasSameObjectId(added_line, base_line) && isSimilar(added_line, base_line)) {
        skip = true;
        break;
      }
    }
    if (!skip) {
      combined.push_back(added_line);
    }
  }

  return combined;
}

lanelet::ConstLanelets getAdjacentLane(
  const lanelet::ConstLanelet & current_lane,
  const std::shared_ptr<const PlannerData> & planner_data,
  const std::shared_ptr<AvoidanceParameters> & parameters, const bool is_right_shift)
{
  const auto & rh = planner_data->route_handler;
  const auto & forward_distance = parameters->object_check_max_forward_distance;
  const auto & backward_distance = parameters->safety_check_backward_distance;
  const auto & vehicle_pose = planner_data->self_odometry->pose.pose;

  const auto ego_succeeding_lanes =
    rh->getLaneletSequence(current_lane, vehicle_pose, backward_distance, forward_distance);

  lanelet::ConstLanelets lanes{};

  const auto exist = [&lanes](const auto id) {
    const auto itr = std::find_if(
      lanes.begin(), lanes.end(), [&id](const auto & lane) { return lane.id() == id; });
    return itr != lanes.end();
  };

  for (const auto & lane : ego_succeeding_lanes) {
    const auto opt_left_lane = rh->getLeftLanelet(lane, true, false);
    if (!is_right_shift && opt_left_lane) {
      lanes.push_back(opt_left_lane.value());
    }

    const auto opt_right_lane = rh->getRightLanelet(lane, true, false);
    if (is_right_shift && opt_right_lane) {
      lanes.push_back(opt_right_lane.value());
    }

    const auto left_opposite_lanes = rh->getLeftOppositeLanelets(lane);
    if (!is_right_shift && !left_opposite_lanes.empty()) {
      lanes.push_back(left_opposite_lanes.front());

      for (const auto & prev_lane : rh->getPreviousLanelets(left_opposite_lanes.front())) {
        if (!exist(prev_lane.id())) {
          lanes.push_back(prev_lane);
        }
      }
    }

    const auto right_opposite_lanes = rh->getRightOppositeLanelets(lane);
    if (is_right_shift && !right_opposite_lanes.empty()) {
      lanes.push_back(right_opposite_lanes.front());

      for (const auto & prev_lane : rh->getPreviousLanelets(right_opposite_lanes.front())) {
        if (!exist(prev_lane.id())) {
          lanes.push_back(prev_lane);
        }
      }
    }
  }

  for (const auto & lane : lanes) {
    for (const auto & next_lane : rh->getNextLanelets(lane)) {
      if (!exist(next_lane.id())) {
        lanes.push_back(next_lane);
      }
    }
  }

  return lanes;
}

std::vector<ExtendedPredictedObject> getSafetyCheckTargetObjects(
  const AvoidancePlanningData & data, const std::shared_ptr<const PlannerData> & planner_data,
  const std::shared_ptr<AvoidanceParameters> & parameters, const bool has_left_shift,
  const bool has_right_shift, DebugData & debug)
{
  const auto & p = parameters;
  const auto check_right_lanes =
    (has_right_shift && p->check_shift_side_lane) || (has_left_shift && p->check_other_side_lane);
  const auto check_left_lanes =
    (has_left_shift && p->check_shift_side_lane) || (has_right_shift && p->check_other_side_lane);

  std::vector<ExtendedPredictedObject> target_objects;

  const auto time_horizon = std::max(
    parameters->ego_predicted_path_params.time_horizon_for_front_object,
    parameters->ego_predicted_path_params.time_horizon_for_rear_object);

  const auto append = [&](const auto & objects) {
    std::for_each(objects.objects.begin(), objects.objects.end(), [&](const auto & object) {
      target_objects.push_back(utils::path_safety_checker::transform(
        object, time_horizon, parameters->ego_predicted_path_params.time_resolution));
    });
  };

  const auto to_predicted_objects = [&parameters](const auto & objects) {
    PredictedObjects ret{};
    std::for_each(objects.begin(), objects.end(), [&ret, &parameters](const auto & object) {
      if (filtering_utils::isSafetyCheckTargetObjectType(object.object, parameters)) {
        // check only moving objects
        if (filtering_utils::isMovingObject(object, parameters) || !object.is_parked) {
          ret.objects.push_back(object.object);
        }
      }
    });
    return ret;
  };

  const auto unavoidable_objects = [&data]() {
    ObjectDataArray ret;
    std::for_each(data.target_objects.begin(), data.target_objects.end(), [&](const auto & object) {
      if (!object.is_avoidable) {
        ret.push_back(object);
      }
    });
    return ret;
  }();

  lanelet::ConstLanelet closest_lanelet;
  const auto & ego_pose = planner_data->self_odometry->pose.pose;
  if (!nero_geom_fast::getClosestLanelet(
        data.current_lanelets, ego_pose, &closest_lanelet)) {
    return {};
  }

  const auto is_moving = [&parameters](const auto & object) {
    const auto & object_twist = object.kinematics.initial_twist_with_covariance.twist;
    const auto object_vel_norm = std::hypot(object_twist.linear.x, object_twist.linear.y);
    const auto object_parameter =
      parameters->object_parameters.at(utils::getHighestProbLabel(object.classification));
    return object_vel_norm > object_parameter.moving_speed_threshold;
  };

  const auto filter =
    [&is_moving](const auto & object, const auto & lanelet, [[maybe_unused]] const auto unused) {
      // filter by yaw deviation only when the object is moving because the head direction is not
      // reliable while object is stopping.
      const auto yaw_threshold = is_moving(object) ? M_PI_2 : M_PI;
      return utils::path_safety_checker::isCentroidWithinLanelet(object, lanelet, yaw_threshold);
    };

  // check right lanes
  if (check_right_lanes) {
    const auto check_lanes = getAdjacentLane(closest_lanelet, planner_data, p, true);

    if (p->check_other_object) {
      const auto [targets, others] = utils::path_safety_checker::separateObjectsByLanelets(
        to_predicted_objects(data.other_objects), check_lanes, filter);
      append(targets);
    }

    if (p->check_unavoidable_object) {
      const auto [targets, others] = utils::path_safety_checker::separateObjectsByLanelets(
        to_predicted_objects(unavoidable_objects), check_lanes, filter);
      append(targets);
    }

    debug.safety_check_lanes.insert(
      debug.safety_check_lanes.end(), check_lanes.begin(), check_lanes.end());
  }

  // check left lanes
  if (check_left_lanes) {
    const auto check_lanes = getAdjacentLane(closest_lanelet, planner_data, p, false);

    if (p->check_other_object) {
      const auto [targets, others] = utils::path_safety_checker::separateObjectsByLanelets(
        to_predicted_objects(data.other_objects), check_lanes, filter);
      append(targets);
    }

    if (p->check_unavoidable_object) {
      const auto [targets, others] = utils::path_safety_checker::separateObjectsByLanelets(
        to_predicted_objects(unavoidable_objects), check_lanes, filter);
      append(targets);
    }

    debug.safety_check_lanes.insert(
      debug.safety_check_lanes.end(), check_lanes.begin(), check_lanes.end());
  }

  // check current lanes
  if (p->check_current_lane) {
    const auto check_lanes = data.current_lanelets;

    if (p->check_other_object) {
      const auto [targets, others] = utils::path_safety_checker::separateObjectsByLanelets(
        to_predicted_objects(data.other_objects), check_lanes, filter);
      append(targets);
    }

    if (p->check_unavoidable_object) {
      const auto [targets, others] = utils::path_safety_checker::separateObjectsByLanelets(
        to_predicted_objects(unavoidable_objects), check_lanes, filter);
      append(targets);
    }

    debug.safety_check_lanes.insert(
      debug.safety_check_lanes.end(), check_lanes.begin(), check_lanes.end());
  }

  return target_objects;
}

std::pair<PredictedObjects, PredictedObjects> separateObjectsByPath(
  const PathWithLaneId & reference_path, const PathWithLaneId & spline_path,
  const std::shared_ptr<const PlannerData> & planner_data, const AvoidancePlanningData & data,
  const std::shared_ptr<AvoidanceParameters> & parameters,
  const double object_check_forward_distance, DebugData & debug)
{
  PredictedObjects target_objects;
  PredictedObjects other_objects;

  if (reference_path.points.empty() || spline_path.points.empty()) {
    return std::make_pair(target_objects, other_objects);
  }

  double max_offset = 0.0;
  for (const auto & object_parameter : parameters->object_parameters) {
    const auto p = object_parameter.second;
    const auto lateral_hard_margin =
      std::max(p.lateral_hard_margin, p.lateral_hard_margin_for_parked_vehicle);
    const auto offset =
      2.0 * p.envelope_buffer_margin + lateral_hard_margin + p.lateral_soft_margin;
    max_offset = std::max(max_offset, offset);
  }

  const auto detection_area =
    createVehiclePolygon(planner_data->parameters.vehicle_info, max_offset);
  const auto ego_idx = planner_data->findEgoIndex(reference_path.points);
  const auto arc_length_array =
    utils::calcPathArcLengthArray(reference_path, 0L, reference_path.points.size(), 0.0);

  const auto points_size = std::min(reference_path.points.size(), spline_path.points.size());

  std::vector<Polygon2d> detection_areas;
  Pose p_reference_ego_front = reference_path.points.front().point.pose;
  Pose p_spline_ego_front = spline_path.points.front().point.pose;
  double next_longitudinal_distance = parameters->resample_interval_for_output;
  const auto offset = arc_length_array.at(ego_idx);
  for (size_t i = 0; i < points_size; ++i) {
    if (arc_length_array.at(i) > object_check_forward_distance + offset) {
      break;
    }

    if (arc_length_array.at(i) < next_longitudinal_distance) {
      continue;
    }

    const auto & p_reference_ego_back = reference_path.points.at(i).point.pose;
    const auto & p_spline_ego_back = spline_path.points.at(i).point.pose;

    detection_areas.push_back(createOneStepPolygon(
      p_reference_ego_front, p_reference_ego_back, p_spline_ego_front, p_spline_ego_back,
      detection_area));

    p_reference_ego_front = p_reference_ego_back;
    p_spline_ego_front = p_spline_ego_back;

    next_longitudinal_distance += parameters->resample_interval_for_output;
  }

  std::for_each(detection_areas.begin(), detection_areas.end(), [&](const auto & detection_area) {
    debug.detection_areas.push_back(toMsg(detection_area, data.reference_pose.position.z));
  });

  const auto within_detection_area = [&](const auto & obj_polygon) {
    for (const auto & detection_area : detection_areas) {
      if (!boost::geometry::disjoint(obj_polygon, detection_area)) {
        return true;
      }
    }

    return false;
  };

  const auto objects = planner_data->dynamic_object->objects;
  std::for_each(objects.begin(), objects.end(), [&](const auto & object) {
    const auto obj_polygon = autoware_utils::to_polygon2d(object);
    if (!within_detection_area(obj_polygon)) {
      other_objects.objects.push_back(object);
    } else {
      target_objects.objects.push_back(object);
    }
  });

  return std::make_pair(target_objects, other_objects);
}

DrivableLanes generateNotExpandedDrivableLanes(const lanelet::ConstLanelet & lanelet)
{
  DrivableLanes current_drivable_lanes;
  current_drivable_lanes.left_lane = lanelet;
  current_drivable_lanes.right_lane = lanelet;

  return current_drivable_lanes;
}

DrivableLanes generateExpandedDrivableLanes(
  const lanelet::ConstLanelet & lanelet, const std::shared_ptr<const PlannerData> & planner_data,
  const std::shared_ptr<AvoidanceParameters> & parameters)
{
  const auto & route_handler = planner_data->route_handler;

  DrivableLanes current_drivable_lanes;
  current_drivable_lanes.left_lane = lanelet;
  current_drivable_lanes.right_lane = lanelet;

  if (parameters->use_lane_type == "current_lane") {
    return current_drivable_lanes;
  }

  const auto use_opposite_lane = parameters->use_lane_type == "opposite_direction_lane";

  // 1. get left/right side lanes
  const auto update_left_lanelets = [&](const lanelet::ConstLanelet & target_lane) {
    const auto all_left_lanelets =
      route_handler->getAllLeftSharedLinestringLanelets(target_lane, use_opposite_lane, true);
    if (!all_left_lanelets.empty()) {
      current_drivable_lanes.left_lane = all_left_lanelets.back();  // leftmost lanelet
      pushUniqueVector(
        current_drivable_lanes.middle_lanes,
        lanelet::ConstLanelets(all_left_lanelets.begin(), all_left_lanelets.end() - 1));
    }
  };
  const auto update_right_lanelets = [&](const lanelet::ConstLanelet & target_lane) {
    const auto all_right_lanelets =
      route_handler->getAllRightSharedLinestringLanelets(target_lane, use_opposite_lane, true);
    if (!all_right_lanelets.empty()) {
      current_drivable_lanes.right_lane = all_right_lanelets.back();  // rightmost lanelet
      pushUniqueVector(
        current_drivable_lanes.middle_lanes,
        lanelet::ConstLanelets(all_right_lanelets.begin(), all_right_lanelets.end() - 1));
    }
  };

  update_left_lanelets(lanelet);
  update_right_lanelets(lanelet);

  // 2.1 when there are multiple lanes whose previous lanelet is the same
  const auto get_next_lanes_from_same_previous_lane =
    [&route_handler](const lanelet::ConstLanelet & lane) {
      // get previous lane, and return false if previous lane does not exist
      lanelet::ConstLanelets prev_lanes;
      if (!route_handler->getPreviousLaneletsWithinRoute(lane, &prev_lanes)) {
        return lanelet::ConstLanelets{};
      }

      lanelet::ConstLanelets next_lanes;
      for (const auto & prev_lane : prev_lanes) {
        const auto next_lanes_from_prev = route_handler->getNextLanelets(prev_lane);
        pushUniqueVector(next_lanes, next_lanes_from_prev);
      }
      return next_lanes;
    };

  const auto next_lanes_for_right =
    get_next_lanes_from_same_previous_lane(current_drivable_lanes.right_lane);
  const auto next_lanes_for_left =
    get_next_lanes_from_same_previous_lane(current_drivable_lanes.left_lane);

  // 2.2 look for neighbor lane recursively, where end line of the lane is connected to end line
  // of the original lane
  const auto update_drivable_lanes =
    [&](const lanelet::ConstLanelets & next_lanes, const bool is_left) {
      for (const auto & next_lane : next_lanes) {
        const auto & edge_lane =
          is_left ? current_drivable_lanes.left_lane : current_drivable_lanes.right_lane;
        if (next_lane.id() == edge_lane.id()) {
          continue;
        }

        const auto & left_lane = is_left ? next_lane : edge_lane;
        const auto & right_lane = is_left ? edge_lane : next_lane;
        if (!isEndPointsConnected(left_lane, right_lane)) {
          continue;
        }

        if (is_left) {
          current_drivable_lanes.left_lane = next_lane;
        } else {
          current_drivable_lanes.right_lane = next_lane;
        }

        const auto & middle_lanes = current_drivable_lanes.middle_lanes;
        const auto has_same_lane = std::any_of(
          middle_lanes.begin(), middle_lanes.end(),
          [&edge_lane](const auto & lane) { return lane.id() == edge_lane.id(); });

        if (!has_same_lane) {
          if (is_left) {
            if (current_drivable_lanes.right_lane.id() != edge_lane.id()) {
              current_drivable_lanes.middle_lanes.push_back(edge_lane);
            }
          } else {
            if (current_drivable_lanes.left_lane.id() != edge_lane.id()) {
              current_drivable_lanes.middle_lanes.push_back(edge_lane);
            }
          }
        }

        return true;
      }
      return false;
    };

  const auto expand_drivable_area_recursively =
    [&](const lanelet::ConstLanelets & next_lanes, const bool is_left) {
      // NOTE: set max search num to avoid infinity loop for drivable area expansion
      constexpr size_t max_recursive_search_num = 3;
      for (size_t i = 0; i < max_recursive_search_num; ++i) {
        const bool is_update_kept = update_drivable_lanes(next_lanes, is_left);
        if (!is_update_kept) {
          break;
        }
        if (i == max_recursive_search_num - 1) {
          RCLCPP_DEBUG(
            rclcpp::get_logger(logger_namespace), "Drivable area expansion reaches max iteration.");
        }
      }
    };
  expand_drivable_area_recursively(next_lanes_for_right, false);
  expand_drivable_area_recursively(next_lanes_for_left, true);

  // 3. update again for new left/right lanes
  update_left_lanelets(current_drivable_lanes.left_lane);
  update_right_lanelets(current_drivable_lanes.right_lane);

  // 4. compensate that current_lane is in either of left_lane, right_lane or middle_lanes.
  if (
    current_drivable_lanes.left_lane.id() != lanelet.id() &&
    current_drivable_lanes.right_lane.id() != lanelet.id()) {
    current_drivable_lanes.middle_lanes.push_back(lanelet);
  }

  return current_drivable_lanes;
}

double calcDistanceToAvoidStartLine(
  const lanelet::ConstLanelets & lanelets, const std::shared_ptr<AvoidanceParameters> & parameters,
  const std::optional<double> distance_to_red_traffic)
{
  if (lanelets.empty()) {
    return std::numeric_limits<double>::lowest();
  }

  double distance_to_return_dead_line = std::numeric_limits<double>::lowest();

  // dead line stop factor(traffic light)
  if (parameters->enable_dead_line_for_traffic_light) {
    if (distance_to_red_traffic.has_value()) {
      distance_to_return_dead_line = std::max(
        distance_to_return_dead_line,
        distance_to_red_traffic.value() + parameters->dead_line_buffer_for_traffic_light);
    }
  }

  return distance_to_return_dead_line;
}

double calcDistanceToReturnDeadLine(
  const lanelet::ConstLanelets & lanelets, const PathWithLaneId & path,
  const std::shared_ptr<const PlannerData> & planner_data,
  const std::shared_ptr<AvoidanceParameters> & parameters,
  const std::optional<double> distance_to_red_traffic, const bool is_allowed_goal_modification)
{
  if (lanelets.empty()) {
    return std::numeric_limits<double>::max();
  }

  double distance_to_return_dead_line = std::numeric_limits<double>::max();

  // dead line stop factor(traffic light)
  if (parameters->enable_dead_line_for_traffic_light) {
    if (distance_to_red_traffic.has_value()) {
      distance_to_return_dead_line = std::min(
        distance_to_return_dead_line,
        distance_to_red_traffic.value() - parameters->dead_line_buffer_for_traffic_light);
    }
  }

  // dead line for goal
  if (!is_allowed_goal_modification && parameters->enable_dead_line_for_goal) {
    if (planner_data->route_handler->isInGoalRouteSection(lanelets.back())) {
      const auto & ego_pos = planner_data->self_odometry->pose.pose.position;
      const auto to_goal_distance =
        nero_mu::calcSignedArcLength(path.points, ego_pos, path.points.size() - 1);
      distance_to_return_dead_line = std::min(
        distance_to_return_dead_line, to_goal_distance - parameters->dead_line_buffer_for_goal);
    }
  }

  return distance_to_return_dead_line;
}

double calcErrorEclipseLongRadius(const PoseWithCovariance & pose)
{
  Eigen::Matrix2d xy_covariance;
  const auto cov = pose.covariance;
  xy_covariance(0, 0) = cov[0 * 6 + 0];
  xy_covariance(0, 1) = cov[0 * 6 + 1];
  xy_covariance(1, 0) = cov[1 * 6 + 0];
  xy_covariance(1, 1) = cov[1 * 6 + 1];

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(xy_covariance);

  return std::sqrt(eigensolver.eigenvalues()(1));
}
}  // namespace autoware::behavior_path_planner::utils::static_obstacle_avoidance
