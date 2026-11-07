// Copyright 2026 Nero (ixlpass). Licensed under the Apache License, Version 2.0.
//
// NERO ixlpass (QUEUE IXLPASS): a same-direction VEHICLE that passes the robot on its LEFT while
// the robot waits at / starts its LEFT turn (through the oncoming lane or the left part of the
// robot's lane). The car's blind_spot module (0.44.2) has an area for it (#5635's
// opposite-adjacent strip) but targets BICYCLE / PEDESTRIAN / MOTORCYCLE only (scene.cpp
// isTargetObjectType) and its area ends at the junction entry.
//
// NERO_BS_DETECT_NON_VRU (unset = the car: every new statement is behind it), LEFT turns from a
// lane whose left neighbour is the oncoming lane (or nothing): vehicles (car / truck / bus /
// trailer, highest-probability class) are judged as upstream's re-designed blind_spot judges
// objects (#11079 + #11234 detect_non_vru + #11265 lateral clearance, ported from
// autoware_universe c1db0d536):
// - attention area: from ego's left edge (the approach centreline + half its width) to the blind
//   side's far edge, carried straight on past
//   the junction entry (upstream's virtual lines); the approach is walked back over the turn
//   lanelet's straight predecessors <= 100 m;
// - ego's and each object's time interval across ego's turn path (ego from its smoothed path
//   profile, objects along their predicted paths); unsafe when they overlap with margins
//   ttc_start / ttc_end (2.0 / 1.0 s = the car's intersection not_prioritized margins);
// - a stop at the turn lanelet's map stop line (where the robot stops anyway; as the car's
//   intersection module) or, without one, at the junction entry, when the car's planner limits
//   allow it smoothly, else where ego can stop (standing: where it stands) while that is before the
//   commit point = the old module's critical line (the footprint reaches the oncoming half), for a
//   robot still crawling (<= 1.5 m/s, not past it) up to 1 m beyond it; for a passer squeezing past
//   within 1.0 m of the robot's straight band, a stop before the robot's left edge leaves that
//   band; past the commit point the check is over (the obstacle modules stay); a "where ego can
//   stop" point is latched once chosen; 0.5 s release debounce.
// Differences to upstream: the blind side is the OPPOSITE lane's far kerb (upstream uses the
// leftmost same-direction lane, which on our one-lane approaches excludes every passer);
// upstream's overlap case2 (always true once case0 fails: a hold for any car crossing ever after
// ego enters) is dropped; objects must move with the approach (heading) faster than min_v; a car
// queued behind in the robot's band (a follower, < 3 m/s), a vehicle that moved then stood until it
// is faster than resume_v (it yields), and a vehicle braking hard enough to stop >= 1 m before
// ego's turn path are no conflicts ("be dynamic"); braking / commit use the car's limits, not -6
// m/s2. VRUs and RIGHT turns: the car's code.
//   NERO_BS_DETECT_NON_VRU=<far_extend_m>,<ttc_start_s>,<ttc_end_s>,<debounce_s>,<heading_deg>,
//                          <min_v_mps>,<stand_v_mps>,<resume_v_mps>,<brake_decel_mps2>
//     far_extend_m <= 0: the whole opposite lane; > 0: that width of it from the centre line
//   NERO_BS_DEBUG=1: log lines per module / cycle / object (log only; also with the switch unset)

#include "autoware/behavior_velocity_blind_spot_module/scene.hpp"
#include "autoware/behavior_velocity_blind_spot_module/util.hpp"

#include <autoware/behavior_velocity_planner_common/utilization/boost_geometry_helper.hpp>
#include <autoware/behavior_velocity_planner_common/utilization/trajectory_utils.hpp>
#include <autoware/behavior_velocity_planner_common/utilization/util.hpp>
#include <autoware/motion_utils/trajectory/trajectory.hpp>
#include <autoware_lanelet2_extension/regulatory_elements/road_marking.hpp>
#include <autoware_lanelet2_extension/utility/utilities.hpp>
#include <autoware_utils/geometry/boost_polygon_utils.hpp>
#include <autoware_utils/geometry/geometry.hpp>

#include <boost/geometry/algorithms/correct.hpp>
#include <boost/geometry/algorithms/disjoint.hpp>
#include <boost/geometry/algorithms/distance.hpp>
#include <boost/geometry/algorithms/envelope.hpp>
#include <boost/geometry/algorithms/expand.hpp>
#include <boost/geometry/algorithms/intersection.hpp>
#include <boost/geometry/algorithms/intersects.hpp>
#include <boost/geometry/algorithms/within.hpp>

#include <lanelet2_core/Attribute.h>
#include <lanelet2_core/geometry/Lanelet.h>
#include <lanelet2_core/geometry/LineString.h>
#include <tf2/utils.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace autoware::behavior_velocity_planner
{
namespace bg = boost::geometry;
using autoware_perception_msgs::msg::ObjectClassification;
using autoware_perception_msgs::msg::PredictedObject;

namespace
{
struct NeroBsVehParam
{
  bool enabled{false};
  double far_extend{0.0};
  double ttc_start_margin{2.0};
  double ttc_end_margin{1.0};
  double debounce{0.5};
  double heading_deg{60.0};
  double min_velocity{1.2};
  // below this a vehicle 'stood' (a yield: Ahmed's 'a car that stops is the signal')
  double stand_velocity{0.3};
  // a vehicle that stood is a target again only above this (yieldrevoke-like)
  double resume_velocity{1.5};
  // a vehicle braking at least this hard that stops before ego's turn path: no conflict
  double brake_decel{0.5};
};

const NeroBsVehParam & nero_veh_param()
{
  static const NeroBsVehParam param = []() {
    NeroBsVehParam p;
    const char * env = std::getenv("NERO_BS_DETECT_NON_VRU");
    if (!env || !*env) {
      return p;
    }
    std::vector<double> v;
    std::stringstream ss(env);
    std::string item;
    while (std::getline(ss, item, ',')) {
      try {
        v.push_back(std::stod(item));
      } catch (...) {
        v.push_back(std::numeric_limits<double>::quiet_NaN());
      }
    }
    auto get = [&](size_t i, double def) {
      return (i < v.size() && std::isfinite(v.at(i))) ? v.at(i) : def;
    };
    p.enabled = true;
    p.far_extend = get(0, p.far_extend);
    p.ttc_start_margin = get(1, p.ttc_start_margin);
    p.ttc_end_margin = get(2, p.ttc_end_margin);
    p.debounce = get(3, p.debounce);
    p.heading_deg = get(4, p.heading_deg);
    p.min_velocity = get(5, p.min_velocity);
    p.stand_velocity = get(6, p.stand_velocity);
    p.resume_velocity = get(7, p.resume_velocity);
    p.brake_decel = get(8, p.brake_decel);
    return p;
  }();
  return param;
}

bool nero_bs_debug()
{
  static const bool on = []() {
    const char * env = std::getenv("NERO_BS_DEBUG");
    return env && std::string(env) == "1";
  }();
  return on;
}

std::string short_id(const PredictedObject & o)
{
  char buf[16];
  std::snprintf(
    buf, sizeof(buf), "%02x%02x%02x", o.object_id.uuid.at(0), o.object_id.uuid.at(1),
    o.object_id.uuid.at(2));
  return std::string(buf);
}

uint8_t highest_label(const PredictedObject & o)
{
  uint8_t label = ObjectClassification::UNKNOWN;
  double best = -1.0;
  for (const auto & c : o.classification) {
    if (c.probability > best) {
      best = c.probability;
      label = c.label;
    }
  }
  return label;
}

const char * label_name(const uint8_t l)
{
  switch (l) {
    case ObjectClassification::CAR:
      return "CAR";
    case ObjectClassification::TRUCK:
      return "TRUCK";
    case ObjectClassification::BUS:
      return "BUS";
    case ObjectClassification::TRAILER:
      return "TRAILER";
    case ObjectClassification::MOTORCYCLE:
      return "MOTORCYCLE";
    case ObjectClassification::BICYCLE:
      return "BICYCLE";
    case ObjectClassification::PEDESTRIAN:
      return "PEDESTRIAN";
    default:
      return "UNKNOWN";
  }
}

bool is_vehicle_label(const uint8_t l)
{
  return l == ObjectClassification::CAR || l == ObjectClassification::TRUCK ||
         l == ObjectClassification::BUS || l == ObjectClassification::TRAILER;
}

lanelet::BasicLineString2d to_bls2d(const lanelet::ConstLineString3d & ls)
{
  lanelet::BasicLineString2d out;
  for (const auto & p : ls) {
    out.emplace_back(p.x(), p.y());
  }
  return out;
}

lanelet::LineString3d make_ls(const std::vector<lanelet::BasicPoint3d> & pts)
{
  lanelet::Points3d points;
  for (const auto & p : pts) {
    points.emplace_back(lanelet::InvalId, p);
  }
  return lanelet::LineString3d{lanelet::InvalId, points};
}

// upstream helper::find_intersection_point
template <typename L1, typename L2>
std::optional<lanelet::BasicPoint2d> find_intersection_point(L1 && line1, L2 && line2)
{
  std::vector<lanelet::BasicPoint2d> intersection_points;
  bg::intersection(line1, line2, intersection_points);
  if (intersection_points.empty()) {
    return std::nullopt;
  }
  return intersection_points.front();
}

// upstream clip_virtual_line_to_intersection_bound (LEFT turn: the farthest bound is the turn
// lanelet's right bound)
std::vector<lanelet::BasicPoint3d> clip_to_bound(
  const lanelet::BasicPoint3d & start, const lanelet::BasicPoint3d & end,
  const lanelet::ConstLineString3d & farthest_bound)
{
  lanelet::BasicLineString2d virtual_line{{start.x(), start.y()}, {end.x(), end.y()}};
  std::vector<lanelet::BasicPoint2d> intersected;
  for (size_t i = 0; i + 1 < farthest_bound.size(); ++i) {
    lanelet::BasicLineString2d seg{
      {farthest_bound[i].x(), farthest_bound[i].y()},
      {farthest_bound[i + 1].x(), farthest_bound[i + 1].y()}};
    bg::intersection(virtual_line, seg, intersected);
  }
  if (intersected.empty()) {
    return {start, end};
  }
  return {start, {intersected.front().x(), intersected.front().y(), end.z()}};
}

// upstream compute_time_interval_for_passing_line (ego footprint along its future profile)
std::optional<std::pair<double, double>> ego_passage_interval(
  const std::vector<std::pair<autoware_internal_planning_msgs::msg::PathPointWithLaneId, double>> &
    future,
  const autoware_utils::LinearRing2d & footprint, const lanelet::BasicLineString2d & line1,
  const lanelet::BasicLineString2d & line2)
{
  std::optional<double> entry{};
  for (const auto & [p, t] : future) {
    const auto fp =
      autoware_utils::transform_vector(footprint, autoware_utils::pose2transform(p.point.pose));
    if (bg::intersects(fp, line1)) {
      entry = t;
      break;
    }
  }
  if (!entry) {
    return std::nullopt;
  }
  std::optional<double> exit{};
  for (auto it = future.rbegin(); it != future.rend(); ++it) {
    const auto fp = autoware_utils::transform_vector(
      footprint, autoware_utils::pose2transform(it->first.point.pose));
    if (bg::intersects(fp, line2)) {
      exit = it->second;
      break;
    }
  }
  if (!exit) {
    // ego's path ends (goal / path end) inside the band before it reaches the far line: it stays in
    // the band until the path's last point (upstream returns no interval -> InternalError; planning
    // sim fb7: the goal 1.5 m short of it)
    exit = future.back().second;
  }
  return std::make_pair(entry.value(), exit.value());
}

// the predicted path resampled at 0.1 s (upstream: object_recognition_utils::resamplePredictedPath;
// linear here)
std::vector<geometry_msgs::msg::Pose> resample_path(
  const autoware_perception_msgs::msg::PredictedPath & path, const double dt, double * horizon)
{
  std::vector<geometry_msgs::msg::Pose> out;
  const double step = rclcpp::Duration(path.time_step).seconds();
  if (path.path.size() < 2 || step <= 0.0) {
    *horizon = 0.0;
    return out;
  }
  *horizon = step * static_cast<double>(path.path.size() - 1);
  for (double t = 0.0; t <= *horizon + 1e-6; t += dt) {
    const size_t i = std::min(static_cast<size_t>(t / step), path.path.size() - 2);
    const double r = std::clamp((t - step * static_cast<double>(i)) / step, 0.0, 1.0);
    const auto & a = path.path.at(i);
    const auto & b = path.path.at(i + 1);
    geometry_msgs::msg::Pose p = a;
    p.position.x = a.position.x + r * (b.position.x - a.position.x);
    p.position.y = a.position.y + r * (b.position.y - a.position.y);
    p.position.z = a.position.z + r * (b.position.z - a.position.z);
    const double ya = tf2::getYaw(a.orientation);
    double dy = tf2::getYaw(b.orientation) - ya;
    while (dy > M_PI) dy -= 2.0 * M_PI;
    while (dy < -M_PI) dy += 2.0 * M_PI;
    p.orientation = autoware_utils::create_quaternion_from_yaw(ya + r * dy);
    out.push_back(p);
  }
  return out;
}

double lane_direction_at(const lanelet::LineString3d & cl, const double x, const double y)
{
  double best = std::numeric_limits<double>::max();
  double dir = 0.0;
  for (size_t i = 0; i + 1 < cl.size(); ++i) {
    const double ax = cl[i].x();
    const double ay = cl[i].y();
    const double dx = cl[i + 1].x() - ax;
    const double dy = cl[i + 1].y() - ay;
    const double l2 = dx * dx + dy * dy;
    if (l2 < 1e-9) {
      continue;
    }
    const double t = std::clamp(((x - ax) * dx + (y - ay) * dy) / l2, 0.0, 1.0);
    const double d = std::hypot(ax + t * dx - x, ay + t * dy - y);
    if (d < best) {
      best = d;
      dir = std::atan2(dy, dx);
    }
  }
  return dir;
}

// signed lateral offset (left +) of (x, y) from the centreline's nearest segment line
double lane_lateral_at(const lanelet::LineString3d & cl, const double x, const double y)
{
  double best = std::numeric_limits<double>::max();
  double lat = 0.0;
  for (size_t i = 0; i + 1 < cl.size(); ++i) {
    const double ax = cl[i].x();
    const double ay = cl[i].y();
    const double dx = cl[i + 1].x() - ax;
    const double dy = cl[i + 1].y() - ay;
    const double l2 = dx * dx + dy * dy;
    if (l2 < 1e-9) {
      continue;
    }
    const double t = std::clamp(((x - ax) * dx + (y - ay) * dy) / l2, 0.0, 1.0);
    const double d = std::hypot(ax + t * dx - x, ay + t * dy - y);
    if (d < best) {
      best = d;
      lat = (dx * (y - ay) - dy * (x - ax)) / std::sqrt(l2);
    }
  }
  return lat;
}

// scene.cpp's file-static insertPointIndex (+ getDuplicatedPointIdx), copied: returns (index,
// inserted)
std::optional<std::pair<size_t, bool>> nero_insert_point_index(
  const geometry_msgs::msg::Pose & in_pose,
  autoware_internal_planning_msgs::msg::PathWithLaneId * inout_path,
  const double ego_nearest_dist_threshold, const double ego_nearest_yaw_threshold)
{
  for (size_t i = 0; i < inout_path->points.size(); i++) {
    const auto & p = inout_path->points.at(i).point.pose.position;
    constexpr double min_dist = 0.001;
    if (autoware_utils::calc_distance2d(p, in_pose.position) < min_dist) {
      return std::make_pair(i, false);
    }
  }
  const size_t closest_idx = autoware::motion_utils::findFirstNearestIndexWithSoftConstraints(
    inout_path->points, in_pose, ego_nearest_dist_threshold, ego_nearest_yaw_threshold);
  size_t insert_idx = closest_idx;
  autoware_internal_planning_msgs::msg::PathPointWithLaneId inserted_point =
    inout_path->points.at(closest_idx);
  if (planning_utils::isAheadOf(in_pose, inout_path->points.at(closest_idx).point.pose)) {
    ++insert_idx;
  } else {
    const size_t prior_ind = closest_idx > 0 ? closest_idx - 1 : 0;
    inserted_point.point.longitudinal_velocity_mps =
      inout_path->points.at(prior_ind).point.longitudinal_velocity_mps;
  }
  inserted_point.point.pose = in_pose;
  inout_path->points.insert(inout_path->points.begin() + insert_idx, inserted_point);
  return std::make_pair(insert_idx, true);
}

double norm_angle(double a)
{
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}
}  // namespace

std::optional<BlindSpotModule::NeroVehGeometry> BlindSpotModule::neroVehBuildGeometry(
  const std::vector<lanelet::Id> & lane_ids_upto_intersection) const
{
  const auto & rh = planner_data_->route_handler_;
  const auto map = rh->getLaneletMapPtr();
  const auto graph = rh->getRoutingGraphPtr();
  const auto & prm = nero_veh_param();
  const auto turn_lane = map->laneletLayer.get(lane_id_);
  NeroVehGeometry g;
  std::vector<lanelet::BasicPoint3d> far_pts;
  std::vector<lanelet::BasicPoint3d> cl_pts;
  // the approach: walk back from the turn lanelet over its predecessors (straight / untagged first:
  // upstream previous_lane_straight_priority, the road a passer comes along whatever ego's route),
  // up to ~100 m (upstream backward_attention_length); stop before a left / right turn lanelet. The
  // route's lanes are not used: at 61556 they end 27 m back at a previous junction lanelet (a fast
  // passer would enter the area ~2.5 s before the junction)
  (void)lane_ids_upto_intersection;
  std::vector<lanelet::ConstLanelet> lanes;
  {
    double len = 0.0;
    lanelet::ConstLanelet cur = turn_lane;
    while (len < 100.0) {
      const auto prevs = graph->previous(cur);
      std::optional<lanelet::ConstLanelet> pick{};
      for (const auto & pv : prevs) {
        const std::string td = pv.attributeOr("turn_direction", "else");
        if (td != "left" && td != "right") {
          pick = pv;
          break;
        }
      }
      if (!pick) {
        break;
      }
      lanes.insert(lanes.begin(), pick.value());
      len += lanelet::geometry::length2d(pick.value());
      cur = pick.value();
    }
  }
  if (lanes.empty()) {
    g.kind = "no approach lane";
    return std::nullopt;
  }
  // a same-direction lane on the left of the last approach lane: not this check's road (upstream
  // #11813's inner turn lane / multi-lane approaches stay stock); further back such a lane ends the
  // area
  {
    const auto & last = lanes.back();
    if (graph->adjacentLeft(last) || graph->left(last)) {
      g.kind = "same-direction left lane on " + std::to_string(last.id());
      return std::nullopt;
    }
    size_t keep = 0;
    for (size_t k = lanes.size(); k-- > 0;) {
      if (graph->adjacentLeft(lanes.at(k)) || graph->left(lanes.at(k))) {
        keep = k + 1;
        break;
      }
    }
    lanes.erase(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(keep));
  }
  // far edge per lane: the opposite lane's far kerb (or far_extend of it); a lane without a shared
  // opposite lane takes its centreline at the lateral offset of the nearest lane that has one (a
  // smooth edge through junction lanelets)
  std::vector<std::optional<lanelet::ConstLanelet>> opps;
  for (const auto & lane : lanes) {
    std::optional<lanelet::ConstLanelet> opposite{};
    for (const auto & u : map->laneletLayer.findUsages(lane.leftBound().invert())) {
      if (u.id() != lane.id()) {
        opposite = u;
        break;
      }
    }
    opps.push_back(opposite);
  }
  auto lane_width = [](const lanelet::ConstLanelet & ll) {
    const auto cl = ll.centerline2d();
    return bg::area(ll.polygon2d().basicPolygon()) /
           std::max(static_cast<double>(bg::length(cl)), 1e-3);
  };
  std::vector<std::optional<double>> offs(lanes.size());
  for (size_t k = 0; k < lanes.size(); ++k) {
    if (opps.at(k)) {
      const double wo = lane_width(opps.at(k).value());
      offs.at(k) =
        lane_width(lanes.at(k)) / 2.0 + (prm.far_extend <= 0.0 ? wo : std::min(prm.far_extend, wo));
    }
  }
  for (size_t k = 0; k < lanes.size(); ++k) {
    const auto & lane = lanes.at(k);
    g.road_lanelets.push_back(lane);
    lanelet::ConstLineString3d far_edge;
    if (opps.at(k)) {
      const auto opp_inv = opps.at(k)->invert();  // the opposite lane in ego's direction
      if (prm.far_extend <= 0.0) {
        far_edge = opp_inv.leftBound();  // its far kerb
      } else {
        const double width = lane_width(opp_inv);
        far_edge = lanelet::utils::getCenterlineWithOffset(
          opp_inv, -width / 2.0 + std::min(prm.far_extend, width));
      }
      g.kind += (g.kind.empty() ? "" : "+") + std::string("opp") + std::to_string(opps.at(k)->id());
    } else {
      std::optional<double> off{};
      for (size_t d = 1; d < lanes.size() && !off; ++d) {
        if (k + d < lanes.size() && offs.at(k + d)) {
          off = offs.at(k + d);
        } else if (k >= d && offs.at(k - d)) {
          off = offs.at(k - d);
        }
      }
      far_edge = off ? lanelet::utils::getCenterlineWithOffset(lane, off.value())
                     : lanelet::ConstLineString3d(lane.leftBound());
      g.kind += (g.kind.empty() ? "" : "+") + std::string(off ? "offset" : "own") +
                std::to_string(lane.id());
    }
    for (const auto & p : far_edge) {
      if (!far_pts.empty() && (far_pts.back() - p.basicPoint()).norm() < 0.05) {
        continue;
      }
      far_pts.push_back(p.basicPoint());
    }
    for (const auto & p : lane.centerline()) {
      if (!cl_pts.empty() && (cl_pts.back() - p.basicPoint()).norm() < 0.05) {
        continue;
      }
      cl_pts.push_back(p.basicPoint());
    }
  }
  if (far_pts.size() < 2 || cl_pts.size() < 2) {
    return std::nullopt;
  }
  g.far_before = make_ls(far_pts);
  g.road_centerline = make_ls(cl_pts);
  // upstream generate_virtual_blind_side_boundary_after_turning: the far edge's last segment
  // carried on by the turn lanelet's length, clipped at the turn lanelet's right (far) bound
  const double extend = lanelet::geometry::length3d(turn_lane);
  const auto & p1 = far_pts.at(far_pts.size() - 2);
  const auto & p2 = far_pts.back();
  const auto dir = (p2 - p1).normalized();
  const lanelet::BasicPoint3d p3 = p2 + dir * extend;
  g.far_after = make_ls(clip_to_bound(p2, p3, turn_lane.rightBound()));
  return g;
}

BlindSpotModule::NeroVehDecision BlindSpotModule::neroVehicleCheck(
  const InterpolatedPathInfo & interpolated_path_info,
  autoware_internal_planning_msgs::msg::PathWithLaneId * path)
{
  NeroVehDecision dec;
  const auto & prm = nero_veh_param();
  if (!prm.enabled || turn_direction_ != TurnDirection::LEFT) {
    return dec;
  }
  const rclcpp::Time now = clock_->now();
  if (!nero_veh_geometry_ && !nero_veh_geometry_failed_) {
    const auto ids = find_lane_ids_upto(*path, lane_id_);
    nero_veh_geometry_ = neroVehBuildGeometry(ids);
    if (!nero_veh_geometry_) {
      nero_veh_geometry_failed_ = true;
    }
    RCLCPP_INFO(
      logger_, "[NERO_BS] lane %ld vehicle check %s", lane_id_,
      nero_veh_geometry_
        ? ("on: " + nero_veh_geometry_->kind).c_str()
        : "off: no opposite-lane geometry (a same-direction left lane / no lanes)");
  }
  if (!nero_veh_geometry_) {
    dec.note = "no geometry";
    if (nero_bs_debug()) {
      RCLCPP_INFO_THROTTLE(
        logger_, *clock_, 1000, "[NERO_BS] lane %ld veh off: %s", lane_id_, dec.note.c_str());
    }
    return dec;
  }
  if (nero_veh_over_pass_judge_) {
    dec.note = "over the pass judge line";
    if (nero_bs_debug()) {
      RCLCPP_INFO_THROTTLE(
        logger_, *clock_, 1000, "[NERO_BS] lane %ld veh off: %s", lane_id_, dec.note.c_str());
    }
    return dec;
  }
  const auto & geo = nero_veh_geometry_.value();
  const auto & vi = planner_data_->vehicle_info_;
  const double ego_width = vi.vehicle_width_m;
  const auto & current_pose = planner_data_->current_odometry->pose;
  const double velocity = planner_data_->current_velocity->twist.linear.x;
  const double acceleration = planner_data_->current_acceleration
                                ? planner_data_->current_acceleration->accel.accel.linear.x
                                : 0.0;
  const auto map = planner_data_->route_handler_->getLaneletMapPtr();
  const auto turn_lane = map->laneletLayer.get(lane_id_);
  const auto & path_ip = interpolated_path_info.path;
  if (!interpolated_path_info.lane_id_interval) {
    dec.note = "no lane interval";
    if (nero_bs_debug()) {
      RCLCPP_INFO_THROTTLE(
        logger_, *clock_, 1000, "[NERO_BS] lane %ld veh off: %s", lane_id_, dec.note.c_str());
    }
    return dec;
  }
  const auto [lane_start, lane_end] = interpolated_path_info.lane_id_interval.value();

  // --- the turn lanelet's entry line, ego's virtual straight path after turning (upstream helpers)
  // ---
  const lanelet::BasicLineString2d entry_line_tl{
    {turn_lane.leftBound().front().x(), turn_lane.leftBound().front().y()},
    {turn_lane.rightBound().front().x(), turn_lane.rightBound().front().y()}};
  lanelet::BasicLineString2d path_ls;
  for (const auto & p : path_ip.points) {
    path_ls.emplace_back(p.point.pose.position.x, p.point.pose.position.y);
  }
  auto entry_pt = find_intersection_point(path_ls, entry_line_tl);
  if (entry_pt) {
    nero_veh_entry_pt_ =
      entry_pt;  // cached: a robot past the entry may get a path that starts after it
  } else if (nero_veh_entry_pt_) {
    entry_pt = nero_veh_entry_pt_;
  }
  if (!entry_pt) {
    dec.note = "path does not cross the turn lanelet's entry line";
    if (nero_bs_debug()) {
      RCLCPP_INFO_THROTTLE(
        logger_, *clock_, 1000, "[NERO_BS] lane %ld veh off: %s", lane_id_, dec.note.c_str());
    }
    return dec;
  }
  const lanelet::BasicPoint2d e0 = entry_line_tl.front();
  const lanelet::BasicPoint2d e1 = entry_line_tl.back();
  const double ewidth = (e1 - e0).norm();
  const double s_left =
    (entry_pt.value() - e0).norm() - ego_width / 2.0;  // LEFT turn: ego's left edge
  if (ewidth < 1e-3) {
    dec.note = "degenerate entry line";
    if (nero_bs_debug()) {
      RCLCPP_INFO_THROTTLE(
        logger_, *clock_, 1000, "[NERO_BS] lane %ld veh off: %s", lane_id_, dec.note.c_str());
    }
    return dec;
  }
  const lanelet::BasicPoint2d vs2 = e0 + (e1 - e0) / ewidth * s_left;
  const lanelet::BasicPoint2d nrm{-(e1 - e0).y() / ewidth, (e1 - e0).x() / ewidth};  // forward
  const double extend = lanelet::geometry::length3d(turn_lane);
  const lanelet::BasicPoint3d vstart{vs2.x(), vs2.y(), current_pose.position.z};
  const lanelet::BasicPoint3d vend{
    vs2.x() + nrm.x() * extend, vs2.y() + nrm.y() * extend, current_pose.position.z};
  const auto virtual_straight = clip_to_bound(vstart, vend, turn_lane.rightBound());
  const lanelet::BasicLineString2d virtual_straight_2d{
    {virtual_straight.front().x(), virtual_straight.front().y()},
    {virtual_straight.back().x(), virtual_straight.back().y()}};

  // --- attention area (upstream generate_attention_area with our far edge) ---
  lanelet::BasicPolygon2d attention;
  for (const auto & p : geo.far_before) {
    attention.emplace_back(p.x(), p.y());
  }
  for (const auto & p : geo.far_after) {
    attention.emplace_back(p.x(), p.y());
  }
  // near side, reversed: virtual straight path (far end -> start), ego's path left edge back to the
  // path's start, then the road centreline offset by half ego's width back to the road's start
  std::vector<lanelet::BasicPoint2d> near;
  {
    const double off = ego_width / 2.0;
    std::vector<lanelet::BasicPoint2d> road_near;
    for (size_t i = 0; i + 1 < geo.road_centerline.size(); ++i) {
      const auto a = geo.road_centerline[i].basicPoint2d();
      const auto b = geo.road_centerline[i + 1].basicPoint2d();
      const auto d = (b - a).normalized();
      const lanelet::BasicPoint2d n{-d.y(), d.x()};
      road_near.push_back(a + n * off);
      if (i + 2 == geo.road_centerline.size()) {
        road_near.push_back(b + n * off);
      }
    }
    // the approach lanes' centreline + half ego's width up to the junction entry (upstream takes
    // ego's path edge here: once ego is past the entry its path can start after it and that edge
    // followed the turn: lp56366 v11_lft_dp2 lost the 11 m/s passer from the area while it still
    // came), then the virtual straight line
    near.insert(near.end(), road_near.begin(), road_near.end());
    near.emplace_back(virtual_straight.front().x(), virtual_straight.front().y());
    near.emplace_back(virtual_straight.back().x(), virtual_straight.back().y());
  }
  for (auto it = near.rbegin(); it != near.rend(); ++it) {
    attention.push_back(*it);
  }
  bg::correct(attention);
  debug_data_.nero_veh_attention_area = attention;

  // --- ego's path polygon in the turn lanelet (upstream generate_ego_path_polygon) ---
  lanelet::BasicLineString2d ego_left;
  lanelet::BasicLineString2d ego_right;
  for (size_t i = lane_start; i <= lane_end && i < path_ip.points.size(); ++i) {
    const auto & pose = path_ip.points.at(i).point.pose;
    const double yaw = tf2::getYaw(pose.orientation);
    ego_left.emplace_back(
      pose.position.x + std::cos(yaw + M_PI / 2.0) * ego_width / 2.0,
      pose.position.y + std::sin(yaw + M_PI / 2.0) * ego_width / 2.0);
    ego_right.emplace_back(
      pose.position.x + std::cos(yaw - M_PI / 2.0) * ego_width / 2.0,
      pose.position.y + std::sin(yaw - M_PI / 2.0) * ego_width / 2.0);
  }
  if (ego_left.size() < 2) {
    dec.note = "short turn path";
    if (nero_bs_debug()) {
      RCLCPP_INFO_THROTTLE(
        logger_, *clock_, 1000, "[NERO_BS] lane %ld veh off: %s", lane_id_, dec.note.c_str());
    }
    return dec;
  }
  const lanelet::BasicLineString2d ego_entry{ego_left.front(), ego_right.front()};
  {
    lanelet::BasicPolygon2d conflict(ego_left.begin(), ego_left.end());
    conflict.insert(conflict.end(), ego_right.rbegin(), ego_right.rend());
    debug_data_.nero_veh_conflict_area = conflict;
  }

  // --- ego's future profile (upstream calculate_future_profile) and its passage interval: computed
  // only when a vehicle passes the cheap filters below (most cycles have none: no smoother pass)
  // ---
  const double time_to_restart =
    nero_veh_last_unsafe_
      ? std::max(0.0, prm.debounce - (now - nero_veh_last_unsafe_.value()).seconds())
      : 0.0;
  std::optional<std::optional<std::pair<double, double>>> ego_cache{};
  auto compute_ego = [&]() -> std::optional<std::pair<double, double>> {
    std::vector<std::pair<autoware_internal_planning_msgs::msg::PathPointWithLaneId, double>>
      future;
    {
      const auto closest_idx = autoware::motion_utils::findFirstNearestIndexWithSoftConstraints(
        path->points, current_pose, planner_data_->ego_nearest_dist_threshold,
        planner_data_->ego_nearest_yaw_threshold);
      autoware_internal_planning_msgs::msg::PathWithLaneId reference_path;
      reference_path.header = path->header;
      bool assigned_lane_found = false;
      for (size_t i = 0; i < path->points.size(); ++i) {
        auto rp = path->points.at(i);
        if (i < closest_idx) {
          rp.point.longitudinal_velocity_mps = static_cast<float>(velocity);
        }
        reference_path.points.push_back(rp);
        const bool has_id =
          std::find(rp.lane_ids.begin(), rp.lane_ids.end(), lane_id_) != rp.lane_ids.end();
        if (assigned_lane_found && !has_id) {
          break;
        }
        assigned_lane_found = has_id;
      }
      if (reference_path.points.size() < 3 || !assigned_lane_found) {
        dec.note = "no reference path";
        return std::nullopt;
      }
      auto smoothed = reference_path;
      if (!smoothPath(reference_path, smoothed, planner_data_)) {
        dec.note = "smoothPath failed";
        return std::nullopt;
      }
      const auto c2 = autoware::motion_utils::findFirstNearestIndexWithSoftConstraints(
        smoothed.points, current_pose, planner_data_->ego_nearest_dist_threshold,
        planner_data_->ego_nearest_yaw_threshold);
      constexpr double minimum_default_velocity = 1.388;  // upstream blind_spot.param.yaml
      double t = time_to_restart;
      future.emplace_back(smoothed.points.at(c2), t);
      for (size_t i = c2 + 1; i + 1 < smoothed.points.size(); ++i) {
        const auto & q1 = smoothed.points.at(i);
        const auto & q2 = smoothed.points.at(i + 1);
        const double dist = autoware_utils::calc_distance2d(q1, q2);
        const double avg =
          (q1.point.longitudinal_velocity_mps + q2.point.longitudinal_velocity_mps) / 2.0;
        t += dist / std::max(avg, minimum_default_velocity);
        future.emplace_back(q1, t);
      }
    }
    lanelet::BasicLineString2d far_after_2d;
    for (const auto & p : geo.far_after) {
      far_after_2d.emplace_back(p.x(), p.y());
    }
    const auto ego_interval =
      ego_passage_interval(future, vi.createFootprint(0.0, 0.0), virtual_straight_2d, far_after_2d);
    if (!ego_interval) {
      dec.note = "ego does not cross the attention area";
    }
    return ego_interval;
  };
  auto ego_interval_lazy = [&]() {
    if (!ego_cache) {
      ego_cache = compute_ego();
    }
    return ego_cache.value();
  };
  double ego_in = -1.0;
  double ego_out = -1.0;

  // --- lateral clearance (upstream #11265): ego's left side to the far edge before turning ---
  const auto ego_fp = autoware_utils::transform_vector(
    vi.createFootprint(), autoware_utils::pose2transform(current_pose));
  const lanelet::BasicLineString2d ego_side{
    {ego_fp[vehicle_info_utils::VehicleInfo::FrontLeftIndex].x(),
     ego_fp[vehicle_info_utils::VehicleInfo::FrontLeftIndex].y()},
    {ego_fp[vehicle_info_utils::VehicleInfo::RearLeftIndex].x(),
     ego_fp[vehicle_info_utils::VehicleInfo::RearLeftIndex].y()}};
  const auto far_before_2d = to_bls2d(geo.far_before);
  const double lat_gap = bg::distance(ego_side, far_before_2d);

  // --- the robot's straight band (followers): the frame at ego with x along the approach lane; the
  // band's left edge = the robot's leftmost footprint corner. A vehicle behind the robot's front
  // whose right side reaches into the band (+0.1 m) is a car queued behind the robot (IXFOLLOW's
  // shape), not a passer: it cannot pass without moving out first ---
  const double ego_lane_dir =
    lane_direction_at(geo.road_centerline, current_pose.position.x, current_pose.position.y);
  const double ego_rel_yaw =
    std::abs(norm_angle(tf2::getYaw(current_pose.orientation) - ego_lane_dir));
  const double cd = std::cos(ego_lane_dir);
  const double sd = std::sin(ego_lane_dir);
  auto to_band = [&](const double x, const double y) {
    const double dx = x - current_pose.position.x;
    const double dy = y - current_pose.position.y;
    return std::make_pair(dx * cd + dy * sd, -dx * sd + dy * cd);
  };
  // the band is taken at the robot's REAR corners (its trail: a robot already turned at its line
  // swings its nose left, which made a passer squeezing past it a 'follower': planning sim lp56366
  // v5_mid_dp2 contact); a follower is behind its rear
  double band_left = std::numeric_limits<double>::lowest();
  double band_rear = std::numeric_limits<double>::max();
  for (const size_t ci :
       {vehicle_info_utils::VehicleInfo::RearLeftIndex,
        vehicle_info_utils::VehicleInfo::RearRightIndex}) {
    const auto [bx, by] = to_band(ego_fp[ci].x(), ego_fp[ci].y());
    band_left = std::max(band_left, by);
    band_rear = std::min(band_rear, bx);
  }
  constexpr double follower_margin = 0.1;
  // a passer this close to the robot's straight band squeezes past it; measured across the lane
  // (lateral offsets from the approach's centreline), not in the straight frame at ego: a passer
  // 24 m back on a bending approach read 1.0 m closer there (lp56366 v5_onc_dm2, an oncoming-lane
  // passer taken for a squeezer)
  constexpr double squeeze_clearance = 1.0;
  double band_left_lane = std::numeric_limits<double>::lowest();
  for (const size_t ci :
       {vehicle_info_utils::VehicleInfo::RearLeftIndex,
        vehicle_info_utils::VehicleInfo::RearRightIndex}) {
    band_left_lane = std::max(
      band_left_lane, lane_lateral_at(geo.road_centerline, ego_fp[ci].x(), ego_fp[ci].y()));
  }
  double min_unsafe_clear = std::numeric_limits<double>::max();

  // --- objects (upstream filter_attention_objects + collect_unsafe_objects, vehicles only) ---
  const double heading_thr = prm.heading_deg * M_PI / 180.0;
  std::optional<std::tuple<double, PredictedObject, double, double>> worst{};
  for (const auto & object : planner_data_->predicted_objects->objects) {
    const auto label = highest_label(object);
    if (!is_vehicle_label(label)) {
      continue;
    }
    const auto & pose = object.kinematics.initial_pose_with_covariance.pose;
    const lanelet::BasicPoint2d pos{pose.position.x, pose.position.y};
    if (!bg::within(pos, attention)) {
      continue;
    }
    const double v = object.kinematics.initial_twist_with_covariance.twist.linear.x;
    const double lane_dir = lane_direction_at(geo.road_centerline, pos.x(), pos.y());
    const double dyaw = norm_angle(tf2::getYaw(pose.orientation) - lane_dir);
    const bool heading_ok = std::abs(dyaw) <= heading_thr;
    const bool speed_ok = v >= prm.min_velocity;
    const bool gap_ok = object.shape.dimensions.y < lat_gap;
    // yield memory (per track id, this module instance): a vehicle seen MOVING (>= 1.0 m/s) that
    // then stood (< stand_v) is exempt until it moves faster than resume_v again (a new track is
    // often born at 0 m/s: that is no yield)
    const std::string oid = short_id(object);
    bool yielded = false;
    {
      auto & st = nero_veh_stood_[oid];
      if (std::abs(v) >= 1.0) {
        st.first = true;
      }
      if (st.first && std::abs(v) < prm.stand_velocity) {
        st.second = true;
      } else if (st.second && v >= prm.resume_velocity) {
        st.second = false;
      }
      yielded = st.second;
    }
    // braking: deceleration from the track's own reported speed over ~1 s (>= 0.6 s of history)
    double decel = 0.0;
    {
      auto & hist = nero_veh_vhist_[oid];
      const double tn = now.seconds();
      hist.emplace_back(tn, v);
      while (!hist.empty() && tn - hist.front().first > 1.5) {
        hist.pop_front();
      }
      const std::pair<double, double> * old = nullptr;
      for (const auto & hv : hist) {
        if (tn - hv.first >= 1.0 - 1e-3) {
          old = &hv;
        }
      }
      if (!old && !hist.empty() && tn - hist.front().first >= 0.6) {
        old = &hist.front();
      }
      if (old && tn - old->first > 1e-3) {
        decel = (old->second - v) / (tn - old->first);
      }
    }
    bool follower = false;
    double band_clear = 0.0;     // its right side's clearance to the robot's straight band
    double squeeze_clear = 0.0;  // the same across the lane (squeezer test)
    {
      double min_y = std::numeric_limits<double>::max();
      double max_x = std::numeric_limits<double>::lowest();
      double min_lat = std::numeric_limits<double>::max();
      // the polygon is named: a range-for over to_polygon2d(...).outer() iterated a destroyed
      // temporary (garbage corners in some runs: a passer read as a squeezer / follower)
      const auto obj_poly = autoware_utils::to_polygon2d(pose, object.shape);
      for (const auto & c : obj_poly.outer()) {
        const auto [ox, oy] = to_band(c.x(), c.y());
        min_y = std::min(min_y, oy);
        max_x = std::max(max_x, ox);
        min_lat = std::min(min_lat, lane_lateral_at(geo.road_centerline, c.x(), c.y()));
      }
      band_clear = min_y - band_left;
      squeeze_clear = min_lat - band_left_lane;
      // a car queued behind creeps (IXFOLLOW: 0.5-0.9 m/s); one coming fast into the robot's band
      // is no follower, unless it is deep in the band (> 0.2 m across the lane) behind a robot that
      // is still along its lane (< 20 deg): it cannot pass without moving out first, whatever its
      // speed (lp56366 fol1: a follower 1 m left of the robot's centre approaching at 3.0 m/s was a
      // target until it slowed; a robot turned further in has no straight band behind it)
      follower =
        max_x < band_rear + 0.5 && ((squeeze_clear < -0.2 && ego_rel_yaw < 20.0 * M_PI / 180.0) ||
                                    (band_clear < follower_margin && std::abs(v) < 3.0));
    }
    std::string iv;
    bool unsafe = false;
    double crit = std::numeric_limits<double>::max();
    double o_in_best = 0.0;
    double o_out_best = 0.0;
    bool candidate = heading_ok && speed_ok && gap_ok && !follower && !yielded;
    if (candidate) {
      const auto ei = ego_interval_lazy();
      if (!ei) {
        candidate = false;  // ego never crosses the attention area on its path (dec.note): no
                            // conflict possible
      } else {
        ego_in = ei->first;
        ego_out = ei->second;
      }
    }
    if (candidate) {
      for (const auto & pp : object.kinematics.predicted_paths) {
        double horizon = 0.0;
        const auto poses = resample_path(pp, 0.1, &horizon);
        if (poses.size() < 2) {
          continue;
        }
        std::optional<double> o_in{};
        size_t k_in = 0;
        for (size_t k = 0; k < poses.size(); ++k) {
          const auto poly = autoware_utils::to_polygon2d(poses.at(k), object.shape);
          if (bg::intersects(poly, ego_left) || bg::intersects(poly, ego_entry)) {
            o_in = 0.1 * static_cast<double>(k);
            k_in = k;
            break;
          }
        }
        if (!o_in) {
          continue;
        }
        std::optional<double> o_out{};
        for (size_t k = poses.size(); k-- > k_in;) {
          const auto poly = autoware_utils::to_polygon2d(poses.at(k), object.shape);
          if (bg::intersects(poly, ego_right)) {
            o_out = 0.1 * static_cast<double>(k);
            break;
          }
        }
        if (!o_out) {
          continue;
        }
        // a vehicle braking hard enough to stop >= 1 m before it reaches ego's turn path will not
        // meet ego ("be dynamic": its own motion shows it): no conflict on this path this cycle;
        // checked again as soon as it stops braking
        double d_in = 0.0;
        for (size_t k = 1; k <= k_in && k < poses.size(); ++k) {
          d_in += std::hypot(
            poses.at(k).position.x - poses.at(k - 1).position.x,
            poses.at(k).position.y - poses.at(k - 1).position.y);
        }
        if (decel >= prm.brake_decel && v * v / (2.0 * decel) + 1.0 < d_in) {
          char bb[96];
          std::snprintf(
            bb, sizeof(bb), " path(%.2f)[%.1f,%.1f]BRAKES(stop %.1f < %.1f m)", pp.confidence,
            o_in.value(), o_out.value(), v * v / (2.0 * decel), d_in);
          iv += bb;
          continue;
        }
        // overlap with margins (upstream case0 / case1; case2 dropped, see the file header)
        const bool gone_before_ego = ego_in > o_out.value() + prm.ttc_end_margin;
        const bool enters_while_ego_in = o_in.value() - prm.ttc_start_margin < ego_out;
        const bool u = !gone_before_ego && enters_while_ego_in;
        char buf[96];
        std::snprintf(
          buf, sizeof(buf), " path(%.2f)[%.1f,%.1f]%s", pp.confidence, o_in.value(), o_out.value(),
          u ? "!" : "");
        iv += buf;
        if (u) {
          unsafe = true;
          if (o_in.value() < crit) {
            crit = o_in.value();
            o_in_best = o_in.value();
            o_out_best = o_out.value();
          }
        }
      }
    }
    if (nero_bs_debug()) {
      RCLCPP_INFO(
        logger_,
        "[NERO_BS] t %.2f lane %ld veh obj %s %s v %.2f dyaw %+.0f%s width %.2f gap %.2f%s clear "
        "%.2f%s ego "
        "[%.2f,%.2f]%s "
        "-> %s",
        std::fmod(now.seconds(), 1000.0), lane_id_, short_id(object).c_str(), label_name(label), v,
        dyaw * 180.0 / M_PI, heading_ok ? "" : "(no)", object.shape.dimensions.y, lat_gap,
        gap_ok ? "" : "(no)", squeeze_clear,
        follower ? " FOLLOWER" : (yielded ? " STOOD(yield)" : ""), ego_in, ego_out,
        iv.empty() ? " no crossing" : iv.c_str(),
        unsafe ? "UNSAFE"
               : (heading_ok && speed_ok && gap_ok && !follower && !yielded ? "safe" : "ignored"));
    }
    if (unsafe) {
      min_unsafe_clear = std::min(min_unsafe_clear, squeeze_clear);
    }
    if (unsafe && (!worst || crit < std::get<0>(worst.value()))) {
      worst = std::make_tuple(crit, object, o_in_best, o_out_best);
    }
  }
  dec.active = true;
  if (worst) {
    dec.unsafe_now = true;
    nero_veh_last_unsafe_ = now;
    nero_veh_last_object_ = std::get<1>(worst.value());
  }
  dec.hold = dec.unsafe_now || (nero_veh_last_unsafe_ &&
                                (now - nero_veh_last_unsafe_.value()).seconds() < prm.debounce);
  if (dec.hold) {
    dec.object = dec.unsafe_now ? std::get<1>(worst.value()) : nero_veh_last_object_;
  }

  // --- stop points (upstream generate_stop_points + the smooth-stop choice, with the car's limits)
  // ---
  const auto footprint = vi.createFootprint(0.0, 0.0);
  const size_t search_start = static_cast<size_t>(std::max<int>(
    0, static_cast<int>(lane_start) -
         static_cast<int>(std::ceil(vi.vehicle_length_m / interpolated_path_info.ds))));
  std::optional<size_t> default_ip{};
  for (size_t i = search_start; i <= lane_end && i < path_ip.points.size(); ++i) {
    const auto fp = autoware_utils::transform_vector(
      footprint, autoware_utils::pose2transform(path_ip.points.at(i).point.pose));
    const bool hit = bg::intersects(fp, entry_line_tl);
    if (i == search_start && hit) {
      break;
    }
    if (hit) {
      default_ip = i;
      break;
    }
  }
  // commit point: where ego's footprint first reaches the far side's lane, i.e. the old module's
  // critical line (footprint on the straight sibling's LEFT bound = the centre line carried into
  // the junction: the oncoming half begins); without a straight sibling the virtual straight line
  // of ego's left edge (+5 cm). Upstream's critical line (ego's left edge leaves its straight band,
  // -1 m) lies at the junction entry on our tight turns, before the robot's own intersection stand
  // (logsim b7 lp_b7_fix2_a: the stand 0.6 m past it, the check latched before the passer came).
  std::optional<size_t> critical_ip{};
  {
    lanelet::BasicLineString2d cline;
    if (sibling_straight_lanelet_) {
      for (const auto & p : sibling_straight_lanelet_->leftBound()) {
        cline.emplace_back(p.x(), p.y());
      }
    } else {
      const lanelet::BasicPoint2d shift{-nrm.y() * 0.05, nrm.x() * 0.05};
      cline = {virtual_straight_2d.front() + shift, virtual_straight_2d.back() + shift};
    }
    for (size_t i = search_start; i <= lane_end && i < path_ip.points.size(); ++i) {
      const auto fp = autoware_utils::transform_vector(
        footprint, autoware_utils::pose2transform(path_ip.points.at(i).point.pose));
      if (bg::intersects(fp, cline)) {
        critical_ip = i;
        break;
      }
    }
  }
  // the turn lanelet's map stop line, as the car's intersection module takes it
  // (getStopLineIndexFromMap: the first path segment crossing the road_marking stop_line extended
  // by stop_line_extend_length, minus base_link-to-front): the robot stops there anyway (the
  // stop_line module, same line, hold_stop_margin_distance 2.0 m). A hold at the entry instead
  // (56366: 2.2 m before it) made the robot creep and stop again after the hold (+4-6 s, lp56366
  // v2_onc_dp0). Used for passers that are no squeezers (the squeezer keeps its straight wait).
  std::optional<size_t> map_stop_ip{};
  {
    const auto turn_ll =
      planner_data_->route_handler_->getLaneletMapPtr()->laneletLayer.get(lane_id_);
    for (const auto & rm : turn_ll.regulatoryElementsAs<lanelet::autoware::RoadMarking>()) {
      const std::string type = rm->roadMarking().attributeOr(lanelet::AttributeName::Type, "none");
      if (type != lanelet::AttributeValueString::StopLine) {
        continue;
      }
      const autoware_utils::LineString2d ext = planning_utils::extendLine(
        rm->roadMarking().front(), rm->roadMarking().back(),
        planner_data_->stop_line_extend_length);
      const int b2f =
        static_cast<int>(std::ceil(vi.max_longitudinal_offset_m / interpolated_path_info.ds));
      for (size_t i = lane_start; i < lane_end && i + 1 < path_ip.points.size(); ++i) {
        const auto & pf = path_ip.points.at(i).point.pose.position;
        const auto & pb = path_ip.points.at(i + 1).point.pose.position;
        const autoware_utils::LineString2d seg{{pf.x, pf.y}, {pb.x, pb.y}};
        std::vector<autoware_utils::Point2d> cps;
        bg::intersection(ext, seg, cps);
        if (!cps.empty()) {
          if (static_cast<int>(i) - b2f >= 0) {
            map_stop_ip = static_cast<size_t>(static_cast<int>(i) - b2f);
          }
          break;
        }
      }
      break;  // only one stop line exists (as the intersection module)
    }
  }
  const double braking_dist = planning_utils::calcJudgeLineDistWithJerkLimit(
    velocity, acceleration, planner_data_->max_stop_acceleration_threshold,
    planner_data_->max_stop_jerk_threshold, planner_data_->delay_response_time);
  const auto closest_ip = autoware::motion_utils::findFirstNearestIndexWithSoftConstraints(
    path_ip.points, current_pose, planner_data_->ego_nearest_dist_threshold,
    planner_data_->ego_nearest_yaw_threshold);
  auto ahead_dist = [&](const size_t idx) {
    return (static_cast<double>(idx) - static_cast<double>(closest_ip)) * interpolated_path_info.ds;
  };
  const size_t instant_ip = std::min(
    static_cast<size_t>(closest_ip + std::ceil(braking_dist / interpolated_path_info.ds)),
    path_ip.points.size() - 1);
  // for a passer squeezing past (its right side < squeeze_clearance = 1.0 m from the robot's
  // straight band): a stop before the robot's left edge leaves its straight band by 0.1 m (minus
  // 0.2 m), so that the standing robot's nose is not in its way (at 56366 the robot stands at its
  // stop line already 11 deg into the turn: lp56366 v5_mid_dp0 contact while standing)
  std::optional<size_t> band_ip{};
  if (dec.hold && min_unsafe_clear < squeeze_clearance) {
    const lanelet::BasicPoint2d shift{-nrm.y() * 0.1, nrm.x() * 0.1};
    const lanelet::BasicLineString2d vline{
      virtual_straight_2d.front() + shift, virtual_straight_2d.back() + shift};
    for (size_t i = search_start; i <= lane_end && i < path_ip.points.size(); ++i) {
      const auto fp = autoware_utils::transform_vector(
        footprint, autoware_utils::pose2transform(path_ip.points.at(i).point.pose));
      if (bg::intersects(fp, vline)) {
        band_ip = static_cast<size_t>(std::max<int>(
          static_cast<int>(search_start),
          static_cast<int>(i) - static_cast<int>(std::ceil(0.2 / interpolated_path_info.ds))));
        break;
      }
    }
  }
  // the latest stop: the commit point (the old critical line); a robot not yet past it and still
  // slow (<= 1.5 m/s) may stop up to 1.0 m beyond it (from a crawl it stops within ~1 m: better
  // than crossing slowly in front of the passer; lp56366 v11_onc_dp2: the robot pulling off at
  // 0.7-0.9 m/s, its reachable stop 0.2 m past the commit point, turned into an 11 m/s passer).
  // Faster or past it: committed (upstream: no stop inside the conflict; the obstacle modules
  // stay).
  std::optional<size_t> limit_ip = critical_ip;
  if (critical_ip && closest_ip <= critical_ip.value() && velocity <= 1.5) {
    limit_ip =
      critical_ip.value() + static_cast<size_t>(std::ceil(1.0 / interpolated_path_info.ds));
  }
  // a squeezer's own commit point: a MOVING robot (it pulled off before the squeezer was seen) is
  // committed once its stop would lie where its footprint reaches the squeezer's line (its right
  // side, min_unsafe_clear left of the straight band, -0.1 m): stopped there it stands in the
  // squeezer's way, where the car's robot (no stop) turns clear (lp56366 v8_lft_dp3: the car no
  // contact, a stop 1.4 m into the turn: contact). A standing robot still holds where it stands (it
  // never pulls off into a passer).
  if (dec.hold && min_unsafe_clear < squeeze_clearance && velocity > 0.1) {
    const double sq = std::max(min_unsafe_clear - 0.1, 0.05);
    const lanelet::BasicPoint2d shift{-nrm.y() * sq, nrm.x() * sq};
    const lanelet::BasicLineString2d sline{
      virtual_straight_2d.front() + shift, virtual_straight_2d.back() + shift};
    std::optional<size_t> sq_ip{};
    for (size_t i = search_start; i <= lane_end && i < path_ip.points.size(); ++i) {
      const auto fp = autoware_utils::transform_vector(
        footprint, autoware_utils::pose2transform(path_ip.points.at(i).point.pose));
      if (bg::intersects(fp, sline)) {
        sq_ip = i;
        break;
      }
    }
    if (sq_ip && (!limit_ip || sq_ip.value() < limit_ip.value())) {
      limit_ip = sq_ip;
    }
  }
  std::optional<size_t> chosen_ip{};
  const char * where = "none";
  // the map stop line (else the entry line), for a squeezer the earliest of the entry line and the
  // band point, if ego can still stop there
  // smoothly before the limit; else where ego can stop (a standing robot: where it stands) if that
  // is before the limit; else none (committed)
  std::optional<size_t> first_ip{};
  const char * first_where = "none";
  if (default_ip && default_ip.value() >= closest_ip) {
    first_ip = default_ip;
    first_where = "entry";
  }
  // no squeezer: the map stop line instead, if it lies past the entry (where ego stops anyway)
  if (
    !(dec.hold && min_unsafe_clear < squeeze_clearance) && map_stop_ip &&
    map_stop_ip.value() >= closest_ip &&
    (!default_ip || map_stop_ip.value() > default_ip.value())) {
    first_ip = map_stop_ip;
    first_where = "stopline";
  }
  if (
    band_ip && band_ip.value() >= closest_ip && (!first_ip || band_ip.value() < first_ip.value())) {
    first_ip = band_ip;
    first_where = "band";
  }
  if (
    first_ip && ahead_dist(first_ip.value()) > braking_dist &&
    (!limit_ip || first_ip.value() <= limit_ip.value())) {
    chosen_ip = first_ip;
    where = first_where;
  } else if (!limit_ip || instant_ip <= limit_ip.value()) {
    chosen_ip = instant_ip;
    where = "instant";
  }
  // "where ego can stop" is latched once chosen (recomputed from ego each cycle it moves on with a
  // rolling robot and is never reached: lp56366 v11_lft_dp2 rolled 1 s at 0.9 m/s toward a stop
  // always 0.8-1.0 m ahead); it is kept while it is still ahead of ego, and dropped when the hold
  // ends. The entry / band points are fixed lines and stay per cycle (latched from the hold's first
  // cycle, 15 m back, the entry stop lay 0.3 m earlier: 2.2 m before the stop line, outside
  // stop_line's hold_stop_margin_distance 2.0 m, so the robot crept and stopped again: +6 s)
  const bool chosen_instant = (chosen_ip && std::string(where) == "instant");
  if (dec.hold && nero_veh_stop_pose_) {
    std::optional<size_t> li{};
    double best = 0.5;  // m
    for (size_t i = (closest_ip > 10 ? closest_ip - 10 : 0); i < path_ip.points.size(); ++i) {
      const double d = autoware_utils::calc_distance2d(
        path_ip.points.at(i).point.pose.position, nero_veh_stop_pose_->position);
      if (d < best) {
        best = d;
        li = i;
      }
    }
    if (li && li.value() + 1 >= closest_ip) {
      chosen_ip = std::max(li.value(), closest_ip);
      where = "latched";
    } else {
      nero_veh_stop_pose_.reset();
    }
  }
  if (dec.hold && chosen_instant && !nero_veh_stop_pose_) {
    nero_veh_stop_pose_ = path_ip.points.at(chosen_ip.value()).point.pose;
  }
  if (!dec.hold) {
    nero_veh_stop_pose_.reset();
  }
  if (!dec.hold) {
    // over the pass judge line once ego is past the commit point (it will not be checked again, as
    // upstream's latch)
    if (critical_ip && closest_ip > critical_ip.value()) {
      nero_veh_over_pass_judge_ = true;
    }
  } else if (chosen_ip) {
    // production line (no debug switch): once per second while the vehicle check holds the robot
    RCLCPP_INFO_THROTTLE(
      logger_, *clock_, 1000,
      "[NERO_BS] lane %ld vehicle hold: obj %s (%s) ego [%.1f, %.1f] s, stop %s %.1f m ahead",
      lane_id_, dec.object ? short_id(dec.object.value()).c_str() : "-",
      dec.unsafe_now ? "crossing ego's turn path" : "debounce", ego_in, ego_out, where,
      ahead_dist(chosen_ip.value()));
    const auto inserted = nero_insert_point_index(
      path_ip.points.at(chosen_ip.value()).point.pose, path,
      planner_data_->ego_nearest_dist_threshold, planner_data_->ego_nearest_yaw_threshold);
    if (inserted) {
      dec.stop_line_idx = inserted->first;
      dec.inserted = inserted->second;
    }
  }
  if (nero_bs_debug()) {
    RCLCPP_INFO(
      logger_,
      "[NERO_BS] t %.2f lane %ld veh %s%s ego v %.2f [%.2f,%.2f] restart %.2f gap %.2f stop %s "
      "(entry %s stopline %s crit "
      "%s instant %.1f m, brake %.2f m)%s%s",
      std::fmod(now.seconds(), 1000.0), lane_id_,
      dec.unsafe_now ? "UNSAFE" : (dec.hold ? "HOLD(debounce)" : "safe"),
      dec.object ? (" by " + short_id(dec.object.value())).c_str() : "", velocity, ego_in, ego_out,
      time_to_restart, lat_gap, where,
      default_ip ? (std::to_string(ahead_dist(default_ip.value())).substr(0, 5) + " m").c_str()
                 : "-",
      map_stop_ip ? (std::to_string(ahead_dist(map_stop_ip.value())).substr(0, 5) + " m").c_str()
                  : "-",
      critical_ip ? (std::to_string(ahead_dist(critical_ip.value())).substr(0, 5) + " m").c_str()
                  : "-",
      ahead_dist(instant_ip), braking_dist, nero_veh_over_pass_judge_ ? " OVER-PASS-JUDGE" : "",
      (dec.hold && !chosen_ip) ? " TOO-LATE(no stop)" : "");
  }
  return dec;
}

void BlindSpotModule::neroDebugOldArea(
  const lanelet::ConstLanelets & blind_spot_lanelets,
  const geometry_msgs::msg::Pose & stop_line_pose, const lanelet::CompoundPolygon3d & area,
  const double ego_time_to_reach_stop_line, const bool vru_collision) const
{
  if (!nero_bs_debug()) {
    return;
  }
  const auto stop_line_arc_ego =
    lanelet::utils::getArcCoordinates(blind_spot_lanelets, stop_line_pose).length;
  int n = 0;
  for (const auto & object : planner_data_->predicted_objects->objects) {
    const auto & pos = object.kinematics.initial_pose_with_covariance.pose.position;
    if (!bg::within(to_bg2d(pos), lanelet::utils::to2D(area))) {
      continue;
    }
    ++n;
    const auto arc = lanelet::utils::getArcCoordinates(
                       blind_spot_lanelets, object.kinematics.initial_pose_with_covariance.pose)
                       .length;
    const double v = object.kinematics.initial_twist_with_covariance.twist.linear.x;
    const double ttc = ego_time_to_reach_stop_line - (arc - stop_line_arc_ego) / v;
    const bool target = isTargetObjectType(object);
    RCLCPP_INFO(
      logger_,
      "[NERO_BS] t %.2f lane %ld old-area obj %s class[0] %s (best %s) v %.2f s-stop %.1f ttc %.2f "
      "%s%s",
      std::fmod(clock_->now().seconds(), 1000.0), lane_id_, short_id(object).c_str(),
      label_name(object.classification.at(0).label), label_name(highest_label(object)), v,
      arc - stop_line_arc_ego, ttc,
      target ? "TARGET" : "DROPPED-by-class (scene.cpp isTargetObjectType)",
      (target && ttc > planner_param_.ttc_min && ttc < planner_param_.ttc_max) ? " in-ttc" : "");
  }
  RCLCPP_INFO(
    logger_, "[NERO_BS] t %.2f lane %ld old-area objects %d ego_t %.2f vru %s",
    std::fmod(clock_->now().seconds(), 1000.0), lane_id_, n, ego_time_to_reach_stop_line,
    vru_collision ? "STOP" : "go");
}

}  // namespace autoware::behavior_velocity_planner
