// Copyright 2026 Nero
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

#ifndef NERO_SQUEEZE_HPP_
#define NERO_SQUEEZE_HPP_

#include "autoware/behavior_path_static_obstacle_avoidance_module/data_structs.hpp"

#include <autoware_utils/geometry/boost_polygon_utils.hpp>
#include <autoware_utils/math/normalization.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/utils.h>

#include <autoware_perception_msgs/msg/predicted_objects.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <unique_identifier_msgs/msg/uuid.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/algorithms/distance.hpp>

#include <autoware_perception_msgs/msg/object_classification.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace nero_squeeze
{
// NERO_SOA_SQUEEZE=<veh_clear_m>,<band_m>,<v_mps>,<decel>,<range_m>,<window_m>,<gate_m> (mainlane-12 2026-11-14; Ahmed 11-12 13:5x "that place has
// cars so we have to squeeze through": the 80325 work zone with cars queued in 80344 beside the barrier). Off unless field 1 > 0.
// Active only in a SQUEEZE: a footprint on the module's reference path (the lane centre) within <range_m> ahead comes within
// <gate_m> (default 1.5) of BOTH a still UNKNOWN target (or a piece still UNSTABLE) that needs room (its overhang inside the stock needs-avoiding line: half the
// robot's width + the UNKNOWN hard margin) and a standing same-way vehicle (those vehicles = the squeeze's vehicles; the 11-08
// trips: never outside the 80325 work zone, research/walls/squeeze/census/REPORT.md). Then:
//  (1) each standing object's clip of the module's drivable bound uses that object's own hard margin (stock passes
//      vehicle_width / 2 into generateObstaclePolygonsForDrivableArea, which subtracts vehicle_width / 2 again: the clip was the
//      hard margin + a quarter of the robot's width, 0.2 m wider than configured; upstream main the same). The barrier pieces'
//      shoulder distance, avoid margin and the shift-line validity check all see the corrected room; and a still UNKNOWN
//      target's room is the narrowest (clipped) bound over its next <window_m> (default 8) along the path: its shift is held
//      there (a sharp return is trimmed), so a piece just before a queued car must not ask for more than the car leaves;
//  (2) a squeeze vehicle (STANDING: < 0.3 m/s, the safety check's own stopped threshold; heading the same way, < 45 deg) is judged by the
//      planned body's clearance to its box along the candidate path (>= <veh_clear_m>, default 0.3 = the module's hard margin for
//      vehicles; obstacle_stop's is 0.1) instead of RSS's lateral band (0.4 x hysteresis 1.2): a car queued beside the squeeze is no
//      reaction-time case. A vehicle that moves (>= 0.3 m/s) or comes the other way keeps the stock RSS check;
//  (3) the module's output path is capped at <v_mps> wherever the robot's footprint comes within <band_m> of BOTH a still UNKNOWN
//      target and a squeeze vehicle (the squeeze itself), reached at <decel> m/s2 before it.
//  (4) field 8 <outline_m> (mainlane-13, default 0 = off): a still UNKNOWN target that needs room and whose outline is longer than
//      <outline_m> gets its overhang (and its clip of the drivable bound) from its OWN outline buffered by its envelope margin instead of
//      stock's envelope = ONE box in the frame of the path pose nearest the object: the car's chain publishes the 80325 corner as a
//      polygon that grows to ~18 m along the curved barrier, and that box reached the lane centre (overhang -0.73 -> +0.07 m in 0.5 s)
//      -> INSUFFICIENT_DRIVABLE_SPACE, the shift dropped, obstacle_stop braked -3.9 (m13c run 3). The outline is inside the box, so this
//      only removes room the box invented; the module's stored envelope stays the stock box (nothing carried out of a squeeze).
//  (5) field 9 <centre> (mainlane-13, default 0 = off): in a squeeze, every still UNKNOWN target within the gate (body clearance <
//      <gate_m>) that has room asks for the CENTRE of the room left to the squeeze's bound (at least the stock margin, at most the hard
//      bound limit) and counts as needing avoidance: the robot centres in the gap from the moment the squeeze is seen (~40 m), as a driver
//      squeezes by hand. Stock leaves a barrier 0.85-0.92 m off the path "not needed" and keeps the lane centre until the barrier's inner
//      edge (cones / its base) shows within ~7 m and crosses the needs line (0.71 m): too late (INSUFFICIENT_LONGITUDINAL_DISTANCE,
//      obstacle_stop -3.4 with the field robot's 0.1 m right offset: m13d). The room = from the piece's overhang to the NEAR SIDE of the
//      squeeze's standing vehicles on the other side (their outlines projected on the path; v1 used the window-limited bound, which before
//      the cars entered the window was 4.3 m wide and shifted the robot 1.4 m into a queued car). It asks for MORE room than stock,
//      except where stock's own margin would bring the body within <veh_clear_m> of those vehicles (v3, mainlane-14: a margin on a
//      one-pose box that jumped toward the lane centre made one candidate pass the queued car within 0.3 m -> unsafe -> overtake
//      v6's start-yield dropped the shift -> obstacle_stop at the corner for good): that margin is lowered to the centre (still >=
//      the stock hard margin to the barrier). Where the gap has no room for both (max < min) the stock module decides (yield).
//  (6) fields 10-12 (mainlane-14, default 0 = off): RECOGNITION from the moment the queued car is in range. Field 10 <pair_m>: a standing
//      vehicle pairs with the barrier along the lane stretch [first footprint near a barrier piece - 2 m, last + pair_m] instead of only at
//      the SAME footprint (the barrier's long side beside the queued cars is seen only 10-16 m ahead: the same-footprint pairing switched on
//      ~2 s late, and meanwhile stock RSS treated the STANDING queued car as traffic: unsafe -> yield, and NERO_SOA_GAP approved at a lower
//      approach speed a shift line made on the corner's first-sight box, which later passed the queued car at 0.27 m -> the start-yield ->
//      obstacle_stop for good, 4 of 4 such runs: m14a / m14c). Field 11 <bar_clear_m>: a barrier piece counts for the recognition when it is
//      within bar_clear_m of a centred body (not only inside the needs line: the corner's true edge sits 0.75-1.0 m off the centre and
//      "needs room" flickered). Field 12 <hold_s>: once recognised the squeeze holds hold_s (objects' clock) for its vehicles that still stand.
//      With field 10 the squeeze speed (3) applies where the body passes within band of a squeeze vehicle (the barrier beside it is seen late).
//      A vehicle pairs only with barrier pieces on the OTHER side of the reference path (census research/walls/squeeze14/census_pair: W 10-20
//      add nothing outside 80325 on the 11-08 trips; the only non-work-zone pairings had both on the same side).
//  (7) field 13 <keep> (mainlane-14, default 0 = off): in a squeeze, when this cycle's new lines would pass a standing squeeze vehicle under
//      veh_clear but the registered plan alone keeps it, the new lines are dropped for this cycle (re-judged next cycle) instead of the
//      path counting as unsafe (overtake v6's start-yield then dropped the whole shift: a stand for good).
//  v6 (mainlane-14, with field 9): the centring target SHIFT is held per barrier piece within 0.10 m (the module's 0.1 m shift quantisation):
//      a few-cm flicker of the corner's outline flipped the rounded line 0.1 / 0.2 m before the shift started, the registered line was
//      replaced and for ~0.3 s no shift was approved (obstacle_stop planned at the corner 9-10 m ahead, m14g run 4 / m14j run 1).
//  v7 (mainlane-14, with fields 10 and 12): in a recognised squeeze one cycle in which every target reads "absolutely not avoidable" (a one-
//      frame outline of the corner at -0.30 m) does not CANCEL the not-yet-started registered line (the module's state stays RUNNING for up to
//      hold_s after the last avoidable cycle): the cancel + re-approval left ~0.3 s without a shift (m14j runs 1 / 3).
// Nothing is stopped later, no obstacle_stop / AEB margin changes, moving objects untouched.
struct Env
{
  bool on{false};
  double veh_clear{0.3};
  double band{0.6};
  double v{1.4};
  double dec{0.8};
  double range{40.0};
  double window{8.0};
  double gate{1.5};
  double outline{0.0};
  double centre{0.0};
  double pair{0.0};       // field 10 (mainlane-14): barrier piece and vehicle paired along a stretch of the lane, 0 = the same footprint
  double bar_clear{0.0};  // field 11 (mainlane-14): a barrier piece counts for the recognition within this body clearance, 0 = needs room
  double hold{0.0};       // field 12 (mainlane-14): a recognised squeeze holds this long for its vehicles still standing, 0 = no hold
  double keep{0.0};       // field 13 (mainlane-14): new lines that would pass a squeeze vehicle under veh_clear give way to the registered plan
};

inline const Env & env()
{
  static const Env e = [] {
    Env t;
    if (const char * s = std::getenv("NERO_SOA_SQUEEZE"); s && *s) {
      const int n = std::sscanf(
        s, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &t.veh_clear, &t.band, &t.v, &t.dec, &t.range, &t.window, &t.gate,
        &t.outline, &t.centre, &t.pair, &t.bar_clear, &t.hold, &t.keep);
      t.on = n >= 1 && t.veh_clear > 0.0;
      t.band = std::max(t.band, t.veh_clear);
      t.v = std::max(t.v, 0.3);
      t.dec = std::max(t.dec, 0.1);
      t.range = std::clamp(t.range, 5.0, 80.0);
      t.window = std::clamp(t.window, 0.0, 20.0);
      t.gate = std::clamp(t.gate, t.band, 3.0);
      t.outline = std::clamp(t.outline, 0.0, 50.0);
      t.pair = std::clamp(t.pair, 0.0, 30.0);
      t.bar_clear = std::clamp(t.bar_clear, 0.0, 1.5);
      t.hold = std::clamp(t.hold, 0.0, 5.0);
    }
    if (t.on) {
      RCLCPP_INFO(
        rclcpp::get_logger("static_obstacle_avoidance"),
        "[NERO_SOA_SQUEEZE] on: standing vehicles >= %.2f m from the planned body (no RSS band), squeeze band %.2f m -> %.2f m/s "
        "(at %.1f m/s2), clip at each object's own hard margin, barrier room over the next %.0f m; only where the lane-centre "
        "footprint comes within %.1f m of a still barrier AND a standing vehicle (within %.0f m); outline overhang for barriers > %.1f m "
        "(0 = off), centre in the gap %s; barrier and vehicle paired within %.0f m along the lane (0 = the same footprint), a barrier "
        "counts within %.2f m of the body (0 = needs room), recognition held %.1f s, keep the registered plan over too-close new lines %s",
        t.veh_clear, t.band, t.v, t.dec, t.window, t.gate, t.range, t.outline, t.centre > 0.0 ? "on" : "off", t.pair, t.bar_clear, t.hold,
        t.keep > 0.0 ? "on" : "off");
    }
    return t;
  }();
  return e;
}

inline bool isVehicle(const uint8_t label)
{
  using autoware_perception_msgs::msg::ObjectClassification;
  return label == ObjectClassification::CAR || label == ObjectClassification::TRUCK ||
         label == ObjectClassification::BUS || label == ObjectClassification::TRAILER ||
         label == ObjectClassification::MOTORCYCLE;
}

// a still UNKNOWN target that needs room within range ahead (geometric: usable before the module's necessity is filled)
inline bool needsRoom(
  const autoware::behavior_path_planner::ObjectData & o, const double vehicle_width, const double unknown_hard_margin)
{
  using autoware_perception_msgs::msg::ObjectClassification;
  if (o.object.classification.empty()) return false;
  const auto best = std::max_element(
    o.object.classification.begin(), o.object.classification.end(),
    [](const auto & a, const auto & b) { return a.probability < b.probability; });
  if (best->label != ObjectClassification::UNKNOWN) return false;
  if (o.longitudinal < -5.0 || o.longitudinal > env().range || o.overhang_points.empty()) return false;
  return std::abs(o.overhang_points.front().first) < 0.5 * vehicle_width + unknown_hard_margin + 0.05;
}

// signed lateral offset of (x, y) from the path (nearest segment, left +); one pass over the path, no path copies (nero_lint)
template <class Points>
inline double signedLat(const Points & ref, const double x, const double y)
{
  double best = std::numeric_limits<double>::max();
  double lat = 0.0;
  for (size_t i = 1; i < ref.size(); ++i) {
    const auto & a = ref.at(i - 1).point.pose.position;
    const auto & b = ref.at(i).point.pose.position;
    const double sx = b.x - a.x;
    const double sy = b.y - a.y;
    const double l2 = sx * sx + sy * sy;
    if (l2 < 1.0e-12) continue;
    const double t = std::clamp(((x - a.x) * sx + (y - a.y) * sy) / l2, 0.0, 1.0);
    const double d = std::hypot(x - (a.x + t * sx), y - (a.y + t * sy));
    if (d < best) {
      best = d;
      lat = (sx * (y - a.y) - sy * (x - a.x)) / std::sqrt(l2);
    }
  }
  return lat;
}

// (5) the nearest side of the squeeze's vehicles on one side of the path (distance from the path, > 0), or nullopt
template <class Points>
inline std::optional<double> vehicleSide(
  const std::vector<unique_identifier_msgs::msg::UUID> & ids, const autoware_perception_msgs::msg::PredictedObjects * objs,
  const Points & ref, const bool left)
{
  if (objs == nullptr || ref.size() < 2) return std::nullopt;
  std::optional<double> out;
  for (const auto & o : objs->objects) {
    if (std::find(ids.begin(), ids.end(), o.object_id) == ids.end()) continue;
    const auto g = autoware_utils::to_polygon2d(o.kinematics.initial_pose_with_covariance.pose, o.shape);
    double near = std::numeric_limits<double>::max();
    bool on_side = false;
    for (const auto & p : g.outer()) {
      const double l = signedLat(ref, p.x(), p.y());
      if (left ? l > 0.0 : l < 0.0) {
        on_side = true;
        near = std::min(near, std::abs(l));
      } else {
        near = 0.0;  // straddles the path: no room on that side
      }
    }
    if (!on_side) continue;
    out = out ? std::min(*out, near) : near;
  }
  return out;
}

// (5) a still UNKNOWN target within the squeeze gate (body clearance < gate) and within range
inline bool inGate(const autoware::behavior_path_planner::ObjectData & o, const double vehicle_width)
{
  using autoware_perception_msgs::msg::ObjectClassification;
  if (o.object.classification.empty() || o.overhang_points.empty()) return false;
  const auto best = std::max_element(
    o.object.classification.begin(), o.object.classification.end(),
    [](const auto & a, const auto & b) { return a.probability < b.probability; });
  if (best->label != ObjectClassification::UNKNOWN) return false;
  if (o.longitudinal < -5.0 || o.longitudinal > env().range) return false;
  return std::abs(o.overhang_points.front().first) < 0.5 * vehicle_width + env().gate;
}

// (4) the object's own outline buffered by `buffer` (round joins: inside the buffered box), when it is a valid polygon longer than
// env().outline; std::nullopt = keep stock's box
inline std::optional<autoware_utils::Polygon2d> outlineEnvelope(
  const autoware::behavior_path_planner::ObjectData & o, const double buffer)
{
  namespace bg = boost::geometry;
  using autoware_utils::Polygon2d;
  if (env().outline <= 0.0) return std::nullopt;
  if (o.object.shape.type != autoware_perception_msgs::msg::Shape::POLYGON) return std::nullopt;
  Polygon2d poly = autoware_utils::to_polygon2d(o.object);
  bg::correct(poly);
  if (poly.outer().size() < 4 || !bg::is_valid(poly)) return std::nullopt;
  bg::model::box<autoware_utils::Point2d> b;
  bg::envelope(poly, b);
  if (std::hypot(b.max_corner().x() - b.min_corner().x(), b.max_corner().y() - b.min_corner().y()) < env().outline) {
    return std::nullopt;
  }
  bg::model::multi_polygon<Polygon2d> out;
  bg::buffer(
    poly, out, bg::strategy::buffer::distance_symmetric<double>(std::max(buffer, 0.0)), bg::strategy::buffer::side_straight(),
    bg::strategy::buffer::join_round(8), bg::strategy::buffer::end_round(8), bg::strategy::buffer::point_circle(8));
  if (out.empty()) return std::nullopt;
  const auto best = std::max_element(out.begin(), out.end(), [](const auto & a, const auto & c) { return bg::area(a) < bg::area(c); });
  Polygon2d r = *best;
  r.inners().clear();
  bg::correct(r);
  if (r.outer().size() < 4) return std::nullopt;
  return r;
}

struct Bx
{
  double x0{1e18}, y0{1e18}, x1{-1e18}, y1{-1e18};
  static Bx of(const autoware_utils::Polygon2d & g)
  {
    Bx b;
    for (const auto & q : g.outer()) {
      b.x0 = std::min(b.x0, q.x());
      b.y0 = std::min(b.y0, q.y());
      b.x1 = std::max(b.x1, q.x());
      b.y1 = std::max(b.y1, q.y());
    }
    return b;
  }
  bool apart(const Bx & o, const double m) const
  {
    return o.x0 > x1 + m || x0 > o.x1 + m || o.y0 > y1 + m || y0 > o.y1 + m;
  }
};

// the squeeze's vehicles (empty = no squeeze): standing same-way vehicles that come within gate of a lane-centre footprint which is also
// within gate of a still UNKNOWN target needing room (reference path sampled every 1 m from ego to range)
template <class Points>
inline std::vector<unique_identifier_msgs::msg::UUID> squeezeVehicles(
  const autoware::behavior_path_planner::ObjectDataArray & targets,
  const autoware::behavior_path_planner::ObjectDataArray & others,
  const autoware_perception_msgs::msg::PredictedObjects * objs, const Points & ref, const geometry_msgs::msg::Pose & ego,
  const double front, const double rear, const double width, const double unknown_hard_margin)
{
  std::vector<unique_identifier_msgs::msg::UUID> out;
  if (!env().on || objs == nullptr || ref.size() < 2) return out;
  // the same inputs give the same answer: the module asks up to 3 times a cycle plus once per NERO_SOA_GAP speed step (isSafePath)
  struct Key
  {
    const void * objs{nullptr};
    int32_t sec{0};
    uint32_t nsec{0};
    double ex{0.0}, ey{0.0}, r0x{0.0}, r0y{0.0}, r1x{0.0}, r1y{0.0};
    size_t nt{0}, no{0}, nr{0};
    bool operator==(const Key & o) const
    {
      return objs == o.objs && sec == o.sec && nsec == o.nsec && ex == o.ex && ey == o.ey && r0x == o.r0x && r0y == o.r0y &&
             r1x == o.r1x && r1y == o.r1y && nt == o.nt && no == o.no && nr == o.nr;
    }
  };
  const Key key{
    objs, objs->header.stamp.sec, objs->header.stamp.nanosec, ego.position.x, ego.position.y,
    ref.front().point.pose.position.x, ref.front().point.pose.position.y, ref.back().point.pose.position.x,
    ref.back().point.pose.position.y, targets.size(), others.size(), ref.size()};
  thread_local Key last_key;
  thread_local std::vector<unique_identifier_msgs::msg::UUID> last_out;
  thread_local bool has_last = false;
  if (has_last && key == last_key) return last_out;
  // field 12: a recognised squeeze holds for hold s (objects' clock) for its vehicles that still stand, so one cycle without a barrier
  // piece (an outline that flickers across the band) cannot hand the queued cars back to RSS (a yield) for that cycle
  static std::vector<unique_identifier_msgs::msg::UUID> held;  // shared: the planner timer may run on any executor thread
  static double held_t = -1.0;
  const double now_t = static_cast<double>(objs->header.stamp.sec) + 1.0e-9 * objs->header.stamp.nanosec;
  const auto finish = [&](std::vector<unique_identifier_msgs::msg::UUID> r) {
    if (!r.empty()) {
      held = r;
      held_t = now_t;
    } else if (env().hold > 0.0 && !held.empty() && now_t >= held_t && now_t - held_t <= env().hold) {
      for (const auto & o : objs->objects) {
        if (std::find(held.begin(), held.end(), o.object_id) == held.end()) continue;
        const auto & tw = o.kinematics.initial_twist_with_covariance.twist.linear;
        if (std::hypot(tw.x, tw.y) < 0.3) r.push_back(o.object_id);
      }
    }
    last_key = key;
    last_out = r;
    has_last = true;
    return r;
  };
  const double gate = env().gate;
  std::vector<std::pair<autoware_utils::Polygon2d, Bx>> bar;
  std::vector<int> bar_side;  // +1 left of the reference path, -1 right (field 10: a vehicle pairs only with a barrier on the other side)
  // field 11: a barrier piece within bar_clear of a centred body counts too (the 80325 corner's true inner edge sits 0.75-1.0 m off the
  // lane centre, around the needs line: "needs room" flickered and the squeeze switched on late / off for cycles)
  const auto counts = [&](const autoware::behavior_path_planner::ObjectData & o) {
    if (needsRoom(o, width, unknown_hard_margin)) return true;
    if (env().bar_clear <= 0.0 || o.object.classification.empty() || o.overhang_points.empty()) return false;
    const auto best = std::max_element(
      o.object.classification.begin(), o.object.classification.end(),
      [](const auto & a, const auto & b) { return a.probability < b.probability; });
    if (best->label != autoware_perception_msgs::msg::ObjectClassification::UNKNOWN) return false;
    if (o.longitudinal < -5.0 || o.longitudinal > env().range) return false;
    return std::abs(o.overhang_points.front().first) < 0.5 * width + env().bar_clear;
  };
  const auto side_of = [&](const autoware_utils::Polygon2d & g) {
    autoware_utils::Point2d c;
    boost::geometry::centroid(g, c);
    return signedLat(ref, c.x(), c.y()) >= 0.0 ? 1 : -1;
  };
  for (const auto & o : targets) {
    if (!counts(o)) continue;
    const auto g = autoware_utils::to_polygon2d(o.object);
    bar.emplace_back(g, Bx::of(g));
    bar_side.push_back(side_of(g));
  }
  // a barrier piece the module still holds back as UNSTABLE (the car's NERO_SOA_UNKNOWN_STABLE_S: 3 s after first sight) already
  // makes the squeeze: the shift toward the gap must not wait for it (RSS would veto the early shift beside the queued car)
  for (const auto & o : others) {
    if (o.info != autoware::behavior_path_planner::ObjectInfo::UNSTABLE_OBJECT) continue;
    if (!counts(o)) continue;
    const auto g = autoware_utils::to_polygon2d(o.object);
    bar.emplace_back(g, Bx::of(g));
    bar_side.push_back(side_of(g));
  }
  if (bar.empty()) return finish(out);
  const double ego_yaw = tf2::getYaw(ego.orientation);
  std::vector<std::pair<autoware_utils::Polygon2d, Bx>> veh;
  std::vector<int> veh_side;
  std::vector<unique_identifier_msgs::msg::UUID> ids;
  for (const auto & o : objs->objects) {
    if (o.classification.empty()) continue;
    const auto best = std::max_element(
      o.classification.begin(), o.classification.end(),
      [](const auto & a, const auto & b) { return a.probability < b.probability; });
    if (!isVehicle(best->label)) continue;
    const auto & tw = o.kinematics.initial_twist_with_covariance.twist.linear;
    if (std::hypot(tw.x, tw.y) >= 0.3) continue;
    const auto & op = o.kinematics.initial_pose_with_covariance.pose;
    if (std::hypot(op.position.x - ego.position.x, op.position.y - ego.position.y) > env().range + 10.0) continue;
    if (std::abs(autoware_utils::normalize_radian(tf2::getYaw(op.orientation) - ego_yaw)) >= M_PI / 4.0) continue;
    const auto g = autoware_utils::to_polygon2d(op, o.shape);
    veh.emplace_back(g, Bx::of(g));
    veh_side.push_back(side_of(g));
    ids.push_back(o.object_id);
  }
  if (veh.empty()) return finish(out);
  size_t i0 = 0;
  double d0 = std::numeric_limits<double>::max();
  for (size_t i = 0; i < ref.size(); ++i) {
    const auto & p = ref.at(i).point.pose.position;
    const double d = std::hypot(p.x - ego.position.x, p.y - ego.position.y);
    if (d < d0) {
      d0 = d;
      i0 = i;
    }
  }
  std::vector<bool> hit(veh.size(), false);
  // lane-centre footprints every 1 m to range; those within gate of a barrier piece; field 10: a vehicle pairs with the barrier along the
  // stretch [first near footprint - 2 m, last near footprint + pair] (the barrier's long side beside the queued cars is seen only 10-16 m
  // ahead, so the same-footprint pairing switched on ~2 s late and the queued car was RSS traffic meanwhile: m14a / m14c)
  struct Fp
  {
    double s;
    autoware_utils::Polygon2d g;
    Bx b;
    bool near_bar;
  };
  std::vector<Fp> fps;
  double s = 0.0;
  double s_last = -1.0e9;
  // per side of the reference path: the stretch of footprints near a barrier piece on that side ([0] right, [1] left)
  double lo_side[2] = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  double hi_side[2] = {-std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()};
  for (size_t i = i0; i < ref.size() && s <= env().range; ++i) {
    if (i > i0) {
      const auto & a = ref.at(i - 1).point.pose.position;
      const auto & b = ref.at(i).point.pose.position;
      s += std::hypot(b.x - a.x, b.y - a.y);
    }
    if (i > i0 && i + 1 < ref.size() && s - s_last < 1.0) continue;
    s_last = s;
    Fp f{s, autoware_utils::to_footprint(ref.at(i).point.pose, front, rear, width), {}, false};
    f.b = Bx::of(f.g);
    for (size_t q = 0; q < bar.size(); ++q) {
      const auto & [g, b] = bar[q];
      if (!f.b.apart(b, gate) && boost::geometry::distance(f.g, g) < gate) {
        f.near_bar = true;
        const int k = bar_side[q] > 0 ? 1 : 0;
        lo_side[k] = std::min(lo_side[k], f.s);
        hi_side[k] = std::max(hi_side[k], f.s);
      }
    }
    fps.push_back(std::move(f));
  }
  for (const auto & f : fps) {
    if (env().pair <= 0.0 && !f.near_bar) continue;
    for (size_t k = 0; k < veh.size(); ++k) {
      if (env().pair > 0.0) {
        // field 10: on the stretch of the barrier pieces on the OTHER side of the reference path (census: every pairing outside the work
        // zone was a barrier-like piece and a parked car on the same side; a narrowing needs them on opposite sides)
        const int o = veh_side[k] > 0 ? 0 : 1;
        if (!(f.s >= lo_side[o] - 2.0 && f.s <= hi_side[o] + env().pair)) continue;
      }
      if (!hit[k] && !f.b.apart(veh[k].second, gate) && boost::geometry::distance(f.g, veh[k].first) < gate) hit[k] = true;
    }
  }
  for (size_t k = 0; k < veh.size(); ++k) {
    if (hit[k]) out.push_back(ids[k]);
  }
  return finish(out);
}

// field 13: set by the safety check when its last unsafe verdict was a squeeze vehicle's clearance (reset at each check)
inline bool & lastUnsafeWasSqueeze()
{
  static bool b = false;
  return b;
}

inline bool contains(const std::vector<unique_identifier_msgs::msg::UUID> & ids, const unique_identifier_msgs::msg::UUID & id)
{
  return std::any_of(ids.begin(), ids.end(), [&](const auto & x) { return x.uuid == id.uuid; });
}
}  // namespace nero_squeeze

#endif  // NERO_SQUEEZE_HPP_
