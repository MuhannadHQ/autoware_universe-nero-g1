// Copyright 2026 Nero (crosshold job, 2026-10-28, QUEUE SLX)
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

// NERO_IX_CROSSING_HOLD=<gap m>,<depth m>,<early s>,<late s>[,<v_min m/s>,<release s>,
//   <delay s>,<acc m/s2>,<slow_r m>,<occ_t s>,<occ_depth m>,<v_slow m/s>]
// (crosshold job, 2026-10-28, QUEUE SLX): at a stop-sign lane where this module has no attention
// lane (the crossing road is not in the map: 59721 / stop line 110666 on e23, whose through road
// south-west of the mouth is unmapped; 16 such stop-sign lanes on e23), the module returns
// InternalError and nothing yields: the stop_line module (core) holds 0.1 s, obstacle_stop sees a
// crossing vehicle only once it is in front of the nose (10-27 #2 / #3: trucks 0.08-1.0 m from the
// nose, 6 operator STOPs). From the moment ego's nose is within 2 m of the stop line until it is
// <depth> m past it, every moving vehicle (the module's target classes, UNKNOWN included; >=
// <v_min> m/s, whatever lane, mapped or not) is checked against ego's own passage from where it is
// (standstill: <delay> s, then <acc> m/s2 up to the path's speed). If its predicted path (or a
// straight line at its speed when it has none; a decelerating object also as slowing on at its
// observed rate) comes within 0.5 m of ego's way at all (a crosser, not a car passing beside in its
// own lane), and, while ego would be moving, it comes closer than <gap> m to ego's footprint at any
// time within [t - <late>, t + <early>] of ego's time t there, ego is held at the stop line (or
// where it stands if it is past it) until no object does for <release> s (Apollo's stop-sign CREEP
// check: go only when no obstacle enters the first metres of the path within a time window). A
// moving ego stops for a new conflict only if standing where it would stop keeps it further from
// the object than going on does; otherwise it is committed for that object and goes on
// (obstacle_stop / AEB keep their role). Followers (behind ego, same heading) are not crossers. A
// standing ego with its nose at most 2.5 m past the line also waits while a vehicle (not UNKNOWN:
// those count from <v_min> on) slower than <v_slow> moves near its way and is not leaving it (>= 1
// m in the last 5 s, within <slow_r> m: a car manoeuvring or creeping at the mouth is about to do
// something its prediction does not tell, a parked one does not move; 10-27 #3's 4654 manoeuvred 19
// m off at 0.1-0.7 m/s and turned toward the junction 2 s after a release, 774a crept at the 59677
// mouth before it turned in). And a standing ego does not start while a vehicle will be across its
// first <occ_depth> m past the line within <occ_t> s (within 0.5 m of its footprint there: no
// creeping toward a passer, even one that would pass before ego gets there). Unset = the car
// (nothing runs). NERO_IX_CROSSING_HOLD_DEBUG=1: one line per cycle while in scope (log only).
//
// NERO_IX_CROSSING_HOLD_YIELD=<stood s>,<approach m>[,<near m>] (crosswait job, 2026-10-29, QUEUE
// STK1846): the slow-mover wait above ends for a vehicle that is letting ego go. 10-28 18:46 at
// 62867 it held a standing ego 10 s after the last crosser, with no end, for a car stopping beside
// / behind ego into the other lane of its own road (pointing away) and a truck waiting at the mouth
// for ego (a mutual wait) until the operator drove off. With this switch a slow mover does not hold
// a standing ego (1) while it is within <near> m of ego's way, wholly behind ego's nose, points
// backward relative to ego and has not moved forward along ego's heading by <approach> in the last
// <stood> s (it cannot come into ego's way ahead without turning around), or (2) once ego has stood
// <stood> s, while it is within <near> m of ego's way and has not come <approach> m closer to it in
// the last <stood> s (it
// stopped at the mouth, or moves along / away; one manoeuvring further off keeps holding as before:
// Ahmed 10-24, a driver slowing down and waiting is the signal for the robot to go; Apollo's
// stop-sign STOP stage waits only for vehicles that arrived first and drops the wait after
// stop_timeout_sec). Both are re-judged every cycle from the vehicle's motion, and a vehicle that
// let ego go and then moves again (>= <v_min> two samples in a row, not away from ego's way: the
// crazy driver who pulls out after all) no longer lets it go, at any speed: it
// holds ego standing with its nose up to <occ_depth> past the line, and stops a moving ego whose
// comfortable stop (0.3 s + 1 m/s2 + 0.3 m) still ends within <occ_depth> (else ego is committed
// for it, as for a crosser), until it is behind ego, leaves the way or has stood again. The
// crosser / no-start rules are unchanged (a vehicle >= <v_min> whose path comes into ego's way
// holds it whatever this switch says). Unset = the above exactly.

#include "scene_intersection.hpp"

#include <autoware/behavior_velocity_planner_common/utilization/boost_geometry_helper.hpp>
#include <autoware/behavior_velocity_planner_common/utilization/util.hpp>
#include <autoware/motion_utils/trajectory/trajectory.hpp>
#include <autoware_utils/geometry/boost_polygon_utils.hpp>
#include <autoware_utils/geometry/geometry.hpp>
#include <autoware_utils/math/normalization.hpp>
#include <autoware_utils/ros/uuid_helper.hpp>

#include <boost/geometry.hpp>

#include <lanelet2_core/primitives/BasicRegulatoryElements.h>
#include <tf2/utils.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <iterator>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace
{
struct NeroCrossingHoldParam
{
  bool on{false};
  double gap{2.0};
  double depth{12.0};
  double early{2.0};
  double late{1.0};
  double v_min{0.5};
  double release{0.5};
  double delay{0.5};
  double acc{0.5};
  double slow_r{20.0};
  double occ_t{3.0};
  double occ_depth{5.0};
  double v_slow{2.0};
};

const NeroCrossingHoldParam & nero_crossing_hold_param()
{
  static const NeroCrossingHoldParam p = [] {
    NeroCrossingHoldParam r;
    const char * e = std::getenv("NERO_IX_CROSSING_HOLD");
    if (e == nullptr || std::string(e).empty()) return r;
    double v[12] = {r.gap,   r.depth, r.early,  r.late,  r.v_min,     r.release,
                    r.delay, r.acc,   r.slow_r, r.occ_t, r.occ_depth, r.v_slow};
    const int n = std::sscanf(
      e, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &v[0], &v[1], &v[2], &v[3], &v[4],
      &v[5], &v[6], &v[7], &v[8], &v[9], &v[10], &v[11]);
    if (n < 4) return r;
    r.on = true;
    r.gap = std::max(0.0, v[0]);
    r.depth = std::max(1.0, v[1]);
    r.early = std::max(0.0, v[2]);
    r.late = std::max(0.0, v[3]);
    r.v_min = std::max(0.0, v[4]);
    r.release = std::max(0.0, v[5]);
    r.delay = std::max(0.0, v[6]);
    r.acc = std::max(0.1, v[7]);
    r.slow_r = std::max(0.0, v[8]);
    r.occ_t = std::max(0.0, v[9]);
    r.occ_depth = std::max(0.0, v[10]);
    r.v_slow = std::max(0.0, v[11]);
    return r;
  }();
  return p;
}

struct NeroCrossingYieldParam
{
  bool on{false};
  double stood{2.0};
  double approach{0.5};
  double near{12.0};
};

const NeroCrossingYieldParam & nero_crossing_yield_param()
{
  static const NeroCrossingYieldParam p = [] {
    NeroCrossingYieldParam r;
    const char * e = std::getenv("NERO_IX_CROSSING_HOLD_YIELD");
    if (e == nullptr || std::string(e).empty()) return r;
    double v[3] = {r.stood, r.approach, r.near};
    const int n = std::sscanf(e, "%lf,%lf,%lf", &v[0], &v[1], &v[2]);
    if (n < 1) return r;
    r.on = true;
    r.stood = std::max(0.5, v[0]);
    r.approach = std::max(0.0, v[1]);
    r.near = std::max(0.0, v[2]);
    return r;
  }();
  return p;
}

bool nero_crossing_hold_debug()
{
  static const bool on = [] {
    const char * v = std::getenv("NERO_IX_CROSSING_HOLD_DEBUG");
    return v != nullptr && std::string(v) == "1";
  }();
  return on;
}

// the stop lines of the lane's stop signs (traffic_sign whose sign is a stop_sign -> its ref_line)
std::vector<lanelet::ConstLineString3d> nero_stop_sign_lines(const lanelet::ConstLanelet & ll)
{
  std::vector<lanelet::ConstLineString3d> out;
  for (const auto & reg : ll.regulatoryElements()) {
    if (std::string(reg->attributeOr(lanelet::AttributeName::Subtype, "none")) != "traffic_sign")
      continue;
    bool stop_sign = false;
    for (const auto & ls : reg->getParameters<lanelet::ConstLineString3d>("refers")) {
      if (std::string(ls.attributeOr(lanelet::AttributeName::Subtype, "none")) == "stop_sign")
        stop_sign = true;
    }
    if (!stop_sign) continue;
    for (const auto & ls : reg->getParameters<lanelet::ConstLineString3d>("ref_line"))
      out.push_back(ls);
  }
  return out;
}

// the object's pose at path time t [s] along a predicted path (linear between its points, straight
// on at speed v past the end)
geometry_msgs::msg::Pose nero_pose_on_path(
  const std::vector<geometry_msgs::msg::Pose> & path, const double dt, const double t,
  const double v)
{
  if (path.size() < 2 || dt <= 0.0) {
    return path.empty() ? geometry_msgs::msg::Pose{} : path.front();
  }
  const double k = std::max(0.0, t) / dt;
  const size_t i = static_cast<size_t>(std::floor(k));
  if (i + 1 < path.size()) {
    const double r = k - static_cast<double>(i);
    auto p = path.at(i);
    p.position.x += r * (path.at(i + 1).position.x - p.position.x);
    p.position.y += r * (path.at(i + 1).position.y - p.position.y);
    return p;
  }
  const auto & a = path.at(path.size() - 2);
  const auto & b = path.back();
  const double yaw = std::atan2(b.position.y - a.position.y, b.position.x - a.position.x);
  const double extra = (t - dt * static_cast<double>(path.size() - 1)) * v;
  auto p = b;
  if (std::hypot(b.position.x - a.position.x, b.position.y - a.position.y) > 1e-3) {
    p.position.x += extra * std::cos(yaw);
    p.position.y += extra * std::sin(yaw);
  }
  return p;
}
}  // namespace

namespace autoware::behavior_velocity_planner
{
namespace bg = boost::geometry;

void IntersectionModule::neroCrossingHold(PathWithLaneId * path)
{
  const auto & p = nero_crossing_hold_param();
  if (!p.on) return;
  auto reset = [&]() {
    if (nero_xh_holding_) {
      RCLCPP_INFO(logger_, "[NERO_IX] crossing_hold: lane %ld GO (out of the hold zone)", lane_id_);
    }
    nero_xh_holding_ = false;
    nero_xh_committed_.clear();
    nero_xh_ego_stand_since_ = -1.0;
    nero_xh_letgo_.clear();
    nero_xh_revoked_.clear();
    nero_xh_again_stop_.reset();
    nero_xh_again_held_ = false;
  };
  // scope: a stop-sign lane where this module has no attention lane to check (it returned
  // InternalError)
  if (
    !intersection_lanelets_ || intersection_lanelets_->first_attention_area() ||
    has_traffic_light_) {
    return reset();
  }
  const auto lanelet_map_ptr = planner_data_->route_handler_->getLaneletMapPtr();
  const auto & assigned_lanelet = lanelet_map_ptr->laneletLayer.get(lane_id_);
  const auto lines = nero_stop_sign_lines(assigned_lanelet);
  if (lines.empty()) {
    return reset();
  }
  if (!nero_xh_scope_logged_) {
    nero_xh_scope_logged_ = true;
    RCLCPP_INFO(
      logger_,
      "[NERO_IX] crossing_hold: lane %ld in scope (stop sign, stop line %ld, no attention lane): "
      "gap %.1f m, "
      "depth %.1f m, early %.1f s, late %.1f s, v_min %.1f m/s, release %.1f s, ego delay %.1f s, "
      "acc %.2f m/s2, "
      "slow movers within %.0f m, no start while a vehicle is across the first %.1f m within %.1f "
      "s, slow movers "
      "below %.1f m/s",
      lane_id_, lines.front().id(), p.gap, p.depth, p.early, p.late, p.v_min, p.release, p.delay,
      p.acc, p.slow_r, p.occ_depth, p.occ_t, p.v_slow);
    if (nero_crossing_yield_param().on) {
      RCLCPP_INFO(
        logger_,
        "[NERO_IX] crossing_hold: lane %ld slow movers within %.0f m of its way yield: "
        "behind ego's nose pointing back, or not %.1f m closer to it in %.1f s once ego stood",
        lane_id_, nero_crossing_yield_param().near, nero_crossing_yield_param().approach,
        nero_crossing_yield_param().stood);
    }
  }
  if (path->points.size() < 2) return reset();
  const auto & ego_pose = planner_data_->current_odometry->pose;
  const auto & ego_pos = ego_pose.position;
  const double front = planner_data_->vehicle_info_.max_longitudinal_offset_m;
  const double rear = planner_data_->vehicle_info_.rear_overhang_m;

  // where the path crosses the stop line (arc length from ego's base_link)
  std::optional<double> s_line;
  {
    const auto & ls = lines.front();
    const LineString2d line =
      planning_utils::extendLine(ls.front(), ls.back(), planner_data_->stop_line_extend_length);
    for (size_t i = 0; i + 1 < path->points.size() && !s_line; ++i) {
      const auto & a = path->points.at(i).point.pose.position;
      const auto & b = path->points.at(i + 1).point.pose.position;
      const LineString2d seg{{a.x, a.y}, {b.x, b.y}};
      std::vector<Point2d> ips;
      bg::intersection(line, seg, ips);
      if (ips.empty()) continue;
      geometry_msgs::msg::Point q;
      q.x = ips.front().x();
      q.y = ips.front().y();
      q.z = a.z;
      s_line = autoware::motion_utils::calcSignedArcLength(path->points, ego_pos, q);
    }
  }
  if (!s_line) return reset();
  const double nose_past = front - s_line.value();  // + = ego's nose past the line
  if (nose_past < -2.0 || nose_past > p.depth) return reset();

  // ego's passage from where it is: standstill -> <delay>, then <acc> up to the path speed
  const double v0 = std::abs(planner_data_->current_velocity->twist.linear.x);
  const bool standing = v0 < 0.1;
  if (standing) nero_xh_committed_.clear();  // a standing ego can always wait
  const double delay = standing ? p.delay : 0.0;
  const double s_end =
    s_line.value() + p.depth - front;  // base_link where the nose is <depth> past the line
  // the path's speed at arc length s from ego (stops of other modules skipped: ego's passage when
  // nothing holds it)
  std::vector<std::pair<double, double>> v_of_s;
  {
    const size_t ego_seg = autoware::motion_utils::findNearestSegmentIndex(path->points, ego_pos);
    double last_v = 0.0;
    for (size_t i = ego_seg; i < path->points.size(); ++i) {
      const double s = autoware::motion_utils::calcSignedArcLength(path->points, ego_pos, i);
      const double v = path->points.at(i).point.longitudinal_velocity_mps;
      if (v > 0.1) last_v = v;
      v_of_s.emplace_back(s, last_v > 0.1 ? last_v : 1.0);
      if (s > s_end + 2.0) break;
    }
  }
  auto v_lim = [&](const double s) {
    double v = v_of_s.empty() ? 1.0 : v_of_s.front().second;
    for (const auto & [si, vi] : v_of_s) {
      if (si > s) break;
      v = vi;
    }
    return v;
  };
  struct EgoSample
  {
    double t;
    double s;
    autoware_utils::Polygon2d fp;
    geometry_msgs::msg::Point c;
  };
  const auto base_fp = planner_data_->vehicle_info_.createFootprint(0.0, 0.0);
  auto footprint_at = [&](const geometry_msgs::msg::Pose & pose) {
    autoware_utils::Polygon2d fp;
    const auto tf = autoware_utils::pose2transform(pose);
    for (const auto & q : autoware_utils::transform_vector(base_fp, tf))
      fp.outer().emplace_back(q.x(), q.y());
    bg::correct(fp);
    return fp;
  };
  std::vector<EgoSample> ego;
  {
    const double dt = 0.2;
    double s = 0.0, v = v0, t = 0.0;
    while (s <= s_end && t <= 15.0) {
      if (s >= 0.1) {
        const auto pose =
          autoware::motion_utils::calcLongitudinalOffsetPose(path->points, ego_pos, s);
        if (!pose) break;
        ego.push_back({t, s, footprint_at(pose.value()), pose->position});
      }
      t += dt;
      if (t > delay) {
        v = std::min(v_lim(s), v + p.acc * dt);
        s += v * dt;
      }
    }
  }
  const double t_end = ego.empty() ? 0.0 : ego.back().t;

  // a moving ego: where it would stand if it stopped now (its response + a soft stop + the docking
  // stop's run-on)
  const double d_stop = standing ? 0.0 : v0 * 0.3 + v0 * v0 / (2.0 * 1.0) + 0.3;
  std::optional<autoware_utils::Polygon2d> stop_fp;
  if (!standing) {
    const auto pose =
      autoware::motion_utils::calcLongitudinalOffsetPose(path->points, ego_pos, d_stop);
    if (pose) stop_fp = footprint_at(pose.value());
  }

  // the objects' speed history (for their observed deceleration)
  const double t_obj = rclcpp::Time(planner_data_->predicted_objects->header.stamp).seconds();
  const auto & yp = nero_crossing_yield_param();
  if (!standing) {
    nero_xh_ego_stand_since_ = -1.0;
  } else if (nero_xh_ego_stand_since_ < 0.0) {
    nero_xh_ego_stand_since_ = t_obj;
  }
  const double ego_stood = standing ? t_obj - nero_xh_ego_stand_since_ : 0.0;
  const bool dbg_wait = yp.on && nero_crossing_hold_debug();
  std::string dbg_slow, dbg_hold;
  for (auto it = nero_xh_vhist_.begin(); it != nero_xh_vhist_.end();) {
    if (it->second.empty() || t_obj - it->second.back().first > 2.0)
      it = nero_xh_vhist_.erase(it);
    else
      ++it;
  }
  for (auto it = nero_xh_phist_.begin(); it != nero_xh_phist_.end();) {
    if (it->second.empty() || t_obj - std::get<0>(it->second.back()) > 2.0)
      it = nero_xh_phist_.erase(it);
    else
      ++it;
  }
  for (auto * set : {&nero_xh_letgo_, &nero_xh_revoked_}) {
    for (auto it = set->begin(); it != set->end();) {
      if (nero_xh_phist_.count(*it) == 0)
        it = set->erase(it);
      else
        ++it;
    }
  }

  using Conflict = NeroCrossingCause;
  std::optional<Conflict> conflict;
  int n_targets = 0, n_committed = 0;
  // the farthest footprint point from base_link (the prefilters below measure from base_link)
  const double half_diag_ego =
    std::hypot(std::max(front, rear), planner_data_->vehicle_info_.vehicle_width_m / 2.0);
  for (const auto & object : planner_data_->predicted_objects->objects) {
    if (!isTargetCollisionVehicleType(object)) continue;
    const auto & tw = object.kinematics.initial_twist_with_covariance.twist.linear;
    const double v = std::hypot(tw.x, tw.y);
    const std::string key = uuidKey(object.object_id);
    auto & hist = nero_xh_vhist_[key];
    if (hist.empty() || t_obj - hist.back().first > 0.05) hist.emplace_back(t_obj, v);
    while (!hist.empty() && t_obj - hist.front().first > 1.5) hist.pop_front();
    const auto & opose = object.kinematics.initial_pose_with_covariance.pose;
    if (
      autoware_utils::calc_distance2d(opose.position, ego_pos) >
      std::max(v, 1.0) * (t_end + p.early) + 40.0)
      continue;
    // a follower (behind ego, along ego's heading, in its lane) is not a crosser
    {
      const double s_o =
        autoware::motion_utils::calcSignedArcLength(path->points, ego_pos, opose.position);
      const double lat =
        std::abs(autoware::motion_utils::calcLateralOffset(path->points, opose.position));
      const double dyaw = std::abs(autoware_utils::normalize_radian(
        tf2::getYaw(opose.orientation) - tf2::getYaw(ego_pose.orientation)));
      if (s_o < -rear && dyaw < M_PI / 4.0 && lat < 2.0) continue;
    }
    const std::string id = autoware_utils::to_hex_string(object.object_id).substr(0, 4);
    // slow movers: a vehicle (not UNKNOWN: those count from <v_min> on) slower than <v_slow> that
    // has moved >= 1 m in the last 5 s within <slow_r> of ego's way and is not leaving it holds a
    // standing ego (a car manoeuvring or creeping at the mouth: its prediction says little about
    // what it does next; a parked car does not move)
    {
      const auto label = object.classification.empty() ? 0 : object.classification.front().label;
      if (
        p.slow_r > 0.0 && label != autoware_perception_msgs::msg::ObjectClassification::UNKNOWN &&
        !ego.empty()) {
        auto & ph = nero_xh_phist_[key];
        if (ph.empty() || t_obj - std::get<0>(ph.back()) > 0.05)
          ph.emplace_back(t_obj, opose.position.x, opose.position.y);
        while (!ph.empty() && t_obj - std::get<0>(ph.front()) > 5.0) ph.pop_front();
        // NERO_IX_CROSSING_HOLD_YIELD: the vehicle's motion against ego's way: its distance to it
        // now and 5 s ago, how much closer it came in the last <stood> s, whether it is wholly
        // behind ego's nose pointing backward (and not moving forward along ego's heading), and its
        // speed toward the nearest point of the way
        struct YieldView
        {
          double d_way{0.0}, d_then{0.0}, approach{0.0}, closing{0.0};
          bool behind{false};
        };
        auto yield_view = [&]() {
          YieldView y;
          const double x0 = std::get<1>(ph.front()), y0 = std::get<2>(ph.front());
          const auto ref = std::find_if(ph.begin(), ph.end(), [&](const auto & h) {
            return t_obj - std::get<0>(h) <= yp.stood;
          });
          const double rx = std::get<1>(*ref), ry = std::get<2>(*ref);
          double d_ref = std::numeric_limits<double>::infinity();
          y.d_way = y.d_then = d_ref;
          geometry_msgs::msg::Point near;
          for (const auto & e : ego) {
            const double d = std::hypot(opose.position.x - e.c.x, opose.position.y - e.c.y);
            if (d < y.d_way) {
              y.d_way = d;
              near = e.c;
            }
            y.d_then = std::min(y.d_then, std::hypot(x0 - e.c.x, y0 - e.c.y));
            d_ref = std::min(d_ref, std::hypot(rx - e.c.x, ry - e.c.y));
          }
          y.approach = d_ref - y.d_way;
          const double eyaw = tf2::getYaw(ego_pose.orientation);
          const double ux = std::cos(eyaw), uy = std::sin(eyaw);
          const double nx = ego_pos.x + front * ux, ny = ego_pos.y + front * uy;
          double ahead = -std::numeric_limits<double>::infinity();
          for (const auto & q : autoware_utils::to_polygon2d(opose, object.shape).outer()) {
            ahead = std::max(ahead, (q.x() - nx) * ux + (q.y() - ny) * uy);
          }
          const double dyaw =
            autoware_utils::normalize_radian(tf2::getYaw(opose.orientation) - eyaw);
          const double fwd = (opose.position.x - rx) * ux + (opose.position.y - ry) * uy;
          // (within <near> m too: at a left turn into a through road a car 19 m behind-left, in a
          // parking area by the road, can still drive into ego's way: 59721 scene B's 31aa / 4654)
          y.behind = ahead < 0.0 && std::cos(dyaw) < 0.0 && fwd < yp.approach && y.d_way <= yp.near;
          if (y.d_way > 1e-3) {
            const double vdir = tf2::getYaw(opose.orientation) + std::atan2(tw.y, tw.x);
            y.closing = v *
                        (std::cos(vdir) * (near.x - opose.position.x) +
                         std::sin(vdir) * (near.y - opose.position.y)) /
                        y.d_way;
          }
          return y;
        };
        bool evaluated = false;
        if (
          standing && nose_past <= 2.5 && v < p.v_slow && t_obj - std::get<0>(ph.front()) >= 2.5 &&
          (!conflict || dbg_wait)) {
          const double x0 = std::get<1>(ph.front()), y0 = std::get<2>(ph.front());
          const double moved = std::hypot(opose.position.x - x0, opose.position.y - y0);
          double d_way = std::numeric_limits<double>::infinity(), d_then = d_way;
          for (const auto & e : ego) {
            d_way = std::min(d_way, std::hypot(opose.position.x - e.c.x, opose.position.y - e.c.y));
            d_then = std::min(d_then, std::hypot(x0 - e.c.x, y0 - e.c.y));
          }
          if (moved >= 1.0 && d_way <= p.slow_r && d_way <= d_then + 1.0) {
            // NERO_IX_CROSSING_HOLD_YIELD: a vehicle letting ego go does not hold it: (1) wholly
            // behind ego's nose pointing back, (2) ego stood <stood> s and it has not come
            // <approach> m closer to ego's way since
            const char * yields = nullptr;
            double approach = 0.0;
            if (yp.on) {
              evaluated = true;
              const auto yv = yield_view();
              approach = yv.approach;
              if (yv.behind) {
                yields = "behind";
              } else if (
                ego_stood >= yp.stood && yv.approach < yp.approach && yv.d_way <= yp.near) {
                // (a vehicle manoeuvring further off, e.g. in a parking area 17-24 m from the 59721
                // mouth, is not waiting for ego: it keeps holding as before)
                yields = "stood";
              }
              if (yields) {
                nero_xh_letgo_.insert(key);
                nero_xh_revoked_.erase(key);
              }
            }
            if (dbg_wait) {
              char b[96];
              std::snprintf(
                b, sizeof(b), " %s(d %.1f appr %+.2f v %.1f %s)", id.c_str(), d_way, approach, v,
                yields ? yields : "HOLD");
              dbg_slow += b;
            }
            if (!yields && !conflict) conflict = Conflict{id, "slow", v, 0.0, 0.0, 0.0, d_way};
          }
        }
        // NERO_IX_CROSSING_HOLD_YIELD: a vehicle that let ego go and moves again (>= <v_min> two
        // samples in a row, not away from ego's way) no longer lets it go, at any
        // speed and also while ego moves: it holds ego standing with its nose up to <occ_depth>
        // past the line, and stops a moving ego whose comfortable stop is still within <occ_depth>
        // (else ego is committed for it), until it is behind ego, leaves the way or stood again
        const bool letgo = nero_xh_letgo_.count(key) > 0;
        bool revoked = nero_xh_revoked_.count(key) > 0;
        if (yp.on && !evaluated && (letgo || revoked)) {
          const auto yv = yield_view();
          // moving again: reported >= <v_min> in its last two samples, and not away from the way (a
          // truck pulling out along a line parallel to ego's exit lane closes on it only a little)
          const bool moving_again = hist.size() >= 2 && std::prev(hist.end())->second >= p.v_min &&
                                    std::prev(hist.end(), 2)->second >= p.v_min;
          if (!revoked && !yv.behind && moving_again && yv.closing > 0.05) {
            nero_xh_letgo_.erase(key);
            nero_xh_revoked_.insert(key);
            revoked = true;
            RCLCPP_INFO(
              logger_,
              "[NERO_IX] crossing_hold: lane %ld %s moves toward ego's way again (%.1f m/s, "
              "closing %.2f m/s, %.1f m off; ego v %.2f, nose %+.2f): it no longer lets ego go",
              lane_id_, id.c_str(), v, yv.closing, yv.d_way, v0, nose_past);
          }
          if (revoked) {
            const bool leaving = yv.d_way > p.slow_r || yv.d_way > yv.d_then + 1.0;
            const bool stood =
              standing && ego_stood >= yp.stood && yv.approach < yp.approach && yv.d_way <= yp.near;
            const char * what = "again";
            if (yv.behind || leaving || stood) {
              nero_xh_revoked_.erase(key);
              if (!leaving) nero_xh_letgo_.insert(key);
              what = yv.behind ? "behind" : (leaving ? "leaving" : "stood");
            } else if (
              !nero_xh_again_held_ &&
              (standing ? nose_past > p.occ_depth : nose_past + d_stop > p.occ_depth)) {
              what = "committed";
            } else {
              // stop / hold for it (once decided, until the hold ends: the stop's run-on can carry
              // ego past <occ_depth>)
              nero_xh_again_held_ = true;
              if (!conflict) conflict = Conflict{id, "again", v, 0.0, 0.0, 0.0, yv.d_way};
            }
            if (dbg_wait) {
              char b[96];
              std::snprintf(
                b, sizeof(b), " %s(d %.1f appr %+.2f close %+.2f v %.1f %s)", id.c_str(), yv.d_way,
                yv.approach, yv.closing, v, what);
              dbg_slow += b;
            }
          }
        }
      }
    }
    if (v < p.v_min) continue;
    if (nero_xh_committed_.count(key)) {
      ++n_committed;
      continue;
    }
    ++n_targets;
    // its observed deceleration over the last second (> 0.3 m/s2 slowing: it may also be slower
    // than its prediction)
    double a_obs = 0.0;
    {
      const auto ref = std::find_if(
        hist.begin(), hist.end(), [&](const auto & h) { return t_obj - h.first <= 1.0; });
      if (ref != hist.end() && t_obj - ref->first >= 0.5)
        a_obs = (v - ref->second) / (t_obj - ref->first);
    }
    const bool slowing = a_obs < -0.3;
    // the path time an object slowing at a_obs (down to 0.3 v) reaches at real time x: it covers
    // s_dec(x) = v x + a x^2 / 2
    auto path_time_slowing = [&](const double x) {
      const double x_floor = (0.7 * v) / (-a_obs);  // time to fall to 0.3 v
      const double xs = std::min(x, x_floor);
      const double s =
        v * xs + 0.5 * a_obs * xs * xs + (x > x_floor ? 0.3 * v * (x - x_floor) : 0.0);
      return s / std::max(0.1, v);
    };
    // the farthest point of the object's shape from its centre (a polygon UNKNOWN carries no
    // dimensions)
    double half_diag_obj = 0.5;
    {
      const auto & sh = object.shape;
      if (sh.type == autoware_perception_msgs::msg::Shape::POLYGON) {
        for (const auto & q : sh.footprint.points) {
          half_diag_obj = std::max(half_diag_obj, static_cast<double>(std::hypot(q.x, q.y)));
        }
      } else if (sh.type == autoware_perception_msgs::msg::Shape::CYLINDER) {
        half_diag_obj = std::max(half_diag_obj, sh.dimensions.x / 2.0);
      } else {
        half_diag_obj = std::max(half_diag_obj, std::hypot(sh.dimensions.x, sh.dimensions.y) / 2.0);
      }
    }
    // hypotheses: the predicted paths, or straight on at its speed when none moves it
    std::vector<std::pair<std::vector<geometry_msgs::msg::Pose>, double>> hyps;
    for (const auto & pp : object.kinematics.predicted_paths) {
      if (
        pp.path.size() < 2 ||
        pp.confidence < planner_param_.collision_detection.min_predicted_path_confidence)
        continue;
      const double dt = rclcpp::Duration(pp.time_step).seconds();
      const double span =
        autoware_utils::calc_distance2d(pp.path.front().position, pp.path.back().position);
      if (dt <= 0.0 || span < 0.25 * v * dt * static_cast<double>(pp.path.size() - 1))
        continue;  // stationary
      hyps.emplace_back(std::vector<geometry_msgs::msg::Pose>(pp.path.begin(), pp.path.end()), dt);
    }
    if (hyps.empty()) {
      const double yaw = tf2::getYaw(opose.orientation) + std::atan2(tw.y, tw.x);
      std::vector<geometry_msgs::msg::Pose> line;
      for (int k = 0; k <= 30; ++k) {
        auto q = opose;
        q.position.x += v * 0.5 * k * std::cos(yaw);
        q.position.y += v * 0.5 * k * std::sin(yaw);
        line.push_back(q);
      }
      hyps.emplace_back(line, 0.5);
    }
    std::optional<Conflict> c_obj;
    double gap_go = std::numeric_limits<double>::infinity();  // closest ego would come going on
    double gap_stop =
      std::numeric_limits<double>::infinity();  // closest it would come to ego standing at d_stop
    for (const auto & [ppath, pdt] : hyps) {
      // a crosser: this path comes within 0.5 m of ego's way (its footprints to <depth> past the
      // line) within the horizon; a car passing beside it in its own lane (an oncoming car in the
      // next lane, one turning off past ego's side) or going away is not one (10-27 #11: 4785 drove
      // off beside the robot at 1.3-1.9 m after it had turned out)
      bool crosses = false;
      for (double tau = 0.0; tau <= t_end + p.early + 1e-6 && !crosses; tau += 0.25) {
        const auto op = nero_pose_on_path(ppath, pdt, tau, v);
        const auto opoly = autoware_utils::to_polygon2d(op, object.shape);
        for (const auto & e : ego) {
          if (
            std::hypot(op.position.x - e.c.x, op.position.y - e.c.y) >
            0.5 + half_diag_ego + half_diag_obj)
            continue;
          if (bg::distance(e.fp, opoly) < 0.5) {
            crosses = true;
            break;
          }
        }
      }
      if (!crosses) continue;
      for (const auto & e : ego) {
        const double lo =
          slowing ? path_time_slowing(std::max(0.0, e.t - p.late)) : std::max(0.0, e.t - p.late);
        for (double tau = lo; tau <= e.t + p.early + 1e-6; tau += 0.25) {
          const auto op = nero_pose_on_path(ppath, pdt, tau, v);
          const double dc = std::hypot(op.position.x - e.c.x, op.position.y - e.c.y);
          if (dc > p.gap + half_diag_ego + half_diag_obj) continue;
          const double g = bg::distance(e.fp, autoware_utils::to_polygon2d(op, object.shape));
          gap_go = std::min(gap_go, g);
          if (g < p.gap && (!c_obj || e.t < c_obj->t)) {
            c_obj = Conflict{id, "", v, e.t, tau, e.s, g};
            c_obj->cls = std::to_string(
              object.classification.empty() ? 0 : object.classification.front().label);
          }
        }
      }
      if (standing && p.occ_t > 0.0) {
        // a vehicle that will be across ego's first <occ_depth> m within <occ_t> s: no start toward
        // it
        const double s_occ = s_line.value() + p.occ_depth - front;
        for (double tau = 0.0; tau <= p.occ_t + 1e-6 && !(c_obj && c_obj->cls == "occ");
             tau += 0.25) {
          const auto op = nero_pose_on_path(ppath, pdt, tau, v);
          const auto opoly = autoware_utils::to_polygon2d(op, object.shape);
          for (const auto & e : ego) {
            if (e.s > s_occ) break;
            if (
              std::hypot(op.position.x - e.c.x, op.position.y - e.c.y) >
              0.5 + half_diag_ego + half_diag_obj)
              continue;
            const double g = bg::distance(e.fp, opoly);
            if (g < 0.5) {
              c_obj = Conflict{id, "occ", v, 0.0, tau, e.s, g};
              break;
            }
          }
        }
      }
      if (stop_fp) {
        for (double tau = 0.0; tau <= t_end + p.early + 1e-6; tau += 0.25) {
          const auto op = nero_pose_on_path(ppath, pdt, tau, v);
          gap_stop = std::min(
            gap_stop, static_cast<double>(bg::distance(
                        stop_fp.value(), autoware_utils::to_polygon2d(op, object.shape))));
        }
      }
    }
    if (!c_obj) continue;
    if (dbg_wait) {
      char b[96];
      std::snprintf(
        b, sizeof(b), " %s(%s gap %.2f t %.1f)", id.c_str(), c_obj->cls == "occ" ? "occ" : "cross",
        c_obj->gap, c_obj->t);
      dbg_hold += b;
    }
    if (!standing && gap_stop < gap_go) {
      // stopping would leave ego closer to it than going on: committed for this object
      nero_xh_committed_.insert(key);
      ++n_committed;
      RCLCPP_INFO(
        logger_,
        "[NERO_IX] crossing_hold: lane %ld committed for %s (%.1f m/s): ego (v %.2f, nose %+.2f) "
        "going on keeps %.2f m, "
        "stopping (%.2f m) %.2f m",
        lane_id_, id.c_str(), v, v0, nose_past, gap_go, d_stop, gap_stop);
      continue;
    }
    if (!conflict || c_obj->t < conflict->t) conflict = c_obj;
  }

  const double now = clock_->now().seconds();
  const bool was = nero_xh_holding_;
  if (conflict) {
    nero_xh_last_conflict_ = now;
    nero_xh_holding_ = true;
    nero_xh_cause_ = *conflict;
  } else if (
    nero_xh_holding_ &&
    (now - nero_xh_last_conflict_ >= p.release || (!standing && n_committed > 0))) {
    nero_xh_holding_ = false;
  }
  if (!nero_xh_holding_) nero_xh_again_held_ = false;
  if (nero_xh_holding_ != was) {
    if (nero_xh_holding_) {
      nero_xh_hold_since_ = now;
      RCLCPP_INFO(
        logger_,
        "[NERO_IX] crossing_hold: lane %ld HOLD (nose %+.2f m past the line, v %.2f) for %s (class "
        "%s, %.1f m/s): gap %.2f m "
        "at ego t %.1f s (s %.1f m), object t %.2f s",
        lane_id_, nose_past, v0, conflict->id.c_str(), conflict->cls.c_str(), conflict->v,
        conflict->gap, conflict->t, conflict->s, conflict->tau);
    } else {
      RCLCPP_INFO(
        logger_, "[NERO_IX] crossing_hold: lane %ld GO (nose %+.2f m, held %.1f s%s)", lane_id_,
        nose_past, now - nero_xh_hold_since_, n_committed > 0 ? ", committed" : "");
    }
  }
  if (nero_crossing_hold_debug()) {
    const std::string what = conflict ? conflict->id + " gap " + std::to_string(conflict->gap) +
                                          " at t " + std::to_string(conflict->t)
                                      : "clear";
    RCLCPP_INFO(
      logger_,
      "[NERO_IX] crossing_hold dbg: lane %ld nose %+.2f v %.2f targets %d committed %d ego samples "
      "%zu (to %.1f s) %s",
      lane_id_, nose_past, v0, n_targets, n_committed, ego.size(), t_end, what.c_str());
  }
  if (dbg_wait) {
    RCLCPP_INFO(
      logger_, "[NERO_IX] crossing_wait dbg: lane %ld ego stood %.1f s | crossers:%s | slow:%s",
      lane_id_, ego_stood, dbg_hold.empty() ? " none" : dbg_hold.c_str(),
      dbg_slow.empty() ? " none" : dbg_slow.c_str());
  }
  char txt[200];
  if (nero_xh_holding_) {
    std::snprintf(
      txt, sizeof(txt), " | NERO crossing hold: HOLD for %s gap %.2f at t %.1f (nose %+.2f)",
      nero_xh_cause_.id.c_str(), nero_xh_cause_.gap, nero_xh_cause_.t, nose_past);
  } else {
    std::snprintf(
      txt, sizeof(txt), " | NERO crossing hold: clear (nose %+.2f, %d targets, %d committed)",
      nose_past, n_targets, n_committed);
  }
  internal_debug_data_.decision_type += txt;
  if (!nero_xh_holding_ || nero_xh_cause_.cls != "again" || standing) nero_xh_again_stop_.reset();
  if (!nero_xh_holding_) return;

  // hold: at the stop line (base_link where the nose meets it), or where ego is if it is past that
  double s_stop = std::max(0.0, s_line.value() - front);
  if (nero_xh_cause_.cls == "again" && !standing) {
    // NERO_IX_CROSSING_HOLD_YIELD: a moving ego stops for a vehicle that no longer lets it go where
    // its comfortable stop ends (latched), not at once
    if (!nero_xh_again_stop_) {
      const auto pose = autoware::motion_utils::calcLongitudinalOffsetPose(
        path->points, ego_pos, std::max(s_stop, d_stop));
      if (pose) nero_xh_again_stop_ = pose->position;
    }
    if (nero_xh_again_stop_) {
      s_stop = std::max(
        s_stop,
        autoware::motion_utils::calcSignedArcLength(path->points, ego_pos, *nero_xh_again_stop_));
    }
  }
  const auto stop_pose =
    autoware::motion_utils::calcLongitudinalOffsetPose(path->points, ego_pos, s_stop);
  if (!stop_pose) return;
  const auto inserted = planning_utils::insertStopPoint(stop_pose->position, *path);
  if (!inserted) return;
  const size_t stop_idx =
    autoware::motion_utils::findNearestIndex(path->points, inserted->position);
  debug_data_.collision_stop_wall_pose = planning_utils::getAheadPose(stop_idx, front, *path);
  setSafe(false);
  setDistance(s_stop);
  planning_factor_interface_->add(
    path->points, ego_pose, inserted.value(),
    autoware_internal_planning_msgs::msg::PlanningFactor::STOP,
    autoware_internal_planning_msgs::msg::SafetyFactorArray{}, true /*is_driving_forward*/, 0.0,
    0.0 /*shift distance*/, "intersection(NERO crossing hold)");
}

}  // namespace autoware::behavior_velocity_planner
