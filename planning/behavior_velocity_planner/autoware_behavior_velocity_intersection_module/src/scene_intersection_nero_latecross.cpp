// Copyright 2026 Nero (latecrosser job, 2026-10-31, QUEUE XHD)
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

// NERO_IX_CROSSING_HOLD_CREEP=<v_creep m/s>,<creep depth m> (latecrosser job, 2026-10-31, QUEUE
// XHD): at a crosshold mouth (a stop-sign lane whose crossing road the map lacks; see
// scene_intersection_nero_crossing.cpp), once the crossing hold has held ego with its nose at most
// <creep depth> past the stop line, ego leaves from where it then stands at no more than <v_creep>
// until its nose is <creep depth> past the line, then as before. A crossing car can become a fast
// crosser in the objects only 3-9 m from ego after the hold let it go (10-27 20:55 62830: 39b2
// driving up the home street was a CenterPoint CAR from 19.7 m but its obstacle points were missing
// until 8.9 m; 10-27 20:26 59721: the turning car re-born under a new id 3 m away; 10-29 15:51
// 51377: an UNKNOWN moving 3.6 m/s reported at 0): with the car's departure (0.45 m/s2 to the 5
// km/h turn tag) ego's nose is +2.45 m past the line at 1.36 m/s 3.3 s after it starts, too deep
// and too fast to stop clear of the crosser's line (0.55-0.95 m). Creeping the first metres keeps
// ego able to stop short of where the crossing traffic passes (stock's peeking creep is 0.83 m/s;
// Apollo's stop sign creeps after its stop). The departure moment does not change (ego leaves when
// the hold lets it go, as now), only the first <creep depth> m are slower; a mouth where nothing
// held ego is driven as now. The crossing hold's own passage model sees the capped speed (the cap
// is put into the path before it). Unset = the car (nothing runs). Never adds a stop, never weakens
// one.
//   3rd field <probe> = 1 (latecrosser v7): the creep must never cost a gap the car takes. In the
// passage model the creep makes ego's whole way through a stream mouth later (59721: ~2.6 s through
// the merge), so a stream gap the car takes (16 s at 59721 with the 5 km/h turn tag) is judged too
// short and ego never leaves. While ego stands held with the creep armed, every 2nd cycle is a
// probe: the cap is not put into the path, so the crossing hold judges the car's own departure
// (ego stands behind the hold's stop: the output is the same). Probes run only while the last
// conflict is < 0.3 s old (the car's release is 0.5 s: the hold cannot let ego go on a probe
// cycle; a probe's conflict holds as the car would). Two probes in a row clear while the capped
// cycles between them still hold = the creep alone holds ego: the creep is dropped for this
// departure (ego leaves as the car; a later hold arms it again). Never while moving.
//   4th field <waiting> = 1 (latecrosser v8): no creep in front of a driver who waits for ego. When
// the hold lets ego go while a vehicle that let ego go (crosswait's let-go, not revoked, not behind
// ego pointing back) STANDS (< 0.5 m/s), the creep is dropped for this departure: that driver is
// the near risk and the creep keeps ego in its way ~2.6 s longer through a turn (59721,
// crosswait's never-slowing pull-out 8 s after the let-go met the creeping robot at nose +5.9 and
// passes the car's 1.3-1.5 m behind it; Ahmed 10-24: a car slowing down is the signal for the robot
// to go through). A let-go vehicle that is leaving (62830's passer at 1.6 m/s) keeps the creep.
//
// NERO_IX_CROSSING_HOLD_VANISHED=<max s>,<near m>[,<v_min m/s>] (latecrosser, QUEUE XHD / HIDF): a
// moving vehicle (the crossing hold's target classes, UNKNOWN included, >= <v_min>) that VANISHES
// from the objects while its straight line at its last velocity still comes within <near> m of
// ego's way (the path to the hold's depth) is judged by the crossing hold as moving on at that
// velocity for up to <max> s, unless another object appears within 2 m of where it would be (a new
// track took it over: judged on its own). 10-27 20:26 at 59721: the car turning in front of the
// robot lost its CAR track (CenterPoint dropped it 3-5 m away), went on as UNKNOWN 1ad0 and was
// re-born as CAR 1cd1 0.40 s after 1ad0's last report: the hold releases 0.5 s after the last
// conflict, so a gap a little longer lets ego roll into it (crosshold's scene B, which lacked 1ad0:
// release in the gap, 0.4 m roll, 0.46-1.56 m). A car that leaves the way and drops out further on
// is not carried. Unset = the car (the planner's objects as they are, no copy).
//
// NERO_IX_CROSSING_HOLD_AGAIN_COMMIT=1 (latecrosser, QUEUE XHD): crosswait's rule stops a moving
// ego for a vehicle that let it go and then moves toward its way again while ego's comfortable stop
// still ends within the first <occ depth> m past the line - also where the crossing hold's own
// commit test (rule 4) finds that stopping leaves ego closer to that vehicle than going on (the
// stop's place is inside the vehicle's way). Then the stop of THIS cycle is dropped and ego goes
// on, as the crossing hold does for any crosser it is committed for (59721, crosswait's
// never-slowing pull-out 8 s after the let-go with ego creeping: going on keeps 1.91 m, stopping
// 0.00 m: contact at +4.4 m). A stop decided in an earlier cycle stays (its run-on), and a vehicle
// whose predicted path does not cross ego's passage keeps the stop as now. Unset = the car (nothing
// runs).

#include "scene_intersection.hpp"

#include <autoware/motion_utils/trajectory/trajectory.hpp>
#include <autoware_utils/geometry/boost_polygon_utils.hpp>
#include <autoware_utils/math/normalization.hpp>
#include <autoware_utils/ros/uuid_helper.hpp>

#include <tf2/utils.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace
{
struct NeroCrossingCreepParam
{
  bool on{false};
  double v{0.8};
  double depth{2.0};
  bool probe{false};
  bool waiting{false};
};

const NeroCrossingCreepParam & nero_crossing_creep_param()
{
  static const NeroCrossingCreepParam p = [] {
    NeroCrossingCreepParam r;
    const char * e = std::getenv("NERO_IX_CROSSING_HOLD_CREEP");
    if (e == nullptr || std::string(e).empty()) return r;
    double v[4] = {r.v, r.depth, 0.0, 0.0};
    const int n = std::sscanf(e, "%lf,%lf,%lf,%lf", &v[0], &v[1], &v[2], &v[3]);
    if (n < 1) return r;
    r.on = true;
    r.v = std::max(0.2, v[0]);
    r.depth = std::max(0.0, v[1]);
    r.probe = v[2] > 0.5;
    r.waiting = v[3] > 0.5;
    return r;
  }();
  return p;
}

struct NeroCrossingVanishedParam
{
  bool on{false};
  double t_max{2.0};
  double near{1.0};
  double v_min{0.5};
};

const NeroCrossingVanishedParam & nero_crossing_vanished_param()
{
  static const NeroCrossingVanishedParam p = [] {
    NeroCrossingVanishedParam r;
    const char * e = std::getenv("NERO_IX_CROSSING_HOLD_VANISHED");
    if (e == nullptr || std::string(e).empty()) return r;
    double v[3] = {r.t_max, r.near, r.v_min};
    const int n = std::sscanf(e, "%lf,%lf,%lf", &v[0], &v[1], &v[2]);
    if (n < 1) return r;
    r.on = true;
    r.t_max = std::max(0.0, v[0]);
    r.near = std::max(0.0, v[1]);
    r.v_min = std::max(0.0, v[2]);
    return r;
  }();
  return p;
}

bool nero_crossing_again_commit_on()
{
  static const bool on = [] {
    const char * e = std::getenv("NERO_IX_CROSSING_HOLD_AGAIN_COMMIT");
    return e != nullptr && std::string(e) == "1";
  }();
  return on;
}

// the farthest point of an object's shape from its centre (a polygon UNKNOWN carries no dimensions)
double nero_half_diag(const autoware_perception_msgs::msg::Shape & sh)
{
  double h = 0.5;
  if (sh.type == autoware_perception_msgs::msg::Shape::POLYGON) {
    for (const auto & q : sh.footprint.points)
      h = std::max(h, static_cast<double>(std::hypot(q.x, q.y)));
  } else if (sh.type == autoware_perception_msgs::msg::Shape::CYLINDER) {
    h = std::max(h, sh.dimensions.x / 2.0);
  } else {
    h = std::max(h, std::hypot(sh.dimensions.x, sh.dimensions.y) / 2.0);
  }
  return h;
}
}  // namespace

namespace autoware::behavior_velocity_planner
{

void IntersectionModule::neroCrossingCreep(
  PathWithLaneId * path, const double s_line, const double nose_past, const double v0)
{
  const auto & p = nero_crossing_creep_param();
  if (!p.on || path->points.size() < 2) return;
  const auto & ego_pos = planner_data_->current_odometry->pose.position;
  // armed by a hold of the crossing hold (last cycle's state) while the nose is not yet <depth>
  // past the line; done once the nose is past it
  if (nose_past > p.depth) {
    if (nero_xc_armed_) {
      RCLCPP_INFO(
        logger_,
        "[NERO_IX] crossing_hold: lane %ld creep ends (nose %+.2f m past the line, v %.2f, %.1f s)",
        lane_id_, nose_past, v0, clock_->now().seconds() - nero_xc_since_);
    }
    nero_xc_armed_ = false;
    nero_xc_stood_ = false;
    nero_xc_dropped_ = false;
    nero_xc_track_ = false;
    nero_xc_probe_ = false;
    nero_xc_held_ = false;
    nero_xc_clear_ = 0;
    return;
  }
  if (nero_xc_dropped_) {
    // dropped by the probe: not armed again while the same hold lasts; its GO ends the drop
    if (nero_xh_holding_) return;
    nero_xc_dropped_ = false;
  }
  if (!nero_xc_armed_ && nero_xh_holding_) {
    nero_xc_armed_ = true;
    nero_xc_stood_ = false;
    nero_xc_track_ = false;
    nero_xc_probe_ = false;
    nero_xc_held_ = false;
    nero_xc_clear_ = 0;
    nero_xc_since_ = clock_->now().seconds();
    RCLCPP_INFO(
      logger_,
      "[NERO_IX] crossing_hold: lane %ld creep armed by the hold for %s (nose %+.2f m past the "
      "line, v %.2f): after it, at most %.2f m/s until the nose is %.1f m past the line",
      lane_id_, nero_xh_cause_.id.c_str(), nose_past, v0, p.v, p.depth);
  }
  if (!nero_xc_armed_) return;
  // the cap starts once ego has stood after the hold (crosshold's "standing", < 0.1 m/s): a hold
  // that comes while ego approaches the line leaves its approach and its stop as they are (capping
  // the approach made it stop 0.1-0.35 m short of the line); the departure from the stand creeps
  if (!nero_xc_stood_) {
    if (v0 >= 0.1) return;
    nero_xc_stood_ = true;
  }
  if (p.waiting && !nero_xh_holding_ && v0 < 0.1) {
    // the hold let ego go (ego still stands): a let-go vehicle standing by the way waits for ego
    const double eyaw = tf2::getYaw(planner_data_->current_odometry->pose.orientation);
    const double ux = std::cos(eyaw), uy = std::sin(eyaw);
    const double front = planner_data_->vehicle_info_.max_longitudinal_offset_m;
    const double nx = ego_pos.x + front * ux, ny = ego_pos.y + front * uy;
    for (const auto & o : planner_data_->predicted_objects->objects) {
      const std::string key = uuidKey(o.object_id);
      if (nero_xh_letgo_.count(key) == 0 || nero_xh_revoked_.count(key) > 0) continue;
      const auto & tw = o.kinematics.initial_twist_with_covariance.twist.linear;
      if (std::hypot(tw.x, tw.y) >= 0.5) continue;
      const auto & op = o.kinematics.initial_pose_with_covariance.pose;
      double ahead = -std::numeric_limits<double>::infinity();
      for (const auto & q : autoware_utils::to_polygon2d(op, o.shape).outer())
        ahead = std::max(ahead, (q.x() - nx) * ux + (q.y() - ny) * uy);
      const double dyaw = autoware_utils::normalize_radian(tf2::getYaw(op.orientation) - eyaw);
      if (ahead < 0.0 && std::cos(dyaw) < 0.0) continue;  // behind ego pointing back
      RCLCPP_INFO(
        logger_,
        "[NERO_IX] crossing_hold: lane %ld creep dropped: %s stands and lets ego go (%.1f m from "
        "ego): ego leaves as the car (no extra time in its way)",
        lane_id_, autoware_utils::to_hex_string(o.object_id).substr(0, 4).c_str(),
        std::hypot(op.position.x - ego_pos.x, op.position.y - ego_pos.y));
      nero_xc_armed_ = false;
      nero_xc_stood_ = false;
      nero_xc_track_ = false;
      internal_debug_data_.decision_type += " | NERO creep dropped (waiting)";
      return;
    }
  }
  if (p.probe) {
    // the last tracked cycle's result: did the crossing hold find a conflict in it?
    const bool was_tracked = nero_xc_track_;
    if (was_tracked) {
      const bool had = nero_xh_last_conflict_ > nero_xc_lc_;
      if (nero_xc_probe_) {
        nero_xc_clear_ = (!had && nero_xc_held_) ? nero_xc_clear_ + 1 : 0;
      } else {
        nero_xc_held_ = had;
        if (!had) nero_xc_clear_ = 0;
      }
    }
    nero_xc_track_ = false;
    if (nero_xc_clear_ >= 2) {
      RCLCPP_INFO(
        logger_,
        "[NERO_IX] crossing_hold: lane %ld creep dropped: only the creep holds ego (for %s, gap "
        "%.2f m at ego t %.1f s; the car's departure is clear): ego leaves as the car",
        lane_id_, nero_xh_cause_.id.c_str(), nero_xh_cause_.gap, nero_xh_cause_.t);
      nero_xc_armed_ = false;
      nero_xc_stood_ = false;
      nero_xc_dropped_ = true;
      nero_xc_clear_ = 0;
      nero_xc_held_ = false;
      nero_xc_probe_ = false;
      internal_debug_data_.decision_type += " | NERO creep dropped";
      return;
    }
    const double now = clock_->now().seconds();
    if (v0 < 0.1 && nero_xh_holding_ && now - nero_xh_last_conflict_ < 0.3) {
      // alternate capped / probe, starting with a capped cycle
      const bool probe = was_tracked && !nero_xc_probe_;
      nero_xc_track_ = true;
      nero_xc_probe_ = probe;
      nero_xc_lc_ = nero_xh_last_conflict_;
      if (probe) {
        internal_debug_data_.decision_type += " | NERO creep probe";
        return;
      }
    } else {
      nero_xc_probe_ = false;
      nero_xc_held_ = false;
      nero_xc_clear_ = 0;
    }
  }

  // cap the path's speed up to where the nose is <depth> past the line (base_link arc length
  // s_line + depth - front, the path's own spacing beyond it at most); the points behind ego too
  // (the trajectory starts behind ego and an uncapped point there leaked into the speed planned at
  // ego: 0.67-0.78 m/s for a 0.6 cap in build c1)
  const double front = planner_data_->vehicle_info_.max_longitudinal_offset_m;
  const double s_end = s_line + p.depth - front;
  const size_t seg = autoware::motion_utils::findNearestSegmentIndex(path->points, ego_pos);
  const double off =
    autoware::motion_utils::calcLongitudinalOffsetToSegment(path->points, seg, ego_pos);
  double s = -off;  // arc length of point seg from ego (points before it: behind ego)
  size_t n_capped = 0;
  for (size_t i = 0; i < seg; ++i) {
    auto & vel = path->points.at(i).point.longitudinal_velocity_mps;
    if (vel > p.v) {
      vel = static_cast<float>(p.v);
      ++n_capped;
    }
  }
  for (size_t i = seg; i < path->points.size(); ++i) {
    if (i > seg) {
      const auto & a = path->points.at(i - 1).point.pose.position;
      const auto & b = path->points.at(i).point.pose.position;
      s += std::hypot(b.x - a.x, b.y - a.y);
    }
    auto & vel = path->points.at(i).point.longitudinal_velocity_mps;
    if (vel > p.v) {
      vel = static_cast<float>(p.v);
      ++n_capped;
    }
    if (s >= s_end) break;  // this point is the first at or beyond the creep end: capped too
  }
  char txt[96];
  std::snprintf(
    txt, sizeof(txt), " | NERO creep <= %.2f to +%.1f (nose %+.2f, %zu pts)", p.v, p.depth,
    nose_past, n_capped);
  internal_debug_data_.decision_type += txt;
}

const std::vector<autoware_perception_msgs::msg::PredictedObject> &
IntersectionModule::neroCrossingObjects(const PathWithLaneId & path, const double s_end)
{
  // (called once per cycle at the start of the crossing hold's object loop)
  nero_xa_held_at_start_ = nero_xh_again_held_;
  const auto & objs = planner_data_->predicted_objects->objects;
  const auto & p = nero_crossing_vanished_param();
  if (!p.on) return objs;
  const double t_obj = rclcpp::Time(planner_data_->predicted_objects->header.stamp).seconds();
  const auto & ego_pos = planner_data_->current_odometry->pose.position;
  const double front = planner_data_->vehicle_info_.max_longitudinal_offset_m;
  const double half_w = planner_data_->vehicle_info_.vehicle_width_m / 2.0;
  // ego's way: the path points from ego to where the nose is at the hold's depth
  std::vector<std::pair<double, double>> way;
  if (path.points.size() >= 2) {
    const size_t seg = autoware::motion_utils::findNearestSegmentIndex(path.points, ego_pos);
    double s = 0.0;
    way.emplace_back(ego_pos.x, ego_pos.y);
    for (size_t i = seg + 1; i < path.points.size() && s <= s_end + front; ++i) {
      const auto & q = path.points.at(i).point.pose.position;
      s += std::hypot(q.x - way.back().first, q.y - way.back().second);
      way.emplace_back(q.x, q.y);
    }
  }
  auto dist_way = [&](const double x, const double y) {
    double d = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i + 1 < way.size(); ++i) {
      const double ax = way[i].first, ay = way[i].second;
      const double bx = way[i + 1].first, by = way[i + 1].second;
      const double l2 = (bx - ax) * (bx - ax) + (by - ay) * (by - ay);
      const double r =
        l2 < 1e-9 ? 0.0 : std::clamp(((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / l2, 0.0, 1.0);
      d = std::min(d, std::hypot(x - (ax + r * (bx - ax)), y - (ay + r * (by - ay))));
    }
    return d;
  };
  // the moving vehicles seen now (their last state is kept)
  std::set<std::string> present;
  for (const auto & o : objs) {
    if (!isTargetCollisionVehicleType(o)) continue;
    const std::string key = uuidKey(o.object_id);
    present.insert(key);
    const auto & tw = o.kinematics.initial_twist_with_covariance.twist.linear;
    const auto & op = o.kinematics.initial_pose_with_covariance.pose.position;
    if (
      std::hypot(tw.x, tw.y) >= p.v_min && std::hypot(op.x - ego_pos.x, op.y - ego_pos.y) <= 30.0) {
      auto & e = nero_xv_[key];
      e.t_last = t_obj;
      e.ghost = false;
      e.logged = false;
      e.last = o;
    } else {
      nero_xv_.erase(key);
    }
  }
  std::vector<autoware_perception_msgs::msg::PredictedObject> ghosts;
  for (auto it = nero_xv_.begin(); it != nero_xv_.end();) {
    if (present.count(it->first)) {
      ++it;
      continue;
    }
    auto & e = it->second;
    const double age = t_obj - e.t_last;
    const auto & lp = e.last.kinematics.initial_pose_with_covariance.pose;
    const auto & tw = e.last.kinematics.initial_twist_with_covariance.twist.linear;
    const double v = std::hypot(tw.x, tw.y);
    const double dir = tf2::getYaw(lp.orientation) + std::atan2(tw.y, tw.x);
    const std::string id = autoware_utils::to_hex_string(e.last.object_id).substr(0, 4);
    if (age > p.t_max || way.size() < 2) {
      if (e.ghost) {
        RCLCPP_INFO(
          logger_, "[NERO_IX] crossing_hold: lane %ld vanished %s no longer carried on (%.1f s)",
          lane_id_, id.c_str(), age);
      }
      it = nero_xv_.erase(it);
      continue;
    }
    const double gx = lp.position.x + v * age * std::cos(dir);
    const double gy = lp.position.y + v * age * std::sin(dir);
    // a new object where it would be: a new track took it over (judged on its own)
    bool taken = false;
    std::string by;
    for (const auto & o : objs) {
      if (!isTargetCollisionVehicleType(o)) continue;
      const auto & q = o.kinematics.initial_pose_with_covariance.pose.position;
      if (std::hypot(q.x - gx, q.y - gy) <= 2.0) {
        taken = true;
        by = autoware_utils::to_hex_string(o.object_id).substr(0, 4);
        break;
      }
    }
    if (taken) {
      if (e.ghost) {
        RCLCPP_INFO(
          logger_, "[NERO_IX] crossing_hold: lane %ld vanished %s taken over by %s after %.1f s",
          lane_id_, id.c_str(), by.c_str(), age);
      }
      it = nero_xv_.erase(it);
      continue;
    }
    if (!e.ghost && !e.logged) {
      // decided once, on its first missing cycle: does its straight line still come next to ego's
      // way?
      const double reach = nero_half_diag(e.last.shape) + half_w + p.near;
      double d_min = std::numeric_limits<double>::infinity();
      for (double tau = 0.0; tau <= p.t_max + 1e-6; tau += 0.25) {
        d_min = std::min(
          d_min, dist_way(
                   lp.position.x + v * (age + tau) * std::cos(dir),
                   lp.position.y + v * (age + tau) * std::sin(dir)));
      }
      e.logged = true;
      e.ghost = d_min <= reach;
      if (e.ghost) {
        RCLCPP_INFO(
          logger_,
          "[NERO_IX] crossing_hold: lane %ld %s (class %d, %.1f m/s) vanished %.1f m from ego's "
          "way (centre %.1f m from ego): judged as moving on at its last velocity for up to %.1f s",
          lane_id_, id.c_str(),
          e.last.classification.empty() ? 0 : e.last.classification.front().label, v, d_min,
          std::hypot(lp.position.x - ego_pos.x, lp.position.y - ego_pos.y), p.t_max);
      }
    }
    if (!e.ghost) {
      it = nero_xv_.erase(it);
      continue;
    }
    auto g = e.last;
    g.kinematics.initial_pose_with_covariance.pose.position.x = gx;
    g.kinematics.initial_pose_with_covariance.pose.position.y = gy;
    // the crossing hold then takes the straight line at its speed
    g.kinematics.predicted_paths.clear();
    ghosts.push_back(std::move(g));
    ++it;
  }
  if (ghosts.empty()) return objs;
  nero_xv_objects_ = objs;
  for (auto & g : ghosts) nero_xv_objects_.push_back(std::move(g));
  char txt[64];
  std::snprintf(txt, sizeof(txt), " | NERO vanished carried: %zu", ghosts.size());
  internal_debug_data_.decision_type += txt;
  return nero_xv_objects_;
}

void IntersectionModule::neroAgainCommit(
  std::optional<NeroCrossingCause> & conflict, const std::string & id, const double v0,
  const double nose_past, const double gap_go, const double gap_stop)
{
  if (!nero_crossing_again_commit_on()) return;
  // only the stop crosswait's rule set for this very object in this cycle (it sets one only when
  // no other conflict was set before it, so dropping it restores the cycle's state exactly)
  if (!conflict || conflict->id != id || conflict->cls != "again" || nero_xa_held_at_start_) return;
  conflict.reset();
  nero_xh_again_held_ = false;
  const double now = clock_->now().seconds();
  auto & last = nero_xa_logged_[id];
  if (now - last >= 2.0) {
    last = now;
    if (gap_go >= 0.0) {
      RCLCPP_INFO(
        logger_,
        "[NERO_IX] crossing_hold: lane %ld %s moves again but ego is committed for it (ego v %.2f, "
        "nose %+.2f; going on keeps %.2f m, stopping %.2f m): no stop inside its way",
        lane_id_, id.c_str(), v0, nose_past, gap_go, gap_stop);
    } else {
      RCLCPP_INFO(
        logger_,
        "[NERO_IX] crossing_hold: lane %ld %s moves again but ego is committed for it (ego v %.2f, "
        "nose %+.2f; committed earlier): no stop inside its way",
        lane_id_, id.c_str(), v0, nose_past);
    }
  }
}

}  // namespace autoware::behavior_velocity_planner
