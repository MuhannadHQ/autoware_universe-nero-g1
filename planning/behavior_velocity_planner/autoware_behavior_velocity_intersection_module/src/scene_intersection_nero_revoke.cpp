// Copyright 2026 Nero (yieldrevoke job, 2026-11-03, QUEUE YREV)
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

// NERO_IX_YIELD_REVOKE part 3 (nero_yield_revoke.hpp): the permanent GO latched while a yielder was left out is not
// final for him. Once he is revoked and unsafe, ego stops where it is if the car's own stop from here ends with its nose
// at least <margin> m short of his way; else ego keeps the commit (a stop that would end in his way is never made).
// Decided once per car (latched): the module never flips between stopping and going for him. Before the commit the
// module's own collision stop stands as it is (v3 replaced it by "go on" where it would end within <margin> of his way:
// sj54545 slowgo / tjminor slowgo, ego at 0.8-0.9 m/s cannot clear his way, contact where the car stopped).
// His way = everywhere he can drive from where he is: his own box, every predicted path of his, the path that made him
// unsafe, and his lanes from where he is on through every following lanelet for 30 m (map; straight on where the map
// ends), each lane sample his box widened by 0.25 m a side (drivers off the centre line). The pull-away model's and the
// stock check's lane paths end with his attention lane, which can end before the conflict (mr66508 go3: 43978 ends 6.3
// m before ego's path; his way on is the straight 66598, crossing at ego's nose): a scan on them alone stopped ego
// inside his way (contact, v3).

#include "nero_yield_revoke.hpp"
#include "scene_intersection.hpp"
#include "util.hpp"

#include <autoware_lanelet2_extension/utility/utilities.hpp>

#include <autoware_utils/geometry/boost_polygon_utils.hpp>
#include <autoware_utils/geometry/geometry.hpp>

#include <boost/geometry/algorithms/correct.hpp>
#include <boost/geometry/algorithms/intersects.hpp>

#include <lanelet2_core/geometry/LaneletMap.h>
#include <lanelet2_routing/RoutingGraph.h>
#include <tf2/utils.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace
{
std::string nero_yr_hex(const autoware_perception_msgs::msg::PredictedObject & o)
{
  char b[9];
  std::snprintf(
    b, sizeof(b), "%02x%02x%02x%02x", o.object_id.uuid[0], o.object_id.uuid[1], o.object_id.uuid[2],
    o.object_id.uuid[3]);
  return std::string(b);
}
}  // namespace

namespace autoware::behavior_velocity_planner
{
namespace bg = boost::geometry;

void IntersectionModule::neroYieldRevokeCommit()
{
  if (!nero_yield_revoke::param().on) return;
  const double t_now = rclcpp::Time(planner_data_->predicted_objects->header.stamp).seconds();
  for (const auto & k : nero_yi_yielders_) nero_yr_yielder_seen_[k] = t_now;
  for (auto it = nero_yr_yielder_seen_.begin(); it != nero_yr_yielder_seen_.end();) {
    if (t_now - it->second > 3.0 || t_now < it->second) {
      it = nero_yr_yielder_seen_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = nero_yr_decided_.begin(); it != nero_yr_decided_.end();) {
    if (t_now - it->second.second > 2.0 || t_now < it->second.second) {
      it = nero_yr_decided_.erase(it);
    } else {
      ++it;
    }
  }
  if (is_permanent_go_ && !nero_yr_prev_pgo_) {
    // the GO latched this cycle: on the "safe" of a cycle whose yielders were left out (within the last second)
    for (const auto & kv : nero_yr_yielder_seen_) {
      if (t_now - kv.second <= 1.0) nero_yr_commit_on_.insert(kv.first);
    }
    if (!nero_yr_commit_on_.empty()) {
      RCLCPP_INFO(
        logger_,
        "[NERO_IX] yield_revoke: lane %ld permanent GO while %zu yielder(s) were left out: they stay watched", lane_id_,
        nero_yr_commit_on_.size());
    }
  }
  if (!is_permanent_go_) nero_yr_commit_on_.clear();
  nero_yr_prev_pgo_ = is_permanent_go_;
}

bool IntersectionModule::neroYieldRevokeDecide(
  const InterpolatedPathInfo & interpolated_path_info, const ObjectInfo & object_info, const double s_line,
  const char * where)
{
  const auto & p = nero_yield_revoke::param();
  const auto & obj = object_info.predicted_object();
  const auto key = uuidKey(obj.object_id);
  const double t_now = rclcpp::Time(planner_data_->predicted_objects->header.stamp).seconds();
  auto dit = nero_yr_decided_.find(key);
  if (dit != nero_yr_decided_.end()) {
    dit->second.second = t_now;
    return dit->second.first;
  }
  const double v = planner_data_->current_velocity->twist.linear.x;
  const double a0 = planner_data_->current_acceleration->accel.accel.linear.x;
  // where ego really comes to rest: its own stop from here, never before the stop line it is given
  const double d_need = std::max(std::max(0.0, s_line), nero_yield_revoke::stop_distance(v, a0));
  size_t n_lanes = 0;
  const double s_touch =
    neroYieldRevokeTouch(interpolated_path_info, obj, object_info.unsafe_info().value(), &n_lanes);
  const bool stop = d_need + p.margin <= s_touch;
  nero_yr_decided_.emplace(key, std::make_pair(stop, t_now));
  RCLCPP_INFO(
    logger_,
    "[NERO_IX] yield_revoke: lane %ld %s drives on after its yield (%s; ego %.2f m/s, acc %.2f): ego comes to rest in "
    "%.2f m, its nose meets his way in %.2f m (margin %.2f; his way: %zu lane sequence(s)): %s",
    lane_id_, nero_yr_hex(obj).c_str(), where, v, a0, d_need, s_touch, p.margin, n_lanes,
    stop ? "STOP short of his way" : "go on (a stop would end in his way)");
  return stop;
}

std::optional<size_t> IntersectionModule::neroYieldRevokeStop(
  const InterpolatedPathInfo & interpolated_path_info, const size_t closest_idx)
{
  if (!nero_yield_revoke::param().on || nero_yr_commit_on_.empty()) return std::nullopt;
  bool stop = false;
  for (const auto & object_info : object_info_manager_.attentionObjects()) {
    const auto key = uuidKey(object_info->predicted_object().object_id);
    if (nero_yr_commit_on_.count(key) == 0 || nero_yi_yielders_.count(key) > 0) continue;  // still yields
    if (!object_info->unsafe_info()) continue;
    stop = neroYieldRevokeDecide(interpolated_path_info, *object_info, 0.0, "after ego committed") || stop;
  }
  return stop ? std::make_optional(closest_idx) : std::nullopt;
}

double IntersectionModule::neroYieldRevokeTouch(
  const InterpolatedPathInfo & interpolated_path_info,
  const autoware_perception_msgs::msg::PredictedObject & obj, const CollisionInterval & unsafe,
  size_t * n_lanes) const
{
  constexpr double kWayAhead = 30.0;  // [m] of his lanes from where he is
  constexpr double kStep = 0.5;       // [m] between his footprints along a lane
  constexpr double kWiden = 0.25;     // [m] a side, his box on a lane sample
  constexpr size_t kMaxSamples = 3000;
  // his way: his box, every predicted path of his, the whole path that made him unsafe ...
  std::vector<autoware_utils::Polygon2d> his;
  his.push_back(autoware_utils::to_polygon2d(obj.kinematics.initial_pose_with_covariance.pose, obj.shape));
  for (const auto & q : unsafe.path) his.push_back(autoware_utils::to_polygon2d(q, obj.shape));
  for (const auto & pp : obj.kinematics.predicted_paths) {
    for (const auto & q : pp.path) his.push_back(autoware_utils::to_polygon2d(q, obj.shape));
  }
  // ... and his lanes on through every following lanelet (the map), straight on where the map ends
  auto wide = obj.shape;
  if (wide.type == autoware_perception_msgs::msg::Shape::BOUNDING_BOX) {
    wide.dimensions.x += 2.0 * kWiden;
    wide.dimensions.y += 2.0 * kWiden;
  }
  const auto add = [&](const double x, const double y, const double yaw) {
    if (his.size() >= kMaxSamples) return;
    geometry_msgs::msg::Pose q;
    q.position.x = x;
    q.position.y = y;
    q.orientation = autoware_utils::create_quaternion_from_yaw(yaw);
    his.push_back(autoware_utils::to_polygon2d(q, wide));
  };
  const auto straight = [&](const double x, const double y, const double yaw, const double len) {
    for (double d = kStep; d <= len + 1e-6; d += kStep) add(x + d * std::cos(yaw), y + d * std::sin(yaw), yaw);
  };
  const auto mpose = util::getObjectPoseWithVelocityDirection(obj.kinematics);  // his direction of motion
  const double myaw = tf2::getYaw(mpose.orientation);
  size_t n_seq = 0;
  const auto lanelet_map = planner_data_->route_handler_->getLaneletMapPtr();
  const auto routing_graph = planner_data_->route_handler_->getRoutingGraphPtr();
  if (lanelet_map && routing_graph) {
    std::function<void(const lanelet::ConstLanelet &, double, double, int)> walk =
      [&](const lanelet::ConstLanelet & ll, const double s_from, const double left, const int depth) {
        const auto cl = ll.centerline2d();
        const double len = lanelet::geometry::length(cl);
        if (len < 1e-3) return;
        const double s_end = std::min(len, s_from + left);
        for (double s = s_from; s <= s_end + 1e-6; s += kStep) {
          const auto a = lanelet::geometry::interpolatedPointAtDistance(cl, std::max(0.0, s - 0.5));
          const auto b = lanelet::geometry::interpolatedPointAtDistance(cl, std::min(len, s + 0.5));
          const auto c = lanelet::geometry::interpolatedPointAtDistance(cl, s);
          add(c.x(), c.y(), std::atan2(b.y() - a.y(), b.x() - a.x()));
        }
        const double rest = left - (len - s_from);
        if (rest <= 0.0) {
          ++n_seq;
          return;
        }
        const auto next = routing_graph->following(ll, false);
        if (next.empty() || depth >= 6) {
          const auto a = lanelet::geometry::interpolatedPointAtDistance(cl, std::max(0.0, len - 1.0));
          const auto b = cl.back();
          straight(b.x(), b.y(), std::atan2(b.y() - a.y(), b.x() - a.x()), rest);
          ++n_seq;
          return;
        }
        for (const auto & n : next) walk(n, 0.0, rest, depth + 1);
      };
    const lanelet::BasicPoint2d p2(mpose.position.x, mpose.position.y);
    for (const auto & near : lanelet::geometry::findWithin2d(lanelet_map->laneletLayer, p2, 0.5)) {
      const lanelet::ConstLanelet ll = near.second;
      if (ll.hasAttribute(lanelet::AttributeName::Subtype) &&
          ll.attribute(lanelet::AttributeName::Subtype).value() == lanelet::AttributeValueString::Crosswalk) {
        continue;
      }
      const double ll_yaw = lanelet::utils::getLaneletAngle(ll, mpose.position);
      if (std::fabs(autoware_utils::normalize_radian(ll_yaw - myaw)) > M_PI / 4.0) continue;
      const auto cl = ll.centerline2d();
      const double s0 = lanelet::geometry::toArcCoordinates(cl, p2).length;
      walk(ll, std::max(0.0, s0), kWayAhead, 0);
    }
  }
  if (n_seq == 0) {  // on no lane of his heading: straight on along his direction of motion
    straight(mpose.position.x, mpose.position.y, myaw, kWayAhead);
  }
  if (n_lanes) *n_lanes = n_seq;
  const double his_r = 0.5 * std::hypot(wide.dimensions.x, wide.dimensions.y);

  // ego's footprints along the (dense) interpolated path from where it is
  const auto & pts = interpolated_path_info.path.points;
  if (pts.size() < 2) return 0.0;
  const auto & ego = planner_data_->current_odometry->pose.position;
  size_t i0 = 0;
  double d_best = std::numeric_limits<double>::max();
  for (size_t i = 0; i < pts.size(); ++i) {
    const auto & q = pts.at(i).point.pose.position;
    const double d = std::hypot(q.x - ego.x, q.y - ego.y);
    if (d < d_best) d_best = d, i0 = i;
  }
  const auto base_fp = planner_data_->vehicle_info_.createFootprint(0.0, 0.0);
  const double ego_r = 0.5 * std::hypot(
                               planner_data_->vehicle_info_.vehicle_length_m,
                               planner_data_->vehicle_info_.vehicle_width_m) +
                       planner_data_->vehicle_info_.vehicle_length_m;  // generous: base_link is not the centre
  double s = 0.0, s_prev = 0.0;
  for (size_t j = i0; j < pts.size() && s <= 25.0; ++j) {
    if (j > i0) {
      const auto & a = pts.at(j - 1).point.pose.position;
      const auto & b = pts.at(j).point.pose.position;
      s_prev = s;
      s += std::hypot(b.x - a.x, b.y - a.y);
    }
    const auto & pose = pts.at(j).point.pose;
    autoware_utils::Polygon2d fp;
    bool fp_made = false;
    for (const auto & h : his) {
      const auto & c = h.outer().front();
      if (std::hypot(c.x() - pose.position.x, c.y() - pose.position.y) > ego_r + 2.0 * his_r) continue;
      if (!fp_made) {
        const auto tf = autoware_utils::pose2transform(pose);
        for (const auto & q : autoware_utils::transform_vector(base_fp, tf)) fp.outer().emplace_back(q.x(), q.y());
        bg::correct(fp);
        fp_made = true;
      }
      if (bg::intersects(fp, h)) return j == i0 ? 0.0 : s_prev;  // the last free sample (conservative)
    }
  }
  return std::numeric_limits<double>::infinity();
}

}  // namespace autoware::behavior_velocity_planner
