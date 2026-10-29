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

#ifndef NERO_PROJ_HPP_
#define NERO_PROJ_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <autoware/motion_utils/trajectory/trajectory.hpp>
#include <autoware_utils/geometry/geometry.hpp>

#include <geometry_msgs/msg/point.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace nero_proj
{
// NERO_SOA_PROJ_FAST=1 (2026-10-28, trajgap job, QUEUE TRJ; the 10-28 17:25-17:28 drive: the trajectory fell under 5 Hz
// 8 times -> 6 emergency-stop MRMs): autoware::motion_utils::calcLongitudinalOffsetToSegment and calcLateralOffset copy
// the WHOLE path on every call (removeOverlapPoints: one PathPointWithLaneId with its lane_ids vector per point), and
// findNearestSegmentIndex calls the first. The module's NERO loops project many points per cycle: the return window
// (NERO_SOA_RETURN_NO_NEED: every object's envelope outline every 0.5 m, 3 path copies per sample), the corridor
// give-way's object extents (2 per corner) and the path-from-ego's previous plan (3 per point). In a give-way hold with
// ~10 objects that was 3-8 ms of NeroStation CPU per cycle (0.01 ms otherwise); on the car the planner fell under 5 Hz.
// Proj gives the SAME numbers for one path without the copies: the same nearest-point scan, the same overlap rules
// (eps 1e-8 per axis against the last kept point), the same Eigen arithmetic; the overlap-removed index list is built
// once per path. Unset = the motion_utils calls, as before.
inline bool fast()
{
  static const bool on = [] {
    const char * v = std::getenv("NERO_SOA_PROJ_FAST");
    const bool r = v && std::atoi(v) > 0;
    std::fprintf(stderr, "[NERO_SOA] proj_fast=%d\n", r);
    return r;
  }();
  return on;
}

template <class T>
class Proj
{
public:
  explicit Proj(const T & points) : pts_(points)
  {
    // removeOverlapPoints(points, 0): keep a point unless it lies within eps (per axis) of the last kept one
    kept_.reserve(pts_.size());
    for (size_t i = 0; i < pts_.size(); ++i) {
      if (!kept_.empty() && same(pt(kept_.back()), pt(i))) continue;
      kept_.push_back(i);
    }
  }

  // = autoware::motion_utils::findNearestSegmentIndex(points, p)
  size_t nearestSegment(const geometry_msgs::msg::Point & p) const
  {
    const size_t nearest_idx = autoware::motion_utils::findNearestIndex(pts_, p);  // a scan, no copy
    if (nearest_idx == 0) {
      return 0;
    }
    if (nearest_idx == pts_.size() - 1) {
      return pts_.size() - 2;
    }
    const double signed_length = lonToSegment(nearest_idx, p);
    if (signed_length <= 0) {
      return nearest_idx - 1;
    }
    return nearest_idx;
  }

  // = autoware::motion_utils::calcLongitudinalOffsetToSegment(points, seg_idx, p) (no-throw: NaN where it gives NaN)
  double lonToSegment(const size_t seg_idx, const geometry_msgs::msg::Point & p) const
  {
    if (pts_.empty() || seg_idx >= pts_.size() - 1) {
      return std::nan("");
    }
    // removeOverlapPoints(points, seg_idx) keeps points[0..seg_idx]; its element seg_idx + 1 is the first later point
    // not within eps of points[seg_idx] (the last kept one until then)
    const auto p_front = pt(seg_idx);
    size_t j = seg_idx + 1;
    while (j < pts_.size() && same(p_front, pt(j))) ++j;
    if (j >= pts_.size()) {
      return std::nan("");  // "not supported for the same points"
    }
    const auto p_back = pt(j);
    const Eigen::Vector3d segment_vec{p_back.x - p_front.x, p_back.y - p_front.y, 0};
    const Eigen::Vector3d target_vec{p.x - p_front.x, p.y - p_front.y, 0};
    return segment_vec.dot(target_vec) / segment_vec.norm();
  }

  // = autoware::motion_utils::calcLateralOffset(points, p, seg_idx) (no-throw: NaN where it gives NaN); seg_idx counts
  // in the overlap-removed points there, as it does here
  double lateral(const geometry_msgs::msg::Point & p, const size_t seg_idx) const
  {
    if (kept_.size() <= 1) {
      return std::nan("");
    }
    const auto p_indices = kept_.size() - 2;
    const auto p_front_idx = (p_indices > seg_idx) ? seg_idx : p_indices;
    const auto p_back_idx = p_front_idx + 1;
    const auto p_front = pt(kept_.at(p_front_idx));
    const auto p_back = pt(kept_.at(p_back_idx));
    const Eigen::Vector3d segment_vec{p_back.x - p_front.x, p_back.y - p_front.y, 0.0};
    const Eigen::Vector3d target_vec{p.x - p_front.x, p.y - p_front.y, 0.0};
    const Eigen::Vector3d cross_vec = segment_vec.cross(target_vec);
    return cross_vec(2) / segment_vec.norm();
  }
  // (no calcLateralOffset(points, p) twin: that one searches its segment among the overlap-removed points)

private:
  geometry_msgs::msg::Point pt(const size_t i) const { return autoware_utils::get_point(pts_.at(i)); }
  static bool same(const geometry_msgs::msg::Point & a, const geometry_msgs::msg::Point & b)
  {
    constexpr double eps = 1.0E-08;
    return std::abs(a.x - b.x) < eps && std::abs(a.y - b.y) < eps;
  }

  const T & pts_;
  std::vector<size_t> kept_;
};
}  // namespace nero_proj

#endif  // NERO_PROJ_HPP_
