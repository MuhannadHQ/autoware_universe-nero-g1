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

#ifndef AUTOWARE__BEHAVIOR_PATH_PLANNER_COMMON__UTILS__NERO_MU_HPP_
#define AUTOWARE__BEHAVIOR_PATH_PLANNER_COMMON__UTILS__NERO_MU_HPP_

// NERO_BPP_PROJ_FAST (bppstall job 2026-11-03, QUEUE TRJ2: 11-02 13:24-13:28 behavior_path_planner ran at 3-7 Hz in the
// 72467 -> 44311 street, the trajectory's 10-message rate fell under 5 Hz -> emergency stops).
// autoware_core 1.1.0's motion_utils::calcLongitudinalOffsetToSegment and calcLateralOffset copy the WHOLE path on every
// call (removeOverlapPoints: every PathPointWithLaneId with its lane_ids vector), and findNearestSegmentIndex(points,
// point) / every calcSignedArcLength with a point call the first. Profiled at 11-02 13:24 D (logsim, no load): 49 % of
// behavior_path_planner's CPU was inside those copies (lane change's per-object filter, static avoidance's per-object
// loops). Upstream fixed it in autoware_core b6048885 / PR #1099 (2026-06-17, "avoid per-call full-container copies in
// trajectory offset helpers": allocation-free scans with the exact overlap rules); the car's core is not forked, so the
// same scans live here for the universe callers that run per object / per point.
//   unset / 0: the motion_utils calls (the car exactly);
//   1: the copy-free functions below (the same numbers: same nearest-point scan, the same overlap rule (eps 1e-8 per axis
//      against the last kept point), the same Eigen arithmetic, the same error returns);
//   2: check mode (research only): both, compared bit for bit; mismatches counted and printed; returns the motion_utils one.
// Upstream reference: autoware_core common/autoware_motion_utils/include/autoware/motion_utils/trajectory/trajectory.hpp
// at b6048885 (calcLongitudinalOffsetToSegment, detail::computeOverlapRemovedLateralIndices, calcLateralOffset).

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <autoware/motion_utils/trajectory/trajectory.hpp>
#include <autoware_utils/geometry/geometry.hpp>
#include <autoware/route_handler/route_handler.hpp>
#include <autoware_utils/system/backtrace.hpp>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nero_mu
{
inline int mode()
{
  static const int m = [] {
    const char * v = std::getenv("NERO_BPP_PROJ_FAST");
    const int r = v ? std::atoi(v) : 0;
    if (r > 0) {
      std::fprintf(stderr, "[NERO_BPP_PROJ_FAST] %d (1: copy-free path offsets, 2: check)\n", r);
    }
    return r;
  }();
  return m;
}

namespace detail
{
constexpr double overlap_eps = 1.0E-08;

inline bool same_bits(const double a, const double b)
{
  return (std::isnan(a) && std::isnan(b)) || std::memcmp(&a, &b, sizeof(double)) == 0;
}

struct CheckCount
{
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> bad{0};
  ~CheckCount()
  {
    if (calls.load() > 0) {
      std::fprintf(stderr, "[NERO_BPP_PROJ_FAST] check total: %lu calls, %lu mismatches\n",
                   static_cast<unsigned long>(calls.load()), static_cast<unsigned long>(bad.load()));
    }
  }
};

inline CheckCount & check_count()
{
  static CheckCount c;
  return c;
}

inline void check(const char * fn, const double fast, const double ref)
{
  auto & calls = check_count().calls;
  auto & bad = check_count().bad;
  const auto n = ++calls;
  if (!same_bits(fast, ref)) {
    const auto b = ++bad;
    if (b <= 20 || b % 1000 == 0) {
      std::fprintf(stderr, "[NERO_BPP_PROJ_FAST] check MISMATCH %s fast %.17g ref %.17g (%lu of %lu)\n", fn, fast, ref,
                   static_cast<unsigned long>(b), static_cast<unsigned long>(n));
    }
  }
  if (n % 20000 == 0) {
    std::fprintf(stderr, "[NERO_BPP_PROJ_FAST] check: %lu calls, %lu mismatches\n", static_cast<unsigned long>(n),
                 static_cast<unsigned long>(bad.load()));
  }
}

inline void check(const char * fn, const size_t fast, const size_t ref)
{
  check(fn, static_cast<double>(fast), static_cast<double>(ref));
}

// = autoware::motion_utils::calcLongitudinalOffsetToSegment (core 1.1.0) without removeOverlapPoints' copy
template <class T>
double lon_to_segment(
  const T & points, const size_t seg_idx, const geometry_msgs::msg::Point & p_target,
  const bool throw_exception)
{
  if (seg_idx >= points.size() - 1) {
    const std::string error_message(
      "[autoware_motion_utils] calcLongitudinalOffsetToSegment: Failed to calculate longitudinal offset because the "
      "given segment index is out of the points size.");
    autoware_utils::print_backtrace();
    if (throw_exception) {
      throw std::out_of_range(error_message);
    }
    return std::nan("");
  }
  // removeOverlapPoints(points, seg_idx) keeps points[0..seg_idx]; it is empty only when points is
  if (throw_exception) {
    autoware::motion_utils::validateNonEmpty(points);
  } else {
    try {
      autoware::motion_utils::validateNonEmpty(points);
    } catch (const std::exception &) {
      return std::nan("");
    }
  }
  // its element seg_idx + 1 = the first later point not within eps of points[seg_idx] (the last kept point until then)
  const auto p_front = autoware_utils::get_point(points.at(seg_idx));
  size_t back_idx = seg_idx + 1;
  while (back_idx < points.size()) {
    const auto candidate = autoware_utils::get_point(points.at(back_idx));
    if (std::abs(p_front.x - candidate.x) >= overlap_eps || std::abs(p_front.y - candidate.y) >= overlap_eps) {
      break;
    }
    ++back_idx;
  }
  if (back_idx >= points.size()) {
    const std::string error_message(
      "[autoware_motion_utils] calcLongitudinalOffsetToSegment: Longitudinal offset calculation is not supported for "
      "the same points.");
    autoware_utils::print_backtrace();
    if (throw_exception) {
      throw std::runtime_error(error_message);
    }
    return std::nan("");
  }
  const auto p_back = autoware_utils::get_point(points.at(back_idx));

  const Eigen::Vector3d segment_vec{p_back.x - p_front.x, p_back.y - p_front.y, 0};
  const Eigen::Vector3d target_vec{p_target.x - p_front.x, p_target.y - p_front.y, 0};

  return segment_vec.dot(target_vec) / segment_vec.norm();
}

// = autoware::motion_utils::findNearestSegmentIndex(points, point) (core 1.1.0) with lon_to_segment
template <class T>
size_t nearest_segment(const T & points, const geometry_msgs::msg::Point & point)
{
  const size_t nearest_idx = autoware::motion_utils::findNearestIndex(points, point);  // a scan, no copy
  if (nearest_idx == 0) {
    return 0;
  }
  if (nearest_idx == points.size() - 1) {
    return points.size() - 2;
  }
  const double signed_length = lon_to_segment(points, nearest_idx, point, false);
  if (signed_length <= 0) {
    return nearest_idx - 1;
  }
  return nearest_idx;
}

// = calcSignedArcLength(points, src_point, dst_idx) (core 1.1.0)
template <class T>
double arc_point_to_idx(const T & points, const geometry_msgs::msg::Point & src_point, const size_t dst_idx)
{
  try {
    autoware::motion_utils::validateNonEmpty(points);
  } catch (const std::exception &) {
    return 0.0;
  }
  const size_t src_seg_idx = nearest_segment(points, src_point);
  const double signed_length_on_traj = autoware::motion_utils::calcSignedArcLength(points, src_seg_idx, dst_idx);
  const double signed_length_src_offset = lon_to_segment(points, src_seg_idx, src_point, false);
  return signed_length_on_traj - signed_length_src_offset;
}

// = calcSignedArcLength(points, src_point, dst_point) (core 1.1.0)
template <class T>
double arc_point_to_point(
  const T & points, const geometry_msgs::msg::Point & src_point, const geometry_msgs::msg::Point & dst_point)
{
  try {
    autoware::motion_utils::validateNonEmpty(points);
  } catch (const std::exception &) {
    return 0.0;
  }
  const size_t src_seg_idx = nearest_segment(points, src_point);
  const size_t dst_seg_idx = nearest_segment(points, dst_point);
  const double signed_length_on_traj = autoware::motion_utils::calcSignedArcLength(points, src_seg_idx, dst_seg_idx);
  const double signed_length_src_offset = lon_to_segment(points, src_seg_idx, src_point, false);
  const double signed_length_dst_offset = lon_to_segment(points, dst_seg_idx, dst_point, false);
  return signed_length_on_traj - signed_length_src_offset + signed_length_dst_offset;
}

// = calcLateralOffset(points, p_target, seg_idx) (core 1.1.0): seg_idx counts in the overlap-removed points
// (removeOverlapPoints(points, 0)); upstream b6048885's detail::computeOverlapRemovedLateralIndices finds the two
// original indices in one forward pass
template <class T>
double lateral_seg(
  const T & points, const geometry_msgs::msg::Point & p_target, const size_t seg_idx, const bool throw_exception)
{
  if (throw_exception) {
    autoware::motion_utils::validateNonEmpty(points);
  } else {
    try {
      autoware::motion_utils::validateNonEmpty(points);
    } catch (const std::exception &) {
      return std::nan("");
    }
  }
  size_t kept_count = 1;
  size_t last_kept = 0;
  size_t prev_kept = 0;
  size_t front_rec = 0;
  size_t back_rec = 0;
  auto last_p = autoware_utils::get_point(points.at(0));
  for (size_t i = 1; i < points.size(); ++i) {
    const auto curr_p = autoware_utils::get_point(points.at(i));
    if (std::abs(last_p.x - curr_p.x) < overlap_eps && std::abs(last_p.y - curr_p.y) < overlap_eps) {
      continue;
    }
    prev_kept = last_kept;
    last_kept = i;
    last_p = curr_p;
    const size_t kept_pos = kept_count;
    ++kept_count;
    if (kept_pos == seg_idx) {
      front_rec = i;
    }
    if (kept_pos == seg_idx + 1) {
      back_rec = i;
    }
  }
  if (kept_count == 1) {
    const std::string error_message(
      "[autoware_motion_utils] calcLateralOffset: Lateral offset calculation is not supported for the same points.");
    autoware_utils::print_backtrace();
    if (throw_exception) {
      throw std::runtime_error(error_message);
    }
    return std::nan("");
  }
  size_t front_idx = 0;
  size_t back_idx = 0;
  if (kept_count - 2 > seg_idx) {  // p_front_idx = seg_idx, p_back_idx = seg_idx + 1
    front_idx = front_rec;
    back_idx = back_rec;
  } else {  // clamped to the last two kept points
    front_idx = prev_kept;
    back_idx = last_kept;
  }
  const auto p_front = autoware_utils::get_point(points.at(front_idx));
  const auto p_back = autoware_utils::get_point(points.at(back_idx));

  const Eigen::Vector3d segment_vec{p_back.x - p_front.x, p_back.y - p_front.y, 0.0};
  const Eigen::Vector3d target_vec{p_target.x - p_front.x, p_target.y - p_front.y, 0.0};

  const Eigen::Vector3d cross_vec = segment_vec.cross(target_vec);
  return cross_vec(2) / segment_vec.norm();
}

// = calcLateralOffset(points, p_target) (core 1.1.0): the segment is searched among the overlap-removed points (an index
// list here instead of a copy of the points), then calcLateralOffset(points, p_target, seg_idx) on the original ones
template <class T>
double lateral(const T & points, const geometry_msgs::msg::Point & p_target, const bool throw_exception)
{
  if (throw_exception) {
    autoware::motion_utils::validateNonEmpty(points);
  } else {
    try {
      autoware::motion_utils::validateNonEmpty(points);
    } catch (const std::exception &) {
      return std::nan("");
    }
  }
  std::vector<size_t> kept;
  kept.reserve(points.size());
  kept.push_back(0);
  auto last_p = autoware_utils::get_point(points.at(0));
  for (size_t i = 1; i < points.size(); ++i) {
    const auto curr_p = autoware_utils::get_point(points.at(i));
    if (std::abs(last_p.x - curr_p.x) < overlap_eps && std::abs(last_p.y - curr_p.y) < overlap_eps) {
      continue;
    }
    kept.push_back(i);
    last_p = curr_p;
  }
  if (kept.size() == 1) {
    const std::string error_message(
      "[autoware_motion_utils] calcLateralOffset: Lateral offset calculation is not supported for the same points.");
    autoware_utils::print_backtrace();
    if (throw_exception) {
      throw std::runtime_error(error_message);
    }
    return std::nan("");
  }
  // findNearestSegmentIndex(overlap_removed_points, p_target)
  double min_dist = std::numeric_limits<double>::max();
  size_t nearest = 0;
  for (size_t k = 0; k < kept.size(); ++k) {
    const auto dist = autoware_utils::calc_squared_distance2d(points.at(kept[k]), p_target);
    if (dist < min_dist) {
      min_dist = dist;
      nearest = k;
    }
  }
  size_t seg_idx = nearest;
  if (nearest == 0) {
    seg_idx = 0;
  } else if (nearest == kept.size() - 1) {
    seg_idx = kept.size() - 2;
  } else {
    // calcLongitudinalOffsetToSegment(overlap_removed_points, nearest, p): consecutive kept points never overlap
    const auto p_front = autoware_utils::get_point(points.at(kept[nearest]));
    const auto p_back = autoware_utils::get_point(points.at(kept[nearest + 1]));
    const Eigen::Vector3d segment_vec{p_back.x - p_front.x, p_back.y - p_front.y, 0};
    const Eigen::Vector3d target_vec{p_target.x - p_front.x, p_target.y - p_front.y, 0};
    const double signed_length = segment_vec.dot(target_vec) / segment_vec.norm();
    seg_idx = (signed_length <= 0) ? nearest - 1 : nearest;
  }
  return lateral_seg(points, p_target, seg_idx, throw_exception);
}
}  // namespace detail

// ---- the drop-in calls (same signatures as autoware::motion_utils) ----

template <class T>
double calcLongitudinalOffsetToSegment(
  const T & points, const size_t seg_idx, const geometry_msgs::msg::Point & p_target,
  const bool throw_exception = false)
{
  const int m = mode();
  if (m == 1) return detail::lon_to_segment(points, seg_idx, p_target, throw_exception);
  const double ref =
    autoware::motion_utils::calcLongitudinalOffsetToSegment(points, seg_idx, p_target, throw_exception);
  if (m == 2) detail::check("lon_to_segment", detail::lon_to_segment(points, seg_idx, p_target, false), ref);
  return ref;
}

template <class T>
size_t findNearestSegmentIndex(const T & points, const geometry_msgs::msg::Point & point)
{
  const int m = mode();
  if (m == 1) return detail::nearest_segment(points, point);
  const size_t ref = autoware::motion_utils::findNearestSegmentIndex(points, point);
  if (m == 2) detail::check("nearest_segment", detail::nearest_segment(points, point), ref);
  return ref;
}

// = findNearestSegmentIndex(points, pose, max_dist, max_yaw) (core 1.1.0): the nearest index by distance + yaw, then the
// same segment rule with the copy-free offset
template <class T>
std::optional<size_t> findNearestSegmentIndex(
  const T & points, const geometry_msgs::msg::Pose & pose,
  const double max_dist = std::numeric_limits<double>::max(),
  const double max_yaw = std::numeric_limits<double>::max())
{
  const auto fast = [&]() -> std::optional<size_t> {
    const auto nearest_idx = autoware::motion_utils::findNearestIndex(points, pose, max_dist, max_yaw);
    if (!nearest_idx) {
      return std::nullopt;
    }
    if (*nearest_idx == 0) {
      return 0;
    }
    if (*nearest_idx == points.size() - 1) {
      return points.size() - 2;
    }
    const double signed_length = detail::lon_to_segment(points, *nearest_idx, pose.position, false);
    if (signed_length <= 0) {
      return *nearest_idx - 1;
    }
    return *nearest_idx;
  };
  const int m = mode();
  if (m == 1) return fast();
  const auto ref = autoware::motion_utils::findNearestSegmentIndex(points, pose, max_dist, max_yaw);
  if (m == 2) {
    const auto f = fast();
    detail::check(
      "nearest_segment_pose", f ? static_cast<double>(*f) : -1.0, ref ? static_cast<double>(*ref) : -1.0);
  }
  return ref;
}

template <class T>
double calcSignedArcLength(const T & points, const geometry_msgs::msg::Point & src_point, const size_t dst_idx)
{
  const int m = mode();
  if (m == 1) return detail::arc_point_to_idx(points, src_point, dst_idx);
  const double ref = autoware::motion_utils::calcSignedArcLength(points, src_point, dst_idx);
  if (m == 2) detail::check("arc_point_to_idx", detail::arc_point_to_idx(points, src_point, dst_idx), ref);
  return ref;
}

template <class T>
double calcSignedArcLength(const T & points, const size_t src_idx, const geometry_msgs::msg::Point & dst_point)
{
  const int m = mode();
  // = -calcSignedArcLength(points, dst_point, src_idx) after the same empty check (core 1.1.0)
  const auto fast = [&]() -> double {
    try {
      autoware::motion_utils::validateNonEmpty(points);
    } catch (const std::exception &) {
      return 0.0;
    }
    return -detail::arc_point_to_idx(points, dst_point, src_idx);
  };
  if (m == 1) return fast();
  const double ref = autoware::motion_utils::calcSignedArcLength(points, src_idx, dst_point);
  if (m == 2) detail::check("arc_idx_to_point", fast(), ref);
  return ref;
}

template <class T>
double calcSignedArcLength(
  const T & points, const geometry_msgs::msg::Point & src_point, const geometry_msgs::msg::Point & dst_point)
{
  const int m = mode();
  if (m == 1) return detail::arc_point_to_point(points, src_point, dst_point);
  const double ref = autoware::motion_utils::calcSignedArcLength(points, src_point, dst_point);
  if (m == 2) detail::check("arc_point_to_point", detail::arc_point_to_point(points, src_point, dst_point), ref);
  return ref;
}

// index-only overload: no copy in motion_utils; forwarded so a replaced call site keeps compiling for any argument mix
template <class T>
double calcSignedArcLength(const T & points, const size_t src_idx, const size_t dst_idx)
{
  return autoware::motion_utils::calcSignedArcLength(points, src_idx, dst_idx);
}

// the segment-index overloads (core 1.1.0, trajectory.hpp ~1966-2055): validateNonEmpty THROWS there (no catch); the
// offsets to the given segments copy-free
template <class T>
double calcSignedArcLength(
  const T & points, const geometry_msgs::msg::Point & src_point, const size_t src_seg_idx,
  const geometry_msgs::msg::Point & dst_point, const size_t dst_seg_idx)
{
  const auto fast = [&]() -> double {
    autoware::motion_utils::validateNonEmpty(points);
    const double signed_length_on_traj = autoware::motion_utils::calcSignedArcLength(points, src_seg_idx, dst_seg_idx);
    const double signed_length_src_offset = detail::lon_to_segment(points, src_seg_idx, src_point, false);
    const double signed_length_dst_offset = detail::lon_to_segment(points, dst_seg_idx, dst_point, false);
    return signed_length_on_traj - signed_length_src_offset + signed_length_dst_offset;
  };
  const int m = mode();
  if (m == 1) return fast();
  const double ref =
    autoware::motion_utils::calcSignedArcLength(points, src_point, src_seg_idx, dst_point, dst_seg_idx);
  if (m == 2) detail::check("arc_pseg_to_pseg", fast(), ref);
  return ref;
}

template <class T>
double calcSignedArcLength(
  const T & points, const geometry_msgs::msg::Point & src_point, const size_t src_seg_idx, const size_t dst_idx)
{
  const auto fast = [&]() -> double {
    autoware::motion_utils::validateNonEmpty(points);
    const double signed_length_on_traj = autoware::motion_utils::calcSignedArcLength(points, src_seg_idx, dst_idx);
    const double signed_length_src_offset = detail::lon_to_segment(points, src_seg_idx, src_point, false);
    return signed_length_on_traj - signed_length_src_offset;
  };
  const int m = mode();
  if (m == 1) return fast();
  const double ref = autoware::motion_utils::calcSignedArcLength(points, src_point, src_seg_idx, dst_idx);
  if (m == 2) detail::check("arc_pseg_to_idx", fast(), ref);
  return ref;
}

template <class T>
double calcSignedArcLength(
  const T & points, const size_t src_idx, const geometry_msgs::msg::Point & dst_point, const size_t dst_seg_idx)
{
  const auto fast = [&]() -> double {
    autoware::motion_utils::validateNonEmpty(points);
    const double signed_length_on_traj = autoware::motion_utils::calcSignedArcLength(points, src_idx, dst_seg_idx);
    const double signed_length_dst_offset = detail::lon_to_segment(points, dst_seg_idx, dst_point, false);
    return signed_length_on_traj + signed_length_dst_offset;
  };
  const int m = mode();
  if (m == 1) return fast();
  const double ref = autoware::motion_utils::calcSignedArcLength(points, src_idx, dst_point, dst_seg_idx);
  if (m == 2) detail::check("arc_idx_to_pseg", fast(), ref);
  return ref;
}

namespace detail
{
inline bool same_pose(const std::optional<geometry_msgs::msg::Pose> & a, const std::optional<geometry_msgs::msg::Pose> & b)
{
  if (!a || !b) return !a && !b;
  return same_bits(a->position.x, b->position.x) && same_bits(a->position.y, b->position.y) &&
         same_bits(a->position.z, b->position.z) && same_bits(a->orientation.x, b->orientation.x) &&
         same_bits(a->orientation.y, b->orientation.y) && same_bits(a->orientation.z, b->orientation.z) &&
         same_bits(a->orientation.w, b->orientation.w);
}
inline bool same_point(const std::optional<geometry_msgs::msg::Point> & a, const std::optional<geometry_msgs::msg::Point> & b)
{
  if (!a || !b) return !a && !b;
  return same_bits(a->x, b->x) && same_bits(a->y, b->y) && same_bits(a->z, b->z);
}
}  // namespace detail

// = calcLongitudinalOffsetPose(points, src_point, offset, set_orientation) (core 1.1.0): the segment and the offset to it
// copy-free, then the index overload (no copy) as there
template <class T>
std::optional<geometry_msgs::msg::Pose> calcLongitudinalOffsetPose(
  const T & points, const geometry_msgs::msg::Point & src_point, const double offset,
  const bool set_orientation_from_position_direction = true)
{
  const auto fast = [&]() -> std::optional<geometry_msgs::msg::Pose> {
    try {
      autoware::motion_utils::validateNonEmpty(points);
    } catch (const std::exception &) {
      return {};
    }
    const size_t src_seg_idx = detail::nearest_segment(points, src_point);
    const double signed_length_src_offset = detail::lon_to_segment(points, src_seg_idx, src_point, false);
    return autoware::motion_utils::calcLongitudinalOffsetPose(
      points, src_seg_idx, offset + signed_length_src_offset, set_orientation_from_position_direction);
  };
  const int m = mode();
  if (m == 1) return fast();
  const auto ref = autoware::motion_utils::calcLongitudinalOffsetPose(
    points, src_point, offset, set_orientation_from_position_direction);
  if (m == 2) detail::check("offset_pose", detail::same_pose(fast(), ref) ? 0.0 : 1.0, 0.0);
  return ref;
}

template <class T>
std::optional<geometry_msgs::msg::Pose> calcLongitudinalOffsetPose(
  const T & points, const size_t src_idx, const double offset,
  const bool set_orientation_from_position_direction = true, const bool throw_exception = false)
{
  return autoware::motion_utils::calcLongitudinalOffsetPose(
    points, src_idx, offset, set_orientation_from_position_direction, throw_exception);
}

// = calcLongitudinalOffsetPoint(points, src_point, offset) (core 1.1.0); a negative offset reverses a copy of the path
// there: left to motion_utils (unchanged)
template <class T>
std::optional<geometry_msgs::msg::Point> calcLongitudinalOffsetPoint(
  const T & points, const geometry_msgs::msg::Point & src_point, const double offset)
{
  const auto fast = [&]() -> std::optional<geometry_msgs::msg::Point> {
    try {
      autoware::motion_utils::validateNonEmpty(points);
    } catch (const std::exception &) {
      return {};
    }
    const size_t src_seg_idx = detail::nearest_segment(points, src_point);
    const double signed_length_src_offset = detail::lon_to_segment(points, src_seg_idx, src_point, false);
    return autoware::motion_utils::calcLongitudinalOffsetPoint(points, src_seg_idx, offset + signed_length_src_offset);
  };
  const int m = mode();
  if (m == 1 && offset >= 0.0) return fast();
  const auto ref = autoware::motion_utils::calcLongitudinalOffsetPoint(points, src_point, offset);
  if (m == 2 && offset >= 0.0) detail::check("offset_point", detail::same_point(fast(), ref) ? 0.0 : 1.0, 0.0);
  return ref;
}

template <class T>
std::optional<geometry_msgs::msg::Point> calcLongitudinalOffsetPoint(
  const T & points, const size_t src_idx, const double offset, const bool throw_exception = false)
{
  return autoware::motion_utils::calcLongitudinalOffsetPoint(points, src_idx, offset, throw_exception);
}

// = findFirstNearestSegmentIndexWithSoftConstraints (core 1.1.0) with the copy-free offset
template <class T>
size_t findFirstNearestSegmentIndexWithSoftConstraints(
  const T & points, const geometry_msgs::msg::Pose & pose,
  const double dist_threshold = std::numeric_limits<double>::max(),
  const double yaw_threshold = std::numeric_limits<double>::max())
{
  const auto fast = [&]() -> size_t {
    const size_t nearest_idx = autoware::motion_utils::findFirstNearestIndexWithSoftConstraints(
      points, pose, dist_threshold, yaw_threshold);
    if (nearest_idx == 0) {
      return 0;
    }
    if (nearest_idx == points.size() - 1) {
      return points.size() - 2;
    }
    const double signed_length = detail::lon_to_segment(points, nearest_idx, pose.position, false);
    if (signed_length <= 0) {
      return nearest_idx - 1;
    }
    return nearest_idx;
  };
  const int m = mode();
  if (m == 1) return fast();
  const size_t ref = autoware::motion_utils::findFirstNearestSegmentIndexWithSoftConstraints(
    points, pose, dist_threshold, yaw_threshold);
  if (m == 2) detail::check("first_nearest_segment_soft", fast(), ref);
  return ref;
}

template <class T>
double calcLateralOffset(
  const T & points, const geometry_msgs::msg::Point & p_target, const size_t seg_idx,
  const bool throw_exception = false)
{
  const int m = mode();
  if (m == 1) return detail::lateral_seg(points, p_target, seg_idx, throw_exception);
  const double ref = autoware::motion_utils::calcLateralOffset(points, p_target, seg_idx, throw_exception);
  if (m == 2) detail::check("lateral_seg", detail::lateral_seg(points, p_target, seg_idx, false), ref);
  return ref;
}

template <class T>
double calcLateralOffset(const T & points, const geometry_msgs::msg::Point & p_target, const bool throw_exception = false)
{
  const int m = mode();
  if (m == 1) return detail::lateral(points, p_target, throw_exception);
  const double ref = autoware::motion_utils::calcLateralOffset(points, p_target, throw_exception);
  if (m == 2) detail::check("lateral", detail::lateral(points, p_target, false), ref);
  return ref;
}

// = route_handler.getClosestLaneletWithinRoute(pose, &lanelet) (a linear scan with polygon distances over every route
// lanelet: 65 segments on the 11-02 trip), memoised by the route's UUID and the pose's exact bits. Static avoidance asks it
// per object up to 4 times (getRoadShoulderDistance for the object and its flipped twins) with the reference-path pose
// nearest to the object, and lane change per object: the same pose gives the same answer (a pure function of the route
// and the pose), so a repeat is free. The memo is cleared on a new route UUID and above 20000 entries.
namespace detail
{
struct PoseBits
{
  std::array<uint64_t, 7> b{};
  bool operator==(const PoseBits & o) const { return b == o.b; }
};
struct PoseBitsHash
{
  size_t operator()(const PoseBits & k) const
  {
    uint64_t h = 1469598103934665603ULL;
    for (const auto v : k.b) {
      h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return static_cast<size_t>(h);
  }
};
inline PoseBits pose_bits(const geometry_msgs::msg::Pose & p)
{
  const double v[7] = {p.position.x,    p.position.y,    p.position.z,   p.orientation.x,
                       p.orientation.y, p.orientation.z, p.orientation.w};
  PoseBits k;
  std::memcpy(k.b.data(), v, sizeof(v));
  return k;
}
struct RouteLaneletMemo
{
  std::mutex m;
  std::array<uint8_t, 16> uuid{};
  const void * handler{nullptr};
  std::unordered_map<PoseBits, std::pair<bool, lanelet::ConstLanelet>, PoseBitsHash> map;
};
inline RouteLaneletMemo & route_lanelet_memo()
{
  static RouteLaneletMemo memo;
  return memo;
}
}  // namespace detail

inline bool getClosestLaneletWithinRoute(
  const autoware::route_handler::RouteHandler & rh, const geometry_msgs::msg::Pose & pose,
  lanelet::ConstLanelet * closest_lanelet)
{
  const int m = mode();
  if (m != 1 && m != 2) {
    return rh.getClosestLaneletWithinRoute(pose, closest_lanelet);
  }
  const auto fast = [&]() -> std::pair<bool, lanelet::ConstLanelet> {
    auto & memo = detail::route_lanelet_memo();
    std::lock_guard<std::mutex> lock(memo.m);
    const auto uuid = rh.getRouteUuid().uuid;
    if (uuid != memo.uuid || memo.handler != static_cast<const void *>(&rh) || memo.map.size() > 20000) {
      memo.map.clear();
      memo.uuid = uuid;
      memo.handler = &rh;
    }
    const auto key = detail::pose_bits(pose);
    const auto it = memo.map.find(key);
    if (it != memo.map.end()) {
      return it->second;
    }
    lanelet::ConstLanelet ll;
    const bool found = rh.getClosestLaneletWithinRoute(pose, &ll);
    memo.map.emplace(key, std::make_pair(found, ll));
    return {found, ll};
  };
  if (m == 1) {
    const auto r = fast();
    if (r.first) {
      *closest_lanelet = r.second;
    }
    return r.first;
  }
  lanelet::ConstLanelet ref_ll;
  const bool ref = rh.getClosestLaneletWithinRoute(pose, &ref_ll);
  const auto r = fast();
  detail::check(
    "closest_route_lanelet", (r.first == ref && (!ref || r.second.id() == ref_ll.id())) ? 0.0 : 1.0, 0.0);
  if (ref) {
    *closest_lanelet = ref_ll;
  }
  return ref;
}
}  // namespace nero_mu

#endif  // AUTOWARE__BEHAVIOR_PATH_PLANNER_COMMON__UTILS__NERO_MU_HPP_
