// Copyright 2026 Nero (trjrate-2, 2026-11-09, QUEUE TRJ3)
//
// autoware::motion_utils::resamplePath for PathWithLaneId (autoware_core 1.1.0, common/autoware_motion_utils/src/resample/
// resample.cpp, both overloads) with the SAME output, without its two costs that grow with the number of samples:
//   - interpolation::calc_closest_segment_indices scans the input keys from the END for every sample (O(n x m): a 300 m
//     path splined at 0.1 m = 3000 samples x ~150 input points, once per call); here the same index by binary search: the
//     test "keys[j-1] - threshold < q && q < keys[j]", largest j above the previous index, holds exactly on [jB, jA] where
//     jA = the last j with keys[j-1] - threshold < q (that difference is monotone in j) and jB = the first j with q < keys[j];
//   - the lane-id insertion for input points with several lane ids scans the samples linearly: on a sorted sample list the
//     first j with s[j-1] <= d < s[j] is upper_bound(d) (an unsorted list keeps upstream's scan);
//   - the per-sample lane-id vectors are moved into the output instead of copied.
// Every other step calls the same core functions in the same order (validators, calcSignedArcLength, resamplePoseVector,
// lerp, zero_order_hold), so the output is bit-identical; callers ship it with a check mode that compares every field.
#ifndef AUTOWARE__BEHAVIOR_VELOCITY_INTERSECTION_MODULE__NERO_RESAMPLE_HPP_
#define AUTOWARE__BEHAVIOR_VELOCITY_INTERSECTION_MODULE__NERO_RESAMPLE_HPP_

#include <autoware/interpolation/interpolation_utils.hpp>
#include <autoware/interpolation/linear_interpolation.hpp>
#include <autoware/interpolation/zero_order_hold.hpp>
#include <autoware/motion_utils/constants.hpp>
#include <autoware/motion_utils/resample/resample.hpp>
#include <autoware/motion_utils/resample/resample_utils.hpp>
#include <autoware/motion_utils/trajectory/trajectory.hpp>
#include <autoware_utils/geometry/geometry.hpp>

#include <autoware_internal_planning_msgs/msg/path_with_lane_id.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace autoware::behavior_velocity_planner::nero_resample
{
using autoware_internal_planning_msgs::msg::PathWithLaneId;

// = autoware::interpolation::calc_closest_segment_indices (the same validation / exceptions, the same indices)
inline std::vector<size_t> closest_segment_indices_of(
  const std::vector<double> & base_keys, const std::vector<double> & query_keys,
  const double overlap_threshold = 1e-3)
{
  const auto validated_query_keys = autoware::interpolation::validateKeys(base_keys, query_keys);
  const size_t n = base_keys.size();
  std::vector<double> lo(n, 0.0);  // lo[j] = base_keys[j - 1] - overlap_threshold, the same arithmetic as upstream's test
  for (size_t j = 1; j < n; ++j) lo[j] = base_keys[j - 1] - overlap_threshold;
  std::vector<size_t> closest_segment_indices(validated_query_keys.size());
  size_t closest_segment_idx = 0;
  for (size_t i = 0; i < validated_query_keys.size(); ++i) {
    const double q = validated_query_keys[i];
    if (base_keys.back() - overlap_threshold < q) {
      closest_segment_idx = n - 1;
    } else {
      // jA: the last j in [1, n-1] with lo[j] < q (lo is non-decreasing)
      const size_t first_not_below =
        static_cast<size_t>(std::lower_bound(lo.begin() + 1, lo.end(), q) - lo.begin());
      // jB: the first j in [1, n-1] with q < base_keys[j]
      const size_t jB =
        static_cast<size_t>(std::upper_bound(base_keys.begin() + 1, base_keys.end(), q) - base_keys.begin());
      if (first_not_below >= 2 && jB <= n - 1) {
        const size_t jA = first_not_below - 1;
        if (jA >= jB && jA > closest_segment_idx) {
          closest_segment_idx = jA - 1;
        }
      }
    }
    closest_segment_indices[i] = closest_segment_idx;
  }
  return closest_segment_indices;
}

// = autoware::motion_utils::resamplePath(input_path, resampled_arclength, ...)
inline PathWithLaneId resamplePath(
  const PathWithLaneId & input_path, const std::vector<double> & resampled_arclength,
  const bool use_akima_spline_for_xy = false, const bool use_lerp_for_z = true,
  const bool use_zero_order_hold_for_v = true)
{
  auto resampling_arclength = resampled_arclength;

  // Add resampling_arclength to insert input points which have multiple lane_ids
  // (upstream: calcSignedArcLength(points, 0, i) per such point = the sum of the segment lengths 0..i-1 added in order from
  // 0.0; here the same sum kept running: the same additions in the same order, the same value)
  double running_s = 0.0;
  size_t running_i = 0;
  for (size_t i = 0; i < input_path.points.size(); ++i) {
    if (input_path.points.at(i).lane_ids.size() < 2) {
      continue;
    }

    for (; running_i < i; ++running_i) {
      running_s += autoware_utils::calc_distance2d(
        input_path.points.at(running_i), input_path.points.at(running_i + 1));
    }
    const double distance_to_resampling_point = running_s;
    size_t j_hit = 0;  // 0 = none (upstream's loop starts at 1)
    if (std::is_sorted(resampling_arclength.begin(), resampling_arclength.end())) {
      const size_t u = static_cast<size_t>(
        std::upper_bound(
          resampling_arclength.begin(), resampling_arclength.end(), distance_to_resampling_point) -
        resampling_arclength.begin());
      if (u >= 1 && u < resampling_arclength.size()) j_hit = u;
    } else {
      for (size_t j = 1; j < resampling_arclength.size(); ++j) {
        if (
          resampling_arclength.at(j - 1) <= distance_to_resampling_point &&
          distance_to_resampling_point < resampling_arclength.at(j)) {
          j_hit = j;
          break;
        }
      }
    }
    if (j_hit != 0) {
      const size_t j = j_hit;
      const double dist_to_prev_point =
        std::fabs(distance_to_resampling_point - resampling_arclength.at(j - 1));
      const double dist_to_following_point =
        std::fabs(resampling_arclength.at(j) - distance_to_resampling_point);
      if (dist_to_prev_point < autoware::motion_utils::overlap_threshold) {
        resampling_arclength.at(j - 1) = distance_to_resampling_point;
      } else if (dist_to_following_point < autoware::motion_utils::overlap_threshold) {
        resampling_arclength.at(j) = distance_to_resampling_point;
      } else {
        resampling_arclength.insert(resampling_arclength.begin() + j, distance_to_resampling_point);
      }
    }
  }

  // validate arguments
  if (!resample_utils::validate_arguments(input_path.points, resampling_arclength)) {
    return input_path;
  }

  // Input Path Information
  std::vector<double> input_arclength;
  std::vector<geometry_msgs::msg::Pose> input_pose;
  std::vector<double> v_lon;
  std::vector<double> v_lat;
  std::vector<double> heading_rate;
  std::vector<bool> is_final;
  input_arclength.reserve(input_path.points.size());
  input_pose.reserve(input_path.points.size());
  v_lon.reserve(input_path.points.size());
  v_lat.reserve(input_path.points.size());
  heading_rate.reserve(input_path.points.size());
  is_final.reserve(input_path.points.size());

  input_arclength.push_back(0.0);
  input_pose.push_back(input_path.points.front().point.pose);
  v_lon.push_back(input_path.points.front().point.longitudinal_velocity_mps);
  v_lat.push_back(input_path.points.front().point.lateral_velocity_mps);
  heading_rate.push_back(input_path.points.front().point.heading_rate_rps);
  is_final.push_back(input_path.points.front().point.is_final);
  for (size_t i = 1; i < input_path.points.size(); ++i) {
    const auto & prev_pt = input_path.points.at(i - 1).point;
    const auto & curr_pt = input_path.points.at(i).point;
    const double ds = autoware_utils::calc_distance2d(prev_pt.pose.position, curr_pt.pose.position);
    input_arclength.push_back(ds + input_arclength.back());
    input_pose.push_back(curr_pt.pose);
    v_lon.push_back(curr_pt.longitudinal_velocity_mps);
    v_lat.push_back(curr_pt.lateral_velocity_mps);
    heading_rate.push_back(curr_pt.heading_rate_rps);
    is_final.push_back(curr_pt.is_final);
  }

  if (input_arclength.back() < resampling_arclength.back()) {
    std::cerr << "[autoware_motion_utils]: resampled path length is longer than input path length"
              << std::endl;
    return input_path;
  }

  // Interpolate
  const auto lerp = [&](const auto & input) {
    return autoware::interpolation::lerp(input_arclength, input, resampling_arclength);
  };

  const auto closest_segment_indices = closest_segment_indices_of(input_arclength, resampling_arclength);

  const auto zoh = [&](const auto & input) {
    return autoware::interpolation::zero_order_hold(input_arclength, input, closest_segment_indices);
  };

  const auto interpolated_pose = autoware::motion_utils::resamplePoseVector(
    input_pose, resampling_arclength, use_akima_spline_for_xy, use_lerp_for_z);
  const auto interpolated_v_lon = use_zero_order_hold_for_v ? zoh(v_lon) : lerp(v_lon);
  const auto interpolated_v_lat = use_zero_order_hold_for_v ? zoh(v_lat) : lerp(v_lat);
  const auto interpolated_heading_rate = lerp(heading_rate);
  const auto interpolated_is_final = zoh(is_final);

  // interpolate lane_ids
  std::vector<std::vector<int64_t>> interpolated_lane_ids;
  interpolated_lane_ids.resize(resampling_arclength.size());
  constexpr double epsilon = 1e-6;
  for (size_t i = 0; i < resampling_arclength.size(); ++i) {
    const size_t seg_idx = std::min(closest_segment_indices.at(i), input_path.points.size() - 2);
    const auto & prev_lane_ids = input_path.points.at(seg_idx).lane_ids;
    const auto & next_lane_ids = input_path.points.at(seg_idx + 1).lane_ids;

    if (std::abs(input_arclength.at(seg_idx) - resampling_arclength.at(i)) <= epsilon) {
      interpolated_lane_ids.at(i).insert(
        interpolated_lane_ids.at(i).end(), prev_lane_ids.begin(), prev_lane_ids.end());
    } else if (std::abs(input_arclength.at(seg_idx + 1) - resampling_arclength.at(i)) <= epsilon) {
      interpolated_lane_ids.at(i).insert(
        interpolated_lane_ids.at(i).end(), next_lane_ids.begin(), next_lane_ids.end());
    } else {
      // extract lane_ids those prev_lane_ids and next_lane_ids have in common
      for (const auto target_lane_id : prev_lane_ids) {
        if (
          std::find(next_lane_ids.begin(), next_lane_ids.end(), target_lane_id) !=
          next_lane_ids.end()) {
          interpolated_lane_ids.at(i).push_back(target_lane_id);
        }
      }
      // If there are no common lane_ids, the prev_lane_ids is assigned.
      if (interpolated_lane_ids.at(i).empty()) {
        interpolated_lane_ids.at(i).insert(
          interpolated_lane_ids.at(i).end(), prev_lane_ids.begin(), prev_lane_ids.end());
      }
    }
  }

  if (interpolated_pose.size() != resampling_arclength.size()) {
    std::cerr
      << "[autoware_motion_utils]: Resampled pose size is different from resampled arclength"
      << std::endl;
    return input_path;
  }

  PathWithLaneId resampled_path;
  resampled_path.header = input_path.header;
  resampled_path.left_bound = input_path.left_bound;
  resampled_path.right_bound = input_path.right_bound;
  resampled_path.points.resize(interpolated_pose.size());
  for (size_t i = 0; i < resampled_path.points.size(); ++i) {
    autoware_planning_msgs::msg::PathPoint path_point;
    path_point.pose = interpolated_pose.at(i);
    path_point.longitudinal_velocity_mps = interpolated_v_lon.at(i);
    path_point.lateral_velocity_mps = interpolated_v_lat.at(i);
    path_point.heading_rate_rps = interpolated_heading_rate.at(i);
    path_point.is_final = interpolated_is_final.at(i);
    resampled_path.points.at(i).point = path_point;
    resampled_path.points.at(i).lane_ids = std::move(interpolated_lane_ids.at(i));
  }

  return resampled_path;
}

// = autoware::motion_utils::resamplePath(input_path, resample_interval, ...)
inline PathWithLaneId resamplePath(
  const PathWithLaneId & input_path, const double resample_interval,
  const bool use_akima_spline_for_xy, const bool use_lerp_for_z,
  const bool use_zero_order_hold_for_v, const bool resample_input_path_stop_point)
{
  // validate arguments
  if (!resample_utils::validate_arguments(input_path.points, resample_interval)) {
    return input_path;
  }

  // transform input_path
  std::vector<autoware_planning_msgs::msg::PathPoint> transformed_input_path(
    input_path.points.size());
  for (size_t i = 0; i < input_path.points.size(); ++i) {
    transformed_input_path.at(i) = input_path.points.at(i).point;
  }
  // compute path length
  const double input_path_len = autoware::motion_utils::calcArcLength(transformed_input_path);

  std::vector<double> resampling_arclength;
  for (double s = 0.0; s < input_path_len; s += resample_interval) {
    resampling_arclength.push_back(s);
  }
  if (resampling_arclength.empty()) {
    std::cerr << "[autoware_motion_utils]: resampling arclength is empty" << std::endl;
    return input_path;
  }

  // Insert terminal point
  if (input_path_len - resampling_arclength.back() < autoware::motion_utils::overlap_threshold) {
    resampling_arclength.back() = input_path_len;
  } else {
    resampling_arclength.push_back(input_path_len);
  }

  // Insert stop point
  if (resample_input_path_stop_point) {
    const auto distance_to_stop_point =
      autoware::motion_utils::calcDistanceToForwardStopPoint(transformed_input_path, 0);
    if (distance_to_stop_point && !resampling_arclength.empty()) {
      for (size_t i = 1; i < resampling_arclength.size(); ++i) {
        if (
          resampling_arclength.at(i - 1) <= *distance_to_stop_point &&
          *distance_to_stop_point < resampling_arclength.at(i)) {
          const double dist_to_prev_point =
            std::fabs(*distance_to_stop_point - resampling_arclength.at(i - 1));
          const double dist_to_following_point =
            std::fabs(resampling_arclength.at(i) - *distance_to_stop_point);
          if (dist_to_prev_point < autoware::motion_utils::overlap_threshold) {
            resampling_arclength.at(i - 1) = *distance_to_stop_point;
          } else if (dist_to_following_point < autoware::motion_utils::overlap_threshold) {
            resampling_arclength.at(i) = *distance_to_stop_point;
          } else {
            resampling_arclength.insert(resampling_arclength.begin() + i, *distance_to_stop_point);
          }
          break;
        }
      }
    }
  }

  return resamplePath(
    input_path, resampling_arclength, use_akima_spline_for_xy, use_lerp_for_z,
    use_zero_order_hold_for_v);
}

// every field of every point, bit for bit (doubles compared by their bits: -0.0 vs 0.0 or NaN count as different)
inline bool bits_equal(const double a, const double b)
{
  uint64_t x = 0;
  uint64_t y = 0;
  std::memcpy(&x, &a, sizeof(x));
  std::memcpy(&y, &b, sizeof(y));
  return x == y;
}
inline bool same_path(const PathWithLaneId & a, const PathWithLaneId & b)
{
  if (a.points.size() != b.points.size() || a.header != b.header || a.left_bound != b.left_bound ||
      a.right_bound != b.right_bound) {
    return false;
  }
  for (size_t i = 0; i < a.points.size(); ++i) {
    const auto & p = a.points[i].point;
    const auto & q = b.points[i].point;
    const auto & pp = p.pose;
    const auto & qq = q.pose;
    if (
      !bits_equal(pp.position.x, qq.position.x) || !bits_equal(pp.position.y, qq.position.y) ||
      !bits_equal(pp.position.z, qq.position.z) || !bits_equal(pp.orientation.x, qq.orientation.x) ||
      !bits_equal(pp.orientation.y, qq.orientation.y) ||
      !bits_equal(pp.orientation.z, qq.orientation.z) ||
      !bits_equal(pp.orientation.w, qq.orientation.w) ||
      p.longitudinal_velocity_mps != q.longitudinal_velocity_mps ||
      p.lateral_velocity_mps != q.lateral_velocity_mps || p.heading_rate_rps != q.heading_rate_rps ||
      p.is_final != q.is_final || a.points[i].lane_ids != b.points[i].lane_ids) {
      return false;
    }
  }
  return true;
}

// trjrate-2 v9 (intersection only): resamplePath(input, interval, ..., resample_input_path_stop_point = false) where the part
// that depends only on the input POSES and LANE IDS (the sample positions, segment indices, the resampled poses and lane ids)
// is reused when the previous call of this thread had bit-identical poses, lane ids, interval and flags (the module's
// instances spline the same path one after another while none of them inserts a point); the velocities, heading rates and
// is_final are always taken from this call's input. Failures are never reused (they recompute and print as upstream).
struct GeomReuse
{
  bool valid{false};
  uint64_t interval_bits{0};
  bool akima{false}, lerp_z{true};
  std::vector<geometry_msgs::msg::Pose> key_poses;
  std::vector<std::vector<int64_t>> key_lane_ids;
  std::vector<double> input_arclength, resampling_arclength;
  std::vector<size_t> closest;
  std::vector<geometry_msgs::msg::Pose> interpolated_pose;
  std::vector<std::vector<int64_t>> interpolated_lane_ids;
};
inline bool same_pose_bits(const geometry_msgs::msg::Pose & a, const geometry_msgs::msg::Pose & b)
{
  return bits_equal(a.position.x, b.position.x) && bits_equal(a.position.y, b.position.y) &&
         bits_equal(a.position.z, b.position.z) && bits_equal(a.orientation.x, b.orientation.x) &&
         bits_equal(a.orientation.y, b.orientation.y) && bits_equal(a.orientation.z, b.orientation.z) &&
         bits_equal(a.orientation.w, b.orientation.w);
}
inline PathWithLaneId resamplePathReuse(
  const PathWithLaneId & input_path, const double resample_interval, const bool use_akima_spline_for_xy,
  const bool use_lerp_for_z, const bool use_zero_order_hold_for_v, bool * reused = nullptr)
{
  if (reused) *reused = false;
  thread_local GeomReuse g;
  uint64_t ib = 0;
  std::memcpy(&ib, &resample_interval, sizeof(ib));
  bool hit = g.valid && g.interval_bits == ib && g.akima == use_akima_spline_for_xy && g.lerp_z == use_lerp_for_z &&
             g.key_poses.size() == input_path.points.size();
  for (size_t i = 0; hit && i < input_path.points.size(); ++i) {
    hit = same_pose_bits(g.key_poses[i], input_path.points[i].point.pose) &&
          g.key_lane_ids[i] == input_path.points[i].lane_ids;
  }
  if (!hit) {
    g.valid = false;
    // the geometry, exactly as resamplePath(input, interval, ..., false) computes it
    if (!resample_utils::validate_arguments(input_path.points, resample_interval)) {
      return resamplePath(input_path, resample_interval, use_akima_spline_for_xy, use_lerp_for_z, use_zero_order_hold_for_v, false);
    }
    std::vector<autoware_planning_msgs::msg::PathPoint> transformed_input_path(input_path.points.size());
    for (size_t i = 0; i < input_path.points.size(); ++i) transformed_input_path.at(i) = input_path.points.at(i).point;
    const double input_path_len = autoware::motion_utils::calcArcLength(transformed_input_path);
    std::vector<double> resampling_arclength;
    for (double s = 0.0; s < input_path_len; s += resample_interval) resampling_arclength.push_back(s);
    if (resampling_arclength.empty()) {
      return resamplePath(input_path, resample_interval, use_akima_spline_for_xy, use_lerp_for_z, use_zero_order_hold_for_v, false);
    }
    if (input_path_len - resampling_arclength.back() < autoware::motion_utils::overlap_threshold) {
      resampling_arclength.back() = input_path_len;
    } else {
      resampling_arclength.push_back(input_path_len);
    }
    // the vector overload's lane-id point insertion (as resamplePath above)
    double running_s = 0.0;
    size_t running_i = 0;
    for (size_t i = 0; i < input_path.points.size(); ++i) {
      if (input_path.points.at(i).lane_ids.size() < 2) continue;
      for (; running_i < i; ++running_i) {
        running_s += autoware_utils::calc_distance2d(
          input_path.points.at(running_i), input_path.points.at(running_i + 1));
      }
      const double d = running_s;
      size_t j_hit = 0;
      if (std::is_sorted(resampling_arclength.begin(), resampling_arclength.end())) {
        const size_t u = static_cast<size_t>(
          std::upper_bound(resampling_arclength.begin(), resampling_arclength.end(), d) - resampling_arclength.begin());
        if (u >= 1 && u < resampling_arclength.size()) j_hit = u;
      } else {
        for (size_t j = 1; j < resampling_arclength.size(); ++j) {
          if (resampling_arclength.at(j - 1) <= d && d < resampling_arclength.at(j)) {
            j_hit = j;
            break;
          }
        }
      }
      if (j_hit != 0) {
        const size_t j = j_hit;
        const double dist_to_prev_point = std::fabs(d - resampling_arclength.at(j - 1));
        const double dist_to_following_point = std::fabs(resampling_arclength.at(j) - d);
        if (dist_to_prev_point < autoware::motion_utils::overlap_threshold) {
          resampling_arclength.at(j - 1) = d;
        } else if (dist_to_following_point < autoware::motion_utils::overlap_threshold) {
          resampling_arclength.at(j) = d;
        } else {
          resampling_arclength.insert(resampling_arclength.begin() + j, d);
        }
      }
    }
    if (!resample_utils::validate_arguments(input_path.points, resampling_arclength)) {
      return resamplePath(input_path, resample_interval, use_akima_spline_for_xy, use_lerp_for_z, use_zero_order_hold_for_v, false);
    }
    std::vector<double> input_arclength;
    std::vector<geometry_msgs::msg::Pose> input_pose;
    input_arclength.reserve(input_path.points.size());
    input_pose.reserve(input_path.points.size());
    input_arclength.push_back(0.0);
    input_pose.push_back(input_path.points.front().point.pose);
    for (size_t i = 1; i < input_path.points.size(); ++i) {
      const auto & prev_pt = input_path.points.at(i - 1).point;
      const auto & curr_pt = input_path.points.at(i).point;
      const double ds = autoware_utils::calc_distance2d(prev_pt.pose.position, curr_pt.pose.position);
      input_arclength.push_back(ds + input_arclength.back());
      input_pose.push_back(curr_pt.pose);
    }
    if (input_arclength.back() < resampling_arclength.back()) {
      return resamplePath(input_path, resample_interval, use_akima_spline_for_xy, use_lerp_for_z, use_zero_order_hold_for_v, false);
    }
    auto closest = closest_segment_indices_of(input_arclength, resampling_arclength);
    auto interpolated_pose = autoware::motion_utils::resamplePoseVector(
      input_pose, resampling_arclength, use_akima_spline_for_xy, use_lerp_for_z);
    std::vector<std::vector<int64_t>> interpolated_lane_ids(resampling_arclength.size());
    constexpr double epsilon = 1e-6;
    for (size_t i = 0; i < resampling_arclength.size(); ++i) {
      const size_t seg_idx = std::min(closest.at(i), input_path.points.size() - 2);
      const auto & prev_lane_ids = input_path.points.at(seg_idx).lane_ids;
      const auto & next_lane_ids = input_path.points.at(seg_idx + 1).lane_ids;
      if (std::abs(input_arclength.at(seg_idx) - resampling_arclength.at(i)) <= epsilon) {
        interpolated_lane_ids.at(i).insert(interpolated_lane_ids.at(i).end(), prev_lane_ids.begin(), prev_lane_ids.end());
      } else if (std::abs(input_arclength.at(seg_idx + 1) - resampling_arclength.at(i)) <= epsilon) {
        interpolated_lane_ids.at(i).insert(interpolated_lane_ids.at(i).end(), next_lane_ids.begin(), next_lane_ids.end());
      } else {
        for (const auto target_lane_id : prev_lane_ids) {
          if (std::find(next_lane_ids.begin(), next_lane_ids.end(), target_lane_id) != next_lane_ids.end()) {
            interpolated_lane_ids.at(i).push_back(target_lane_id);
          }
        }
        if (interpolated_lane_ids.at(i).empty()) {
          interpolated_lane_ids.at(i).insert(interpolated_lane_ids.at(i).end(), prev_lane_ids.begin(), prev_lane_ids.end());
        }
      }
    }
    if (interpolated_pose.size() != resampling_arclength.size()) {
      return resamplePath(input_path, resample_interval, use_akima_spline_for_xy, use_lerp_for_z, use_zero_order_hold_for_v, false);
    }
    g.interval_bits = ib;
    g.akima = use_akima_spline_for_xy;
    g.lerp_z = use_lerp_for_z;
    g.key_poses.clear();
    g.key_lane_ids.clear();
    for (const auto & p : input_path.points) {
      g.key_poses.push_back(p.point.pose);
      g.key_lane_ids.push_back(p.lane_ids);
    }
    g.input_arclength = std::move(input_arclength);
    g.resampling_arclength = std::move(resampling_arclength);
    g.closest = std::move(closest);
    g.interpolated_pose = std::move(interpolated_pose);
    g.interpolated_lane_ids = std::move(interpolated_lane_ids);
    g.valid = true;
  } else if (reused) {
    *reused = true;
  }
  // the values of THIS input on the (computed or reused) geometry, as resamplePath does
  std::vector<double> v_lon;
  std::vector<double> v_lat;
  std::vector<double> heading_rate;
  std::vector<bool> is_final;
  v_lon.reserve(input_path.points.size());
  v_lat.reserve(input_path.points.size());
  heading_rate.reserve(input_path.points.size());
  is_final.reserve(input_path.points.size());
  for (const auto & p : input_path.points) {
    v_lon.push_back(p.point.longitudinal_velocity_mps);
    v_lat.push_back(p.point.lateral_velocity_mps);
    heading_rate.push_back(p.point.heading_rate_rps);
    is_final.push_back(p.point.is_final);
  }
  const auto lerp = [&](const auto & input) {
    return autoware::interpolation::lerp(g.input_arclength, input, g.resampling_arclength);
  };
  const auto zoh = [&](const auto & input) {
    return autoware::interpolation::zero_order_hold(g.input_arclength, input, g.closest);
  };
  const auto interpolated_v_lon = use_zero_order_hold_for_v ? zoh(v_lon) : lerp(v_lon);
  const auto interpolated_v_lat = use_zero_order_hold_for_v ? zoh(v_lat) : lerp(v_lat);
  const auto interpolated_heading_rate = lerp(heading_rate);
  const auto interpolated_is_final = zoh(is_final);
  PathWithLaneId resampled_path;
  resampled_path.header = input_path.header;
  resampled_path.left_bound = input_path.left_bound;
  resampled_path.right_bound = input_path.right_bound;
  resampled_path.points.resize(g.interpolated_pose.size());
  for (size_t i = 0; i < resampled_path.points.size(); ++i) {
    autoware_planning_msgs::msg::PathPoint path_point;
    path_point.pose = g.interpolated_pose.at(i);
    path_point.longitudinal_velocity_mps = interpolated_v_lon.at(i);
    path_point.lateral_velocity_mps = interpolated_v_lat.at(i);
    path_point.heading_rate_rps = interpolated_heading_rate.at(i);
    path_point.is_final = interpolated_is_final.at(i);
    resampled_path.points.at(i).point = path_point;
    resampled_path.points.at(i).lane_ids = g.interpolated_lane_ids.at(i);
  }
  return resampled_path;
}
}  // namespace autoware::behavior_velocity_planner::nero_resample

#endif  // AUTOWARE__BEHAVIOR_VELOCITY_INTERSECTION_MODULE__NERO_RESAMPLE_HPP_
