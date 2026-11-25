
// Copyright 2024 TIER IV, Inc.
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

#include <algorithm>
#include <array>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cstdio>
#include <string>
#include <autoware/behavior_path_lane_change_module/utils/calculation.hpp>
#include <autoware/behavior_path_planner_common/utils/utils.hpp>
#include <autoware/motion_utils/trajectory/path_shift.hpp>

#include <boost/geometry/algorithms/buffer.hpp>

namespace autoware::behavior_path_planner::utils::lane_change::calculation
{

rclcpp::Logger get_logger()
{
  constexpr const char * name{"lane_change.utils"};
  static rclcpp::Logger logger = rclcpp::get_logger(name);
  return logger;
}

double calc_dist_from_pose_to_terminal_end(
  const CommonDataPtr & common_data_ptr, const lanelet::ConstLanelets & lanes,
  const Pose & src_pose)
{
  if (lanes.empty()) {
    return 0.0;
  }

  const auto in_goal_route_section =
    common_data_ptr->route_handler_ptr->isInGoalRouteSection(lanes.back());
  if (in_goal_route_section) {
    const auto & goal_pose = common_data_ptr->route_handler_ptr->getGoalPose();
    return utils::getSignedDistance(src_pose, goal_pose, lanes);
  }
  return utils::getDistanceToEndOfLane(src_pose, lanes);
}

double calc_stopping_distance(const LCParamPtr & lc_param_ptr)
{
  // v^2 = u^2 + 2ad
  const auto min_lc_vel = lc_param_ptr->trajectory.min_lane_changing_velocity;
  const auto min_lon_acc = lc_param_ptr->trajectory.min_longitudinal_acc;
  const auto min_back_dist = std::abs((min_lc_vel * min_lc_vel) / (2 * min_lon_acc));

  const auto param_back_dist = lc_param_ptr->backward_length_buffer_for_end_of_lane;
  return std::max(min_back_dist, param_back_dist);
}

double calc_dist_to_last_fit_width(
  const lanelet::ConstLanelets & lanelets, const lanelet::BasicPolygon2d & lanelet_polygon,
  const autoware_utils::LineString2d & line_string, const Pose & src_pose,
  const BehaviorPathPlannerParameters & bpp_param, const double margin)
{
  const double buffer_distance = 0.5 * bpp_param.vehicle_width + margin;
  autoware_utils::MultiPolygon2d center_line_polygon;
  namespace strategy = boost::geometry::strategy::buffer;
  boost::geometry::buffer(
    line_string, center_line_polygon, strategy::distance_symmetric<double>(buffer_distance),
    strategy::side_straight(), strategy::join_miter(), strategy::end_flat(),
    strategy::point_square());

  if (center_line_polygon.empty()) return 0.0;

  std::vector<autoware_utils::Point2d> intersection_points;
  boost::geometry::intersection(lanelet_polygon, center_line_polygon, intersection_points);

  if (intersection_points.empty()) {
    return utils::getDistanceToEndOfLane(src_pose, lanelets);
  }

  Pose pose;
  double distance = std::numeric_limits<double>::max();
  for (const auto & point : intersection_points) {
    pose.position.x = boost::geometry::get<0>(point);
    pose.position.y = boost::geometry::get<1>(point);
    distance = std::min(distance, utils::getSignedDistance(src_pose, pose, lanelets));
  }

  return std::max(distance - (bpp_param.base_link2front + margin), 0.0);
}

double calc_dist_to_last_fit_width(
  const CommonDataPtr & common_data_ptr, const PathWithLaneId & path, const double margin)
{
  const auto & current_lanes = common_data_ptr->lanes_ptr->current;
  const auto & current_lanes_polygon = common_data_ptr->lanes_polygon_ptr->current;
  const auto & bpp_param = *common_data_ptr->bpp_param_ptr;

  autoware_utils::LineString2d line_string;
  line_string.reserve(path.points.size() - 1);
  std::for_each(path.points.begin() + 1, path.points.end(), [&line_string](const auto & point) {
    const auto & position = point.point.pose.position;
    boost::geometry::append(line_string, autoware_utils::Point2d(position.x, position.y));
  });

  const auto & src_pose = path.points.front().point.pose;
  return calc_dist_to_last_fit_width(
    current_lanes, current_lanes_polygon, line_string, src_pose, bpp_param, margin);
}

double calc_dist_to_last_fit_width(
  const lanelet::ConstLanelets & lanelets, const Pose & src_pose,
  const BehaviorPathPlannerParameters & bpp_param, const double margin)
{
  if (lanelets.empty()) return 0.0;

  const auto lane_polygon = lanelets.back().polygon2d().basicPolygon();
  const auto center_line = lanelet::utils::generateFineCenterline(lanelets.back(), 1.0);

  if (center_line.size() <= 1) return 0.0;

  autoware_utils::LineString2d line_string;
  line_string.reserve(center_line.size() - 1);
  std::for_each(center_line.begin() + 1, center_line.end(), [&line_string](const auto & point) {
    boost::geometry::append(line_string, autoware_utils::Point2d(point.x(), point.y()));
  });

  return calc_dist_to_last_fit_width(
    lanelets, lane_polygon, line_string, src_pose, bpp_param, margin);
}

double calc_maximum_prepare_length(const CommonDataPtr & common_data_ptr)
{
  const auto max_prepare_duration = common_data_ptr->lc_param_ptr->trajectory.max_prepare_duration;
  const auto ego_max_speed = common_data_ptr->bpp_param_ptr->max_vel;

  return max_prepare_duration * ego_max_speed;
}

double calc_ego_dist_to_lanes_start(
  const CommonDataPtr & common_data_ptr, const lanelet::ConstLanelets & current_lanes,
  const lanelet::ConstLanelets & target_lanes)
{
  const auto & route_handler_ptr = common_data_ptr->route_handler_ptr;

  if (!route_handler_ptr || target_lanes.empty() || current_lanes.empty()) {
    return std::numeric_limits<double>::max();
  }

  const auto & target_bound =
    common_data_ptr->direction == autoware::route_handler::Direction::RIGHT
      ? target_lanes.front().leftBound()
      : target_lanes.front().rightBound();

  if (target_bound.empty()) {
    return std::numeric_limits<double>::max();
  }

  const auto & path = common_data_ptr->current_lanes_path;

  if (path.points.empty()) {
    return std::numeric_limits<double>::max();
  }

  const auto target_front_pt = lanelet::utils::conversion::toGeomMsgPt(target_bound.front());
  const auto ego_position = common_data_ptr->get_ego_pose().position;

  return motion_utils::calcSignedArcLength(path.points, ego_position, target_front_pt);
}

double calc_minimum_acceleration(
  const LaneChangeParameters & lane_change_parameters, const double current_velocity,
  const double min_acc_threshold, const double prepare_duration)
{
  if (prepare_duration < eps) return -std::abs(min_acc_threshold);
  const double min_lc_velocity = lane_change_parameters.trajectory.min_lane_changing_velocity;
  const double acc = (min_lc_velocity - current_velocity) / prepare_duration;
  return std::clamp(acc, -std::abs(min_acc_threshold), -eps);
}

double calc_maximum_acceleration(
  const double prepare_duration, const double current_velocity, const double current_max_velocity,
  const double max_acc_threshold)
{
  if (prepare_duration < eps) return max_acc_threshold;
  const double acc = (current_max_velocity - current_velocity) / prepare_duration;
  return std::clamp(acc, 0.0, max_acc_threshold);
}

std::vector<double> calc_min_lane_change_lengths(
  const LCParamPtr & lc_param_ptr, const std::vector<double> & shift_intervals)
{
  if (shift_intervals.empty()) {
    return {};
  }

  const auto min_vel = lc_param_ptr->trajectory.min_lane_changing_velocity;
  const auto min_max_lat_acc = lc_param_ptr->trajectory.lat_acc_map.find(min_vel);
  const auto max_lat_acc = std::get<1>(min_max_lat_acc);
  const auto lat_jerk = lc_param_ptr->trajectory.lateral_jerk;

  std::vector<double> min_lc_lengths{};
  min_lc_lengths.reserve(shift_intervals.size());

  const auto min_lc_length = [&](const auto shift_interval) {
    const auto t =
      autoware::motion_utils::calc_shift_time_from_jerk(shift_interval, lat_jerk, max_lat_acc);
    return min_vel * t;
  };

  std::transform(
    shift_intervals.cbegin(), shift_intervals.cend(), std::back_inserter(min_lc_lengths),
    min_lc_length);

  return min_lc_lengths;
}

std::vector<double> calc_max_lane_change_lengths(
  const CommonDataPtr & common_data_ptr, const std::vector<double> & shift_intervals)
{
  if (shift_intervals.empty()) {
    return {};
  }

  const auto & params = common_data_ptr->lc_param_ptr->trajectory;
  const auto lat_jerk = params.lateral_jerk;
  const auto t_prepare = params.max_prepare_duration;
  const auto current_velocity = common_data_ptr->get_ego_speed();
  const auto path_velocity = common_data_ptr->transient_data.current_path_velocity;

  const auto max_acc = calc_maximum_acceleration(
    t_prepare, current_velocity, path_velocity, params.max_longitudinal_acc);

  // TODO(Quda, Azu): should probably limit upper bound of velocity as well, but
  // disabled due failing evaluation tests.
  // const auto limit_vel = [&](const auto vel) {
  //   const auto max_global_vel = common_data_ptr->bpp_param_ptr->max_vel;
  //   return std::clamp(vel, params.min_lane_changing_velocity, max_global_vel);
  // };

  auto vel = std::max(common_data_ptr->get_ego_speed(), params.min_lane_changing_velocity);

  std::vector<double> max_lc_lengths{};

  const auto max_lc_length = [&](const auto shift_interval) {
    // prepare section
    const auto prepare_length = vel * t_prepare + 0.5 * max_acc * t_prepare * t_prepare;
    vel = vel + max_acc * t_prepare;

    // lane changing section
    const auto [min_lat_acc, max_lat_acc] = params.lat_acc_map.find(vel);
    const auto t_lane_changing =
      autoware::motion_utils::calc_shift_time_from_jerk(shift_interval, lat_jerk, max_lat_acc);
    const auto lane_changing_length =
      vel * t_lane_changing + 0.5 * max_acc * t_lane_changing * t_lane_changing;

    vel = vel + max_acc * t_lane_changing;
    return prepare_length + lane_changing_length;
  };

  std::transform(
    shift_intervals.cbegin(), shift_intervals.cend(), std::back_inserter(max_lc_lengths),
    max_lc_length);
  return max_lc_lengths;
}

double calc_distance_buffer(const LCParamPtr & lc_param_ptr, const std::vector<double> & lc_lengths)
{
  if (lc_lengths.empty()) {
    return std::numeric_limits<double>::max();
  }

  const auto finish_judge_buffer = lc_param_ptr->lane_change_finish_judge_buffer;
  const auto backward_buffer = calc_stopping_distance(lc_param_ptr);
  const auto lengths_sum = std::accumulate(lc_lengths.begin(), lc_lengths.end(), 0.0);
  const auto num_of_lane_changes = static_cast<double>(lc_lengths.size());
  return lengths_sum + (num_of_lane_changes * finish_judge_buffer) +
         ((num_of_lane_changes - 1.0) * backward_buffer);
}

double nero_change_window(const CommonDataPtr & common_data_ptr, const lanelet::ConstLanelets & lanes)
{
  const auto & target_neighbor = common_data_ptr->lanes_ptr->target_neighbor;
  const auto & first =
    target_neighbor.empty() ? common_data_ptr->lanes_ptr->target.front() : target_neighbor.front();
  if (first.centerline2d().size() < 2) {
    return std::numeric_limits<double>::max();
  }
  return calc_dist_from_pose_to_terminal_end(
    common_data_ptr, lanes, utils::to_geom_msg_pose(first.centerline2d().front(), first));
}

std::vector<double> calc_shift_intervals(
  const CommonDataPtr & common_data_ptr, const lanelet::ConstLanelets & lanes)
{
  if (!common_data_ptr || !common_data_ptr->is_data_available() || lanes.empty()) {
    return {};
  }

  const auto & route_handler_ptr = common_data_ptr->route_handler_ptr;
  const auto direction = common_data_ptr->direction;

  auto intervals = route_handler_ptr->getLateralIntervalsToPreferredLane(lanes.back(), direction);
  // NERO_ABLC_LANES=1 (mainlane 2026-11-11): avoidance_by_lane_change changes from the preferred lane to the lane beside it,
  // so the intervals "to the preferred lane" are empty, every minimum length / distance buffer became DBL_MAX and its
  // request ("object farther than the minimum lane change") could never pass. One change to the lane beside in `direction`
  // (the same centreline-to-centreline measure upstream uses per change). Unset = upstream.
  static const bool nero_ablc_lanes = [] {
    const char * v = std::getenv("NERO_ABLC_LANES");
    return v != nullptr && std::atoi(v) > 0;
  }();
  if (
    nero_ablc_lanes && intervals.empty() && common_data_ptr->lc_type == LaneChangeModuleType::AVOIDANCE_BY_LANE_CHANGE &&
    (direction == Direction::LEFT || direction == Direction::RIGHT)) {
    // the first current lanelet with a lane-changeable neighbour on that side (the current lanes can run on for many
    // lanelets past the object, to one with no neighbour: field logsim w1108_1747, last lanelet 42922 -> DBL_MAX)
    const auto & graph = route_handler_ptr->getRoutingGraphPtr();
    for (const auto & lane : lanes) {
      const auto beside = direction == Direction::LEFT ? graph->left(lane) : graph->right(lane);
      if (beside && !lane.centerline().empty() && !beside->centerline().empty()) {
        const double d = lanelet::geometry::distance2d(
          lanelet::utils::to2D(lane.centerline().front()), lanelet::utils::to2D(beside->centerline().front()));
        intervals.push_back(direction == Direction::LEFT ? d : -d);
        break;
      }
    }
  }
  // NERO_LC_ROUTE_GATE=1 (routelc job 2026-10-26, QUEUE KRS): upstream counts every lane change to the preferred lane
  // as due before the end of the current lanes. Where the first target lane runs on in the route past that end (the
  // KRS road: 91174 -> 93732 -> 93726 -> 47326 -> 42025 beside the preferred 93787 -> 93781 -> 47324 -> 42027), only
  // the first change is due there: the others are due before the end of the target lane, and the module that makes
  // them from the target lane has its own terminal. Then only the first interval counts (gate, waiting stop, the
  // next-change stop and the blinker buffer of one change). Kept as upstream when the target lane ends too soon for
  // the remaining changes (+ the stopping distance between them). Unset = upstream.
  static const bool route_gate = [] {
    const char * v = std::getenv("NERO_LC_ROUTE_GATE");
    return v != nullptr && std::atoi(v) > 0;
  }();
  if (
    !route_gate || intervals.size() < 2 || common_data_ptr->lc_type != LaneChangeModuleType::NORMAL ||
    !common_data_ptr->is_lanes_available() || common_data_ptr->lanes_ptr->target.empty()) {
    return intervals;
  }
  const auto & target_lanes = common_data_ptr->lanes_ptr->target;
  const auto & ego_pose = common_data_ptr->get_ego_pose();
  const auto rest_intervals =
    route_handler_ptr->getLateralIntervalsToPreferredLane(target_lanes.back(), direction);
  const auto rest_lengths = calc_min_lane_change_lengths(common_data_ptr->lc_param_ptr, rest_intervals);
  const double rest_buffer =
    rest_lengths.empty() ? 0.0 : calc_distance_buffer(common_data_ptr->lc_param_ptr, rest_lengths);
  // how far the target lane runs on past the end of the current lanes (measured from that end point, so also right
  // while ego is still behind the target lane's start)
  const double dist_to_current_end = calc_dist_from_pose_to_terminal_end(common_data_ptr, lanes, ego_pose);
  const auto & last = lanes.back();
  const double dist_to_target_end =
    last.centerline2d().size() < 2
      ? 0.0
      : calc_dist_from_pose_to_terminal_end(
          common_data_ptr, target_lanes, utils::to_geom_msg_pose(last.centerline2d().back(), last));
  const double need = rest_buffer + calc_stopping_distance(common_data_ptr->lc_param_ptr);
  const bool first_only = dist_to_target_end >= need;
  static double last_log = 0.0;
  const double now = rclcpp::Clock(RCL_STEADY_TIME).now().seconds();
  if (std::getenv("NERO_LC_DEBUG") != nullptr && now - last_log > 1.0) {
    last_log = now;
    RCLCPP_INFO(
      get_logger(),
      "[NERO_LC_ROUTE_GATE] %zu changes to the preferred lane from %ld; target %ld..%ld runs on %.1f m past the end of "
      "the current lanes (%.1f m ahead), rest %zu (%.1f m) -> %s",
      intervals.size(), lanes.back().id(), target_lanes.front().id(), target_lanes.back().id(), dist_to_target_end,
      dist_to_current_end, rest_intervals.size(), rest_buffer, first_only ? "first change only" : "all (upstream)");
  }
  if (first_only) {
    intervals.resize(1);
  }
  return intervals;
}

std::pair<MinMaxValue, MinMaxValue> calc_lc_length_and_dist_buffer(
  const CommonDataPtr & common_data_ptr, const lanelet::ConstLanelets & lanes)
{
  if (!common_data_ptr || !common_data_ptr->is_data_available() || lanes.empty()) {
    return {};
  }
  const auto shift_intervals = calculation::calc_shift_intervals(common_data_ptr, lanes);
  const auto min_lc_lengths =
    calculation::calc_min_lane_change_lengths(common_data_ptr->lc_param_ptr, shift_intervals);
  const auto min_lc_length =
    !min_lc_lengths.empty() ? min_lc_lengths.front() : std::numeric_limits<double>::max();
  auto min_dist_buffer =
    calculation::calc_distance_buffer(common_data_ptr->lc_param_ptr, min_lc_lengths);
  // NERO_LC_SHORT_WINDOW=1 (routelc job 2026-10-26, QUEUE LCE / 93699): with two or more changes due in a window (from
  // where the target lane begins to the end of the current lanes) shorter than this buffer + the stop offset
  // (backward_length_buffer_for_end_of_lane, 10 m on the car), the car can never plan a change there (93699: window
  // 28.7 m < 32.4 + prepare; the robot stands at the lane end for good). There the room between the changes is the
  // minimum stopping distance (v_min^2 / 2|a_min|, 2.0 m) instead of the 10 m buffer: 24.4 m for two changes. Single
  // changes (the U-turn merge the 10 m was chosen for) and windows that fit are unchanged. Unset = the car.
  static const bool short_window = [] {
    const char * v = std::getenv("NERO_LC_SHORT_WINDOW");
    return v != nullptr && std::atoi(v) > 0;
  }();
  if (
    short_window && min_lc_lengths.size() >= 2 && common_data_ptr->lc_type == LaneChangeModuleType::NORMAL &&
    common_data_ptr->is_lanes_available() && !common_data_ptr->lanes_ptr->target.empty()) {
    const auto & lc_param = *common_data_ptr->lc_param_ptr;
    const double window = nero_change_window(common_data_ptr, lanes);
    const double stop_offset = calc_stopping_distance(common_data_ptr->lc_param_ptr);
    if (window < min_dist_buffer + stop_offset) {
      const double v_min = lc_param.trajectory.min_lane_changing_velocity;
      const double min_stop = std::abs(v_min * v_min / (2.0 * lc_param.trajectory.min_longitudinal_acc));
      const double shorter =
        min_dist_buffer - (static_cast<double>(min_lc_lengths.size()) - 1.0) * (stop_offset - min_stop);
      static double last_log = 0.0;
      const double now = rclcpp::Clock(RCL_STEADY_TIME).now().seconds();
      if (std::getenv("NERO_LC_DEBUG") != nullptr && now - last_log > 1.0) {
        last_log = now;
        RCLCPP_INFO(
          get_logger(), "[NERO_LC_SHORT_WINDOW] %zu changes in a %.1f m window < %.1f + %.1f m: buffer %.1f -> %.1f m",
          min_lc_lengths.size(), window, min_dist_buffer, stop_offset, min_dist_buffer, shorter);
      }
      min_dist_buffer = shorter;
    }
  }

  const auto max_lc_lengths =
    calculation::calc_max_lane_change_lengths(common_data_ptr, shift_intervals);
  const auto max_lc_length =
    !max_lc_lengths.empty() ? max_lc_lengths.front() : std::numeric_limits<double>::max();
  const auto max_dist_buffer =
    calculation::calc_distance_buffer(common_data_ptr->lc_param_ptr, max_lc_lengths);

  return {{min_lc_length, max_lc_length}, {min_dist_buffer, max_dist_buffer}};
}

// NERO_LC_GIVEUP_NOFIT (stops job 2026-11-25, QUEUE NFUTURN-1123): see the header. The same intervals (NERO_LC_ROUTE_GATE
// included: a target lane that runs on keeps one change due) and the same shortest buffer as NERO_LC_SHORT_WINDOW.
bool nero_changes_cannot_fit(
  const CommonDataPtr & common_data_ptr, const lanelet::ConstLanelets & lanes, double & window, double & need,
  size_t & n_changes)
{
  n_changes = 0;
  if (
    !common_data_ptr || !common_data_ptr->is_data_available() || lanes.empty() ||
    common_data_ptr->lc_type != LaneChangeModuleType::NORMAL || !common_data_ptr->is_lanes_available() ||
    common_data_ptr->lanes_ptr->target.empty()) {
    return false;
  }
  const auto shift_intervals = calculation::calc_shift_intervals(common_data_ptr, lanes);
  const auto min_lc_lengths =
    calculation::calc_min_lane_change_lengths(common_data_ptr->lc_param_ptr, shift_intervals);
  n_changes = min_lc_lengths.size();
  if (n_changes < 2) {
    return false;
  }
  const auto & lc_param = *common_data_ptr->lc_param_ptr;
  const double buffer = calculation::calc_distance_buffer(common_data_ptr->lc_param_ptr, min_lc_lengths);
  const double stop_offset = calc_stopping_distance(common_data_ptr->lc_param_ptr);
  const double v_min = lc_param.trajectory.min_lane_changing_velocity;
  const double min_stop = std::abs(v_min * v_min / (2.0 * lc_param.trajectory.min_longitudinal_acc));
  need = buffer - (static_cast<double>(n_changes) - 1.0) * std::max(0.0, stop_offset - min_stop);
  window = nero_change_window(common_data_ptr, lanes);
  return window < need;
}

double calc_phase_length(
  const double velocity, const double maximum_velocity, const double acceleration,
  const double duration)
{
  const auto length_with_acceleration =
    velocity * duration + 0.5 * acceleration * std::pow(duration, 2);
  const auto length_with_max_velocity = maximum_velocity * duration;
  return std::min(length_with_acceleration, length_with_max_velocity);
}

std::pair<double, double> calc_min_max_acceleration(
  const CommonDataPtr & common_data_ptr, const double max_path_velocity,
  const double prepare_duration)
{
  const auto & lc_params = *common_data_ptr->lc_param_ptr;
  const auto & bpp_params = *common_data_ptr->bpp_param_ptr;
  const auto current_ego_velocity = common_data_ptr->get_ego_speed();

  const auto min_accel_threshold =
    std::max(bpp_params.min_acc, lc_params.trajectory.min_longitudinal_acc);
  const auto max_accel_threshold =
    std::min(bpp_params.max_acc, lc_params.trajectory.max_longitudinal_acc);

  // calculate minimum and maximum acceleration
  const auto min_acc = calc_minimum_acceleration(
    lc_params, current_ego_velocity, min_accel_threshold, prepare_duration);
  const auto max_acc = calc_maximum_acceleration(
    prepare_duration, current_ego_velocity, max_path_velocity, max_accel_threshold);

  return {min_acc, max_acc};
}

std::vector<double> calc_acceleration_values(
  const double min_accel, const double max_accel, const double sampling_num)
{
  if (min_accel > max_accel) return {};

  if (max_accel - min_accel < eps) {
    return {min_accel};
  }

  const auto resolution = std::abs(max_accel - min_accel) / sampling_num;

  std::vector<double> sampled_values{min_accel};
  for (double accel = min_accel + resolution; accel < max_accel + eps; accel += resolution) {
    // check whether if we need to add 0.0
    if (sampled_values.back() < -eps && accel > eps) {
      sampled_values.push_back(0.0);
    }

    sampled_values.push_back(accel);
  }
  std::reverse(sampled_values.begin(), sampled_values.end());

  return sampled_values;
}

std::vector<double> calc_lon_acceleration_samples(
  const CommonDataPtr & common_data_ptr, const double max_path_velocity,
  const double prepare_duration)
{
  const auto & transient_data = common_data_ptr->transient_data;
  const auto & current_pose = common_data_ptr->get_ego_pose();
  const auto & target_lanes = common_data_ptr->lanes_ptr->target;
  const auto goal_pose = common_data_ptr->route_handler_ptr->getGoalPose();
  const auto sampling_num = common_data_ptr->lc_param_ptr->trajectory.lon_acc_sampling_num;

  const auto min_max_accel =
    calc_min_max_acceleration(common_data_ptr, max_path_velocity, prepare_duration);
  const auto & min_accel = min_max_accel.first;
  const auto & max_accel = min_max_accel.second;

  const auto is_sampling_required = std::invoke([&]() -> bool {
    if (max_accel < 0.0 || transient_data.is_ego_stuck) return true;

    const auto max_dist_buffer = transient_data.current_dist_buffer.max;
    if (max_dist_buffer > transient_data.dist_to_terminal_end) return true;

    const auto dist_to_target_lane_end =
      common_data_ptr->lanes_ptr->target_lane_in_goal_section
        ? utils::getSignedDistance(current_pose, goal_pose, target_lanes)
        : utils::getDistanceToEndOfLane(current_pose, target_lanes);

    return max_dist_buffer >= dist_to_target_lane_end;
  });

  if (is_sampling_required) {
    return calc_acceleration_values(min_accel, max_accel, sampling_num);
  }

  return {max_accel};
}

double calc_lane_changing_acceleration(
  const CommonDataPtr & common_data_ptr, const double initial_lane_changing_velocity,
  const double max_path_velocity, const double lane_changing_time,
  const double prepare_longitudinal_acc)
{
  if (prepare_longitudinal_acc <= 0.0) {
    const auto & params = common_data_ptr->lc_param_ptr->trajectory;
    const auto lane_changing_acc =
      common_data_ptr->transient_data.is_ego_near_current_terminal_start
        ? prepare_longitudinal_acc * params.lane_changing_decel_factor
        : 0.0;
    return lane_changing_acc;
  }

  return std::clamp(
    (max_path_velocity - initial_lane_changing_velocity) / lane_changing_time, 0.0,
    prepare_longitudinal_acc);
}

double calc_actual_prepare_duration(
  const CommonDataPtr & common_data_ptr, const double current_velocity,
  const double active_signal_duration)
{
  const auto & params = common_data_ptr->lc_param_ptr->trajectory;
  const auto min_lc_velocity = params.min_lane_changing_velocity;

  // need to ensure min prep duration is sufficient to reach minimum lane changing velocity
  const auto min_prepare_duration = std::invoke([&]() -> double {
    if (current_velocity >= min_lc_velocity) {
      return params.min_prepare_duration;
    }
    const auto max_acc =
      std::min(common_data_ptr->bpp_param_ptr->max_acc, params.max_longitudinal_acc);
    if (max_acc < eps) {
      return params.max_prepare_duration;
    }
    return std::max((min_lc_velocity - current_velocity) / max_acc, params.min_prepare_duration);
  });

  return std::max(params.max_prepare_duration - active_signal_duration, min_prepare_duration);
}

bool & nero_prep_long()
{
  static bool f = false;
  return f;
}

std::vector<double> calc_prepare_durations(const CommonDataPtr & common_data_ptr)
{
  const auto & lc_param_ptr = common_data_ptr->lc_param_ptr;
  // NERO_LC_GAP_AHEAD_PREP=<min s>,<step s>[,<commit s>] (mainlane-7 2026-11-11, Ahmed ~20:2x "the lane changes are too stiff you should
  // plan it before the car is beside so you are ready to execute when the car is beside you"): the prepare durations min, min + step,
  // .. max_prepare_duration everywhere (upstream: ONE duration max(max_prepare - signal time, min_prepare, (min_lc_vel - v) / max_lon_acc)
  // away from the lane end = 4.7 s at 0.6 m/s and 6.7 s from a stand with the car's 3.0 / 1.0 / 2.0 m/s / 0.3 m/s2; the signal time is
  // an indicator this robot does not have), tried shortest first (get_prepare_metrics): the change starts as soon as a candidate is
  // safe instead of driving 4.7-6.7 s straight first (sim soa14 93795: approved 7.7 s, first sideways 13.0 s with the target lane free).
  // The safety check judges each candidate with the robot's own launch (NERO_LC_PRED_LAUNCH_ACC). Unset = upstream.
  {
    static const std::array<double, 3> gp = [] {
      std::array<double, 3> r{-1.0, 0.5, -1.0};
      if (const char * e = std::getenv("NERO_LC_GAP_AHEAD_PREP")) {
        std::sscanf(e, "%lf,%lf,%lf", &r[0], &r[1], &r[2]);
      }
      r[1] = std::max(0.1, r[1]);
      return r;
    }();
    // v7 (planner cost: v6 ran 6 durations everywhere, behavior_path_planner p95 6 -> 36 ms, max 109 ms at no load): the stock durations
    // + the two shortest (min, min + step) only, unique, ascending; near the lane end the stock 3.0 .. 0 already holds them
    if (gp[0] >= 0.0) {
      std::vector<double> d;
      const auto threshold0 = common_data_ptr->bpp_param_ptr->base_link2front + lc_param_ptr->min_length_for_turn_signal_activation;
      if (common_data_ptr->transient_data.dist_to_terminal_start >= threshold0) {
        d.push_back(common_data_ptr->transient_data.lane_change_prepare_duration);
      } else {
        for (double t = lc_param_ptr->trajectory.max_prepare_duration; t >= 0.0; t -= 0.5) d.push_back(t);
      }
      d.push_back(gp[0]);
      d.push_back(gp[0] + gp[1]);
      // v8: while a faster target-lane car is coming from behind (nero_prep_long, set by the module each cycle) also the long ones,
      // so a change that starts as that car goes by can be planned while it is still coming (v7 without them never planned K1103's
      // gap behind B: the stock duration had shrunk to 1 s)
      if (nero_prep_long()) {
        d.push_back(lc_param_ptr->trajectory.max_prepare_duration - 1.0);
        d.push_back(lc_param_ptr->trajectory.max_prepare_duration);
      }
      std::sort(d.begin(), d.end());
      d.erase(std::unique(d.begin(), d.end(), [](double a, double b) { return std::abs(a - b) < 1e-3; }), d.end());
      return d;
    }
  }
  const auto threshold = common_data_ptr->bpp_param_ptr->base_link2front +
                         lc_param_ptr->min_length_for_turn_signal_activation;

  // TODO(Azu) this check seems to cause scenario failures.
  if (common_data_ptr->transient_data.dist_to_terminal_start >= threshold) {
    return {common_data_ptr->transient_data.lane_change_prepare_duration};
  }

  const auto max_prepare_duration = lc_param_ptr->trajectory.max_prepare_duration;
  std::vector<double> prepare_durations;
  constexpr double step = 0.5;

  for (double duration = max_prepare_duration; duration >= 0.0; duration -= step) {
    prepare_durations.push_back(duration);
  }

  return prepare_durations;
}

std::vector<PhaseMetrics> calc_prepare_phase_metrics(
  const CommonDataPtr & common_data_ptr, const double current_velocity,
  const double max_path_velocity, const double min_length_threshold,
  const double max_length_threshold)
{
  const auto & min_lc_vel = common_data_ptr->lc_param_ptr->trajectory.min_lane_changing_velocity;
  const auto & max_vel = common_data_ptr->bpp_param_ptr->max_vel;

  std::vector<PhaseMetrics> metrics;

  auto is_skip = [&](const double prepare_length) {
    if (prepare_length > max_length_threshold || prepare_length < min_length_threshold) {
      RCLCPP_DEBUG(
        get_logger(),
        "Skip: prepare length out of expected range. length: %.5f, threshold min: %.5f, max: %.5f",
        prepare_length, min_length_threshold, max_length_threshold);
      return true;
    }
    return false;
  };

  const auto prepare_durations = calc_prepare_durations(common_data_ptr);

  for (const auto & prepare_duration : prepare_durations) {
    const auto lon_accel_samples =
      calc_lon_acceleration_samples(common_data_ptr, max_path_velocity, prepare_duration);
    for (const auto & lon_accel : lon_accel_samples) {
      const auto prepare_velocity =
        std::clamp(current_velocity + lon_accel * prepare_duration, min_lc_vel, max_vel);

      // compute actual longitudinal acceleration
      const double prepare_accel = (prepare_duration < 1e-3)
                                     ? 0.0
                                     : ((prepare_velocity - current_velocity) / prepare_duration);

      const auto prepare_length =
        calc_phase_length(current_velocity, max_vel, prepare_accel, prepare_duration);

      if (is_skip(prepare_length)) continue;

      metrics.emplace_back(
        prepare_duration, prepare_length, prepare_velocity, lon_accel, prepare_accel, 0.0);
    }
  }

  return metrics;
}

// NERO_LC_GRADUAL=<lateral jerk m/s3>,<lat acc min m/s2>,<lat acc max m/s2> (izlanechange-3, 2026-11-23; Ahmed 13:40 "make the
// lane changes more gradual"). The same values as lane change parameters (lateral_jerk 0.5, lateral_acceleration 0.3..0.6) made
// every change gentler (20 km/h: 5 s / 28 m -> 6 s / 34 m, peak a_lat 0.6 -> 0.44 m/s2, jerk 1.0 -> 0.5) but they also set the
// minimum change length, the distance buffers and the latest start: on e16 the right change after the left turn of the
// three-lane road then had no candidate at all in its short lane (dist_to_terminal_start 13.6 m < target lane start 16.4 m;
// run 1123-155408, a 29 s stand). Here the parameter file keeps the car's values (all lengths, buffers and the terminal path
// as on the car) and each prepare length gets two gentle candidates first (min and max of these values, this jerk), then
// the car's samples: the first valid and safe candidate is taken, so a change is gentle where it fits and has the car's
// shape where the room is short or the gentle one is not safe (a shorter exposure). Near the lane end (ego near the terminal
// start) only the car's samples (no extra candidates where the search is already the most expensive). Unset = the car.
namespace
{
struct NeroGradual
{
  bool on{false};
  double jerk{0.5};
  double a_min{0.3};
  double a_max{0.6};
};
const NeroGradual & nero_gradual_param()
{
  static const NeroGradual g = [] {
    NeroGradual r;
    if (const char * v = std::getenv("NERO_LC_GRADUAL"); v && *v && std::string(v) != "0") {
      double j = r.jerk, a0 = r.a_min, a1 = r.a_max;
      std::sscanf(v, "%lf,%lf,%lf", &j, &a0, &a1);
      r.on = j > 0.05 && a0 > 0.05;
      r.jerk = j;
      r.a_min = a0;
      r.a_max = std::max(a0, a1);
    }
    return r;
  }();
  return g;
}
}  // namespace

std::vector<PhaseMetrics> calc_shift_phase_metrics(
  const CommonDataPtr & common_data_ptr, const double shift_length, const double initial_velocity,
  const double max_path_velocity, const double lon_accel, const double max_length_threshold,
  const bool nero_gradual)
{
  const auto & min_lc_vel = common_data_ptr->lc_param_ptr->trajectory.min_lane_changing_velocity;
  const auto & max_vel = common_data_ptr->bpp_param_ptr->max_vel;

  // get lateral acceleration range
  const auto [min_lateral_acc, max_lateral_acc] =
    common_data_ptr->lc_param_ptr->trajectory.lat_acc_map.find(initial_velocity);
  const auto lateral_acc_resolution =
    std::abs(max_lateral_acc - min_lateral_acc) /
    common_data_ptr->lc_param_ptr->trajectory.lat_acc_sampling_num;

  std::vector<PhaseMetrics> metrics;

  auto is_skip = [&](const double lane_changing_length) {
    if (lane_changing_length > max_length_threshold) {
      RCLCPP_DEBUG(
        get_logger(),
        "Skip: lane changing length exceeds maximum threshold. length: %.5f, threshold: %.5f",
        lane_changing_length, max_length_threshold);
      return true;
    }
    return false;
  };

  // NERO_LC_GRADUAL: the gentle candidates first
  if (const auto & g = nero_gradual_param();
      g.on && nero_gradual && !common_data_ptr->transient_data.is_ego_near_current_terminal_start) {
    // NERO_LC_GRADUAL_HEAD=<deg>[,<deg>...] (izlanechange-3 19:4x; Ahmed 19:35 "when i say gradual i mean the sharpness of the turn
    // when changing the lane"): candidates shaped by DISTANCE so the robot's largest heading angle to the lane while crossing stays
    // under each angle, shallowest first: duration = max(the gentle jerk / acc time, 1.75 |shift| / (v tan(angle))) (1.75 = peak /
    // mean lateral speed of the shifted path, measured: 6.0 s / 33 m at 5.6 m/s gave 8.5-8.9 deg). At 20 km/h the car turns in at
    // 10.7-11.3 deg, the jerk/acc gentle shape at 8.5-8.9 deg; at 2-3 m/s after a turn 13-23 deg (time-shaped: 10-15 m long). A
    // candidate longer than the room is skipped as any other (the next, sharper one, then the car's shape).
    static const std::vector<double> heads = [] {
      std::vector<double> h;
      if (const char * v = std::getenv("NERO_LC_GRADUAL_HEAD"); v && *v) {
        std::string str(v);
        size_t pos = 0;
        while (pos < str.size()) {
          const size_t c = str.find(',', pos);
          const double d = std::atof(str.substr(pos, c == std::string::npos ? std::string::npos : c - pos).c_str());
          if (d > 0.5 && d < 45.0) h.push_back(d);
          if (c == std::string::npos) break;
          pos = c + 1;
        }
      }
      return h;
    }();
    const double t_jerk = autoware::motion_utils::calc_shift_time_from_jerk(shift_length, g.jerk, g.a_max);
    // no two gentle candidates of (nearly) the same duration: the same path, evaluated twice (cost)
    const auto dup = [&metrics](const double t) {
      return std::any_of(metrics.begin(), metrics.end(), [t](const auto & m) { return std::abs(m.duration - t) < 0.3; });
    };
    for (const double deg : heads) {
      const double v_h = std::max(initial_velocity, 1.0);
      const double lane_changing_duration =
        std::max(t_jerk, 1.75 * std::abs(shift_length) / (v_h * std::tan(deg * M_PI / 180.0)));
      if (dup(lane_changing_duration)) continue;
      const double lane_changing_accel = calc_lane_changing_acceleration(
        common_data_ptr, initial_velocity, max_path_velocity, lane_changing_duration, lon_accel);
      const auto lane_changing_length = calculation::calc_phase_length(
        initial_velocity, max_vel, lane_changing_accel, lane_changing_duration);
      if (is_skip(lane_changing_length)) continue;
      const auto lane_changing_velocity = std::clamp(
        initial_velocity + lane_changing_accel * lane_changing_duration, min_lc_vel, max_vel);
      metrics.emplace_back(
        lane_changing_duration, lane_changing_length, lane_changing_velocity, lon_accel,
        lane_changing_accel, g.a_min);
    }
    for (const double lat_acc : {g.a_min, g.a_max}) {
      if (lat_acc == g.a_max && std::abs(g.a_max - g.a_min) < 1e-6) continue;
      const auto lane_changing_duration =
        autoware::motion_utils::calc_shift_time_from_jerk(shift_length, g.jerk, lat_acc);
      if (dup(lane_changing_duration)) continue;
      const double lane_changing_accel = calc_lane_changing_acceleration(
        common_data_ptr, initial_velocity, max_path_velocity, lane_changing_duration, lon_accel);
      const auto lane_changing_length = calculation::calc_phase_length(
        initial_velocity, max_vel, lane_changing_accel, lane_changing_duration);
      if (is_skip(lane_changing_length)) continue;
      const auto lane_changing_velocity = std::clamp(
        initial_velocity + lane_changing_accel * lane_changing_duration, min_lc_vel, max_vel);
      metrics.emplace_back(
        lane_changing_duration, lane_changing_length, lane_changing_velocity, lon_accel,
        lane_changing_accel, lat_acc);
    }
  }

  for (double lat_acc = min_lateral_acc; lat_acc < max_lateral_acc + eps;
       lat_acc += lateral_acc_resolution) {
    const auto lane_changing_duration = autoware::motion_utils::calc_shift_time_from_jerk(
      shift_length, common_data_ptr->lc_param_ptr->trajectory.lateral_jerk, lat_acc);

    const double lane_changing_accel = calc_lane_changing_acceleration(
      common_data_ptr, initial_velocity, max_path_velocity, lane_changing_duration, lon_accel);

    const auto lane_changing_length = calculation::calc_phase_length(
      initial_velocity, max_vel, lane_changing_accel, lane_changing_duration);

    if (is_skip(lane_changing_length)) continue;

    const auto lane_changing_velocity = std::clamp(
      initial_velocity + lane_changing_accel * lane_changing_duration, min_lc_vel, max_vel);

    metrics.emplace_back(
      lane_changing_duration, lane_changing_length, lane_changing_velocity, lon_accel,
      lane_changing_accel, lat_acc);
  }

  return metrics;
}
}  // namespace autoware::behavior_path_planner::utils::lane_change::calculation
