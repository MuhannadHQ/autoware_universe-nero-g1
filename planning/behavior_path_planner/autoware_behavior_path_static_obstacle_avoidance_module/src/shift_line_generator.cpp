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

#include "autoware/behavior_path_static_obstacle_avoidance_module/shift_line_generator.hpp"

#include "autoware/behavior_path_planner_common/utils/utils.hpp"
#include "autoware/behavior_path_static_obstacle_avoidance_module/utils.hpp"

#include "nero_proj.hpp"

#include <magic_enum.hpp>

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <cstdlib>
#include <chrono>
#include <string>
#include <utility>
#include <vector>

namespace autoware::behavior_path_planner::utils::static_obstacle_avoidance
{
namespace
{
// NERO_SOA_NO_CHATTER (session 5, upstream #11046 without changing ObjectData's layout, which avoidance_by_lane_change
// shares): object id -> last time it was judged "avoidable by the desired shift length". Remembered while the object
// keeps coming back every cycle (gap < 1 s); remember=true records it, false only reads.
bool nero_avoidable_memory(const ObjectData & object, const bool remember)
{
  static const bool on = [] {
    const char * v = std::getenv("NERO_SOA_NO_CHATTER");
    return v && std::atoi(v) > 0;
  }();
  if (!on) {
    return false;
  }
  static std::unordered_map<std::string, std::chrono::steady_clock::time_point> mem;
  const auto & u = object.object.object_id.uuid;
  const std::string key(u.begin(), u.end());
  const auto now = std::chrono::steady_clock::now();
  if (remember) {
    mem[key] = now;
    return true;
  }
  const auto it = mem.find(key);
  if (it == mem.end()) {
    return false;
  }
  if (now - it->second > std::chrono::seconds(1)) {
    mem.erase(it);
    return false;
  }
  it->second = now;
  return true;
}

// NERO_SOA_SAME_SIDE_RULE_HARMLESS=1 (2026-10-22, captures #1/#2): the same-side rule ("if an object cannot be avoided,
// only avoid objects on its side") vetoes a target on the other side only when the unavoidable object really limits
// that move: not when it is behind ego (the rear traffic is the safety check's job), and not when it is a
// SAME_DIRECTION_SHIFT object (needs no shift) whose allowed shift still covers the move toward it. Stock: the first
// unavoidable object vetoes every target on the other side (#1: a low UNKNOWN 2.4 m left of the lane blocked the move
// round the nose-in car on the right; #2: the stopped follower behind-left did).
bool nero_same_side_harmless()
{
  static const bool on = [] {
    const char * v = std::getenv("NERO_SOA_SAME_SIDE_RULE_HARMLESS");
    const bool r = v && std::atoi(v) > 0;
    std::fprintf(stderr, "[NERO_SOA] same_side_rule_harmless=%d\n", r);
    return r;
  }();
  return on;
}

// NERO_SOA_DECEL_SOFT=1 (2026-10-28, tbx, QUEUE TBX case 4: 10-27 20:48:18): with policy.deceleration "reliable" a
// target whose full margin does not fit under the lateral jerk limit any more (a parked car's box published late,
// 12 m ahead at 15 km/h) is NEED_DECELERATION and gets a stop that ignores the decel limits (field -4.18), even when
// only the soft margin is short. With the switch such a target goes through the best-effort shift (policy.lateral_
// margin "best_effort" is the car's): it is passed with the shift that fits when that keeps the hard margin, or
// without a shift when the robot's line already keeps it; the reliable stop stays whenever the hard margin would not
// be kept.
bool nero_decel_soft()
{
  static const bool on = [] {
    const char * v = std::getenv("NERO_SOA_DECEL_SOFT");
    const bool r = v && std::atoi(v) > 0;
    std::fprintf(stderr, "[NERO_SOA] decel_soft=%d\n", r);
    return r;
  }();
  return on;
}

// NERO_SOA_RETURN_NO_NEED=1 (2026-10-26, QUEUE OLS, capture 10-25 #6): addReturnShiftLine adds no return while any
// target ahead is not avoidable, and a target that needs no shift at all (SAME_DIRECTION_SHIFT: every kerb UNKNOWN,
// bin or car parked off the lane is a target, the UNKNOWN filter has no "enough lateral distance" check) is never
// avoidable: on a street lined with them a robot that is out of its lane (an avoid line with no registered return, a
// re-engage beside the lane) never comes back. With the switch such targets do not withhold the return (they stay
// targets for everything else: the safety check, the same-side rule, the stops).
bool nero_return_no_need()
{
  static const bool on = [] {
    const char * v = std::getenv("NERO_SOA_RETURN_NO_NEED");
    const bool r = v && std::atoi(v) > 0;
    std::fprintf(stderr, "[NERO_SOA] return_no_need=%d\n", r);
    return r;
  }();
  return on;
}

// NERO_SOA_RETURN_DEBUG=1 (log only, 2026-10-26, QUEUE OLS): why addReturnShiftLine adds no return. One line when the
// reason changes and at most once a second otherwise: the reason, ego's shift, the path shifter's base offset and last
// registered end shift, and every target ahead (id, class, info, longitudinal, overhang, side, avoidable).
bool nero_return_debug()
{
  static const bool on = [] {
    const char * v = std::getenv("NERO_SOA_RETURN_DEBUG");
    return v && std::atoi(v) > 0;
  }();
  return on;
}

void nero_return_log(
  const std::string & why, const AvoidancePlanningData & data, const double ego_shift,
  const double base_offset, const std::optional<ShiftLine> & last)
{
  if (!nero_return_debug()) {
    return;
  }
  static std::string last_why;
  static auto last_t = std::chrono::steady_clock::time_point{};
  const auto now = std::chrono::steady_clock::now();
  if (why == last_why && now - last_t < std::chrono::seconds(1)) {
    return;
  }
  last_why = why;
  last_t = now;
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(2) << "[NERO_SOA_RET] " << why << " | ego_shift " << ego_shift
     << " base_offset " << base_offset << " last_end "
     << (last.has_value() ? std::to_string(last.value().end_shift_length).substr(0, 5) : std::string("none"))
     << " | ahead:";
  int n = 0;
  for (const auto & o : data.target_objects) {
    if (o.longitudinal <= 0.0) continue;
    if (++n > 10) {
      ss << " ...";
      break;
    }
    char id[9];
    std::snprintf(
      id, sizeof(id), "%02x%02x%02x%02x", o.object.object_id.uuid[0], o.object.object_id.uuid[1],
      o.object.object_id.uuid[2], o.object.object_id.uuid[3]);
    ss << " " << id << ":" << static_cast<int>(utils::getHighestProbLabel(o.object.classification)) << ":"
       << magic_enum::enum_name(o.info) << ":lon " << o.longitudinal << ":ovh "
       << (o.overhang_points.empty() ? 0.0 : o.overhang_points.front().first) << ":"
       << (isOnRight(o) ? "R" : "L") << (o.is_avoidable ? ":A" : ":a");
  }
  RCLCPP_INFO(rclcpp::get_logger("static_obstacle_avoidance"), "%s", ss.str().c_str());
}

// NERO_SOA_RETURN_NO_NEED: where a return from `shift` to the lane centre over `len` m may start and must end so that
// every standing object's envelope (targets and others: a truck standing in the lane is not a target) keeps its class's
// lateral hard margin from the robot's sides. The robot's lateral place along the return is the smooth S of a
// jerk-limited shift; its body passes a point at s_p while base_link is in [s_p - front, s_p + rear].
// - an object the robot passes out there but the return would come closer to (a truck beside the shifted path): the
//   return starts only once the rear has passed it;
// - an object in the way of the shifted path itself (within obstacle_stop's lateral margin of it: staying out runs into
//   it, e.g. something on the far side of the lane line ahead): the return must be over before the front reaches it;
// - points after the return's end (the lane ahead) are the avoidance's / the stops' business.
// Returns {start, end limit}; nullopt when the start moves past `limit`.
std::optional<std::pair<double, double>> nero_return_window(
  const AvoidancePlanningData & data, const std::shared_ptr<const PlannerData> & planner_data,
  const std::shared_ptr<AvoidanceParameters> & parameters, const double shift, double start,
  const double len, const double limit, std::string & who, std::string & who_end)
{
  const auto & points = data.reference_path.points;
  if (points.size() < 2 || data.arclength_from_ego.size() != points.size()) {
    return std::make_pair(start, limit);
  }
  const double half_w = 0.5 * planner_data->parameters.vehicle_width;
  const double front = planner_data->parameters.base_link2front;
  const double rear = planner_data->parameters.base_link2rear;
  constexpr double LON_MARGIN = 0.5;
  constexpr double LAT_EXTRA = 0.1;  // the S profile is approximated
  struct Pt
  {
    double s;
    double l;
    double m;       // the module's margin (hard) for the return
    double m_stop;  // obstacle_stop's lateral margin on the car (vehicles 0.1 m, others 0.3 m)
    std::string id;
    size_t obj;
  };
  std::vector<Pt> pts;
  size_t n_obj = 0;
  // NERO_SOA_PROJ_FAST: the same projection without a path copy per sample (nero_proj.hpp)
  std::optional<nero_proj::Proj<std::decay_t<decltype(points)>>> proj;
  if (nero_proj::fast()) proj.emplace(points);
  const auto add = [&](const ObjectData & o) {
    const auto type = utils::getHighestProbLabel(o.object.classification);
    const auto & op = parameters->object_parameters.at(type);
    if (o.move_time > op.moving_time_threshold) return;  // moving: not a standing object
    const bool vehicle = type == ObjectClassification::CAR || type == ObjectClassification::TRUCK ||
                         type == ObjectClassification::BUS || type == ObjectClassification::TRAILER ||
                         type == ObjectClassification::MOTORCYCLE;
    char id[9];
    std::snprintf(
      id, sizeof(id), "%02x%02x%02x%02x", o.object.object_id.uuid[0], o.object.object_id.uuid[1],
      o.object.object_id.uuid[2], o.object.object_id.uuid[3]);
    // the envelope's outline every 0.5 m, not only its corners (narrowweave 2026-10-26: an object straddling the line
    // ego returns to, corners on both sides of it, was never in the way: c08's 4bbb26 across the lane centre made the
    // return end in front of the parked row and ego waited behind it for good)
    const auto & outer = o.envelope_poly.outer();
    for (size_t j = 0; j + 1 < outer.size() || (j == 0 && outer.size() == 1); ++j) {
      const auto & a = outer.at(j);
      const auto & b = outer.at(std::min(j + 1, outer.size() - 1));
      const int n_step = std::max(1, static_cast<int>(std::ceil(std::hypot(b.x() - a.x(), b.y() - a.y()) / 0.5)));
      for (int k = 0; k < n_step; ++k) {
        geometry_msgs::msg::Point p;
        p.x = a.x() + (b.x() - a.x()) * k / n_step;
        p.y = a.y() + (b.y() - a.y()) * k / n_step;
        const size_t seg =
          proj ? proj->nearestSegment(p) : autoware::motion_utils::findNearestSegmentIndex(points, p);
        const double s = data.arclength_from_ego.at(seg) +
                         (proj ? proj->lonToSegment(seg, p)
                               : autoware::motion_utils::calcLongitudinalOffsetToSegment(points, seg, p));
        if (s < -rear - LON_MARGIN || s > limit + front) continue;
        pts.push_back(
          {s, proj ? proj->lateral(p, seg) : autoware::motion_utils::calcLateralOffset(points, p, seg),
           half_w + op.lateral_hard_margin + LAT_EXTRA, half_w + (vehicle ? 0.1 : 0.3), id, n_obj});
      }
    }
    ++n_obj;
  };
  for (const auto & o : data.target_objects) add(o);
  for (const auto & o : data.other_objects) add(o);

  // per object ahead: in the way of staying out (a point within obstacle_stop's margin of the shifted path), and in the
  // way of the lane centre (a point within the module's margin of it): an end limit only for one that blocks staying
  // out but not the centre (returning clears it); one that blocks both (a truck in the lane) is the stops' business
  std::vector<char> blocks_stay(n_obj, 0), blocks_centre(n_obj, 0);
  std::vector<double> near(n_obj, 1e9);
  for (const auto & p : pts) {
    if (p.s < front) continue;
    if (std::abs(p.l - shift) < p.m_stop) blocks_stay[p.obj] = 1;
    if (std::abs(p.l) < p.m) blocks_centre[p.obj] = 1;
    near[p.obj] = std::min(near[p.obj], p.s);
  }
  double end_limit = limit;
  for (const auto & p : pts) {
    if (!blocks_stay[p.obj] || blocks_centre[p.obj]) continue;
    if (near[p.obj] - front - LON_MARGIN < end_limit) {
      end_limit = near[p.obj] - front - LON_MARGIN;
      who_end = p.id;
    }
  }
  const auto y = [&](const double s, const double st) {
    if (s <= st) return shift;
    if (s >= st + len) return 0.0;
    const double x = (s - st) / std::max(len, 0.1);
    return shift * (1.0 - x * x * (3.0 - 2.0 * x));
  };
  for (int iter = 0; iter < 20; ++iter) {
    double far = -1e9;
    for (const auto & p : pts) {
      if (blocks_stay[p.obj] && !blocks_centre[p.obj]) continue;  // the end limit above
      if (p.s - front > start + len) continue;         // the lane after the return
      const double ya = y(p.s - front, start);
      const double yb = y(p.s + rear, start);
      // the body's lateral clearance to the point while passing it on the return vs staying out: a conflict only
      // where the return brings the body closer than the margin and closer than it is out there
      const double lo = std::min(ya, yb), hi = std::max(ya, yb);
      const double d_ret = p.l < lo ? lo - p.l : (p.l > hi ? p.l - hi : 0.0);
      const double d_stay = std::abs(p.l - shift);
      if (d_ret < p.m && d_ret < d_stay - 0.01 && p.s > far) {
        far = p.s;
        who = p.id;
      }
    }
    if (far < -1e8) return std::make_pair(start, end_limit);
    start = std::max(start, far + rear + LON_MARGIN);
    if (start > limit) return std::nullopt;
  }
  return std::nullopt;
}
}  // namespace

namespace
{
bool isBestEffort(const std::string & policy)
{
  return policy == "best_effort";
}

bool perManeuver(const std::string & policy)
{
  return policy == "per_avoidance_maneuver";
}

AvoidLine merge(const AvoidLine & line1, const AvoidLine & line2, const UUID id)
{
  AvoidLine ret{};

  ret.start_idx = line1.start_idx;
  ret.start_shift_length = line1.start_shift_length;
  ret.start_longitudinal = line1.start_longitudinal;

  ret.end_idx = line2.end_idx;
  ret.end_shift_length = line2.end_shift_length;
  ret.end_longitudinal = line2.end_longitudinal;

  ret.id = id;
  ret.object = line1.object;

  return ret;
}

AvoidLine fill(const AvoidLine & line1, const AvoidLine & line2, const UUID id)
{
  AvoidLine ret{};

  ret.start_idx = line1.end_idx;
  ret.start_shift_length = line1.end_shift_length;
  ret.start_longitudinal = line1.end_longitudinal;

  ret.end_idx = line2.start_idx;
  ret.end_shift_length = line2.start_shift_length;
  ret.end_longitudinal = line2.start_longitudinal;

  ret.id = id;
  ret.object = line1.object;

  return ret;
}

AvoidLineArray toArray(const AvoidOutlines & outlines)
{
  AvoidLineArray ret{};
  for (const auto & outline : outlines) {
    ret.push_back(outline.avoid_line);
    if (outline.return_line.has_value()) {
      ret.push_back(outline.return_line.value());
    }

    std::for_each(
      outline.middle_lines.begin(), outline.middle_lines.end(),
      [&ret](const auto & line) { ret.push_back(line); });
  }
  return ret;
}
}  // namespace

void ShiftLineGenerator::update(AvoidancePlanningData & data, DebugData & debug)
{
  /**
   * STEP1: Update registered shift line.
   */
  updateRegisteredRawShiftLines(data);

  /**
   * STEP2: Generate avoid outlines.
   * Basically, avoid outlines are generated per target objects.
   */
  const auto outlines = generateAvoidOutline(data, debug);

  /**
   * STEP3: Create rough shift lines.
   */
  raw_ = applyPreProcess(outlines, data, debug);
}

AvoidLineArray ShiftLineGenerator::generate(
  const AvoidancePlanningData & data, DebugData & debug) const
{
  return generateCandidateShiftLine(raw_, data, debug);
}

AvoidOutlines ShiftLineGenerator::generateAvoidOutline(
  AvoidancePlanningData & data, [[maybe_unused]] DebugData & debug) const
{
  // To be consistent with changes in the ego position, the current shift length is considered.
  const auto current_ego_shift = helper_->getEgoShift();

  // Calculate feasible shift length
  const auto get_shift_profile =
    [&](
      auto & object, const auto & desire_shift_length) -> std::optional<std::pair<double, double>> {
    // use each object param
    const auto object_type = utils::getHighestProbLabel(object.object.classification);
    const auto object_parameter = parameters_->object_parameters.at(object_type);
    const auto is_object_on_right = utils::static_obstacle_avoidance::isOnRight(object);

    // use absolute dist for return-to-center, relative dist from current for avoiding.
    const auto avoiding_shift = desire_shift_length - current_ego_shift;
    const auto nominal_avoid_distance = helper_->getMaxAvoidanceDistance(avoiding_shift);

    // calculate remaining distance.
    const auto prepare_distance = helper_->getNominalPrepareDistance();
    const auto constant_distance = helper_->getFrontConstantDistance(object);
    const auto has_enough_distance =
      object.longitudinal > constant_distance + prepare_distance + nominal_avoid_distance;
    const auto remaining_distance = object.longitudinal - constant_distance - prepare_distance;
    const auto avoidance_distance =
      has_enough_distance ? nominal_avoid_distance : remaining_distance;

    // nominal case. avoidable.
    if (has_enough_distance) {
      return std::make_pair(desire_shift_length, avoidance_distance);
    }

    if (!isBestEffort(parameters_->policy_lateral_margin)) {
      return std::make_pair(desire_shift_length, avoidance_distance);
    }

    // ego already has enough positive shift.
    const auto has_enough_positive_shift = avoiding_shift < -1e-3 && desire_shift_length > 1e-3;
    if (is_object_on_right && has_enough_positive_shift) {
      return std::make_pair(desire_shift_length, avoidance_distance);
    }

    // ego already has enough negative shift.
    const auto has_enough_negative_shift = avoiding_shift > 1e-3 && desire_shift_length < -1e-3;
    if (!is_object_on_right && has_enough_negative_shift) {
      return std::make_pair(desire_shift_length, avoidance_distance);
    }

    // don't relax shift length since it can stop in front of the object.
    if (object.is_stoppable && !parameters_->use_shorten_margin_immediately) {
      return std::make_pair(desire_shift_length, avoidance_distance);
    }

    // the avoidance path is already approved
    const auto is_approved =
      (helper_->getShift(object.getPosition()) > 0.0 && is_object_on_right) ||
      (helper_->getShift(object.getPosition()) < 0.0 && !is_object_on_right);
    if (is_approved) {
      return std::make_pair(desire_shift_length, avoidance_distance);
    }

    // prepare distance is not enough. unavoidable.
    if (avoidance_distance < 1e-3) {
      object.info = ObjectInfo::INSUFFICIENT_LONGITUDINAL_DISTANCE;
      if (helper_->isVehicleStopped() && parameters_->policy_close_distance_avoidance != "ignore") {
        object.info = ObjectInfo::CLOSE_DISTANCE_AVOIDANCE;
        return std::make_pair(desire_shift_length, object.longitudinal);
      }
      return std::nullopt;
    }

    // NERO_SOA_NO_CHATTER (upstream #11046): once avoidable by the desired shift, it stays so while the object
    // stays a target — the decision flipped between cycles as ego's speed / distance changed
    if (nero_avoidable_memory(object, false)) {
      return std::make_pair(desire_shift_length, avoidance_distance);
    }

    // calculate lateral jerk.
    const auto required_jerk = autoware::motion_utils::calc_jerk_from_lat_lon_distance(
      avoiding_shift, avoidance_distance, helper_->getAvoidanceEgoSpeed());

    // relax lateral jerk limit. avoidable.
    if (required_jerk < helper_->getLateralMaxJerkLimit()) {
      nero_avoidable_memory(object, true);
      return std::make_pair(desire_shift_length, avoidance_distance);
    }

    constexpr double LON_DIST_BUFFER = 1e-3;

    // NERO_SOA_DECEL_SOFT: under the reliable policy the best-effort shift below is tried before a stop
    const bool nero_soft = !isBestEffort(parameters_->policy_deceleration) && nero_decel_soft();
    const auto nero_hard_margin = object.is_parked
                                    ? object_parameter.lateral_hard_margin_for_parked_vehicle
                                    : object_parameter.lateral_hard_margin;
    const auto nero_line_keeps_hard_margin =
      std::abs(current_ego_shift - object.overhang_points.front().first) >=
      0.5 * data_->parameters.vehicle_width + nero_hard_margin + 1e-3;

    // avoidance distance is not enough. unavoidable.
    if (!isBestEffort(parameters_->policy_deceleration)) {
      if (avoidance_distance < helper_->getMinAvoidanceDistance(avoiding_shift) + LON_DIST_BUFFER) {
        if (
          helper_->isVehicleStopped() && parameters_->policy_close_distance_avoidance != "ignore") {
          object.info = ObjectInfo::CLOSE_DISTANCE_AVOIDANCE;
          return std::make_pair(desire_shift_length, object.longitudinal);
        }
        object.info = ObjectInfo::INSUFFICIENT_LONGITUDINAL_DISTANCE;
        return std::nullopt;
      } else if (!nero_soft) {
        object.info = ObjectInfo::NEED_DECELERATION;
        return std::nullopt;
      }
    }

    // output avoidance path under lateral jerk constraints.
    const auto feasible_relative_shift_length = autoware::motion_utils::calc_lateral_dist_from_jerk(
      avoidance_distance, helper_->getLateralMaxJerkLimit(), helper_->getAvoidanceEgoSpeed());

    if (std::abs(feasible_relative_shift_length) < parameters_->lateral_execution_threshold) {
      // NERO_SOA_DECEL_SOFT: no shift fits; pass only if the robot's line keeps the hard margin, else stop as today
      object.info = nero_soft && !nero_line_keeps_hard_margin ? ObjectInfo::NEED_DECELERATION
                                                              : ObjectInfo::LESS_THAN_EXECUTION_THRESHOLD;
      return std::nullopt;
    }

    const auto feasible_shift_length =
      desire_shift_length > 0.0 ? feasible_relative_shift_length + current_ego_shift
                                : -1.0 * feasible_relative_shift_length + current_ego_shift;

    if (
      avoidance_distance <
      helper_->getMinAvoidanceDistance(feasible_shift_length) + LON_DIST_BUFFER) {
      if (helper_->isVehicleStopped() && parameters_->policy_close_distance_avoidance != "ignore") {
        object.info = ObjectInfo::CLOSE_DISTANCE_AVOIDANCE;
        return std::make_pair(desire_shift_length, object.longitudinal);
      }
      // NERO_SOA_DECEL_SOFT: as above
      object.info = nero_soft && !nero_line_keeps_hard_margin
                      ? ObjectInfo::NEED_DECELERATION
                      : ObjectInfo::INSUFFICIENT_LONGITUDINAL_DISTANCE;
      return std::nullopt;
    }

    const double LAT_DIST_BUFFER = desire_shift_length > 0.0 ? 1e-3 : -1e-3;

    const auto lateral_hard_margin = object.is_parked
                                       ? object_parameter.lateral_hard_margin_for_parked_vehicle
                                       : object_parameter.lateral_hard_margin;
    const auto infeasible =
      std::abs(feasible_shift_length - object.overhang_points.front().first) - LAT_DIST_BUFFER <
      0.5 * data_->parameters.vehicle_width + lateral_hard_margin;
    if (infeasible) {
      RCLCPP_DEBUG(rclcpp::get_logger(""), "feasible shift length is not enough to avoid. ");
      object.info = ObjectInfo::NEED_DECELERATION;
      return std::nullopt;
    }

    return std::make_pair(feasible_shift_length - LAT_DIST_BUFFER, avoidance_distance);
  };

  const auto is_forward_object = [](const auto & object) { return object.longitudinal > 0.0; };

  const auto is_on_path = [this](const auto & object) {
    const auto [overhang, point] = object.overhang_points.front();
    return std::abs(overhang) < 0.5 * data_->parameters.vehicle_width;
  };

  const auto is_valid_shift_line = [](const auto & s) {
    return s.start_longitudinal > 0.0 && s.start_longitudinal < s.end_longitudinal;
  };

  const auto is_approved = [this](const auto & object) {
    return (helper_->getShift(object.getPosition()) > 0.0 && isOnRight(object)) ||
           (helper_->getShift(object.getPosition()) < 0.0 && !isOnRight(object));
  };

  ObjectDataArray unavoidable_objects;

  // target objects are sorted by longitudinal distance.
  AvoidOutlines outlines;
  for (auto & o : data.target_objects) {
    if (!o.avoid_margin.has_value()) {
      if (
        data.red_signal_lane.has_value() &&
        data.red_signal_lane.value().id() == o.overhang_lanelet.id()) {
        o.info = ObjectInfo::LIMIT_DRIVABLE_SPACE_TEMPORARY;
      } else {
        o.info = ObjectInfo::INSUFFICIENT_DRIVABLE_SPACE;
      }
      if (o.avoid_required && is_forward_object(o) && is_on_path(o)) {
        break;
      } else {
        unavoidable_objects.push_back(o);
        continue;
      }
    }

    const auto desire_shift_length =
      helper_->getShiftLength(o, isOnRight(o), o.avoid_margin.value());
    if (utils::static_obstacle_avoidance::isSameDirectionShift(isOnRight(o), desire_shift_length)) {
      o.info = ObjectInfo::SAME_DIRECTION_SHIFT;
      if (o.avoid_required && is_forward_object(o) && is_on_path(o)) {
        break;
      } else {
        unavoidable_objects.push_back(o);
        continue;
      }
    }

    // calculate feasible shift length based on behavior policy
    const auto feasible_shift_profile = get_shift_profile(o, desire_shift_length);
    if (!feasible_shift_profile.has_value()) {
      if (is_approved(o)) {
        // the avoidance path for this object has already approved
        o.is_avoidable = true;
        continue;
      }
      if (o.avoid_required && is_forward_object(o) && is_on_path(o)) {
        break;
      } else {
        unavoidable_objects.push_back(o);
        continue;
      }
    }

    // If there is an object that cannot be avoided, this module only avoids object on the same side
    // as unavoidable object.
    if (!unavoidable_objects.empty() && !nero_same_side_harmless()) {
      if (isOnRight(unavoidable_objects.front()) && !isOnRight(o)) {
        break;
      }
      if (!isOnRight(unavoidable_objects.front()) && isOnRight(o)) {
        break;
      }
    }
    if (!unavoidable_objects.empty() && nero_same_side_harmless()) {
      bool veto = false;
      for (const auto & u : unavoidable_objects) {
        if (isOnRight(u) == isOnRight(o)) continue;  // same side: no conflict
        if (u.longitudinal < 0.0) continue;           // behind ego
        if (u.info == ObjectInfo::SAME_DIRECTION_SHIFT && u.avoid_margin.has_value()) {
          // the furthest ego may go toward u while keeping u's margin
          const auto allowed = helper_->getShiftLength(u, isOnRight(u), u.avoid_margin.value());
          if (std::abs(desire_shift_length) <= std::abs(allowed)) continue;
        }
        veto = true;
        break;
      }
      if (veto) {
        break;
      }
    }

    // use absolute dist for return-to-center, relative dist from current for avoiding.
    const auto feasible_return_distance =
      helper_->getMaxAvoidanceDistance(feasible_shift_profile.value().first);

    AvoidLine al_avoid;
    {
      const auto constant_distance = helper_->getFrontConstantDistance(o);
      const auto to_shift_end = o.info == ObjectInfo::CLOSE_DISTANCE_AVOIDANCE
                                  ? o.longitudinal
                                  : o.longitudinal - constant_distance;
      const auto path_front_to_ego = data.arclength_from_ego.at(data.ego_closest_path_index);

      // start point (use previous linear shift length as start shift length.)
      al_avoid.start_longitudinal = [&]() {
        const auto nearest_avoid_distance =
          std::max(to_shift_end - feasible_shift_profile.value().second, 1e-3);

        if (data.to_start_point > to_shift_end) {
          return nearest_avoid_distance;
        }

        const auto minimum_avoid_distance = helper_->getMinAvoidanceDistance(
          feasible_shift_profile.value().first - current_ego_shift);
        const auto furthest_avoid_distance = std::max(to_shift_end - minimum_avoid_distance, 1e-3);

        return std::clamp(data.to_start_point, nearest_avoid_distance, furthest_avoid_distance);
      }();

      al_avoid.start_idx = utils::static_obstacle_avoidance::findPathIndexFromArclength(
        data.arclength_from_ego, al_avoid.start_longitudinal + path_front_to_ego);
      al_avoid.start = data.reference_path.points.at(al_avoid.start_idx).point.pose;
      al_avoid.start_shift_length = helper_->getLinearShift(al_avoid.start.position);

      // end point
      al_avoid.end_shift_length = feasible_shift_profile.value().first;
      al_avoid.end_longitudinal = to_shift_end;

      // misc
      al_avoid.id = generate_uuid();
      al_avoid.object = o;
      al_avoid.object_on_right = utils::static_obstacle_avoidance::isOnRight(o);
    }

    AvoidLine al_return;
    {
      const auto constant_distance = helper_->getRearConstantDistance(o);
      const auto to_shift_start = o.longitudinal + constant_distance;

      // start point
      al_return.start_shift_length = feasible_shift_profile.value().first;
      al_return.start_longitudinal = to_shift_start;

      // end point
      al_return.end_longitudinal = [&]() {
        if (data.to_return_point > to_shift_start) {
          return std::clamp(
            data.to_return_point, to_shift_start, feasible_return_distance + to_shift_start);
        }

        return to_shift_start + feasible_return_distance;
      }();
      al_return.end_shift_length = 0.0;

      // misc
      al_return.id = generate_uuid();
      al_return.object = o;
      al_return.object_on_right = utils::static_obstacle_avoidance::isOnRight(o);
    }

    const bool skip_return_shift = [&]() {
      if (!utils::isAllowedGoalModification(data_->route_handler)) {
        return false;
      }
      const auto goal_pose = data_->route_handler->getGoalPose();
      const double goal_longitudinal_distance = autoware::motion_utils::calcSignedArcLength(
        data.reference_path.points, 0, goal_pose.position);
      const bool is_return_shift_to_goal =
        std::abs(al_return.end_longitudinal - goal_longitudinal_distance) <
        parameters_->object_check_return_pose_distance;
      if (is_return_shift_to_goal) {
        return true;
      }
      const bool has_object_near_goal =
        autoware_utils::calc_distance2d(goal_pose.position, o.getPosition()) <
        parameters_->object_check_goal_distance;
      return has_object_near_goal;
    }();

    if (skip_return_shift) {
      // if the object is close to goal or ego is about to return near GOAL, do not return
      outlines.emplace_back(al_avoid, std::nullopt);
    } else if (is_valid_shift_line(al_avoid) && is_valid_shift_line(al_return)) {
      outlines.emplace_back(al_avoid, al_return);
    } else if (!is_approved(o)) {
      o.info = ObjectInfo::INVALID_SHIFT_LINE;
      continue;
    }

    o.is_avoidable = true;
  }

  utils::static_obstacle_avoidance::fillAdditionalInfoFromLongitudinal(data, outlines);

  debug.step1_current_shift_line = toArray(outlines);

  return outlines;
}

AvoidLineArray ShiftLineGenerator::applyPreProcess(
  const AvoidOutlines & outlines, const AvoidancePlanningData & data, DebugData & debug) const
{
  AvoidOutlines processed_outlines = outlines;

  /**
   * Step1: Rough merge process.
   * Merge multiple avoid outlines. If an avoid outlines' return shift line conflicts other
   * outline's avoid shift line, those avoid outlines are merged.
   */
  processed_outlines = applyMergeProcess(processed_outlines, data, debug);

  /**
   * Step2: Fill gap process.
   * Create and add new shift line to avoid outline in order to fill gaps between avoid shift line
   * and middle shift lines, return shift line and middle shift lines.
   */
  processed_outlines = applyFillGapProcess(processed_outlines, data, debug);

  /**
   * Step3: Convert to AvoidLineArray from AvoidOutlines.
   */
  AvoidLineArray processed_raw_lines = toArray(processed_outlines);

  /**
   * Step4: Combine process.
   * Use all registered points. For the current points, if the similar one of the current
   * points are already registered, will not use it.
   */
  processed_raw_lines = applyCombineProcess(processed_raw_lines, raw_registered_, debug);

  /*
   * Step5: Add return shift line.
   * Add return-to-center shift point from the last shift point, if needed.
   * If there is no shift points, set return-to center shift from ego.
   */
  processed_raw_lines = addReturnShiftLine(processed_raw_lines, data, debug);

  /*
   * Step6: Fill gap process.
   * Create and add new shift line to avoid lines.
   */
  return applyFillGapProcess(processed_raw_lines, data, debug);
}

AvoidLineArray ShiftLineGenerator::generateCandidateShiftLine(
  const AvoidLineArray & shift_lines, const AvoidancePlanningData & data, DebugData & debug) const
{
  AvoidLineArray processed_shift_lines = shift_lines;

  /**
   * Step1: Merge process.
   * Merge positive shift avoid lines and negative shift avoid lines.
   */
  processed_shift_lines = applyMergeProcess(processed_shift_lines, data, debug);

  /**
   * Step2: Clean up process.
   * Remove noisy shift line and concat same gradient shift lines.
   */
  processed_shift_lines = applyTrimProcess(processed_shift_lines, debug);

  /**
   * Step3: Extract new shift lines.
   * Compare processed shift lines and registered shift lines in order to find new shift lines.
   */
  return findNewShiftLine(processed_shift_lines, debug);
}

void ShiftLineGenerator::generateTotalShiftLine(
  const AvoidLineArray & avoid_lines, const AvoidancePlanningData & data,
  ShiftLineData & shift_line_data) const
{
  const auto & path = data.reference_path;
  const auto & arcs = data.arclength_from_ego;
  const auto N = path.points.size();

  auto & sl = shift_line_data;

  sl.shift_line = std::vector<double>(N, 0.0);
  sl.shift_line_grad = std::vector<double>(N, 0.0);

  sl.pos_shift_line = std::vector<double>(N, 0.0);
  sl.neg_shift_line = std::vector<double>(N, 0.0);

  sl.pos_shift_line_grad = std::vector<double>(N, 0.0);
  sl.neg_shift_line_grad = std::vector<double>(N, 0.0);

  // debug
  sl.shift_line_history = std::vector<std::vector<double>>(avoid_lines.size(), sl.shift_line);

  // take minmax for same directional shift length
  for (size_t j = 0; j < avoid_lines.size(); ++j) {
    const auto & al = avoid_lines.at(j);
    for (size_t i = 0; i < N; ++i) {
      // calc current interpolated shift
      const auto i_shift = utils::static_obstacle_avoidance::lerpShiftLengthOnArc(arcs.at(i), al);

      // update maximum shift for positive direction
      if (i_shift > sl.pos_shift_line.at(i)) {
        sl.pos_shift_line.at(i) = i_shift;
        sl.pos_shift_line_grad.at(i) = al.getGradient();
      }

      // update minumum shift for negative direction
      if (i_shift < sl.neg_shift_line.at(i)) {
        sl.neg_shift_line.at(i) = i_shift;
        sl.neg_shift_line_grad.at(i) = al.getGradient();
      }

      // store for debug print
      sl.shift_line_history.at(j).at(i) = i_shift;
    }
  }

  // Merge shift length of opposite directions.
  for (size_t i = 0; i < N; ++i) {
    sl.shift_line.at(i) = sl.pos_shift_line.at(i) + sl.neg_shift_line.at(i);
    sl.shift_line_grad.at(i) = sl.pos_shift_line_grad.at(i) + sl.neg_shift_line_grad.at(i);
  }

  // overwrite shift with current_ego_shift until ego pose.
  const auto current_shift = helper_->getEgoLinearShift();
  for (size_t i = 0; i < sl.shift_line.size(); ++i) {
    if (data.ego_closest_path_index < i) {
      break;
    }
    sl.shift_line.at(i) = current_shift;
    sl.shift_line_grad.at(i) = 0.0;
  }

  // If the shift point does not have an associated object,
  // use previous value.
  for (size_t i = 1; i < N; ++i) {
    bool has_object = false;
    for (const auto & al : avoid_lines) {
      if (al.start_idx <= i && i <= al.end_idx) {
        has_object = true;
        break;
      }
    }
    if (!has_object) {
      sl.shift_line.at(i) = sl.shift_line.at(i - 1);
    }
  }

  if (avoid_lines.empty()) {
    sl.shift_line_history.push_back(sl.shift_line);
    return;
  }

  const auto grad_first_shift_line = (avoid_lines.front().start_shift_length - current_shift) /
                                     avoid_lines.front().start_longitudinal;

  for (size_t i = data.ego_closest_path_index; i <= avoid_lines.front().start_idx; ++i) {
    sl.shift_line.at(i) = helper_->getLinearShift(get_point(path.points.at(i)));
    sl.shift_line_grad.at(i) = grad_first_shift_line;
  }

  sl.shift_line_history.push_back(sl.shift_line);
}

AvoidLineArray ShiftLineGenerator::extractShiftLinesFromLine(
  const AvoidancePlanningData & data, ShiftLineData & shift_line_data) const
{
  using utils::static_obstacle_avoidance::setEndData;
  using utils::static_obstacle_avoidance::setStartData;

  const auto & path = data.reference_path;
  const auto & arcs = data.arclength_from_ego;
  const auto N = path.points.size();

  auto & sl = shift_line_data;

  // nero: upstream #12105 rewrote this extraction together with the close-vehicle policy (tested together upstream);
  // with the policy "ignore" (NERO_SOA_CLOSE_VEHICLE_POLICY unset) the 0.44.2 forward/backward-difference test stays.
  const bool close_vehicle_on = parameters_->policy_close_distance_avoidance != "ignore";

  const auto backward_grad = [&](const size_t i) {
    if (i == 0) {
      return sl.shift_line_grad.at(i);
    }
    const double ds = arcs.at(i) - arcs.at(i - 1);
    if (ds < 1.0e-5) {
      return sl.shift_line_grad.at(i);
    }  // use theoretical value when ds is too small.
    return (sl.shift_line.at(i) - sl.shift_line.at(i - 1)) / ds;
  };

  const auto forward_grad = [&](const size_t i) {
    if (i == arcs.size() - 1) {
      return sl.shift_line_grad.at(i);
    }
    const double ds = arcs.at(i + 1) - arcs.at(i);
    if (ds < 1.0e-5) {
      return sl.shift_line_grad.at(i);
    }  // use theoretical value when ds is too small.
    return (sl.shift_line.at(i + 1) - sl.shift_line.at(i)) / ds;
  };

  // calculate forward and backward gradient of the shift length.
  // This will be used for grad-change-point check.
  sl.forward_grad = std::vector<double>(N, 0.0);
  sl.backward_grad = std::vector<double>(N, 0.0);
  for (size_t i = 0; i < N - 1; ++i) {
    sl.forward_grad.at(i) = forward_grad(i);
    sl.backward_grad.at(i) = backward_grad(i);
  }

  // Calculate second derivative of shift line directly from sl.shift_line_grad.
  // This detects the start and end points of shift by finding where the gradient changes.
  std::vector<double> shift_line_second_grad(N, 0.0);
  for (size_t i = 0; i < N - 1; ++i) {
    const double ds = arcs.at(i + 1) - arcs.at(i);
    if (ds > 1.0e-5) {
      shift_line_second_grad.at(i) = (sl.shift_line_grad.at(i + 1) - sl.shift_line_grad.at(i)) / ds;
    }
  }

  AvoidLineArray merged_avoid_lines;
  AvoidLine al{};
  bool found_first_start = false;
  constexpr auto CREATE_SHIFT_GRAD_THR = 0.001;
  constexpr auto IS_ALREADY_SHIFTING_THR = 0.001;
  for (size_t i = data.ego_closest_path_index; i < N - 1; ++i) {
    const auto & p = path.points.at(i).point.pose;
    const auto shift = sl.shift_line.at(i);

    // If the vehicle is already on the avoidance (checked by the first point has shift),
    // set a start point at the first path point.
    if (!found_first_start && std::abs(shift) > IS_ALREADY_SHIFTING_THR) {
      setStartData(al, 0.0, p, i, arcs.at(i));  // start length is overwritten later.
      found_first_start = true;
      if (close_vehicle_on) {
        continue;
      }
    }

    // Find the point where the second derivative of shift gradient is significant.
    // This indicates the start or end of a shift maneuver.
    const bool set_shift_line_flag =
      close_vehicle_on
        ? std::abs(shift_line_second_grad.at(i)) > CREATE_SHIFT_GRAD_THR
        : std::abs(sl.forward_grad.at(i) - sl.backward_grad.at(i)) > CREATE_SHIFT_GRAD_THR;

    if (!set_shift_line_flag) {
      continue;
    }

    if (!found_first_start) {
      setStartData(al, 0.0, p, i, arcs.at(i));  // start length is overwritten later.
      found_first_start = true;
    } else {
      setEndData(al, shift, p, i, arcs.at(i));
      al.id = generate_uuid();
      merged_avoid_lines.push_back(al);
      setStartData(al, 0.0, p, i, arcs.at(i));  // start length is overwritten later.
    }
  }

  // Check if the last point still has a gradient (shift is ongoing at the end of path).
  const bool ongoing_at_end = close_vehicle_on
                                ? (N > 1 && std::abs(sl.shift_line_grad.at(N - 1)) > CREATE_SHIFT_GRAD_THR)
                                : std::abs(backward_grad(N - 1)) > CREATE_SHIFT_GRAD_THR;
  if (ongoing_at_end) {
    const auto & p = path.points.at(N - 1).point.pose;
    const auto shift = sl.shift_line.at(N - 1);
    setEndData(al, shift, p, N - 1, arcs.at(N - 1));
    al.id = generate_uuid();
    merged_avoid_lines.push_back(al);
  }

  return merged_avoid_lines;
}

AvoidOutlines ShiftLineGenerator::applyMergeProcess(
  const AvoidOutlines & outlines, const AvoidancePlanningData & data, DebugData & debug) const
{
  AvoidOutlines ret{};

  if (outlines.size() < 2) {
    return outlines;
  }

  const auto no_conflict = [](const auto & line1, const auto & line2) {
    return line1.end_idx < line2.start_idx || line2.end_idx < line1.start_idx;
  };

  const auto same_side_shift = [](const auto & line1, const auto & line2) {
    return line1.object_on_right == line2.object_on_right;
  };

  const auto within = [](const auto & line, const size_t idx) {
    return line.start_idx < idx && idx < line.end_idx;
  };

  ret.push_back(outlines.front());

  for (size_t i = 1; i < outlines.size(); i++) {
    auto & last_outline = ret.back();
    auto & next_outline = outlines.at(i);

    const auto & return_line = last_outline.return_line;
    if (!return_line.has_value()) {
      ret.push_back(outlines.at(i));
      break;
    }

    const auto & avoid_line = next_outline.avoid_line;
    if (no_conflict(return_line.value(), avoid_line)) {
      ret.push_back(outlines.at(i));
      continue;
    }

    const auto merged_shift_line = merge(return_line.value(), avoid_line, generate_uuid());

    if (!helper_->isComfortable(AvoidLineArray{merged_shift_line})) {
      ret.push_back(outlines.at(i));
      continue;
    }

    if (same_side_shift(return_line.value(), avoid_line)) {
      last_outline.middle_lines.push_back(merged_shift_line);
      last_outline.return_line = next_outline.return_line;
      debug.step1_merged_shift_line.push_back(merged_shift_line);
      continue;
    }

    if (
      within(return_line.value(), avoid_line.end_idx) &&
      within(avoid_line, return_line->start_idx)) {
      last_outline.middle_lines.push_back(merged_shift_line);
      last_outline.return_line = next_outline.return_line;
      debug.step1_merged_shift_line.push_back(merged_shift_line);
      continue;
    }

    if (
      within(return_line.value(), avoid_line.start_idx) &&
      within(avoid_line, return_line->end_idx)) {
      last_outline.middle_lines.push_back(merged_shift_line);
      last_outline.return_line = next_outline.return_line;
      debug.step1_merged_shift_line.push_back(merged_shift_line);
      continue;
    }
  }

  utils::static_obstacle_avoidance::fillAdditionalInfoFromLongitudinal(data, ret);
  utils::static_obstacle_avoidance::fillAdditionalInfoFromLongitudinal(
    data, debug.step1_merged_shift_line);

  return ret;
}

AvoidOutlines ShiftLineGenerator::applyFillGapProcess(
  const AvoidOutlines & outlines, const AvoidancePlanningData & data, DebugData & debug) const
{
  AvoidOutlines ret = outlines;

  for (auto & outline : ret) {
    if (outline.middle_lines.empty()) {
      const auto new_line =
        outline.return_line.has_value()
          ? fill(outline.avoid_line, outline.return_line.value(), generate_uuid())
          : outline.avoid_line;
      outline.middle_lines.push_back(new_line);
      debug.step1_filled_shift_line.push_back(new_line);
    }

    helper_->alignShiftLinesOrder(outline.middle_lines, false);

    if (outline.avoid_line.end_longitudinal < outline.middle_lines.front().start_longitudinal) {
      const auto new_line = fill(outline.avoid_line, outline.middle_lines.front(), generate_uuid());
      outline.middle_lines.push_back(new_line);
      debug.step1_filled_shift_line.push_back(new_line);
    }

    helper_->alignShiftLinesOrder(outline.middle_lines, false);

    if (
      outline.return_line.has_value() &&
      outline.middle_lines.back().end_longitudinal < outline.return_line->start_longitudinal) {
      const auto new_line =
        fill(outline.middle_lines.back(), outline.return_line.value(), generate_uuid());
      outline.middle_lines.push_back(new_line);
      debug.step1_filled_shift_line.push_back(new_line);
    }

    helper_->alignShiftLinesOrder(outline.middle_lines, false);
  }

  utils::static_obstacle_avoidance::fillAdditionalInfoFromLongitudinal(data, ret);
  utils::static_obstacle_avoidance::fillAdditionalInfoFromLongitudinal(
    data, debug.step1_filled_shift_line);

  return ret;
}

AvoidLineArray ShiftLineGenerator::applyFillGapProcess(
  const AvoidLineArray & shift_lines, const AvoidancePlanningData & data, DebugData & debug) const
{
  AvoidLineArray sorted = shift_lines;

  helper_->alignShiftLinesOrder(sorted, false);

  AvoidLineArray ret = sorted;

  if (shift_lines.empty()) {
    return ret;
  }

  // fill gap between ego and nearest shift line.
  if (sorted.front().start_longitudinal > 0.0) {
    AvoidLine ego_line{};
    utils::static_obstacle_avoidance::setEndData(
      ego_line, helper_->getEgoLinearShift(), data.reference_pose, data.ego_closest_path_index,
      0.0);

    const auto new_line = fill(ego_line, sorted.front(), generate_uuid());
    ret.push_back(new_line);
    debug.step1_front_shift_line.push_back(new_line);
  }

  helper_->alignShiftLinesOrder(sorted, false);

  // fill gap among shift lines.
  for (size_t i = 0; i < sorted.size() - 1; ++i) {
    if (sorted.at(i + 1).start_longitudinal < sorted.at(i).end_longitudinal + 1e-3) {
      continue;
    }

    const auto new_line = fill(sorted.at(i), sorted.at(i + 1), generate_uuid());
    ret.push_back(new_line);
    debug.step1_front_shift_line.push_back(new_line);
  }

  helper_->alignShiftLinesOrder(ret, false);

  utils::static_obstacle_avoidance::fillAdditionalInfoFromLongitudinal(data, ret);
  utils::static_obstacle_avoidance::fillAdditionalInfoFromLongitudinal(
    data, debug.step1_front_shift_line);

  return ret;
}

AvoidLineArray ShiftLineGenerator::applyCombineProcess(
  const AvoidLineArray & shift_lines, const AvoidLineArray & registered_lines,
  DebugData & debug) const
{
  debug.step1_registered_shift_line = registered_lines;
  return utils::static_obstacle_avoidance::combineRawShiftLinesWithUniqueCheck(
    registered_lines, shift_lines);
}

AvoidLineArray ShiftLineGenerator::applyMergeProcess(
  const AvoidLineArray & shift_lines, const AvoidancePlanningData & data, DebugData & debug) const
{
  // Generate shift line by merging shift_lines.
  ShiftLineData shift_line_data;
  generateTotalShiftLine(shift_lines, data, shift_line_data);

  // Re-generate shift points by detecting gradient-change point of the shift line.
  auto merged_shift_lines = extractShiftLinesFromLine(data, shift_line_data);

  // set parent id
  for (auto & al : merged_shift_lines) {
    al.parent_ids = utils::static_obstacle_avoidance::calcParentIds(shift_lines, al);
  }

  // sort by distance from ego.
  helper_->alignShiftLinesOrder(merged_shift_lines);

  // debug visualize
  {
    debug.pos_shift = shift_line_data.pos_shift_line;
    debug.neg_shift = shift_line_data.neg_shift_line;
    debug.total_shift = shift_line_data.shift_line;
    debug.pos_shift_grad = shift_line_data.pos_shift_line_grad;
    debug.neg_shift_grad = shift_line_data.neg_shift_line_grad;
    debug.total_forward_grad = shift_line_data.forward_grad;
    debug.total_backward_grad = shift_line_data.backward_grad;
    debug.step2_merged_shift_line = merged_shift_lines;
  }

  return merged_shift_lines;
}

AvoidLineArray ShiftLineGenerator::applyTrimProcess(
  const AvoidLineArray & shift_lines, DebugData & debug) const
{
  if (shift_lines.empty()) {
    return shift_lines;
  }

  AvoidLineArray sl_array_trimmed = shift_lines;

  // sort shift points from front to back.
  helper_->alignShiftLinesOrder(sl_array_trimmed);

  // - Change the shift length to the previous one if the deviation is small.
  {
    constexpr double SHIFT_DIFF_THRES = 1.0;
    applySmallShiftFilter(sl_array_trimmed, SHIFT_DIFF_THRES);
  }

  // - Combine avoid points that have almost same gradient.
  // this is to remove the noise.
  {
    const auto THRESHOLD = parameters_->th_similar_grad_1;
    applySimilarGradFilter(sl_array_trimmed, THRESHOLD);
    debug.step3_grad_filtered_1st = sl_array_trimmed;
  }

  // - Quantize the shift length to reduce the shift point noise
  // This is to remove the noise coming from detection accuracy, interpolation, resampling, etc.
  {
    const auto THRESHOLD = parameters_->quantize_size;
    applyQuantizeProcess(sl_array_trimmed, THRESHOLD);
    debug.step3_quantize_filtered = sl_array_trimmed;
  }

  // - Change the shift length to the previous one if the deviation is small.
  {
    constexpr double SHIFT_DIFF_THRES = 1.0;
    applySmallShiftFilter(sl_array_trimmed, SHIFT_DIFF_THRES);
    debug.step3_noise_filtered = sl_array_trimmed;
  }

  // - Combine avoid points that have almost same gradient (again)
  {
    const auto THRESHOLD = parameters_->th_similar_grad_2;
    applySimilarGradFilter(sl_array_trimmed, THRESHOLD);
    debug.step3_grad_filtered_2nd = sl_array_trimmed;
  }

  // - Combine avoid points that have almost same gradient (again)
  {
    const auto THRESHOLD = parameters_->th_similar_grad_3;
    applySimilarGradFilter(sl_array_trimmed, THRESHOLD);
    debug.step3_grad_filtered_3rd = sl_array_trimmed;
  }

  return sl_array_trimmed;
}

void ShiftLineGenerator::applyQuantizeProcess(
  AvoidLineArray & shift_lines, const double threshold) const
{
  if (threshold < 1.0e-5) {
    return;  // no need to process
  }

  for (auto & sl : shift_lines) {
    sl.end_shift_length = std::round(sl.end_shift_length / threshold) * threshold;
  }

  helper_->alignShiftLinesOrder(shift_lines);
}

void ShiftLineGenerator::applySmallShiftFilter(
  AvoidLineArray & shift_lines, const double threshold) const
{
  if (shift_lines.empty()) {
    return;
  }

  AvoidLineArray input = shift_lines;
  shift_lines.clear();

  for (const auto & s : input) {
    if (s.getRelativeLongitudinal() < threshold) {
      continue;
    }

    if (!helper_->isEnoughPrepareDistance(s.start_longitudinal)) {
      continue;
    }

    shift_lines.push_back(s);
  }
}

void ShiftLineGenerator::applySimilarGradFilter(
  AvoidLineArray & avoid_lines, const double threshold) const
{
  if (avoid_lines.empty()) {
    return;
  }

  AvoidLineArray input = avoid_lines;
  avoid_lines.clear();
  avoid_lines.push_back(input.front());  // Take the first one anyway (think later)

  AvoidLine base_line = input.front();

  AvoidLineArray combine_buffer;
  combine_buffer.push_back(input.front());

  constexpr auto SHIFT_THR = 1e-3;
  const auto is_negative_shift = [&](const auto & s) {
    return s.getRelativeLength() < -1.0 * SHIFT_THR;
  };

  const auto is_positive_shift = [&](const auto & s) { return s.getRelativeLength() > SHIFT_THR; };

  for (size_t i = 1; i < input.size(); ++i) {
    AvoidLine combine{};

    utils::static_obstacle_avoidance::setStartData(
      combine, base_line.start_shift_length, base_line.start, base_line.start_idx,
      base_line.start_longitudinal);
    utils::static_obstacle_avoidance::setEndData(
      combine, input.at(i).end_shift_length, input.at(i).end, input.at(i).end_idx,
      input.at(i).end_longitudinal);

    combine.parent_ids = utils::static_obstacle_avoidance::concatParentIds(
      base_line.parent_ids, input.at(i).parent_ids);

    combine_buffer.push_back(input.at(i));

    const auto violates = [&]() {
      if (is_negative_shift(input.at(i)) && is_positive_shift(base_line)) {
        return true;
      }

      if (is_negative_shift(base_line) && is_positive_shift(input.at(i))) {
        return true;
      }

      const auto lon_combine = combine.getRelativeLongitudinal();
      const auto base_length = base_line.getGradient() * lon_combine;
      return std::abs(combine.getRelativeLength() - base_length) > threshold;
    }();

    if (violates) {
      avoid_lines.push_back(input.at(i));
      base_line = input.at(i);
      combine_buffer.clear();
      combine_buffer.push_back(input.at(i));
    } else {
      avoid_lines.back() = combine;
    }
  }

  helper_->alignShiftLinesOrder(avoid_lines);
}

AvoidLineArray ShiftLineGenerator::addReturnShiftLine(
  const AvoidLineArray & shift_lines, const AvoidancePlanningData & data, DebugData & debug) const
{
  AvoidLineArray ret = shift_lines;

  constexpr double ep = 1.0e-3;
  constexpr double RETURN_SHIFT_THRESHOLD = 0.1;
  const bool has_candidate_point = !ret.empty();
  const bool has_registered_point = last_.has_value();

  if (utils::isAllowedGoalModification(data_->route_handler)) {
    const auto has_object_near_goal =
      std::any_of(data.target_objects.begin(), data.target_objects.end(), [&](const auto & o) {
        return autoware_utils::calc_distance2d(
                 data_->route_handler->getGoalPose().position, o.getPosition()) <
               parameters_->object_check_goal_distance;
      });
    if (has_object_near_goal) {
      RCLCPP_DEBUG(rclcpp::get_logger(""), "object near goal exists so skip adding return shift");
      nero_return_log("object near goal", data, helper_->getEgoShift(), base_offset_, last_);
      return ret;
    }
  }

  // NERO_SOA_RETURN_NO_NEED: a target that needs no shift even from the lane centre (SAME_DIRECTION_SHIFT: kerb
  // objects, bins, cars parked off the lane) does not withhold the return; the return ends at the centre, which keeps
  // its margin by that very verdict, and every shift on the way lies between ego's and the centre's
  const auto exist_unavoidable_object = std::any_of(
    data.target_objects.begin(), data.target_objects.end(), [](const auto & o) {
      if (nero_return_no_need() && o.info == ObjectInfo::SAME_DIRECTION_SHIFT) return false;
      return !o.is_avoidable && o.longitudinal > 0.0;
    });

  if (exist_unavoidable_object) {
    nero_return_log("unavoidable target ahead", data, helper_->getEgoShift(), base_offset_, last_);
    return ret;
  }

  if (last_.has_value()) {
    if (std::abs(last_.value().end_shift_length) < RETURN_SHIFT_THRESHOLD) {
      nero_return_log("last registered line already ends at the centre", data, helper_->getEgoShift(), base_offset_, last_);
      return ret;
    }
  } else {
    // If the return-to-center shift points are already registered, do nothing.
    if (std::abs(base_offset_) < ep) {
      nero_return_log("no base offset", data, helper_->getEgoShift(), base_offset_, last_);
      return ret;
    }
  }

  // From here, the return-to-center is not registered. But perhaps the candidate is
  // already generated.

  // If it has a shift point, add return shift from the existing last shift point.
  // If not, add return shift from ego point. (prepare distance is considered for both.)
  ShiftLine last_sl;  // the return-shift will be generated after the last shift point.
  {
    // avoidance points: Yes, shift points: No -> select last avoidance point.
    if (has_candidate_point && !has_registered_point) {
      helper_->alignShiftLinesOrder(ret, false);
      last_sl = ret.back();
    }

    // avoidance points: No, shift points: Yes -> select last shift point.
    if (!has_candidate_point && has_registered_point) {
      last_sl =
        utils::static_obstacle_avoidance::fillAdditionalInfo(data, AvoidLine{last_.value()});
    }

    // avoidance points: Yes, shift points: Yes -> select the last one from both.
    if (has_candidate_point && has_registered_point) {
      helper_->alignShiftLinesOrder(ret, false);
      const auto & al = ret.back();
      const auto & sl =
        utils::static_obstacle_avoidance::fillAdditionalInfo(data, AvoidLine{last_.value()});
      last_sl = (sl.end_longitudinal > al.end_longitudinal) ? sl : al;
    }

    // avoidance points: No, shift points: No -> set the ego position to the last shift point
    // so that the return-shift will be generated from ego position.
    if (!has_candidate_point && !has_registered_point) {
      last_sl.end_idx = data.ego_closest_path_index;
      last_sl.end = data.reference_path.points.at(last_sl.end_idx).point.pose;
      last_sl.end_shift_length = base_offset_;
    }
  }

  // if last shift line is near the objects, do not add return shift.
  if (utils::isAllowedGoalModification(data_->route_handler)) {
    const bool has_last_shift_near_goal =
      std::any_of(data.target_objects.begin(), data.target_objects.end(), [&](const auto & o) {
        return autoware_utils::calc_distance2d(last_sl.end.position, o.getPosition()) <
               parameters_->object_check_goal_distance;
      });
    if (has_last_shift_near_goal) {
      RCLCPP_DEBUG(rclcpp::get_logger(""), "last shift line is near the objects");
      nero_return_log("last shift line near the goal", data, helper_->getEgoShift(), base_offset_, last_);
      return ret;
    }
  }

  // There already is a shift point candidates to go back to center line, but it could be too sharp
  // due to detection noise or timing.
  // Here the return-shift from ego is added for the in case.
  if (std::fabs(last_sl.end_shift_length) < RETURN_SHIFT_THRESHOLD) {
    const auto current_base_shift = helper_->getEgoShift();
    if (std::abs(current_base_shift) < ep) {
      nero_return_log("ego not shifted", data, current_base_shift, base_offset_, last_);
      return ret;
    }

    // Is there a shift point in the opposite direction of the current_base_shift?
    //   No  -> we can overwrite the return shift, because the other shift points that decrease
    //          the shift length are for return-shift.
    //   Yes -> we can NOT overwrite, because it might be not a return-shift, but a avoiding
    //          shift to the opposite direction which can not be overwritten by the return-shift.
    for (const auto & sl : ret) {
      if (
        (current_base_shift > 0.0 && sl.end_shift_length < -ep) ||
        (current_base_shift < 0.0 && sl.end_shift_length > ep)) {
        nero_return_log("a line to the other side exists", data, current_base_shift, base_offset_, last_);
        return ret;
      }
    }

    // If return shift already exists in candidate or registered shift lines, skip adding return
    // shift.
    if (has_candidate_point || has_registered_point) {
      nero_return_log("a return line already exists", data, current_base_shift, base_offset_, last_);
      return ret;
    }

    // set the return-shift from ego.
    last_sl.end_idx = data.ego_closest_path_index;
    last_sl.end = data.reference_path.points.at(last_sl.end_idx).point.pose;
    last_sl.end_shift_length = current_base_shift;
  }

  const auto & arclength_from_ego = data.arclength_from_ego;

  const auto nominal_prepare_distance = helper_->getNominalPrepareDistance();
  const auto nominal_avoid_distance = helper_->getMaxAvoidanceDistance(last_sl.end_shift_length);

  if (arclength_from_ego.empty()) {
    return ret;
  }

  const auto remaining_distance =
    utils::isAllowedGoalModification(data_->route_handler)
      ? data.to_return_point
      : std::min(
          arclength_from_ego.back() - parameters_->dead_line_buffer_for_goal, data.to_return_point);

  // If the avoidance point has already been set, the return shift must be set after the point.
  const auto last_sl_distance = data.arclength_from_ego.at(last_sl.end_idx);

  // check if there is enough distance for return.
  if (last_sl_distance > remaining_distance) {  // tmp: add some small number (+1.0)
    RCLCPP_DEBUG(rclcpp::get_logger(""), "No enough distance for return.");
    nero_return_log(
      "no room for the return: last line ends " + std::to_string(last_sl_distance).substr(0, 6) +
        " m, remaining " + std::to_string(remaining_distance).substr(0, 6) + " m",
      data, helper_->getEgoShift(), base_offset_, last_);
    return ret;
  }

  // If the remaining distance is not enough, the return shift needs to be shrunk.
  // (or another option is just to ignore the return-shift.)
  // But we do not want to change the last shift point, so we will shrink the distance after
  // the last shift point.
  //
  //  The line "===" is fixed, "---" is scaled.
  //
  // [Before Scaling]
  //  ego              last_sl_end             prepare_end            path_end    avoid_end
  // ==o====================o----------------------o----------------------o------------o
  //   |            prepare_dist                   |          avoid_dist               |
  //
  // [After Scaling]
  // ==o====================o------------------o--------------------------o
  //   |        prepare_dist_scaled            |    avoid_dist_scaled     |
  //
  const double variable_prepare_distance =
    std::max(nominal_prepare_distance - last_sl_distance, 0.0);

  double prepare_distance_scaled = std::max(
    helper_->getNominalPrepareDistance(), std::max(nominal_prepare_distance, last_sl_distance));
  double avoid_distance_scaled = nominal_avoid_distance;
  if (remaining_distance < prepare_distance_scaled + avoid_distance_scaled) {
    const auto scale = (remaining_distance - last_sl_distance) /
                       std::max(nominal_avoid_distance + variable_prepare_distance, 0.1);
    prepare_distance_scaled = std::max(
      helper_->getNominalPrepareDistance(), last_sl_distance + scale * nominal_prepare_distance);
    avoid_distance_scaled *= scale;
  }

  // NERO_SOA_RETURN_NO_NEED: the return starts only where it clears every standing object it would come closer to,
  // and is over before an object that blocks staying out (sharper if needed, at most the module's sharpest)
  bool nero_moved_start = false;
  if (nero_return_no_need()) {
    std::string who, who_end;
    const auto window = nero_return_window(
      data, data_, parameters_, last_sl.end_shift_length, prepare_distance_scaled, avoid_distance_scaled,
      remaining_distance, who, who_end);
    if (!window) {
      nero_return_log(
        "return would touch standing object " + who + ": not before the dead line", data,
        helper_->getEgoShift(), base_offset_, last_);
      return ret;
    }
    if (window->first > prepare_distance_scaled + 1e-3) {
      nero_return_log(
        "return start moved " + std::to_string(prepare_distance_scaled).substr(0, 5) + " -> " +
          std::to_string(window->first).substr(0, 5) + " m past standing object " + who,
        data, helper_->getEgoShift(), base_offset_, last_);
      prepare_distance_scaled = window->first;
      nero_moved_start = true;
    }
    if (prepare_distance_scaled + avoid_distance_scaled > window->second) {
      const double sharp = helper_->getSharpAvoidanceDistance(std::abs(last_sl.end_shift_length));
      const double fit = std::max(sharp, window->second - prepare_distance_scaled);
      nero_return_log(
        "return over " + std::to_string(avoid_distance_scaled).substr(0, 5) + " -> " + std::to_string(fit).substr(0, 5) +
          " m to end before " + who_end + " (in the way of staying out" +
          (fit > window->second - prepare_distance_scaled ? ", the sharpest does not fit)" : ")"),
        data, helper_->getEgoShift(), base_offset_, last_);
      avoid_distance_scaled = std::min(avoid_distance_scaled, fit);
    }
  }

  // shift point for prepare distance: from last shift to return-start point.
  if (nominal_prepare_distance > last_sl_distance || nero_moved_start) {
    AvoidLine al;
    al.id = generate_uuid();
    al.start_idx = last_sl.end_idx;
    al.start = last_sl.end;
    al.start_longitudinal = arclength_from_ego.at(al.start_idx);
    al.end_idx = utils::static_obstacle_avoidance::findPathIndexFromArclength(
      arclength_from_ego, prepare_distance_scaled);
    al.end = data.reference_path.points.at(al.end_idx).point.pose;
    al.end_longitudinal = prepare_distance_scaled;
    al.end_shift_length = last_sl.end_shift_length;
    al.start_shift_length = last_sl.end_shift_length;
    ret.push_back(al);
    debug.step1_return_shift_line.push_back(al);
  }

  // shift point for return to center line
  {
    AvoidLine al;
    al.id = generate_uuid();
    al.start_idx = utils::static_obstacle_avoidance::findPathIndexFromArclength(
      arclength_from_ego, prepare_distance_scaled);
    al.start = data.reference_path.points.at(al.start_idx).point.pose;
    al.start_longitudinal = prepare_distance_scaled;
    al.end_idx = utils::static_obstacle_avoidance::findPathIndexFromArclength(
      arclength_from_ego, prepare_distance_scaled + avoid_distance_scaled);
    al.end = data.reference_path.points.at(al.end_idx).point.pose;
    al.end_longitudinal = arclength_from_ego.at(al.end_idx);
    al.end_shift_length = 0.0;
    al.start_shift_length = last_sl.end_shift_length;
    ret.push_back(al);
    debug.step1_return_shift_line.push_back(al);
  }

  nero_return_log(
    "return added from " + std::to_string(last_sl.end_shift_length).substr(0, 5) + " at " +
      std::to_string(prepare_distance_scaled).substr(0, 5) + " m over " +
      std::to_string(avoid_distance_scaled).substr(0, 5) + " m",
    data, helper_->getEgoShift(), base_offset_, last_);
  return ret;
}

AvoidLineArray ShiftLineGenerator::findNewShiftLine(
  const AvoidLineArray & shift_lines, DebugData & debug) const
{
  if (shift_lines.empty()) {
    return {};
  }

  // add small shift lines.
  const auto add_straight_shift =
    [&, this](auto & subsequent, bool has_large_shift, const size_t start_idx) {
      for (size_t i = start_idx; i < shift_lines.size(); ++i) {
        if (
          std::abs(shift_lines.at(i).getRelativeLength()) >
          parameters_->lateral_small_shift_threshold) {
          if (has_large_shift) {
            return;
          }

          has_large_shift = true;
        }

        if (!helper_->isComfortable(AvoidLineArray{shift_lines.at(i)})) {
          return;
        }

        subsequent.push_back(shift_lines.at(i));
      }
    };

  // get subsequent shift lines.
  const auto get_subsequent_shift = [&, this](size_t i) {
    AvoidLineArray subsequent{shift_lines.at(i)};

    if (!helper_->isComfortable(subsequent)) {
      return subsequent;
    }

    if (shift_lines.size() == i + 1) {
      return subsequent;
    }

    if (!helper_->isComfortable(AvoidLineArray{shift_lines.at(i + 1)})) {
      return subsequent;
    }

    if (
      std::abs(shift_lines.at(i).getRelativeLength()) <
      parameters_->lateral_small_shift_threshold) {
      const auto has_large_shift =
        shift_lines.at(i + 1).getRelativeLength() > parameters_->lateral_small_shift_threshold;

      // candidate.at(i) is small length shift line. add large length shift line.
      subsequent.push_back(shift_lines.at(i + 1));
      add_straight_shift(subsequent, has_large_shift, i + 2);
    } else {
      // candidate.at(i) is large length shift line. add small length shift lines.
      add_straight_shift(subsequent, true, i + 1);
    }

    return subsequent;
  };

  // check ignore or not.
  const auto is_ignore_shift = [this](const auto & s) {
    return std::abs(helper_->getRelativeShiftToPath(s)) < parameters_->lateral_execution_threshold;
  };

  for (size_t i = 0; i < shift_lines.size(); ++i) {
    const auto & candidate = shift_lines.at(i);

    // prevent sudden steering.
    if (!helper_->isEnoughPrepareDistance(candidate.start_longitudinal)) {
      break;
    }

    if (is_ignore_shift(candidate)) {
      continue;
    }

    if (perManeuver(parameters_->policy_approval)) {
      debug.step4_new_shift_line = shift_lines;
      return shift_lines;
    }

    const auto new_shift_lines = get_subsequent_shift(i);
    debug.step4_new_shift_line = new_shift_lines;
    return new_shift_lines;
  }

  return {};
}

void ShiftLineGenerator::updateRegisteredRawShiftLines(const AvoidancePlanningData & data)
{
  utils::static_obstacle_avoidance::fillAdditionalInfoFromPoint(data, raw_registered_);

  AvoidLineArray avoid_lines;

  const auto has_large_offset = [this](const auto & s) {
    constexpr double THRESHOLD = 0.1;
    const auto ego_shift_length = helper_->getEgoLinearShift();

    const auto start_to_ego_longitudinal = -1.0 * s.start_longitudinal;

    if (start_to_ego_longitudinal < 0.0) {
      return false;
    }

    const auto reg_shift_length =
      s.getGradient() * start_to_ego_longitudinal + s.start_shift_length;

    return std::abs(ego_shift_length - reg_shift_length) > THRESHOLD;
  };

  const auto ego_idx = data.ego_closest_path_index;

  for (const auto & s : raw_registered_) {
    // invalid
    if (s.end_idx < ego_idx) {
      continue;
    }

    // invalid
    if (has_large_offset(s)) {
      continue;
    }

    // valid
    avoid_lines.push_back(s);
  }

  raw_registered_ = avoid_lines;
}

void ShiftLineGenerator::setRawRegisteredShiftLine(
  const AvoidLineArray & shift_lines, const AvoidancePlanningData & data)
{
  if (shift_lines.empty()) {
    RCLCPP_ERROR(rclcpp::get_logger(""), "future is empty! return.");
    return;
  }

  auto future_with_info = shift_lines;
  utils::static_obstacle_avoidance::fillAdditionalInfoFromPoint(data, future_with_info);

  // sort by longitudinal
  std::sort(future_with_info.begin(), future_with_info.end(), [](auto a, auto b) {
    return a.end_longitudinal < b.end_longitudinal;
  });

  // calc relative lateral length
  future_with_info.front().start_shift_length = base_offset_;
  for (size_t i = 1; i < future_with_info.size(); ++i) {
    future_with_info.at(i).start_shift_length = future_with_info.at(i - 1).end_shift_length;
  }

  const auto is_registered = [this](const auto id) {
    const auto & r = raw_registered_;
    return std::any_of(r.begin(), r.end(), [id](const auto & s) { return s.id == id; });
  };

  const auto same_id_shift_line = [this](const auto id) {
    const auto & r = raw_;
    const auto itr = std::find_if(r.begin(), r.end(), [id](const auto & s) { return s.id == id; });
    if (itr != r.end()) {
      return *itr;
    }
    throw std::logic_error("not found same id current raw shift line.");
  };

  for (const auto & s : future_with_info) {
    if (s.parent_ids.empty()) {
      RCLCPP_ERROR(rclcpp::get_logger(""), "avoid line for path_shifter must have parent_id.");
    }

    for (const auto id : s.parent_ids) {
      if (is_registered(id)) {
        continue;
      }

      raw_registered_.push_back(same_id_shift_line(id));
    }
  }
}
}  // namespace autoware::behavior_path_planner::utils::static_obstacle_avoidance
