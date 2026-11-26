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
#include "autoware/behavior_path_planner_common/utils/nero_mu.hpp"  // NERO_BPP_PROJ_FAST

#include "autoware/behavior_path_planner_common/utils/utils.hpp"
#include "autoware/behavior_path_static_obstacle_avoidance_module/utils.hpp"

#include "nero_proj.hpp"

#include <magic_enum.hpp>
#include <tf2/utils.h>

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace autoware::behavior_path_planner::utils::static_obstacle_avoidance
{
namespace nero_frag
{
bool keepApprovedShortcut(
  const ObjectData & o, const double approved_shift, const double vehicle_width,
  const std::shared_ptr<AvoidanceParameters> & parameters);
double planNeed(
  const ObjectData & o, const double vehicle_width, const std::shared_ptr<AvoidanceParameters> & parameters);
}  // namespace nero_frag

namespace nero_pipe
{
// NERO_SOA_PIPE_DEBUG=1 (ownlane 2026-11-25, LOG ONLY): every 0.5 s the shift-line pipeline's stages as
// [start>end from>to id] (metres from ego along the path, shift lengths, the line's object id): the outlines per target after
// the outline merge, the raw lines after the pre-process (registered lines combined), the candidate after the total-line merge
// and the trim, and the new lines. Unset = nothing logged.
bool on()
{
  static const bool v = [] {
    const char * e = std::getenv("NERO_SOA_PIPE_DEBUG");
    return e != nullptr && std::atoi(e) > 0;
  }();
  return v;
}
bool due(const char * stage)
{
  static std::unordered_map<std::string, std::chrono::steady_clock::time_point> last;
  const auto now = std::chrono::steady_clock::now();
  auto it = last.find(stage);
  if (it != last.end() && now - it->second < std::chrono::milliseconds(480)) return false;
  last[stage] = now;
  return true;
}
std::string lines(const AvoidLineArray & a)
{
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(2);
  for (const auto & l : a) {
    const auto & u = l.object.object.object_id.uuid;
    ss << " [" << l.start_longitudinal << ">" << l.end_longitudinal << " " << l.start_shift_length << ">" << l.end_shift_length
       << " " << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(u[0]) << std::setw(2)
       << static_cast<int>(u[1]) << std::dec << std::setfill(' ') << "]";
  }
  return ss.str();
}
void log(const char * stage, const std::string & body)
{
  RCLCPP_INFO(rclcpp::get_logger("static_obstacle_avoidance"), "[NERO_SOA_PIPE] %s:%s", stage, body.c_str());
}
}  // namespace nero_pipe

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
          proj ? proj->nearestSegment(p) : nero_mu::findNearestSegmentIndex(points, p);
        const double s = data.arclength_from_ego.at(seg) +
                         (proj ? proj->lonToSegment(seg, p)
                               : nero_mu::calcLongitudinalOffsetToSegment(points, seg, p));
        if (s < -rear - LON_MARGIN || s > limit + front) continue;
        pts.push_back(
          {s, proj ? proj->lateral(p, seg) : nero_mu::calcLateralOffset(points, p, seg),
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
  neroReplanUpdate(data);  // NERO_SOA_REPLAN_HORIZON: split the registered lines into committed / re-plannable

  /**
   * STEP2: Generate avoid outlines.
   * Basically, avoid outlines are generated per target objects.
   */
  const auto outlines = generateAvoidOutline(data, debug);
  neroReplanRaw(outlines, data);  // NERO_SOA_REPLAN_HORIZON: a target's smaller need replaces its not-started registered lines
  if (nero_ra::cfg().on) {
    // NERO_SOA_REPLAN_ANCHOR: the memory of a smaller / vanished need is dropped, started or not. A target's memory is compared with
    // this cycle's lines in the form the memory has them (outlines merged and filled: a merged line belongs to the first of its two
    // targets and carries the second one's level)
    DebugData scratch;
    neroAnchorMemory(toArray(applyFillGapProcess(applyMergeProcess(outlines, data, scratch), data, scratch)), data);
  }

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
    // NERO_SOA_AMBIGUOUS_FRAGMENT (utils.cpp nero_frag): a vehicle first avoided as its fragment is "approved" only
    // when the approved shift reaches what the vehicle needs
    if (
      is_approved && nero_frag::keepApprovedShortcut(
                       object, helper_->getShift(object.getPosition()),
                       data_->parameters.vehicle_width, parameters_)) {
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

  // NERO_SOA_FLOOR_SPAN=1 (ownlane-2 2026-11-25, QUEUE OWNLANE-1122; Ahmed 11-21 "it has to return to its lane after the
  // overtake"): the #11302 floor (NERO_SOA_FIX_11302) and every avoid line's start read the registered lines
  // (getLinearShift) only where registered lines exist. Beyond the furthest registered line's end the path shifter just
  // HOLDS that line's end level, because the return of the car that needed it is not registered yet; the floor read that
  // hold as if it were a line (route_b2 93.4-94.6 s: 078e's avoid line ends 13.9 m ahead at 1.40 m, 6ea3's line end lies
  // 36 m ahead -> "6ea3 floored 1.40 (needs 0.37)", the next floor read 6ea3's raised line, held ~12 s over the lane line).
  // There it reads a nominal return from the held level (getMaxAvoidanceDistance of it, from the registered end, never
  // behind ego). The earlier car's own outline (plateau + return) is still max-merged by the generator, so its margin
  // is kept. Within the registered lines nothing changes (24 Nov 13:01: a4d5's registered return). Unset = the car.
  static const bool nero_floor_span = [] {
    const char * v = std::getenv("NERO_SOA_FLOOR_SPAN");
    return v != nullptr && std::atoi(v) > 0;
  }();
  double nero_span_end = std::numeric_limits<double>::infinity();
  double nero_span_hold = 0.0;
  double nero_span_len = 0.0;
  // NERO_SOA_REPLAN_HORIZON: with re-plannable lines registered, "the registered lines" are the committed ones only
  const auto & nero_span_line = nero_rh_active_ ? nero_rh_furthest_ : nero_furthest_;
  if (nero_floor_span && nero_span_line.has_value() && !data.reference_path.points.empty()) {
    const auto idx =
      autoware::motion_utils::findNearestIndex(data.reference_path.points, nero_span_line.value().end.position);
    nero_span_end = std::max(data.arclength_from_ego.at(idx), 0.0);
    nero_span_hold = neroRegShift(nero_span_line.value().end.position);
    nero_span_len = helper_->getMaxAvoidanceDistance(std::abs(nero_span_hold));
  }
  // target objects are sorted by longitudinal distance.
  AvoidOutlines outlines;

  // NERO_SOA_REPLAN_HORIZON: beyond the committed lines, "the registered path" a floor or a line start reads is what the dropped lines
  // are re-proposed as: the current outlines of the targets before this one (never this target's own earlier line). Reading the
  // committed lines alone lost a nearer target's still valid return (24 Nov 13:01 sim 1125-203315: the row read a4d5's held 1.90
  // instead of its return, lines went 1.58 -> 1.75 -> 0.90 and the blinker flickered)
  const auto nero_prior_shift = [&outlines, this](const double lon, double & out) {
    const auto lerp = [lon](const AvoidLine & l) {
      const double len = l.end_longitudinal - l.start_longitudinal;
      if (len < 1.0e-3) return l.end_shift_length;
      return l.start_shift_length + (l.end_shift_length - l.start_shift_length) * (lon - l.start_longitudinal) / len;
    };
    bool found = false;
    for (const auto & ol : outlines) {
      const auto & a = ol.avoid_line;
      // only a target whose avoidance is under way (a registered raw line of it starts within the horizon): the robot is at its level
      // and can leave it only along its return. A target ahead whose lines have not started floors nothing (mk_great sim 1125-204105:
      // pedestrian 70c7's outline floored 0fc8 at 1.65 m before anything for 70c7 had started -> out to 1.68 m)
      const auto & u = a.object.object.object_id.uuid;
      const std::string key(u.begin(), u.end());
      // NERO_SOA_REPLAN_ANCHOR: every nearer target's current outline counts (a level that is not needed any more is re-planned,
      // started or not, so nothing sticks)
      if (
        !nero_ra::cfg().on &&
        std::find(nero_rh_started_.begin(), nero_rh_started_.end(), key) == nero_rh_started_.end()) {
        continue;
      }
      const double last = ol.return_line ? ol.return_line.value().end_longitudinal : std::numeric_limits<double>::max();
      if (lon < a.start_longitudinal || lon > last) continue;
      double v = a.end_shift_length;
      if (lon < a.end_longitudinal) {
        v = lerp(a);
      } else if (ol.return_line && lon >= ol.return_line.value().start_longitudinal) {
        v = lerp(ol.return_line.value());
      } else {
        for (const auto & m : ol.middle_lines) {
          if (lon < m.start_longitudinal) break;
          v = lon < m.end_longitudinal ? lerp(m) : m.end_shift_length;
        }
      }
      if (!found || std::abs(v) > std::abs(out)) {
        out = v;
        found = true;
      }
    }
    return found;
  };
  // NERO_SOA_REPLAN_ANCHOR: nothing floors a line (fb8). A target's line ends at its own need and starts from the robot's level
  // (below). What the robot cannot do is not written into the levels: the levels of the targets it is beside / past stay in the
  // memory (their plateau and return are max-merged as ever), and a line back toward the centre that is sharper than the jerk limit
  // is lengthened where the plan is joined to the robot's state (scene.cpp neroAnchorResolve: the sharpest comfortable return from
  // the robot's shift and rate). fb7 floored every line on the nominal return from ego's shift: slower than the robot can come
  // back, and moving with ego every cycle (n24 sim 1125-215337: "8fde needs 0.74, floor 1.41", the row 8.6 s over the line, a
  // re-plan every 0.7 s).
  const double nero_ra_ego = nero_ra::cfg().on ? helper_->getEgoLinearShift() : 0.0;
  const auto nero_registered_shift = [&](const double lon, const geometry_msgs::msg::Point & p) {
    if (nero_ra::cfg().on) {
      return 0.0;
    }
    double reg = neroRegShift(p);  // NERO_SOA_REPLAN_HORIZON (unset: helper_->getLinearShift(p))
    if (
      nero_floor_span && lon > nero_span_end + 0.5 && std::abs(nero_span_hold) >= 0.01 && nero_span_len >= 0.5) {
      reg = nero_span_hold * std::max(0.0, 1.0 - (lon - nero_span_end) / nero_span_len);
    }
    double prior = 0.0;
    if (nero_rh_active_ && nero_prior_shift(lon, prior) && std::abs(prior) > std::abs(reg)) {
      return prior;
    }
    return reg;
  };
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

    double desire_shift_length = helper_->getShiftLength(o, isOnRight(o), o.avoid_margin.value());
    // NERO_SOA_FRAG_PLAN=1 (ownlane-2 2026-11-25, QUEUE SOAFRAG-SIM; 24 Nov 13:01:57 "it stopped" at c4df): a fragment-carried vehicle is
    // planned to what NERO_SOA_AMBIGUOUS_FRAGMENT checks it against (hard + approved_tol_m), rounded UP to the module's quantize grid. The
    // module planned c4df to its own margin (hard only, no soft room: 0.71) and rounded to the nearest 0.1 m (0.92 -> 0.90), the fragment
    // rule asked hard + 0.1 (0.81 + overhang) and, as the angled car's box grew 0.83 -> 1.04 m beside the robot, refused 0.83-0.85 m every
    // cycle: slow-downs, stands of 2-30 s (sims 1125-171307, -180735, -182501), the field's wait-stop -1.77. The hard margin and the check
    // are unchanged; the line is only higher (the body up to one grid step further out at such a car). Unset = the car.
    static const bool nero_frag_plan = [] {
      const char * v = std::getenv("NERO_SOA_FRAG_PLAN");
      return v != nullptr && std::atoi(v) > 0;
    }();
    if (nero_frag_plan) {
      const double need = nero_frag::planNeed(o, data_->parameters.vehicle_width, parameters_);
      const double q = std::max(parameters_->quantize_size, 1.0e-3);
      const double planned = std::copysign(std::ceil(std::abs(need) / q - 1.0e-6) * q, need);
      if (need * desire_shift_length > 0.0 && std::abs(planned) > std::abs(desire_shift_length) + 1.0e-3) {
        static rclcpp::Clock fp_clock(RCL_STEADY_TIME);
        RCLCPP_INFO_THROTTLE(
          rclcpp::get_logger("static_obstacle_avoidance"), fp_clock, 500,
          "[NERO_SOA_FRAG_PLAN] %02x%02x: carried vehicle planned to %.2f (the fragment rule's %.2f on the grid; own margin %.2f)",
          o.object.object_id.uuid[0], o.object.object_id.uuid[1], planned, need, desire_shift_length);
        desire_shift_length = planned;
      }
    }
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
    // NERO_SOA_AMBIGUOUS_FRAGMENT: as in get_shift_profile (a vehicle first avoided as its fragment)
    const auto approved_covers = [&]() {
      return is_approved(o) &&
             nero_frag::keepApprovedShortcut(
               o, helper_->getShift(o.getPosition()), data_->parameters.vehicle_width, parameters_);
    };
    if (!feasible_shift_profile.has_value()) {
      if (approved_covers()) {
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
      const double nero_reg_start = al_avoid.start_shift_length;
      al_avoid.start_shift_length =
        nero_registered_shift(al_avoid.start_longitudinal, al_avoid.start.position);  // NERO_SOA_FLOOR_SPAN

      // end point
      // NERO_SOA_FIX_11302=1: port of upstream #11302: a best-effort (feasible) end shift never moves ego back toward
      // the object: the avoid line ends at least at the shift the path already has at its end point (the larger shift
      // away from the object), and the return line starts from that shift. Unset = the car.
      static const bool fix_11302 = [] {
        const char * v = std::getenv("NERO_SOA_FIX_11302");
        return v != nullptr && std::atoi(v) > 0;
      }();
      if (fix_11302) {
        const auto end_idx = utils::static_obstacle_avoidance::findPathIndexFromArclength(
          data.arclength_from_ego, to_shift_end);
        const auto end = data.reference_path.points.at(end_idx).point.pose;
        const double nero_floor = nero_registered_shift(to_shift_end, end.position);  // NERO_SOA_FLOOR_SPAN
        if (utils::static_obstacle_avoidance::isOnRight(o)) {
          al_avoid.end_shift_length = std::max(feasible_shift_profile.value().first, nero_floor);
        } else {
          al_avoid.end_shift_length = std::min(feasible_shift_profile.value().first, nero_floor);
        }
        if (nero_ra::cfg().on) {
          const double reg_end = helper_->getLinearShift(end.position);
          if (std::abs(reg_end) > std::abs(feasible_shift_profile.value().first) + 0.05) {
            static rclcpp::Clock ra_clock(RCL_STEADY_TIME);
            RCLCPP_INFO_THROTTLE(
              rclcpp::get_logger("static_obstacle_avoidance"), ra_clock, 500,
              "[NERO_SOA_ANCHOR_FLOOR] %02x%02x: line %.1f>%.1f m ends at its need %.2f (the registered path there: %.2f, not a floor)",
              o.object.object_id.uuid[0], o.object.object_id.uuid[1], al_avoid.start_longitudinal, to_shift_end,
              al_avoid.end_shift_length, reg_end);
          }
        } else if (nero_floor_span) {
          const double reg_end = helper_->getLinearShift(end.position);
          if (std::abs(reg_end - nero_floor) > 0.01 || std::abs(nero_reg_start - al_avoid.start_shift_length) > 0.01) {
            static rclcpp::Clock fs_clock(RCL_STEADY_TIME);
            RCLCPP_INFO_THROTTLE(
              rclcpp::get_logger("static_obstacle_avoidance"), fs_clock, 500,
              "[NERO_SOA_FLOOR_SPAN] %02x%02x: line %.1f>%.1f m lies beyond the registered lines (end %.1f m, held %.2f): "
              "start %.2f (hold %.2f), floor %.2f (hold %.2f), needs %.2f -> end %.2f",
              o.object.object_id.uuid[0], o.object.object_id.uuid[1], al_avoid.start_longitudinal, to_shift_end,
              nero_span_end, nero_span_hold, al_avoid.start_shift_length, nero_reg_start, nero_floor, reg_end,
              feasible_shift_profile.value().first, al_avoid.end_shift_length);
          }
        }
      } else {
        al_avoid.end_shift_length = feasible_shift_profile.value().first;
      }
      al_avoid.end_longitudinal = to_shift_end;
      if (nero_ra::cfg().on) {
        // NERO_SOA_REPLAN_ANCHOR: the line starts where the robot's state lets the plan be at its start: the robot's own level held
        // when the line rises or holds (never a dip down the nominal return first), the floor when it has to come down
        const double floor_start = al_avoid.start_shift_length;
        const double end_level = al_avoid.end_shift_length;
        al_avoid.start_shift_length = utils::static_obstacle_avoidance::isOnRight(o)
                                        ? std::max(floor_start, std::min(nero_ra_ego, end_level))
                                        : std::min(floor_start, std::max(nero_ra_ego, end_level));
      }

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
      al_return.start_shift_length = al_avoid.end_shift_length;  // NERO_SOA_FIX_11302 (unset: the same value as before)
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
      const double goal_longitudinal_distance = nero_mu::calcSignedArcLength(
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
      // NERO_SOA_STEP_DOWN=3 (ownlane 2026-11-25, QUEUE OWNLANE-1122, Ahmed 11-21 "it has to return to its lane after the
      // overtake"): when #11302's floor (NERO_SOA_FIX_11302) lifted this target's line end above its own need (it ends on an
      // earlier, bigger target's return: 24 Nov 13:01 8fde needs 0.74 m, floored at 1.67 m on a4d5's return), the outline descends
      // from that floor to the target's own level along the nominal return length for the difference, starting at the target's
      // front (never below its own need while beside it), and returns to the centre from its own level. Without this the floored
      // level was the outline's plateau AND its return's start, the next car's floor read that registered line, and the robot held
      // 1.70 m (body ~1 m over the centre line) through a whole row of cars that each fit in the lane. Everything stays on or under
      // the registered ramp (no line conflicts). Any other value / unset = the car.
      static const int nero_step_down_mode = [] {
        const char * v = std::getenv("NERO_SOA_STEP_DOWN");
        return v != nullptr ? std::atoi(v) : 0;
      }();
      bool split = false;
      if (
        nero_step_down_mode == 3 && al_avoid.end_shift_length * desire_shift_length > 0.0 &&
        std::abs(al_avoid.end_shift_length) > std::abs(desire_shift_length) + 0.05 &&
        std::abs(feasible_shift_profile.value().first - desire_shift_length) < 1.0e-3) {
        const double floor_s = al_avoid.end_shift_length;
        const double full_len = helper_->getMaxAvoidanceDistance(std::abs(floor_s - desire_shift_length));
        const double room = al_return.start_longitudinal - al_avoid.end_longitudinal;
        if (room > 0.5 && full_len > 0.5) {
          const double len = std::min(full_len, room);
          AvoidLine m1 = al_avoid;
          m1.start_longitudinal = al_avoid.end_longitudinal;
          m1.start_shift_length = floor_s;
          m1.end_longitudinal = al_avoid.end_longitudinal + len;
          m1.end_shift_length = floor_s + (desire_shift_length - floor_s) * (len / full_len);
          m1.id = generate_uuid();
          al_return.start_shift_length = m1.end_shift_length;
          al_return.end_longitudinal =
            al_return.start_longitudinal + helper_->getMaxAvoidanceDistance(std::abs(m1.end_shift_length));
          AvoidOutline ol(al_avoid, al_return);
          ol.middle_lines.push_back(m1);
          outlines.push_back(ol);
          split = true;
          static rclcpp::Clock sd_clock(RCL_STEADY_TIME);
          RCLCPP_INFO_THROTTLE(
            rclcpp::get_logger("static_obstacle_avoidance"), sd_clock, 500,
            "[NERO_SOA_STEP_DOWN] %02x%02x: floored %.2f (needs %.2f) -> down to %.2f over %.1f m beside it, return from there",
            o.object.object_id.uuid[0], o.object.object_id.uuid[1], floor_s, desire_shift_length, m1.end_shift_length, len);
        }
      }
      if (!split) {
        outlines.emplace_back(al_avoid, al_return);
      }
    } else if (!approved_covers()) {
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
  if (nero_pipe::on() && nero_pipe::due("outlines")) {
    std::string b;
    for (const auto & ol : processed_outlines) {
      b += " {A" + nero_pipe::lines(AvoidLineArray{ol.avoid_line}) + " M" + nero_pipe::lines(ol.middle_lines) + " R" +
           (ol.return_line ? nero_pipe::lines(AvoidLineArray{ol.return_line.value()}) : std::string(" -")) + "}";
    }
    nero_pipe::log("outlines", b);
  }

  /**
   * Step3: Convert to AvoidLineArray from AvoidOutlines.
   */
  AvoidLineArray processed_raw_lines = toArray(processed_outlines);

  /**
   * Step4: Combine process.
   * Use all registered points. For the current points, if the similar one of the current
   * points are already registered, will not use it.
   */
  if (nero_pipe::on() && nero_pipe::due("registered")) {
    nero_pipe::log("registered", nero_pipe::lines(raw_registered_));
  }
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
  if (nero_pipe::on() && nero_pipe::due("raw")) {
    nero_pipe::log("raw", nero_pipe::lines(shift_lines));
    nero_pipe::log("merged", nero_pipe::lines(processed_shift_lines));
  }
  if (nero_ra::cfg().on) {
    return neroAnchorCandidates(processed_shift_lines, data, debug);  // NERO_SOA_REPLAN_ANCHOR
  }
  processed_shift_lines = applyTrimProcess(processed_shift_lines, debug);
  if (nero_pipe::on() && nero_pipe::due("trimmed")) {
    nero_pipe::log("trimmed", nero_pipe::lines(processed_shift_lines));
  }

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
      auto i_shift = utils::static_obstacle_avoidance::lerpShiftLengthOnArc(arcs.at(i), al);
      // NERO_SOA_REPLAN_ANCHOR fb18: a line that ends exactly on a path point has its end level there (stock reads 0 at a
      // line's end, "arc < end"). Stock raw lines never end on a path point; the anchor's memory chords do, and so do all lines
      // while the robot stands (the path does not move): c02 sim 1126-105036, 62.99 s, the chord [0.93>2.13 0.20>1.50] read 0
      // at 2.13 m, the merged needs there fell to 1.18 from the next raw line, "the registered path is at 1.50 at 2.13 m, the
      // merged needs at 1.18", and the level hunted 1.50 -> 1.20 -> 1.00 -> 0.70 in 0.3 s at v = 0
      if (
        nero_ra::cfg().on && al.end_longitudinal > al.start_longitudinal + 1.0e-3 &&
        std::abs(arcs.at(i) - al.end_longitudinal) < 1.0e-3) {
        i_shift = al.end_shift_length;
      }

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
    sl.shift_line.at(i) = neroRegShift(get_point(path.points.at(i)));  // NERO_SOA_REPLAN_HORIZON (unset: getLinearShift)
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
  if (nero_ra::cfg().on) {
    // NERO_SOA_REPLAN_ANCHOR fb16: a target's current line that needs MORE than its similar registered line (same object, start
    // and end within 1 m, end shift within 0.5 m: the stock unique check) is not dropped: the larger current need joins the merge
    // (max of the lines). Stock drops it, so a need that grows by up to 0.5 m at the same place never reaches the plan: n24 sim
    // 1125-223906, 27.23 s, the carried vehicle c4df planned 0.74 -> 0.90 (NERO_SOA_FRAG_PLAN), its plateau [21.75>28.12 0.90]
    // dropped as similar to the registered [21.82>28.12 0.74]; 31.23 s, 1.00 [8.25>14.85] dropped against [9.04>15.65 0.65>0.70]:
    // the plan beside c4df stayed under its need, the fragment rule stopped the robot ("the approved shift at it 0.69 m does
    // not keep the hard margin"). A current line that needs the same or less is still dropped (the stock memory).
    const double more = std::max(parameters_->lateral_execution_threshold, 0.5 * parameters_->quantize_size);
    const auto similar = [](const AvoidLine & a, const AvoidLine & b) {
      return autoware_utils::calc_distance2d(a.start, b.start) <= 1.0 &&
             autoware_utils::calc_distance2d(a.end, b.end) <= 1.0 &&
             std::abs(a.end_shift_length - b.end_shift_length) <= 0.5;
    };
    auto combined = registered_lines;
    for (const auto & added : shift_lines) {
      bool skip = false;
      for (const auto & base : registered_lines) {
        if (added.object.object.object_id == base.object.object.object_id && similar(added, base)) {
          const bool larger = added.end_shift_length * base.end_shift_length >= 0.0 &&
                              std::abs(added.end_shift_length) >= std::abs(base.end_shift_length) + more - 1.0e-3;
          if (!larger) {
            skip = true;
            break;
          }
        }
      }
      if (!skip) combined.push_back(added);
    }
    return combined;
  }
  return utils::static_obstacle_avoidance::combineRawShiftLinesWithUniqueCheck(
    registered_lines, shift_lines);
}

AvoidLineArray ShiftLineGenerator::applyMergeProcess(
  const AvoidLineArray & shift_lines, const AvoidancePlanningData & data, DebugData & debug) const
{
  // Generate shift line by merging shift_lines.
  ShiftLineData shift_line_data;
  generateTotalShiftLine(shift_lines, data, shift_line_data);
  if (nero_ra::cfg().on) {
    neroAnchorHold(data, shift_line_data);  // NERO_SOA_REPLAN_ANCHOR fb16: WEAVE (b), a dip between two close needs is held
  }

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

    // NERO_SOA_REPLAN_ANCHOR: the fresh plan is complete from ego (the anchor joins it to the robot's state)
    if (!nero_ra_keep_near_ && !helper_->isEnoughPrepareDistance(s.start_longitudinal)) {
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
      // NERO_SOA_REPLAN_ANCHOR: the line from ego (what the robot is on now: the rest of a plateau beside a car, the fill before the
      // first line) is never combined with the lines after it: its end is where the plan may change (combined with a return after
      // it, the plan came back toward the car before the plateau's end)
      if (nero_ra_keep_near_ && i == 1 && input.front().start_longitudinal < 0.5) {
        return true;
      }
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
  // NERO_SOA_REPLAN_ANCHOR: the return is planned from the lines of this cycle (current outlines + the memory that is still needed) and
  // from the robot's own shift, not from the registered lines (a registered level that nothing needs any more was re-proposed here as
  // "the last registered line" and its return)
  const std::optional<ShiftLine> last_ = nero_ra::cfg().on ? std::optional<ShiftLine>{} : this->last_;
  const double base_offset_ = nero_ra::cfg().on ? helper_->getEgoLinearShift() : this->base_offset_;
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
    // NERO_SOA_REPLAN_HORIZON: a re-plan (the first line that differs starts beyond the horizon) registers the candidate from its first
    // line beyond the horizon, so every not-started registered line it replaces is removed (addNewShiftLines drops the registered lines
    // that start at / after the new start, and those beyond the horizon that reach across it); the stock new line starting later left a
    // stale line that started before it in place (mk_great 10.2 s: [20.6>40.2 0.50>1.40] kept beside the new [24.5>30.2 0.70>0.00])
    if (nero_rh_active_ && candidate.start_longitudinal > nero_rh_horizon_) {
      size_t j = i;
      while (j > 0 && shift_lines.at(j - 1).start_longitudinal > nero_rh_horizon_) {
        --j;
      }
      if (j < i) {
        AvoidLineArray ext(shift_lines.begin() + static_cast<std::ptrdiff_t>(j), shift_lines.begin() + static_cast<std::ptrdiff_t>(i));
        ext.insert(ext.end(), new_shift_lines.begin(), new_shift_lines.end());
        debug.step4_new_shift_line = ext;
        return ext;
      }
    }
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

// NERO_SOA_REPLAN_HORIZON=<m> (ownlane-3 2026-11-25, QUEUE OWNLANE-1122; Ahmed 11-21 "it should go to the other lane only if there is no
// space on its lane", "it has to return to its lane after the overtake"): registered shift lines were append-only. Every cycle the
// generator max-merged each target's registered raw line (applyCombineProcess), the #11302 floor (NERO_SOA_FIX_11302) and every avoid
// line's start read the registered path (getLinearShift), so a line made for a target's earlier, bigger need was re-carried until the robot
// drove through it (24 Nov 13:01 the row held at 1.70 m; 21 Nov 19:4x route_b2 6ea3 "floored 1.40 (needs 0.37)"; mk_great 35cd 1.46 ->
// 0.74 m with its 1.5 m line kept). With the switch, a registered line is COMMITTED once it starts within ego + the nominal prepare
// distance; the generator reads the registered path made of the committed lines only, and the raw registered lines of a target whose
// manoeuvre has not started (all its lines start beyond the horizon) are replaced by its current outline when that need is at least <m>
// smaller (the hysteresis: a box's 0.1 m jitter never re-plans). The candidate is still compared with the REAL registered path by the stock
// findNewShiftLine, so a re-plan proposes a line only when the difference passes the execution threshold, and the stock addNewShiftLines
// removes the registered lines starting at / after it; lines that started stay, the path at ego never steps. A bigger need is the stock
// path (a new line). Logs [NERO_SOA_REPLAN]. Unset / 0 = the car.
namespace
{
double nero_replan_hyst()
{
  static const double v = [] {
    if (nero_ra::cfg().on) return 0.0;  // NERO_SOA_REPLAN_ANCHOR replaces the horizon
    const char * e = std::getenv("NERO_SOA_REPLAN_HORIZON");
    return e != nullptr ? std::max(0.0, std::atof(e)) : 0.0;
  }();
  return v;
}
std::string nero_uuid_key(const AvoidLine & al)
{
  const auto & u = al.object.object.object_id.uuid;
  return std::string(u.begin(), u.end());
}
bool nero_uuid_zero(const AvoidLine & al)
{
  const auto & u = al.object.object.object_id.uuid;
  return std::all_of(u.begin(), u.end(), [](const auto c) { return c == 0; });
}
}  // namespace

double ShiftLineGenerator::neroRegShift(const geometry_msgs::msg::Point & p) const
{
  if (!nero_rh_active_ || nero_committed_.path.points.empty()) {
    return helper_->getLinearShift(p);
  }
  const auto idx = autoware::motion_utils::findNearestIndex(nero_committed_.path.points, p);
  return nero_committed_.shift_length.at(idx);
}

void ShiftLineGenerator::neroReplanUpdate(const AvoidancePlanningData & data)
{
  nero_rh_active_ = false;
  nero_rh_n_committed_ = 0;
  nero_rh_n_replannable_ = 0;
  nero_rh_furthest_ = std::nullopt;
  nero_rh_started_.clear();
  if (nero_replan_hyst() <= 0.0 || !helper_ || data.reference_path.points.empty()) {
    return;
  }
  nero_rh_horizon_ = helper_->getNominalPrepareDistance();
  nero_rh_started_.clear();
  for (const auto & al : raw_registered_) {
    const auto & u = al.object.object.object_id.uuid;
    if (std::all_of(u.begin(), u.end(), [](const auto c) { return c == 0; })) continue;
    if (al.start_longitudinal <= nero_rh_horizon_) {
      const std::string key(u.begin(), u.end());
      if (std::find(nero_rh_started_.begin(), nero_rh_started_.end(), key) == nero_rh_started_.end()) {
        nero_rh_started_.push_back(key);
      }
    }
  }
  const auto & lines = nero_ps_.getShiftLines();
  ShiftLineArray committed;
  for (const auto & sl : lines) {
    const auto idx = autoware::motion_utils::findNearestIndex(data.reference_path.points, sl.start.position);
    if (data.arclength_from_ego.at(idx) <= nero_rh_horizon_) {
      committed.push_back(sl);
      if (!nero_rh_furthest_ || sl.end_idx > nero_rh_furthest_.value().end_idx) {
        nero_rh_furthest_ = sl;
      }
    } else {
      ++nero_rh_n_replannable_;
    }
  }
  nero_rh_n_committed_ = committed.size();
  nero_rh_committed_end_ = -1.0e9;
  if (nero_rh_furthest_.has_value()) {
    const auto idx =
      autoware::motion_utils::findNearestIndex(data.reference_path.points, nero_rh_furthest_.value().end.position);
    nero_rh_committed_end_ = data.arclength_from_ego.at(idx);
  }
  if (nero_rh_n_replannable_ == 0) {
    return;  // nothing beyond the horizon: the registered path is the committed one
  }
  PathShifter ps = nero_ps_;
  ps.setShiftLines(committed);
  ShiftedPath sp = utils::static_obstacle_avoidance::toShiftedPath(data.reference_path);
  if (!ps.generate(&sp, true, SHIFT_TYPE::LINEAR)) {
    return;  // keep the stock reading
  }
  nero_committed_ = sp;
  nero_rh_active_ = true;
  static rclcpp::Clock rh_clock(RCL_STEADY_TIME);
  if (nero_pipe::on()) {
    RCLCPP_INFO_THROTTLE(
      rclcpp::get_logger("static_obstacle_avoidance"), rh_clock, 500,
      "[NERO_SOA_REPLAN] horizon %.1f m: %zu committed line(s), %zu re-plannable", nero_rh_horizon_, nero_rh_n_committed_,
      nero_rh_n_replannable_);
  }
}

void ShiftLineGenerator::neroReplanRaw(const AvoidOutlines & outlines, const AvoidancePlanningData & data)
{
  const double hyst = nero_replan_hyst();
  if (hyst <= 0.0 || !helper_ || raw_registered_.empty()) {
    return;
  }
  const double horizon = helper_->getNominalPrepareDistance();
  struct Reg
  {
    bool started{false};
    double level{0.0};
    double first_start{std::numeric_limits<double>::max()};
  };
  std::unordered_map<std::string, Reg> reg;
  for (const auto & al : raw_registered_) {
    if (nero_uuid_zero(al)) {
      continue;
    }
    auto & r = reg[nero_uuid_key(al)];
    r.started = r.started || al.start_longitudinal <= horizon;
    r.level = std::max(r.level, std::abs(al.end_shift_length));
    r.first_start = std::min(r.first_start, al.start_longitudinal);
  }
  if (reg.empty()) {
    return;
  }
  const auto current = toArray(outlines);
  std::unordered_map<std::string, std::pair<double, AvoidLineArray>> cur;
  for (const auto & al : current) {
    if (nero_uuid_zero(al)) {
      continue;
    }
    auto & c = cur[nero_uuid_key(al)];
    c.first = std::max(c.first, std::abs(al.end_shift_length));
    c.second.push_back(al);
  }
  for (const auto & [key, r] : reg) {
    // its manoeuvre has begun: one of its raw lines starts within the horizon, or a committed (started) line of the path shifter
    // reaches over its lines (a merged line carries several targets: mk_great s1 13.7 s, cfc0's raw lines started 9.2 m ahead inside
    // the ramp [-2.0>24.4 0.43>1.70] that had started; re-planning it put a lower line inside the started ramp -> a 0.67 m step at ego)
    if (r.started || r.first_start <= nero_rh_committed_end_ + 0.5) {
      continue;
    }
    const auto it = cur.find(key);
    if (it == cur.end() || it->second.first > r.level - hyst) {
      continue;  // not a target this cycle (stock memory), or no smaller need
    }
    raw_registered_.erase(
      std::remove_if(
        raw_registered_.begin(), raw_registered_.end(),
        [&key](const auto & al) { return !nero_uuid_zero(al) && nero_uuid_key(al) == key; }),
      raw_registered_.end());
    for (const auto & al : it->second.second) {
      raw_registered_.push_back(al);
    }
    static rclcpp::Clock rr_clock(RCL_STEADY_TIME);
    RCLCPP_INFO_THROTTLE(
      rclcpp::get_logger("static_obstacle_avoidance"), rr_clock, 200,
      "[NERO_SOA_REPLAN] %02x%02x: registered %.2f (its lines start %.1f m ahead, horizon %.1f m) re-planned to its current need %.2f",
      static_cast<int>(static_cast<unsigned char>(key[0])), static_cast<int>(static_cast<unsigned char>(key[1])), r.level,
      r.first_start, horizon, it->second.first);
  }
  (void)data;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// NERO_SOA_REPLAN_ANCHOR (see shift_line_generator.hpp). The generator's side:
//  1. neroAnchorMemory: the raw registered lines (the generator's memory, max-merged into every cycle's plan) of a target are dropped
//     when its current outline needs at least hyst_m less, started or not, and when the target is still ahead of the robot but has not
//     been a target for absent_s (a pedestrian that flickers in the list for a cycle neither adds nor drops anything: its lines stay
//     through gaps shorter than absent_s). A target the robot is beside / past keeps its plateau and return (no current outline: its
//     avoid line is behind the robot). Lines without a target (returns, fills) that have not started go with them: they are re-made.
//  2. the floor / line start (generateAvoidOutline) and the return (addReturnShiftLine) read the robot's state, not the registered lines.
//  3. neroAnchorCandidates: the fresh plan, complete from ego (the trim without its "starts within the prepare distance" drop), is
//     compared with the registered path at its own line ends and at the registered lines' ends; when they differ the WHOLE fresh plan
//     is returned and the scene joins it to the robot's state (neroAnchorResolve) and registers it in place of the registered lines.
//  4. neroAnchorSetMemory: the memory after such a registration is the raw parents of the registered plan only.
// ---------------------------------------------------------------------------------------------------------------------------------
void ShiftLineGenerator::neroAnchorMemory(const AvoidLineArray & current, const AvoidancePlanningData & data)
{
  const auto & c = nero_ra::cfg();
  if (!c.on || !helper_ || !data_ || !data_->self_odometry) {
    return;
  }
  const double now = rclcpp::Time(data_->self_odometry->header.stamp).seconds();
  std::vector<std::string> targets;
  for (const auto & o : data.target_objects) {
    const auto & u = o.object.object_id.uuid;
    targets.emplace_back(u.begin(), u.end());
    nero_ra_seen_[targets.back()] = now;
  }
  for (auto it = nero_ra_seen_.begin(); it != nero_ra_seen_.end();) {  // the clock went back (another run) / long gone
    it = (it->second > now + 1.0 || now - it->second > 60.0) ? nero_ra_seen_.erase(it) : std::next(it);
  }
  if (raw_registered_.empty()) {
    return;
  }
  struct Reg
  {
    double level{0.0};
    bool ahead{false};
    double lon{0.0};
    bool pedestrian{false};
  };
  std::unordered_map<std::string, Reg> reg;
  const auto ego = helper_->getEgoPosition();
  const double ego_yaw = tf2::getYaw(helper_->getEgoPose().orientation);
  const double ego_cos = std::cos(ego_yaw), ego_sin = std::sin(ego_yaw);
  // the level a target's lines reach: their ends, and the starts of the lines that have not started (a return starts at its
  // target's level; the start of a line in progress is the past)
  const auto level_of = [](const AvoidLine & al) {
    return std::max(std::abs(al.end_shift_length), al.start_longitudinal > 0.0 ? std::abs(al.start_shift_length) : 0.0);
  };
  for (const auto & al : raw_registered_) {
    if (nero_uuid_zero(al)) continue;
    auto & r = reg[nero_uuid_key(al)];
    r.level = std::max(r.level, level_of(al));
    r.pedestrian = utils::getHighestProbLabel(al.object.object.classification) == ObjectClassification::PEDESTRIAN;
    // where the object stood when the line was made: still ahead = the robot's front has not reached its near end. Along the
    // robot's heading (fb18: no projection on the path per memory line; the question is asked of objects a few metres to a few
    // tens of metres ahead, where the path is straight enough for "ahead of the front or not")
    const auto op = al.object.getPosition();
    const double centre = ego_cos * (op.x - ego.x) + ego_sin * (op.y - ego.y);
    r.lon = centre - 0.5 * al.object.length;
    r.ahead = r.lon > data_->parameters.base_link2front;
  }
  std::unordered_map<std::string, double> cur;
  for (const auto & al : current) {
    if (nero_uuid_zero(al)) continue;
    auto & v = cur[nero_uuid_key(al)];
    v = std::max(v, level_of(al));
  }
  // a vehicle (anything but a pedestrian) that dropped out of the target list keeps its lines until passed (the stock memory: its
  // track was lost or re-born under another id, the vehicle has not left; mk_great s1 sim 1125-220152: 35cd out of the list
  // 15 m ahead, re-born d7d5 not yet a target, the robot came back and passed it at 0.20 m, the car at 0.39). Its level may be
  // carried by a line of the target before it (a merged line belongs to the first of its two targets): such a target's memory is
  // not "smaller now" for that
  double kept_absent_level = 0.0;
  for (const auto & [key, r] : reg) {
    if (!r.pedestrian && std::find(targets.begin(), targets.end(), key) == targets.end()) {
      kept_absent_level = std::max(kept_absent_level, r.level);
    }
  }
  std::vector<std::string> drop;
  std::vector<std::pair<std::string, double>> gone;  // pedestrians dropped as absent, with the level their lines reached
  std::vector<std::pair<std::string, double>> smaller;  // a smaller need: the target's memory lines are capped at it
  for (const auto & [key, r] : reg) {
    const auto name = [&key]() {
      char b[8];
      std::snprintf(
        b, sizeof(b), "%02x%02x", static_cast<int>(static_cast<unsigned char>(key[0])),
        static_cast<int>(static_cast<unsigned char>(key[1])));
      return std::string(b);
    };
    const auto it = cur.find(key);
    if (it != cur.end()) {
      // smaller for absent_s without a break: a need that is smaller for a cycle (a pedestrian out of the list for a moment takes
      // its level out of the line merged with the target before it: mk_great s2 sim 1125-215835, "cfc0: registered for 1.65, needs
      // 0.74 now" at 10.68 and 11.78 s, up and down with the pedestrian, the blinker with it) changes nothing
      const bool low = it->second <= r.level - c.hyst && kept_absent_level < r.level - c.hyst;
      if (!low) {
        nero_ra_low_.erase(key);
        continue;
      }
      auto since = nero_ra_low_.find(key);
      if (since == nero_ra_low_.end() || since->second > now) {
        since = nero_ra_low_.insert_or_assign(key, now).first;
      }
      if (now - since->second >= c.absent_s) {
        smaller.emplace_back(key, it->second);
        nero_ra_low_.erase(key);
        RCLCPP_INFO(
          rclcpp::get_logger("static_obstacle_avoidance"),
          "[NERO_SOA_ANCHOR_MEM] %s: registered for %.2f, needs %.2f for %.1f s now (%.1f m ahead): its memory lines are capped "
          "at %.2f (same places)",
          name().c_str(), r.level, it->second, c.absent_s, r.lon, it->second);
      }
      continue;
    }
    nero_ra_low_.erase(key);
    if (std::find(targets.begin(), targets.end(), key) != targets.end()) {
      continue;  // a target without an outline this cycle (beside the robot, unavoidable for a cycle): the memory speaks
    }
    if (c.absent_s <= 0.0 || !r.ahead || !r.pedestrian) {
      continue;  // the stock memory: a vehicle; beside / past it, its plateau and return complete
    }
    const auto seen = nero_ra_seen_.find(key);
    if (seen == nero_ra_seen_.end()) {
      nero_ra_seen_[key] = now;  // never seen by this run of the rule: the wait starts now
      continue;
    }
    if (now - seen->second >= c.absent_s) {
      drop.push_back(key);
      gone.emplace_back(key, r.level);
      RCLCPP_INFO(
        rclcpp::get_logger("static_obstacle_avoidance"),
        "[NERO_SOA_ANCHOR_MEM] %s: registered for %.2f, not a target for %.1f s and still %.1f m ahead: its lines leave the memory",
        name().c_str(), r.level, now - seen->second, r.lon);
    }
  }
  if (drop.empty() && smaller.empty()) {
    return;
  }
  // a pedestrian's level also sits in lines filed under another target: the line merged with the target before it belongs to that
  // target, and so do the fills up to the pedestrian's own lines (mk_great s0 sim 1125-221909, 14.19 s: "[5.98>16.48 0.70>1.60
  // cfc0] [16.48>21.28 1.65>1.65 cfc0]" stayed when 70c7 left; cfc0, a parked car beside the robot, has no outline to be compared
  // with, and the robot went out to 1.64 m for nobody). With the pedestrian go the lines that reach its level and touch / overlap
  // its lines (and, through them, each other), if they have not started or are a ramp away from the centre. A plateau in progress
  // (the car the robot is beside) stays, and so do the lines of a vehicle that is out of the list (the stock memory).
  std::vector<char> out(raw_registered_.size(), 0);
  if (!gone.empty()) {
    double level_min = std::numeric_limits<double>::max();
    for (size_t i = 0; i < raw_registered_.size(); ++i) {
      const auto & al = raw_registered_.at(i);
      if (nero_uuid_zero(al)) continue;
      for (const auto & [key, level] : gone) {
        if (nero_uuid_key(al) == key) {
          out.at(i) = 1;
          level_min = std::min(level_min, level - c.hyst);
        }
      }
    }
    size_t carried = 0;
    for (bool grew = true; grew;) {
      grew = false;
      for (size_t i = 0; i < raw_registered_.size(); ++i) {
        const auto & m = raw_registered_.at(i);
        if (out.at(i) != 0 || nero_uuid_zero(m) || level_of(m) < level_min) continue;
        const auto owner = reg.find(nero_uuid_key(m));
        if (
          owner != reg.end() && !owner->second.pedestrian &&
          std::find(targets.begin(), targets.end(), owner->first) == targets.end()) {
          continue;  // a vehicle out of the list: its lines stay
        }
        const bool movable = m.start_longitudinal > 0.0 ||
                             std::abs(m.end_shift_length) > std::abs(m.start_shift_length) + c.hyst;
        if (!movable) continue;
        for (size_t j = 0; j < raw_registered_.size(); ++j) {
          const auto & d = raw_registered_.at(j);
          if (out.at(j) == 0) continue;
          if (m.start_longitudinal < d.end_longitudinal + 1.0 && m.end_longitudinal > d.start_longitudinal - 1.0) {
            out.at(i) = 1;
            grew = true;
            ++carried;
            break;
          }
        }
      }
    }
    if (carried > 0) {
      RCLCPP_INFO(
        rclcpp::get_logger("static_obstacle_avoidance"),
        "[NERO_SOA_ANCHOR_MEM] with the pedestrian(s): %zu line(s) filed under other targets that reach the level %.2f and touch "
        "its lines leave the memory too",
        carried, level_min + c.hyst);
    }
  }
  {
    AvoidLineArray kept;
    for (size_t i = 0; i < raw_registered_.size(); ++i) {
      const auto & al = raw_registered_.at(i);
      if (out.at(i) != 0) continue;
      if (nero_uuid_zero(al)) {
        if (al.start_longitudinal > 0.0) continue;  // a return / fill that has not started: re-made from this cycle's lines
      } else if (std::find(drop.begin(), drop.end(), nero_uuid_key(al)) != drop.end()) {
        continue;
      }
      kept.push_back(al);
      // a smaller need: the target's lines stay where they are and reach only as far as the need now (fb12). fb10 put this
      // cycle's lines in their place: their avoid line starts a prepare distance ahead of ego, later and steeper than the
      // registered one (n24 s1 sim 1125-222241, 21.4 s: a4d5 1.90 -> 1.60, the fresh line [4.13>17.03 0.00>1.60] against the
      // registered [1.22>19.95 0.10>1.90] was not comfortable, and the re-plan took a4d5's registered avoidance away)
      if (!nero_uuid_zero(al)) {
        for (const auto & [key, cap] : smaller) {
          if (nero_uuid_key(al) != key) continue;
          auto & m = kept.back();
          const auto capped = [cap = cap](const double level) {
            return std::abs(level) > cap ? std::copysign(cap, level) : level;
          };
          m.start_shift_length = capped(m.start_shift_length);
          m.end_shift_length = capped(m.end_shift_length);
        }
      }
    }
    raw_registered_ = kept;
  }
  (void)current;
}

AvoidLineArray ShiftLineGenerator::neroAnchorCandidates(
  const AvoidLineArray & merged, const AvoidancePlanningData & data, DebugData & debug) const
{
  nero_ra_why_.clear();
  nero_ra_diff_lon_ = 0.0;
  nero_ra_total_ = debug.total_shift.size() == data.reference_path.points.size() ? debug.total_shift : std::vector<double>{};
  nero_ra_keep_near_ = true;
  const auto trimmed = applyTrimProcess(merged, debug);
  nero_ra_keep_near_ = false;
  if (nero_pipe::on() && nero_pipe::due("trimmed")) {
    nero_pipe::log("trimmed", nero_pipe::lines(trimmed));
  }
  const auto plan = neroAnchorCover(trimmed, data);  // fb18: never under the merged needs
  if (plan.empty()) {
    return {};
  }
  // lateral_execution_threshold, and never less than half a quantize step: a level the anchor made off the grid (a carrier's stop,
  // a relaxed return) stays within that of the fresh plan's quantized level and must not read as a difference every cycle
  const double th = std::max(parameters_->lateral_execution_threshold, 0.5 * parameters_->quantize_size) + 0.005;
  // the fresh plan as a function of the distance from ego (linear between its line ends, held between lines)
  const auto level_at = [&plan, this](const double lon) {
    double level = helper_->getEgoLinearShift();
    for (const auto & l : plan) {
      if (lon <= l.start_longitudinal) return level;
      if (lon < l.end_longitudinal) {
        const double len = std::max(l.end_longitudinal - l.start_longitudinal, 1.0e-3);
        return level + (l.end_shift_length - level) * (lon - l.start_longitudinal) / len;
      }
      level = l.end_shift_length;
    }
    return level;
  };
  // fb18: where either plan is steep (the short lines of a crawl / a standstill: 1.3 m sideways over 1.2 m), a line end that
  // falls one path step (0.3 m) apart in the two readings is worth slope x 0.3 m of level: that is no difference. Nothing is
  // added below a slope of 0.2 (every line of normal driving: 1.9 m over 20 m is 0.1)
  const auto & arc_steep = data.arclength_from_ego;
  const auto steep = [&](const double lon) {
    double slope = 0.0;
    double level = helper_->getEgoLinearShift();
    for (const auto & l : plan) {
      if (lon >= l.start_longitudinal - 0.5 && lon <= l.end_longitudinal + 0.5) {
        slope = std::max(
          slope, std::abs(l.end_shift_length - level) / std::max(l.end_longitudinal - l.start_longitudinal, 0.3));
      }
      level = l.end_shift_length;
    }
    for (const auto & sl : nero_ps_.getShiftLines()) {
      if (sl.end_idx >= arc_steep.size() || sl.start_idx >= sl.end_idx) continue;
      const double a = arc_steep.at(sl.start_idx);
      const double b = arc_steep.at(sl.end_idx);
      if (lon >= a - 0.5 && lon <= b + 0.5) {
        slope = std::max(slope, std::abs(sl.end_shift_length - sl.start_shift_length) / std::max(b - a, 0.3));
      }
    }
    return std::max(0.0, slope - 0.2) * 0.6;
  };
  std::ostringstream why;
  why << std::fixed << std::setprecision(2);
  bool differs = false;
  // nothing can be changed any more within the minimum prepare distance: a difference there is no reason to re-plan (fb7 re-planned
  // on "the fresh plan is at 1.80 at 1.28 m, the registered path at 1.86")
  const double near_lon =
    std::max(1.5, std::max(helper_->getEgoSpeed() * parameters_->min_prepare_time, parameters_->min_prepare_distance));
  for (const auto & l : plan) {
    if (l.end_longitudinal <= near_lon) continue;
    const double reg = helper_->getLinearShift(l.end.position);
    // the registered path is UNDER the fresh plan there (fb12). A registered path that is above the fresh plan between its own
    // line ends is no difference: the trim combines a hold and the rise after it into one longer line one cycle and not the
    // next (n24 s0 sim 1125-223015, 21.41 s: "the fresh plan is at 0.10 at 3.85 m, the registered path at 0.54": the registered
    // avoid line [.. 0.10>1.90] had just started, the fresh one rose later to the same end; the re-plan took the started line
    // away). A registered level that the needs no longer have is found at the registered line ends, below.
    const double need = l.end_shift_length;
    const double under = need * reg > 0.0 ? std::abs(need) - std::abs(reg) : std::abs(need);
    if (under >= th + steep(l.end_longitudinal)) {
      differs = true;
      nero_ra_diff_lon_ = l.end_longitudinal;
      why << "the fresh plan is at " << l.end_shift_length << " at " << l.end_longitudinal << " m, the registered path at " << reg;
      break;
    }
  }
  // a registered level this cycle's needs do not have between the fresh plan's line ends (a line up and down again inside a fresh
  // hold). Read against the merged shift itself, not the trimmed lines: the trim combines lines within th_similar_grad (0.5 m on
  // the car), and fb7 replaced a good registered descent by the combined, slower one ("the registered path is at 0.57 at 16.92 m,
  // the fresh plan at 0.81", n24 26.49 s: the merged shift there was 0.59)
  const double th_reg = std::max(th, 0.1);
  const auto & arc = data.arclength_from_ego;
  for (const auto & sl : nero_ps_.getShiftLines()) {
    if (sl.end_idx >= arc.size()) continue;
    const double lon = arc.at(sl.end_idx);
    if (lon <= near_lon || (differs && lon >= nero_ra_diff_lon_)) continue;
    const double reg = helper_->getLinearShift(sl.end.position);
    const double fresh = neroAnchorEnvelope(sl.end_idx, level_at(lon));
    // (fb16) and the fresh plan itself is off the registered level there too: registering it must change something, so a
    // registration closes the difference (a registered line that ends where the trimmed plan agrees with it, but the merged needs
    // do not, re-planned every cycle into the same lines: the mode-1 keep no longer keeps a line that ends at the difference)
    const double st = steep(lon);
    if (std::abs(reg - fresh) >= th_reg + st && std::abs(reg - level_at(lon)) >= th + st) {
      differs = true;
      nero_ra_diff_lon_ = lon;
      why.str("");
      why << "the registered path is at " << reg << " at " << lon << " m, the merged needs at " << fresh;
    }
  }
  if (!differs) {
    return {};
  }
  nero_ra_why_ = why.str();
  debug.step4_new_shift_line = plan;
  return plan;
}

bool ShiftLineGenerator::neroAnchorIsNeed(const AvoidLine & line) const
{
  // fb16: a line end at a held dip (or where the descent into it ends) is a need: the level of the close place after it
  const size_t e = line.end_idx;
  if (
    e < nero_ra_held_.size() &&
    (nero_ra_held_.at(e) != 0 || (e + 1 < nero_ra_held_.size() && nero_ra_held_.at(e + 1) != 0))) {
    return true;
  }
  if (std::find(nero_ra_cover_.begin(), nero_ra_cover_.end(), e) != nero_ra_cover_.end()) {
    return true;  // fb18: a knot the cover put on the merged needs
  }
  for (const auto & id : line.parent_ids) {
    const auto itr = std::find_if(raw_.begin(), raw_.end(), [&id](const auto & s) { return s.id == id; });
    if (itr == raw_.end()) continue;
    if (std::abs(itr->end_longitudinal - line.end_longitudinal) > 1.0) continue;
    if (nero_uuid_zero(*itr) && std::abs(itr->start_longitudinal) <= 0.5) continue;  // the fill / hold from ego: no need of a target
    return true;
  }
  return false;
}

void ShiftLineGenerator::neroAnchorSetMemory(
  const AvoidLineArray & lines, const AvoidancePlanningData & data, const double keep_until_lon, const size_t n_chord)
{
  if (lines.empty()) {
    return;
  }
  auto plan = lines;
  utils::static_obstacle_avoidance::fillAdditionalInfoFromPoint(data, plan);
  utils::static_obstacle_avoidance::fillAdditionalInfoFromPoint(data, raw_registered_);
  AvoidLineArray memory;
  if (keep_until_lon > -1.0e8) {
    // the registered line in progress was kept: so is what the memory holds before its end
    for (const auto & al : raw_registered_) {
      if (al.start_longitudinal < keep_until_lon - 0.5) memory.push_back(al);
    }
  }
  const auto in_memory = [&memory](const UUID & id) {
    return std::any_of(memory.begin(), memory.end(), [&id](const auto & s) { return s.id == id; });
  };
  const auto raw_of = [this](const UUID & id) -> const AvoidLine * {
    const auto itr = std::find_if(raw_.begin(), raw_.end(), [&id](const auto & s) { return s.id == id; });
    return itr != raw_.end() ? &(*itr) : nullptr;
  };
  // the lines the anchor made: their memory is their own chord (the registered path there), owned by the target that needs their
  // level most (so a later, smaller need of it drops the chord). Their raw parents stay in the memory under the chord (a target the
  // module shifts for keeps its lines: NERO_SOA_ROUTE_LC / NERO_SOA_ABLC_HANDOFF read them, and its level stays if the chord's owner
  // goes), never above it (a raw line made before the re-plan may lie above the path that is registered now: it would read as a
  // new, higher need every cycle)
  AvoidLineArray chords;
  for (size_t k = 0; k < std::min(n_chord, plan.size()); ++k) {
    AvoidLine chord = plan.at(k);
    chord.id = generate_uuid();
    chord.parent_ids.clear();
    chord.object = ObjectData{};
    double best = -1.0;
    for (const auto & id : plan.at(k).parent_ids) {
      const auto * p = raw_of(id);
      if (p == nullptr || nero_uuid_zero(*p)) continue;
      if (std::abs(p->end_shift_length) > best) {
        best = std::abs(p->end_shift_length);
        chord.object = p->object;
        chord.object_on_right = p->object_on_right;
      }
    }
    chords.push_back(chord);
    memory.push_back(chord);
  }
  const double chord_end = chords.empty() ? -1.0e9 : chords.back().end_longitudinal;
  const double chord_level = chords.empty() ? 0.0 : chords.back().end_shift_length;
  const auto chord_at = [&chords](const double lon) {
    double level = chords.front().start_shift_length;
    for (const auto & c : chords) {
      if (lon <= c.start_longitudinal) return level;
      if (lon < c.end_longitudinal) {
        const double len = std::max(c.end_longitudinal - c.start_longitudinal, 1.0e-3);
        return c.start_shift_length + (c.end_shift_length - c.start_shift_length) * (lon - c.start_longitudinal) / len;
      }
      level = c.end_shift_length;
    }
    return level;
  };
  const auto under = [](const double level, const double cap) {
    return (level * cap > 0.0 && std::abs(level) > std::abs(cap)) ? cap : level;
  };
  for (size_t k = 0; k < plan.size(); ++k) {
    for (const auto & id : plan.at(k).parent_ids) {
      if (in_memory(id)) continue;
      const auto * p = raw_of(id);
      if (p == nullptr) continue;
      AvoidLine al = *p;
      if (!chords.empty() && al.end_longitudinal <= chord_end + 0.5) {
        if (nero_uuid_zero(al)) continue;  // a fill / return inside the chords: the chord speaks
        al.start_shift_length = under(al.start_shift_length, chord_at(al.start_longitudinal));
        al.end_shift_length = under(al.end_shift_length, chord_at(al.end_longitudinal));
      } else if (!chords.empty() && al.start_longitudinal < chord_end) {
        // reaching across the chords' end: only its part after it, from the level the registered path has there
        al.start_shift_length = chord_level;
        al.start_longitudinal = chord_end;
        al.start_idx = findPathIndexFromArclength(data.arclength_from_ego, chord_end);
        al.start = data.reference_path.points.at(al.start_idx).point.pose;
      }
      memory.push_back(al);
    }
  }
  raw_registered_ = memory;
}

// fb16 WEAVE (b) (Ahmed, CAP1122-WEAVE answered (b) "1 hold the shift"; BRIEF_3b case 1): this cycle's merged shift (the max of the
// targets' current lines and the memory, generateTotalShiftLine) is held where it dips between two places that both need more and
// lie within hold_m of each other: every point between them is raised to the LOWER of the two levels (the higher place on each side
// within hold_m). A held level is therefore never above what a place ahead currently needs, never comes from a line alone, and
// a dip only goes when one of its two sides goes. n24 sim 1125-223906, 27.23 s: a4d5's descent (1.67 at ego) went down to the row's
// 0.58-0.61 and back up to c4df's 0.89 within 12.6 m; the robot was led to 0.58 and could not climb back to c4df's growing need
// (0.9 -> 1.1) in time (fragment stop, 1.0 m/s, turn +6.6 s). The robot's own level at ego counts as a side (the dip is in front of it).
// Not held: a dip shallower than one quantize step (quantize_size, 0.1), a dip across the other side (a shift the other way is a
// barrier), the lane centre itself (< 0.05), and nothing behind ego.
void ShiftLineGenerator::neroAnchorHold(const AvoidancePlanningData & data, ShiftLineData & sl) const
{
  const auto & c = nero_ra::cfg();
  const auto & arcs = data.arclength_from_ego;
  const size_t N = sl.shift_line.size();
  nero_ra_held_.assign(N, 0);
  if (!c.on || c.hold_m <= 0.0 || N < 4 || arcs.size() != N || sl.shift_line_grad.size() != N) {
    return;
  }
  const size_t i0 = std::min(data.ego_closest_path_index, N - 1);
  constexpr double LEVEL_MIN = 0.05;   // [m] the lane centre: nothing to hold
  constexpr double BARRIER = -1.0e-3;  // [m] a shift to the other side: a dip never reaches across it
  const double raise_min = std::max(parameters_->quantize_size, 0.05) - 1.0e-3;
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(2);
  size_t runs = 0;
  for (const double sign : {1.0, -1.0}) {
    std::vector<double> z(N, 0.0);
    for (size_t i = i0; i < N; ++i) z.at(i) = sign * sl.shift_line.at(i);
    std::vector<double> lvl = z;
    for (size_t j = i0; j + 2 < N; ++j) {
      // the pair (j, k) raises every point between them to min(z_j, z_k). A point j followed by one at least as high is covered
      // by that one (same pairs, shorter span), so only the tops of descents are scanned
      if (z.at(j) < LEVEL_MIN || z.at(j + 1) >= z.at(j)) continue;
      size_t kmax = j;
      for (size_t k = j + 1; k < N && arcs.at(k) - arcs.at(j) <= c.hold_m; ++k) {
        if (z.at(k) < BARRIER) break;
        kmax = k;
      }
      if (kmax < j + 2) continue;
      double m = -std::numeric_limits<double>::max();  // the highest level in (i, kmax]
      for (size_t i = kmax; i > j; --i) {
        const double cand = std::min(z.at(j), m);
        if (cand > lvl.at(i)) lvl.at(i) = cand;
        m = std::max(m, z.at(i));
      }
    }
    // the runs of raised points; a run that is not one quantize step deep anywhere is left as it is
    for (size_t a = i0 + 1; a < N;) {
      if (lvl.at(a) <= z.at(a) + 1.0e-3) {
        ++a;
        continue;
      }
      size_t b = a;
      double raise = 0.0, low = z.at(a), level = 0.0;
      while (b < N && lvl.at(b) > z.at(b) + 1.0e-3) {
        raise = std::max(raise, lvl.at(b) - z.at(b));
        low = std::min(low, z.at(b));
        level = std::max(level, lvl.at(b));
        ++b;
      }
      if (raise >= raise_min) {
        for (size_t i = a; i < b; ++i) {
          sl.shift_line.at(i) = sign * lvl.at(i);
          nero_ra_held_.at(i) = 1;
        }
        ++runs;
        ss << " " << sign * low << " held at " << sign * level << " over " << arcs.at(a) << ">" << arcs.at(b - 1) << " m;";
      }
      a = b;
    }
  }
  if (runs == 0) {
    return;
  }
  // the gradient the extraction reads (shift_line_second_grad with the close-vehicle policy) follows the held profile, measured
  // inside the held run only (a flat hold reads exactly 0: one start / end point at each side of it, no sliver line at its edge)
  for (size_t i = i0 + 1; i < N; ++i) {
    if (nero_ra_held_.at(i) == 0) continue;
    const bool prev = nero_ra_held_.at(i - 1) != 0;
    const bool next = i + 1 < N && nero_ra_held_.at(i + 1) != 0;
    const size_t lo = prev ? i - 1 : i;
    const size_t hi = next ? i + 1 : i;
    const double ds = arcs.at(hi) - arcs.at(lo);
    sl.shift_line_grad.at(i) = (hi > lo && ds > 1.0e-5) ? (sl.shift_line.at(hi) - sl.shift_line.at(lo)) / ds : 0.0;
  }
  sl.shift_line_history.push_back(sl.shift_line);
  static rclcpp::Clock clock(RCL_STEADY_TIME);
  RCLCPP_INFO_THROTTLE(
    rclcpp::get_logger("static_obstacle_avoidance"), clock, 300,
    "[NERO_SOA_ANCHOR_HOLD] the merged needs dip between two places within %.0f m that need more: the dip to%s (WEAVE (b): "
    "held at the lower side, never above a need ahead)",
    c.hold_m, ss.str().c_str());
}

// fb18 (n24 s2 sim 1126-111423, 27.30 / 29.59 s; b7_cut sim 1126-111210): the plan that is registered is never under this
// cycle's merged needs. The merged needs (generateTotalShiftLine + the WEAVE hold) are exact; the lines made from them are not:
//  - the extraction (close-vehicle policy) puts a line end where the owning lines' theoretical gradient changes, so a level step
//    between two flat lines is invisible to it: 27.49 s, the hold and c4df's plateau at 0.90 up to 27.37 m, then the memory's
//    older plateaus 0.82 / 0.74 ending 0.3 / 0.6 m later, came out as ONE line [8.77>27.67 0.90>0.74]: 0.70 beside c4df that
//    needed 0.90 ("the approved shift at it 0.70 m does not keep the hard margin", four fragment stop lines);
//  - the similar-gradient combine joins lines whose joined chord is within th_similar_grad (0.1 / 0.2 / 0.5 m) of them:
//    29.59 s, one 34 m return [1.04>35.56 1.00>0.00], about 0.45 at c4df where the merged needs were 0.92;
//  - the short-line drop (< 1 m) starts the next line from the level before it.
// Here the trimmed plan is read as a polyline and compared with the merged needs at every path point; where it is under them by
// more than quantize_size / 2 on two points in a row, a knot is put on the needs at the worst point (quantized), up to 12 times.
// A piece next to such a knot that is too short for a line (< 1 m / 3 points) is joined to its neighbour on the side that
// keeps the plan on or above the needs (a rise starts earlier, a descent ends later). If a piece stays too short the trimmed
// plan is returned as it was. Nothing changes when the trimmed plan is not under the needs (the lines keep their ids).
AvoidLineArray ShiftLineGenerator::neroAnchorCover(
  const AvoidLineArray & trimmed, const AvoidancePlanningData & data) const
{
  nero_ra_cover_.clear();
  const auto & E = nero_ra_total_;
  const auto & arc = data.arclength_from_ego;
  const auto & pts = data.reference_path.points;
  const size_t N = arc.size();
  if (trimmed.empty() || E.size() != N || pts.size() != N) {
    return trimmed;
  }
  const double q = std::max(parameters_->quantize_size, 0.02);
  const double tol = 0.5 * q + 1.0e-3;
  struct Knot
  {
    size_t idx;
    double y;
    int src;  // >= 0: the end of that trimmed line; -1: the plan's start; -2: the end of a gap (level held); -3: put in here
  };
  std::vector<Knot> k;
  for (size_t j = 0; j < trimmed.size(); ++j) {
    const auto & l = trimmed.at(j);
    if (l.end_idx <= l.start_idx || l.end_idx >= N) return trimmed;
    if (k.empty()) {
      k.push_back({l.start_idx, l.start_shift_length, -1});
    } else if (l.start_idx > k.back().idx) {
      k.push_back({l.start_idx, k.back().y, -2});
    } else if (l.start_idx < k.back().idx) {
      return trimmed;  // not a chain
    }
    k.push_back({l.end_idx, l.end_shift_length, static_cast<int>(j)});
  }
  // how far the piece a -> b is under the needs at path point i (same side only). A piece that does not move away from the
  // centre (a return, a hold) is asked for no more than the level it starts from: the cover makes a return end later, it never
  // takes the plan further out than the trim left it (a level the trim dropped altogether is not brought back here)
  const auto away = [](const Knot & a, const Knot & b) { return std::abs(b.y) > std::abs(a.y) + 1.0e-3; };
  const auto under = [&](const Knot & a, const Knot & b, const size_t i) {
    const double len = std::max(arc.at(b.idx) - arc.at(a.idx), 1.0e-3);
    const double p = a.y + (b.y - a.y) * (arc.at(i) - arc.at(a.idx)) / len;
    double e = E.at(i);
    if (std::abs(e) < 1.0e-3) return 0.0;
    if (!away(a, b)) {
      // (asked for the piece's own start level, within the tolerance band: a hold one quantize step under a need is no cut, a
      // return that slopes down through a target's span from its front is: route_b2 sim 1126-112346, 91.08 s, 078e needs
      // 1.14, the plan [-4.37>12.48 0.02>1.10] [12.48>67.98 1.10>0.80] comes down from 078e's FRONT toward 6ea3's 0.80:
      // ~1.07 at its far end, passed at 0.42 m)
      if (e * a.y <= 0.0) return 0.0;
      e = std::copysign(std::min(std::abs(e), std::abs(a.y) + 0.9 * tol), e);
    }
    if (std::abs(p) < 1.0e-3) return std::abs(e);
    return e * p > 0.0 ? std::abs(e) - std::abs(p) : 0.0;
  };
  // the worst point of a piece that is under the needs together with a neighbour point (one point alone is the grid)
  const auto worst_of = [&](const Knot & a, const Knot & b, size_t & at) {
    double worst = 0.0;
    double prev = 0.0;
    for (size_t i = a.idx + 1; i < b.idx; ++i) {
      const double d = under(a, b, i);
      if (d > tol && prev > tol) {
        if (d > worst) {
          worst = d;
          at = i;
        }
        if (prev > worst) {
          worst = prev;
          at = i - 1;
        }
      }
      prev = d;
    }
    return worst;
  };
  // A piece that moves AWAY from the centre is covered (it reaches the level where the needs reach it, not at the end of the
  // plateau after it: b7_cut sim 1126-111210, 19.74 s, one 40 m ramp [15.01>55.21 0.00>0.70] where the needs are 0.64 from
  // 26.4 m on) only when (1) the needs themselves climb to that level over a distance (a level step within two path points is a
  // line the trim drops as noise, "shorter than 1 m": a pedestrian's left-over return in the memory, mk_great sim 1126-110501,
  // 14.16 s, [21.3>21.6 0.74>1.65]; the steep close-distance lines of a standstill stay the stock trim's), and (2) the earlier
  // rise is comfortable (isComfortable: n24 sim 1126-110311, 31.98 s, c4df's own [2.9>6.2 0.90>1.10] is not at 2.5 m/s, the
  // trimmed [2.9>12.8 0.90>1.10] is, and it is what let c4df's growing need register at all). Else the piece stays as trimmed.
  const auto away_ok = [&](const Knot & a, const size_t wi, const double level) {
    const double rise = std::abs(level) - std::abs(a.y);
    if (rise <= tol) return false;
    for (size_t i = a.idx + 2; i <= wi; ++i) {
      if (std::abs(E.at(i)) - std::abs(E.at(i - 2)) > 0.5 * rise) return false;
    }
    // comfortable with room to spare (1.3 x the sharpest comfortable length + two path steps): the join may still move the
    // line's start to the minimum prepare distance, and a re-plan with one uncomfortable line is not registered at all
    AvoidLine probe;
    probe.start_longitudinal = arc.at(a.idx);
    probe.end_longitudinal = arc.at(wi);
    probe.start_shift_length = a.y;
    probe.end_shift_length = level;
    return helper_->isComfortable(AvoidLineArray{probe}) &&
           arc.at(wi) - arc.at(a.idx) >= 1.3 * helper_->getSharpAvoidanceDistance(rise) + 0.6;
  };
  std::vector<size_t> left;  // end indices of the pieces away from the centre that stay as trimmed
  double first_worst = 0.0, first_lon = 0.0;
  size_t put = 0;
  for (int it = 0; it < 16; ++it) {
    double worst = 0.0;
    size_t wi = 0, ws = 0;
    for (size_t s = 0; s + 1 < k.size(); ++s) {
      if (std::find(left.begin(), left.end(), k.at(s + 1).idx) != left.end()) continue;
      size_t at = 0;
      const double w = worst_of(k.at(s), k.at(s + 1), at);
      if (w > worst) {
        worst = w;
        wi = at;
        ws = s;
      }
    }
    if (worst <= tol) break;
    const Knot & a = k.at(ws);
    const Knot & b = k.at(ws + 1);
    // the needs' level on the quantize grid: the step below when that is within the tolerance (0.75 -> 0.70, no bump of half a
    // step over the need), else the step above
    double level_abs = std::floor(std::abs(E.at(wi)) / q + 1.0e-6) * q;
    if (std::abs(E.at(wi)) - level_abs > tol - 1.0e-3) level_abs += q;
    double level = std::copysign(level_abs, E.at(wi));
    if (away(a, b)) {
      if (!away_ok(a, wi, level)) {
        left.push_back(b.idx);
        continue;
      }
    } else {
      level = std::copysign(std::min(std::abs(level), std::abs(a.y)), a.y);
    }
    if (put == 0) {
      first_worst = worst;
      first_lon = arc.at(wi);
    }
    k.insert(k.begin() + static_cast<std::ptrdiff_t>(ws + 1), Knot{wi, level, -3});
    ++put;
  }
  if (put == 0) {
    return trimmed;
  }
  // pieces next to a new knot that are too short for a line
  const auto too_short = [&](const Knot & a, const Knot & b) {
    return b.idx < a.idx + 3 || arc.at(b.idx) - arc.at(a.idx) < 1.0;
  };
  const auto fits = [&](const Knot & a, const Knot & b) {
    size_t at = 0;
    return worst_of(a, b, at) <= tol;
  };
  for (int it = 0; it < 24; ++it) {
    bool changed = false;
    for (size_t s = 0; s + 1 < k.size() && !changed; ++s) {
      const Knot a = k.at(s);
      const Knot b = k.at(s + 1);
      if ((a.src != -3 && b.src != -3) || !too_short(a, b)) continue;
      const bool rising = std::abs(b.y) > std::abs(a.y) + 1.0e-3;
      const bool falling = std::abs(b.y) < std::abs(a.y) - 1.0e-3;
      const bool a_free = s > 0;               // (the plan's start is the robot's level)
      const bool b_free = s + 2 < k.size();    // (the plan's end)
      // drop a: the piece before it runs straight to b (a rise starts earlier)
      const auto drop_a = [&]() {
        if (!a_free || !fits(k.at(s - 1), b)) return false;
        k.erase(k.begin() + static_cast<std::ptrdiff_t>(s));
        return true;
      };
      // drop b: a runs straight to the knot after b (a descent ends later)
      const auto drop_b = [&]() {
        if (!b_free || !fits(a, k.at(s + 2))) return false;
        k.erase(k.begin() + static_cast<std::ptrdiff_t>(s + 1));
        return true;
      };
      // b's level already at a (a rise is complete one piece earlier), b dropped
      const auto raise_a = [&]() {
        if (!a_free || !b_free) return false;
        Knot a2 = a;
        a2.y = b.y;
        a2.src = -3;
        if (!fits(k.at(s - 1), a2) || !fits(a2, k.at(s + 2))) return false;
        k.at(s) = a2;
        k.erase(k.begin() + static_cast<std::ptrdiff_t>(s + 1));
        return true;
      };
      // a's level held up to b (a descent starts one piece later), a dropped
      const auto raise_b = [&]() {
        if (!a_free || !b_free) return false;
        Knot b2 = b;
        b2.y = a.y;
        b2.src = -3;
        if (!fits(k.at(s - 1), b2) || !fits(b2, k.at(s + 2))) return false;
        k.at(s + 1) = b2;
        k.erase(k.begin() + static_cast<std::ptrdiff_t>(s));
        return true;
      };
      if (rising) {
        changed = drop_a() || raise_a() || drop_b();
      } else if (falling) {
        changed = drop_b() || raise_b() || drop_a();
      } else {
        changed = drop_b() || drop_a();
      }
    }
    if (!changed) break;
  }
  // a gap of the trimmed plan (the level is held between two lines, no line there)
  const auto is_gap = [](const Knot & a, const Knot & b) {
    return b.src == -2 && a.src != -3 && std::abs(b.y - a.y) < 1.0e-6;
  };
  for (size_t s = 0; s + 1 < k.size(); ++s) {
    const bool touched = k.at(s).src == -3 || k.at(s + 1).src == -3;
    if (touched && !is_gap(k.at(s), k.at(s + 1)) && k.at(s + 1).idx < k.at(s).idx + 3) {
      static rclcpp::Clock short_clock(RCL_STEADY_TIME);
      RCLCPP_INFO_THROTTLE(
        rclcpp::get_logger("static_obstacle_avoidance"), short_clock, 500,
        "[NERO_SOA_ANCHOR_COVER] the trimmed plan is %.2f m under the merged needs at %.1f m, but a piece of the covered plan "
        "would be too short for a line (%.1f>%.1f m): the trimmed plan is used as it is",
        first_worst, first_lon, arc.at(k.at(s).idx), arc.at(k.at(s + 1).idx));
      return trimmed;
    }
  }
  // the lines of the covered plan
  AvoidLineArray out;
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(2);
  for (size_t s = 0; s + 1 < k.size(); ++s) {
    const auto & a = k.at(s);
    const auto & b = k.at(s + 1);
    if (is_gap(a, b)) {
      continue;
    }
    // the trimmed line this piece lies on
    size_t base = 0;
    const size_t mid = (a.idx + b.idx) / 2;
    for (size_t j = 0; j < trimmed.size(); ++j) {
      if (trimmed.at(j).start_idx <= mid) base = j;
    }
    const auto & t = trimmed.at(base);
    if (b.src == static_cast<int>(base) && a.src != -3 && a.idx == t.start_idx) {
      out.push_back(t);  // untouched
    } else {
      AvoidLine al = t;
      al.id = generate_uuid();
      utils::static_obstacle_avoidance::setStartData(al, a.y, pts.at(a.idx).point.pose, a.idx, arc.at(a.idx));
      utils::static_obstacle_avoidance::setEndData(al, b.y, pts.at(b.idx).point.pose, b.idx, arc.at(b.idx));
      out.push_back(al);
    }
    out.back().start_shift_length = a.y;
    ss << " [" << arc.at(a.idx) << ">" << arc.at(b.idx) << " " << a.y << ">" << b.y << "]";
  }
  for (const auto & kn : k) {
    if (kn.src == -3) nero_ra_cover_.push_back(kn.idx);
  }
  static rclcpp::Clock clock(RCL_STEADY_TIME);
  RCLCPP_INFO_THROTTLE(
    rclcpp::get_logger("static_obstacle_avoidance"), clock, 300,
    "[NERO_SOA_ANCHOR_COVER] the trimmed plan was %.2f m under the merged needs at %.1f m: %zu knot(s) put on the needs, the "
    "fresh plan is%s",
    first_worst, first_lon, put, ss.str().c_str());
  return out.empty() ? trimmed : out;
}
}  // namespace autoware::behavior_path_planner::utils::static_obstacle_avoidance
