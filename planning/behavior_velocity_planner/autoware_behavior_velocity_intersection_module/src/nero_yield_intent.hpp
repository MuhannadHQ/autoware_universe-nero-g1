// Copyright 2026 Nero (yieldsignal job, 2026-11-01, QUEUE YSIG)
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

// NERO_IX_YIELD_INTENT=<v_in m/s>,<t_win s>,<v_stand m/s>,<t_stand s>,<v_crawl m/s>,<t_crawl s>,<acc_th m/s>,
//   <queue m>,<ego_v m/s>,<margin m>
//   (yieldsignal job, 2026-11-01, QUEUE YSIG; e.g. 2.0,6.0,0.5,0.3,1.5,1.0,0.3,6,1.0,0.5)
// Ahmed 2026-11-01: "see when the person slowed down for us that was a signal by him to let us pass". A motor vehicle
// (car, truck, bus, trailer, motorcycle) that BRAKED (its reported speed was >= <v_in> within the last <t_win> s) and
// now stands (reported <= <v_stand> for >= <t_stand> s) or crawls (<= <v_crawl> for >= <t_crawl> s) since that
// braking, without speeding up (<= <acc_th> gained in the last 1 s), with nothing right ahead of it (no vehicle,
// person, bicycle or UNKNOWN >= 0.5 m within <queue> m ahead of its front in its own band: a queue or a pedestrian in
// front of it is not a yield; a vehicle ahead that yields to ego itself makes it a yielding platoon), while ego waits
// (ego slower than <ego_v> when the yield starts; then only the car's own motion ends it), is yielding to ego:
// evidence from its own motion, re-judged every cycle and revoked as soon as it speeds up or moves on. Where it is
// used (the collision check of mapped junctions, crosshold's mouths) its stop must also stay <margin> m clear of ego's
// way.
// yieldroll (2026-11-03, QUEUE YSIG case 11-02 #28): two more fields <roll_v m/s>,<roll_d m> (absent / 0 = as above):
//   a yield may also START while the module's own collision stop is bringing ego to its line: that stop is held,
//   ego is at most <roll_d> m before the line and at most <roll_v> fast (#28: the driver stood while the robot still
//   rolled the last 2 m to its stop point at 1.5-2 m/s; the 1 m/s gate held the release ~0.9 s). A robot at speed
//   far from its stop point keeps the rule above.
//   Fields 13-14 <t_pos s>,<r_pos m> (absent / 0 = off): while ego rolls so (and for a yield started so), a braked
//   vehicle (reported <= v_crawl, not speeding up) whose box centre has stayed within r_pos of its newest place for
//   >= t_pos stands (its own positions: the tracker's reported speed of a car that stopped hard trails its stop by
//   ~1.2 s, #28); its stop reach is then taken at r_pos / t_pos at most. At the line nothing changes.
// Unset = nothing of this runs.

#ifndef NERO_YIELD_INTENT_HPP_
#define NERO_YIELD_INTENT_HPP_

#include <autoware_perception_msgs/msg/predicted_object.hpp>
#include <autoware_perception_msgs/msg/predicted_objects.hpp>

#include <tf2/utils.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <set>
#include <string>
#include <unordered_map>

namespace nero_yield_intent
{
struct Param
{
  bool on{false};
  double v_in{2.0};
  double t_win{6.0};
  double v_stand{0.5};
  double t_stand{0.3};
  double v_crawl{1.5};
  double t_crawl{1.0};
  double acc_th{0.3};
  double queue{6.0};
  double ego_v{1.0};
  double margin{0.5};
  double roll_v{0.0};   // yieldroll: 0 = off (yieldsignal_1101 exactly)
  double roll_d{0.0};
  double t_pos{0.0};    // yieldroll: 0 = no stand from positions
  double r_pos{0.0};
};

inline const Param & param()
{
  static const Param p = [] {
    Param r;
    const char * e = std::getenv("NERO_IX_YIELD_INTENT");
    if (e == nullptr) return r;
    double v[14] = {r.v_in,  r.t_win, r.v_stand, r.t_stand, r.v_crawl, r.t_crawl, r.acc_th,
                    r.queue, r.ego_v, r.margin,  r.roll_v,  r.roll_d,  r.t_pos,   r.r_pos};
    const int n = std::sscanf(
      e, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5],
      &v[6], &v[7], &v[8], &v[9], &v[10], &v[11], &v[12], &v[13]);
    if (n < 1 || v[0] <= 0.0) return r;
    r.on = true;
    r.v_in = v[0];
    r.t_win = std::max(1.0, v[1]);
    r.v_stand = std::max(0.0, v[2]);
    r.t_stand = std::max(0.0, v[3]);
    r.v_crawl = std::max(r.v_stand, v[4]);
    r.t_crawl = std::max(0.0, v[5]);
    r.acc_th = std::max(0.0, v[6]);
    r.queue = std::max(0.0, v[7]);
    r.ego_v = std::max(0.0, v[8]);
    r.margin = v[9];
    if (n >= 12 && v[10] > 0.0 && v[11] > 0.0) {  // yieldroll
      r.roll_v = v[10];
      r.roll_d = v[11];
      if (n >= 14 && v[12] > 0.0 && v[13] > 0.0) {
        r.t_pos = v[12];
        r.r_pos = v[13];
      }
    }
    return r;
  }();
  return p;
}

inline std::string key(const autoware_perception_msgs::msg::PredictedObject & o)
{
  return std::string(reinterpret_cast<const char *>(o.object_id.uuid.data()), o.object_id.uuid.size());
}

// only a motor vehicle's motion is read as a signal (an UNKNOWN's reported speed follows its box: growth, flips)
inline bool is_vehicle(const autoware_perception_msgs::msg::PredictedObject & o)
{
  if (o.classification.empty()) return false;
  using autoware_perception_msgs::msg::ObjectClassification;
  const auto l = o.classification.front().label;
  return l == ObjectClassification::CAR || l == ObjectClassification::TRUCK || l == ObjectClassification::BUS ||
         l == ObjectClassification::TRAILER || l == ObjectClassification::MOTORCYCLE;
}

inline double speed(const autoware_perception_msgs::msg::PredictedObject & o)
{
  const auto & tw = o.kinematics.initial_twist_with_covariance.twist.linear;
  return std::hypot(tw.x, tw.y);
}

struct Hist
{
  std::deque<std::pair<double, double>> s;  // (stamp, reported speed)
  std::deque<std::array<double, 3>> xy;      // yieldroll: (stamp, x, y) of its box centre (only with t_pos)
  double last{0.0};
  double t_fast{-1.0};         // last stamp it was >= v_in
  double stand_since{-1.0};    // start of its current run <= v_stand after the braking (-1: none)
  double crawl_since{-1.0};    // start of its current run <= v_crawl after the braking (-1: none)
};

// a module's yielders (key()s) of the last / this cycle. "Was a yielder" is held per module instance (and per path:
// the mapped check, crosshold) and per cycle: two junction modules on the path judge the same car each with its own
// geometry, and one must never reset the other's verdict (the speed history above is shared: one sample per message)
struct Was
{
  std::set<std::string> prev, now;
  double t{-1.0};
  void next(const double t_now)
  {
    if (t >= 0.0 && t_now - t >= 0.0 && t_now - t <= 0.5) {
      prev.swap(now);
    } else {
      prev.clear();  // the module skipped cycles: start afresh
    }
    now.clear();
    t = t_now;
  }
  bool was(const std::string & k) const { return prev.count(k) > 0; }
};

inline std::unordered_map<std::string, Hist> & hists()
{
  static std::unordered_map<std::string, Hist> h;
  return h;
}

// records the object's reported speed once per objects message (stamp t), whichever caller sees it first
inline Hist & update(const autoware_perception_msgs::msg::PredictedObject & o, const double t)
{
  const auto & p = param();
  auto & h = hists()[key(o)];
  if (h.s.empty() || t > h.s.back().first + 1e-3) {
    const double v = speed(o);
    h.s.emplace_back(t, v);
    // the runs that follow a braking (a stand / crawl lasts as long as it lasts: the braking only has to lead into it
    // within t_win)
    if (v >= p.v_in) {
      h.t_fast = t;
      h.stand_since = h.crawl_since = -1.0;
    } else {
      h.crawl_since = v <= p.v_crawl ? (h.crawl_since < 0.0 ? t : h.crawl_since) : -1.0;
      h.stand_since = v <= p.v_stand ? (h.stand_since < 0.0 ? t : h.stand_since) : -1.0;
    }
    if (p.t_pos > 0.0) {
      const auto & pp = o.kinematics.initial_pose_with_covariance.pose.position;
      h.xy.push_back({t, pp.x, pp.y});
    }
  }
  h.last = t;
  while (!h.s.empty() && t - h.s.front().first > p.t_win + 0.5) h.s.pop_front();
  while (!h.xy.empty() && t - h.xy.front()[0] > p.t_pos + 0.5) h.xy.pop_front();
  static double last_prune = 0.0;
  if (t - last_prune > 5.0 || t < last_prune) {
    last_prune = t;
    for (auto it = hists().begin(); it != hists().end();) {
      if (std::abs(t - it->second.last) > 10.0) it = hists().erase(it); else ++it;
    }
  }
  return h;
}

struct View
{
  bool braked{false};    // >= v_in within the last t_win
  double v_peak{0.0};    // its highest speed in the window
  double since_fast{0.0};  // s since it was last >= v_in
  double stand_s{0.0};   // s it has been <= v_stand since the braking
  double crawl_s{0.0};   // s it has been <= v_crawl since the braking
  double gain_1s{0.0};   // speed gained over the last 1 s (v now - min of the last 1 s)
  double v{0.0};
  bool yielding_motion{false};  // braked, stands / crawls long enough, not speeding up
  double pos_stand_s{0.0};       // yieldroll: s its box centre has stayed within r_pos of its newest place
  bool pos_stand{false};         // yieldroll: braked, reported <= v_crawl, not speeding up, pos_stand_s >= t_pos
};

inline View view(const Hist & h, const double t)
{
  const auto & p = param();
  View w;
  if (h.s.empty()) return w;
  w.v = h.s.back().second;
  for (const auto & [ti, vi] : h.s) {
    if (t - ti <= p.t_win) w.v_peak = std::max(w.v_peak, vi);
  }
  if (h.t_fast < 0.0) return w;
  // braked: it was >= v_in within t_win, or its current crawl began within t_win of that (then it lasts while it
  // crawls)
  w.braked = t - h.t_fast <= p.t_win || (h.crawl_since >= 0.0 && h.crawl_since - h.t_fast <= p.t_win);
  if (!w.braked) return w;
  w.since_fast = t - h.t_fast;
  w.stand_s = h.stand_since >= 0.0 ? t - h.stand_since : 0.0;
  w.crawl_s = h.crawl_since >= 0.0 ? t - h.crawl_since : 0.0;
  double v_min_1s = w.v;
  for (const auto & [ti, vi] : h.s) {
    if (t - ti <= 1.0) v_min_1s = std::min(v_min_1s, vi);
  }
  w.gain_1s = w.v - v_min_1s;
  w.yielding_motion =
    w.gain_1s <= p.acc_th &&
    ((w.stand_s >= p.t_stand && w.v <= p.v_stand) || (w.crawl_s >= p.t_crawl && w.v <= p.v_crawl));
  if (p.t_pos > 0.0 && h.xy.size() >= 3) {
    const auto & n = h.xy.back();
    int k = 0;
    double t_old = n[0];
    for (auto it = h.xy.rbegin(); it != h.xy.rend(); ++it) {
      if (std::hypot((*it)[1] - n[1], (*it)[2] - n[2]) > p.r_pos) break;
      t_old = (*it)[0];
      ++k;
    }
    w.pos_stand_s = k >= 3 ? n[0] - t_old : 0.0;
    w.pos_stand = w.gain_1s <= p.acc_th && w.v <= p.v_crawl && w.pos_stand_s >= p.t_pos;
  }
  return w;
}

// something right ahead of o (within `queue` m of its front, in its own band, standing or moving: a vehicle, a person,
// a bicycle, an UNKNOWN >= 0.5 m): o stops for it (a queue, a pedestrian crossing in front of it), not for ego, unless
// it is a vehicle that yields to ego itself in this module (last cycle: a yielding platoon)
inline bool queued(
  const autoware_perception_msgs::msg::PredictedObject & o,
  const autoware_perception_msgs::msg::PredictedObjects & all, const Was & yielders)
{
  const auto & p = param();
  if (p.queue <= 0.0) return false;
  const auto & q = o.kinematics.initial_pose_with_covariance.pose;
  const double yaw = tf2::getYaw(q.orientation);
  const double c = std::cos(yaw), s = std::sin(yaw);
  const std::string k = key(o);
  for (const auto & b : all.objects) {
    if (key(b) == k || b.classification.empty()) continue;
    using autoware_perception_msgs::msg::ObjectClassification;
    if (
      b.classification.front().label == ObjectClassification::UNKNOWN &&
      std::max(b.shape.dimensions.x, b.shape.dimensions.y) < 0.5)
      continue;  // a small fragment
    const auto & bp = b.kinematics.initial_pose_with_covariance.pose.position;
    const double dx = bp.x - q.position.x, dy = bp.y - q.position.y;
    const double along = dx * c + dy * s, lat = -dx * s + dy * c;
    const double gap = along - o.shape.dimensions.x / 2.0 - b.shape.dimensions.x / 2.0;
    if (along > 0.0 && gap < p.queue && std::abs(lat) < 1.5) {
      // behind a vehicle that yields to ego itself (judged so last cycle): the platoon yields, not a queue
      if (yielders.was(key(b))) continue;
      return true;
    }
  }
  return false;
}
}  // namespace nero_yield_intent

#endif  // NERO_YIELD_INTENT_HPP_
