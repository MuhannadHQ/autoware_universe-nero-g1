// NERO (cutin 2026-10-27): ease off for a car squeezing in beside / just ahead of the robot (QUEUE CUT). Unset = stock.
//   NERO_CRUISE_CUTIN_YIELD="t_s,max_deg,gap_m,max_lat_m,a_max[,cycles]" - for each vehicle (an outside cruise type,
//     >= the outside speed threshold) with its front ahead of ego's base, within max_lat_m of ego's band and heading
//     within max_deg of the trajectory: its box carried along its current velocity (autoware_core #517's constant-
//     velocity cut-in estimate, in the trajectory's frame: its speed along and toward ego's path and its heading to it
//     kept, so a car following a bend beside ego keeps its offset) and ego's footprint carried along the trajectory are
//     compared every 0.1 s for t_s.
//     If they come closer than gap_m while its offset to ego's path shrinks >= 0.3 m within t_s (a car ego catches up
//     with, close and parallel in the next lane, draws nothing; nor does a lead already in the band ahead of ego's
//     nose: cruise / obstacle_stop's) for
//     `cycles` cycles in a row (default 2), ego eases off with the smallest
//     deceleration in 0.25 m/s2 steps up to a_max (at the planning jerk limit) that keeps gap_m, as a slow-down interval
//     in the trajectory (a x 1 s slower 1 s ahead; re-judged every cycle, released once gap_m + 0.3 holds at constant
//     speed). Stock obstacle_cruise keeps no gap to such a car: its
//     lanelet path stays in its own lane until it is in the band, and the RSS distance to a faster car is ~0.
//     Field tracks with a stale heading in bends (QUEUE ANG: 16-37 deg into ego's lane, the "squeezes" of 10-25 #16 /
//     10-26 #13) stay above max_deg; cars centred in the next lane stay beyond max_lat_m whatever their heading noise.
//   NERO_CRUISE_CUTIN_STEEP="min_deg,max_deg,a_stop,max_lat_m,a_max" (needs _YIELD) - the same check for a vehicle
//     heading min_deg..max_deg to the trajectory (a car turning in / crossing from the side; the field ghosts are 16-37 deg),
//     its centre ahead of ego's nose, within max_lat_m of the band, once it can no longer stop before the band at a_stop
//     m/s2 (a driver braking for his give-way line draws nothing), braking up to a_max (velocity limit: the speed after 2 s
//     at the deceleration, so it brakes while the conflict lasts). Stock reacts only once the car is
//     in the band (obstacle_stop below 3.0 m/s along the path, cruise at <= 30 deg): ~1 s before contact for a car entering
//     3 m ahead of a robot at 20 km/h.
//   NERO_CRUISE_CUTIN_DEBUG=1 - log only: every yield (object, deceleration, distances).
//   NERO_CRUISE_YIELD_GAIN="gain_m,critical_m" (mergeahead 2026-10-27, QUEUE CUTY; needs _YIELD) - when no ease-off up to
//     a_max keeps gap_m (the car is already about that close), a squeeze yield STARTS (braking at a_max) only if that
//     adds >= gain_m to the closest approach, or the car would otherwise come within critical_m; a car passing at about its
//     closest now draws nothing; a yield already started is held as before (c56c: released early, its car passed at
//     0.92 m instead of 1.13) (10-25 #20: 330df8 joining the next lane beside ego's nose and pulling away at +2 m/s got
//     -2.0 m/s2 for 1.2 s, 2.8 -> 1.0 m/s, for +0.12 m). The steep branch is unchanged. Unset = cutin_1027.
//   NERO_CRUISE_CUTIN_EARLY="behind_m,dv_min,vlat_min,confirm_m,confirm_s,a_early[,max_lat_m]" (cutinyield 2026-10-27,
//     QUEUE C56M; needs _YIELD) - a squeeze that starts from beside: a car faster than ego along the trajectory by >= dv_min whose
//     front is still up to behind_m behind ego's base_link is checked too (cutin_1027: its front must be ahead of
//     base_link; 10-26e #56's fe8f87 moved over from beside-left for ~1.5 s before that and the yield then started with
//     0.85 m predicted), its ease-off capped at a_early while its front is behind base_link. A faster car whose offset to
//     ego's path has really shrunk by >= confirm_m over the last confirm_s (measured from its box; heading noise alone
//     does not count) is predicted to complete its move at >= vlat_min sideways (constant velocity takes its present small
//     sideways speed; a lane change speeds up mid-move: fe8f87 0.2 -> 0.85 m/s). Its sideways motion is measured against
//     its own lane's centre line from the map (the road lanelet it was in confirm_s ago, with its successors): against
//     ego's path a lane change or a bend of ego's own would look like the car moving in. A car behind base_link, or between
//     the YIELD max_lat_m and max_lat_m (default 2.5) from the band, needs that confirmation to count at all (fe8f87 was
//     2.0-2.15 m from the band while it already moved in).
//   NERO_CRUISE_CUTIN_TAPER="taper_m,hold_gain_m" (cutinyield; needs _YIELD) - a squeeze yield that no ease-off up to a_max
//     can keep at gap_m asks the smallest deceleration step within taper_m of the best clearance any step buys (cutin_1027:
//     a_max on every such cycle), and nothing while the best step buys < hold_gain_m over no ease-off (the yield stays
//     armed); a car predicted within the YIELD_GAIN critical_m still gets a_max. 10-26e #56: a_max for 2.2 s for 0.0-0.05 m
//     of clearance, then the slow-down ran on: 4.4 -> 1.2 m/s, and the next merging car squeezed in beside the crawling
//     robot. The steep branch is unchanged. Unset = cutin_1027 + mergeahead_1027.
#ifndef NERO_CUTIN_HPP_
#define NERO_CUTIN_HPP_

#include <cstdio>
#include <cstdlib>
#include <deque>
#include <string>
#include <unordered_map>

namespace autoware::motion_velocity_planner
{
struct NeroCutinParam
{
  double yield_t_s{0.0};
  double yield_max_deg{12.0};
  double yield_gap_m{1.0};
  double yield_max_lat_m{1.5};
  double yield_a_max{2.0};
  int yield_cycles{2};
  double steep_min_deg{0.0};
  double steep_max_deg{0.0};
  double steep_a_stop{3.0};
  double steep_max_lat_m{6.0};
  double steep_a_max{2.5};
  double gain_m{0.0};
  double gain_critical_m{0.3};
  double early_behind_m{-1.0};  // < 0: NERO_CRUISE_CUTIN_EARLY unset
  double early_dv_min{0.5};
  double early_vlat_min{0.5};
  double early_confirm_m{0.1};
  double early_confirm_s{0.5};
  double early_a{1.0};
  double early_max_lat_m{2.5};
  double taper_m{-1.0};  // < 0: NERO_CRUISE_CUTIN_TAPER unset
  double taper_hold_gain_m{0.05};
  bool debug{false};
};

inline const NeroCutinParam & nero_cutin_param()
{
  static const NeroCutinParam p = [] {
    NeroCutinParam r;
    if (const char * e = std::getenv("NERO_CRUISE_CUTIN_YIELD")) {
      std::sscanf(
        e, "%lf,%lf,%lf,%lf,%lf,%d", &r.yield_t_s, &r.yield_max_deg, &r.yield_gap_m,
        &r.yield_max_lat_m, &r.yield_a_max, &r.yield_cycles);
    }
    if (const char * e = std::getenv("NERO_CRUISE_CUTIN_STEEP")) {
      std::sscanf(
        e, "%lf,%lf,%lf,%lf,%lf", &r.steep_min_deg, &r.steep_max_deg, &r.steep_a_stop,
        &r.steep_max_lat_m, &r.steep_a_max);
    }
    if (const char * e = std::getenv("NERO_CRUISE_YIELD_GAIN")) {
      std::sscanf(e, "%lf,%lf", &r.gain_m, &r.gain_critical_m);
    }
    if (const char * e = std::getenv("NERO_CRUISE_CUTIN_EARLY")) {
      std::sscanf(
        e, "%lf,%lf,%lf,%lf,%lf,%lf,%lf", &r.early_behind_m, &r.early_dv_min, &r.early_vlat_min,
        &r.early_confirm_m, &r.early_confirm_s, &r.early_a, &r.early_max_lat_m);
    }
    if (const char * e = std::getenv("NERO_CRUISE_CUTIN_TAPER")) {
      std::sscanf(e, "%lf,%lf", &r.taper_m, &r.taper_hold_gain_m);
    }
    if (const char * e = std::getenv("NERO_CRUISE_CUTIN_DEBUG")) {
      r.debug = std::string(e) == "1";
    }
    return r;
  }();
  return p;
}

// per-object memory: consecutive conflict cycles (seconds on the node clock)
struct NeroCutinState
{
  struct Hits
  {
    double last{-1e9};
    int n{0};
    double yielded{-1e9};  // the last cycle a yield was output for it
  };
  std::unordered_map<std::string, Hits> hits;

  int count(const std::string & uuid, const double now)
  {
    auto & h = hits[uuid];
    h.n = (now - h.last < 0.25) ? h.n + 1 : 1;
    h.last = now;
    return h.n;
  }
  bool in_conflict(const std::string & uuid, const double now) const
  {
    const auto it = hits.find(uuid);
    return it != hits.end() && now - it->second.last < 0.25;
  }
  void clear(const std::string & uuid) { hits.erase(uuid); }
  void mark_yield(const std::string & uuid, const double now) { hits[uuid].yielded = now; }
  bool yielding(const std::string & uuid, const double now) const
  {
    const auto it = hits.find(uuid);
    return it != hits.end() && now - it->second.yielded < 0.25;
  }
  void prune(const double now)
  {
    for (auto it = hits.begin(); it != hits.end();) {
      it = (now - it->second.last > 10.0) ? hits.erase(it) : std::next(it);
    }
    for (auto it = positions.begin(); it != positions.end();) {
      it = (it->second.empty() || now - it->second.back().t > 2.0) ? positions.erase(it)
                                                                     : std::next(it);
    }
  }

  // NERO_CRUISE_CUTIN_EARLY: each candidate's box centre per cycle (seconds on the node clock, map frame), 2 s kept
  struct Pos
  {
    double t, x, y;
  };
  std::unordered_map<std::string, std::deque<Pos>> positions;
  // record the position now; the one from >= ago_s ago (nullptr when not known that long)
  const Pos * record(const std::string & uuid, const double now, const double x, const double y, const double ago_s)
  {
    auto & h = positions[uuid];
    h.push_back({now, x, y});
    while (!h.empty() && now - h.front().t > 2.0) {
      h.pop_front();
    }
    for (auto it = h.rbegin(); it != h.rend(); ++it) {
      if (now - it->t >= ago_s) {
        return &*it;
      }
    }
    return nullptr;
  }
};
}  // namespace autoware::motion_velocity_planner

#endif  // NERO_CUTIN_HPP_
