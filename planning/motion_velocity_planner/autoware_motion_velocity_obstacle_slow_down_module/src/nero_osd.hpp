// NERO (mergeahead 2026-10-27): no slow-down for a moving vehicle that is not in the robot's way (QUEUE SPD). Unset = stock.
//   NERO_OSD_NOT_IN_WAY="t_s,keep_m,max_deg,min_gap_m,approach_m,min_v[,pass_m]" - for a car / truck / bus / trailer / motorcycle
//     moving >= min_v m/s that the stock filter would slow down for (ahead, within the lateral margin of the band), its
//     box is carried at its current velocity in the trajectory's frame (its speed along and toward ego's path and its
//     heading to it kept, as cutin_1027's check: a car following a bend beside ego keeps its offset) and ego's footprint
//     along the trajectory (from its speed, accelerating at 1 m/s2 up to the trajectory's speed: the closest ego could
//     get) every 0.1 s for t_s. Its speed toward ego's path is the trend of its box's MEASURED offset to the path over
//     the last 1 s (once 0.6 s of it exist; before: from its heading): the tracker's heading lags in a bend (the 79460
//     cars: 12-26 deg to the trajectory while their boxes stay in their lane); its speed changes at its measured
//     acceleration over the same second (-4..+1 m/s2, as ego's +1: a car held at a constant speed while ego accelerates was
//     "caught" 2.5-3 s out: 10-25 #20's 330df8, 4.7 -> 7.9 m/s). The vehicle is not in ego's way, and the
//     stock condition counts as not met, when
//       (separating) the two never come closer than min(now, keep_m) - 0.05 within t_s and are >= min_gap_m apart now:
//         a car merging in ahead faster than ego, one beside ego's nose already moving away, one ego does not reach; or
//       (beside) its heading is within max_deg of the trajectory, its offset to ego's path shrinks < approach_m within
//         t_s and the two never come closer than min_gap_m: a car keeping its own lane next to ego's; or
//       (passing) it is >= 0.5 m/s faster than ego along the trajectory and the two never come closer than pass_m (0 =
//         off): a car merging in beside / ahead of ego's nose that pulls away (10-25 #20's 330df8 came within 0.8 m
//         while 2 m/s faster); closer than pass_m stays stock (and cutin_1027's yield brakes below the same 0.5 m).
//     A vehicle moving toward ego's path, one ego catches up with, an angled track (ANG), a bicycle / pedestrian and
//     every UNKNOWN keep the stock slow-down; the stock 5-cycle hysteresis still applies on entry and exit.
//     Stock takes any vehicle within 1.1 m of the band by its current box and slows to 0.7-2.78 m/s (the car's moving
//     margins 0.2-0.7 m): a car in the 2.0 m wide next lane of the 79460 bend, or one merging in ahead that pulls away,
//     brought the robot from 15-20 km/h to <= 10 km/h (10-25 #20 / #22 / #23, 10-26 #13).
//   NERO_OSD_DEBUG=1 - log only: every candidate vehicle with the numbers above and what the switch would decide
//     ([NERO_OSD]); with NERO_OSD_NOT_IN_WAY unset it uses the default values for the log and changes nothing.
#ifndef NERO_OSD_HPP_
#define NERO_OSD_HPP_

#include <cstdio>
#include <cstdlib>
#include <deque>
#include <string>
#include <unordered_map>
#include <utility>

namespace autoware::motion_velocity_planner
{
struct NeroOsdParam
{
  bool enabled{false};
  double t_s{3.0};
  double keep_m{1.1};
  double max_deg{12.0};
  double min_gap_m{0.2};
  double approach_m{0.3};
  double min_v{1.0};
  double pass_m{0.0};
  bool debug{false};
};

// per-object memory: the measured offset to ego's path (seconds on the node clock, metres; left +)
struct NeroOsdHistory
{
  // (time, offset, speed) per object
  struct Sample
  {
    double t, d, v;
  };
  std::unordered_map<std::string, std::deque<Sample>> offsets;

  void add(const std::string & uuid, const double now, const double d, const double v)
  {
    auto & q = offsets[uuid];
    q.push_back({now, d, v});
    while (!q.empty() && now - q.front().t > 1.0) {
      q.pop_front();
    }
  }
  // least-squares rate of the offset (speed = false) or of the speed (speed = true) over the kept window, if it spans
  // >= 0.6 s
  bool rate(const std::string & uuid, double & r, const bool speed = false) const
  {
    const auto it = offsets.find(uuid);
    if (it == offsets.end() || it->second.size() < 3) {
      return false;
    }
    const auto & q = it->second;
    if (q.back().t - q.front().t < 0.6) {
      return false;
    }
    double st = 0.0, sd = 0.0, stt = 0.0, std_ = 0.0;
    const double t0 = q.front().t;
    for (const auto & smp : q) {
      const double t = smp.t;
      const double d = speed ? smp.v : smp.d;
      st += t - t0;
      sd += d;
      stt += (t - t0) * (t - t0);
      std_ += (t - t0) * d;
    }
    const double n = static_cast<double>(q.size());
    const double den = n * stt - st * st;
    if (den <= 1e-9) {
      return false;
    }
    r = (n * std_ - st * sd) / den;
    return true;
  }
  void prune(const double now)
  {
    for (auto it = offsets.begin(); it != offsets.end();) {
      it = (it->second.empty() || now - it->second.back().t > 2.0) ? offsets.erase(it) : std::next(it);
    }
  }
};

inline const NeroOsdParam & nero_osd_param()
{
  static const NeroOsdParam p = [] {
    NeroOsdParam r;
    if (const char * e = std::getenv("NERO_OSD_NOT_IN_WAY")) {
      r.enabled = std::sscanf(
                    e, "%lf,%lf,%lf,%lf,%lf,%lf,%lf", &r.t_s, &r.keep_m, &r.max_deg, &r.min_gap_m,
                    &r.approach_m, &r.min_v, &r.pass_m) >= 6 &&
                  r.t_s > 0.0;
    }
    if (const char * e = std::getenv("NERO_OSD_DEBUG")) {
      r.debug = std::string(e) == "1";
    }
    return r;
  }();
  return p;
}
}  // namespace autoware::motion_velocity_planner

#endif  // NERO_OSD_HPP_
