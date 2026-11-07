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

// NERO_IX_YIELD_REVOKE=<margin m>,<t_lag s>[,<a_soft m/s2>,<d_soft m>,<a_strong m/s2>]
//   (yieldrevoke job, 2026-11-03, QUEUE YREV; e.g. 0.5,0.4)
// Ahmed 10-23: "in saudi you get the occasional crazy driver that doesnt respect the right of way ... we are still ready
// for him". A driver who stopped for the robot (NERO_IX_YIELD_INTENT's yield) and then drives on after all is re-judged
// on his real motion, also after the robot committed:
//  1. the creep rule's (NERO_IX_CREEP) speed history is recorded for every candidate, also while the yield rule leaves
//     it out: the creep rule judged the revoked yielder on samples from before his yield and exempted him again as a
//     "creeper" in the very cycle the yield rule revoked him for speeding up (+0.3-0.4 s before any stop);
//  2. once the yield rule revokes a yield because the car speeds up, no other exemption takes him as yielding for 3 s
//     (until he yields again): not the creep rule, not the priority rule's "must yield to ego (map stop line)" (it
//     re-exempted him for another 0.2 s), and he is checked with the car's pull-away model (NERO_IX_PULLAWAY_ACC) from
//     the revocation on. The revocation itself is yieldsignal_1101's (no new trigger: a restart read from his box
//     positions would add ~800 false revocations per hour of standing vehicles on the car's tracker for a median
//     +0.1 s; research/f3census);
//  3. after a permanent GO latched while he was left out (its "safe" rested on him), the module stops for him once he
//     drives on and is unsafe, if ego's own stop from here (<t_lag> s at the current acceleration, then the
//     controller's docking stop: <a_soft> for <d_soft> m, then <a_strong>; NERO_IX_SIGHT's model) ends at least
//     <margin> m short of his way; else it keeps the commit (port of upstream e672f2dd9's stop after the commit for
//     merging lanes, generalised to his own way: his box, his predicted paths and his lanes on through every following
//     lanelet). Decided once per driver (latched): never a flip between stop and go. Before the commit the module's
//     own collision stop stands as it is (1 and 2 only bring it sooner).
// Unset = nothing of this runs.

#ifndef NERO_YIELD_REVOKE_HPP_
#define NERO_YIELD_REVOKE_HPP_

#include <autoware_perception_msgs/msg/predicted_object.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>

namespace nero_yield_revoke
{
struct Param
{
  bool on{false};
  double margin{0.5};
  double t_lag{0.4};
  double a_soft{0.8};
  double d_soft{0.5};
  double a_strong{3.4};
};

inline const Param & param()
{
  static const Param p = [] {
    Param r;
    const char * e = std::getenv("NERO_IX_YIELD_REVOKE");
    if (e == nullptr) return r;
    double v[5] = {r.margin, r.t_lag, r.a_soft, r.d_soft, r.a_strong};
    const int n = std::sscanf(e, "%lf,%lf,%lf,%lf,%lf", &v[0], &v[1], &v[2], &v[3], &v[4]);
    if (n < 2 || v[0] < 0.0) return r;
    r.on = true;
    r.margin = v[0];
    r.t_lag = std::max(0.0, v[1]);
    if (n >= 5 && v[2] > 0.0 && v[3] >= 0.0 && v[4] > 0.0) {
      r.a_soft = v[2];
      r.d_soft = v[3];
      r.a_strong = v[4];
    }
    return r;
  }();
  return p;
}

inline std::string key(const autoware_perception_msgs::msg::PredictedObject & o)
{
  return std::string(reinterpret_cast<const char *>(o.object_id.uuid.data()), o.object_id.uuid.size());
}

// objects stamp of each car's last revocation for speeding up (any module; cleared when it yields again)
inline std::unordered_map<std::string, double> & revoked()
{
  static std::unordered_map<std::string, double> r;
  return r;
}

inline void mark(const autoware_perception_msgs::msg::PredictedObject & o, const double t, const bool yields_now)
{
  auto & r = revoked();
  if (yields_now) {
    r.erase(key(o));
  } else {
    r[key(o)] = t;
  }
  static double last_prune = 0.0;
  if (t - last_prune > 5.0 || t < last_prune) {
    last_prune = t;
    for (auto it = r.begin(); it != r.end();) {
      if (std::abs(t - it->second) > 10.0) it = r.erase(it); else ++it;
    }
  }
}

// it was revoked for speeding up within the last 3 s and has not yielded again
inline bool drives_on(const autoware_perception_msgs::msg::PredictedObject & o, const double t)
{
  if (!param().on) return false;
  const auto it = revoked().find(key(o));
  return it != revoked().end() && t - it->second >= -1e-3 && t - it->second <= 3.0;
}

// the distance the car's own stop takes from v (m/s) at acceleration a0 (m/s2): t_lag at a0 (>= 0), then the
// controller's docking stop (a_soft for d_soft, then a_strong); NERO_IX_SIGHT's model of the same stop
inline double stop_distance(const double v, const double a0)
{
  const auto & p = param();
  const double a = std::max(0.0, a0);
  const double v0 = std::max(0.0, v);
  const double v1 = v0 + a * p.t_lag;
  const double d_lag = v0 * p.t_lag + 0.5 * a * p.t_lag * p.t_lag;
  const double d_soft_full = v1 * v1 / (2.0 * p.a_soft);
  const double d_brake = d_soft_full <= p.d_soft
                           ? d_soft_full
                           : p.d_soft + std::max(0.0, v1 * v1 - 2.0 * p.a_soft * p.d_soft) / (2.0 * p.a_strong);
  return d_lag + d_brake;
}
}  // namespace nero_yield_revoke

#endif  // NERO_YIELD_REVOKE_HPP_
