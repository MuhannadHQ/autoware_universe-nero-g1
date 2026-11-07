// Copyright 2026 TIER IV, Inc.
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

// NERO_MOT_SIGN_BELIEF (signbelief, 2026-11-06): port of upstream autoware_universe #13050 (0966df772, merged
// 2026-07-28) "orientation sign belief for heading flip decisions" into the 0.44.2 tracker. Unset = the car exactly.
//
// The 0.44.2 vehicle tracker folds every measured yaw within +-90 deg of the track (updateStatePoseHead), so a track
// keeps the heading SIGN of the box that started it; the only turn is limitStates' reverse-speed flip (or
// NERO_MOT_FLIP_DEFER's judged form of it), which turns the state but not its covariance. With this switch:
//   - every box with an orientation, from a channel that trusts it, votes BEFORE its yaw is folded: agrees (< 90 deg) or
//     disagrees (> 90 deg) with the track, no vote within the dead zone around 90 deg; the votes accumulate in a scalar
//     Kalman estimate of the agreement (+1 right, -1 wrong) that relaxes toward an uninformed prior (time constant tau);
//   - the tracked longitudinal speed adds evidence at decision time only, weighted by speed (saturating at
//     vote_vel_par) and by its own certainty (halved at vote_vel_var), so only sustained, well-estimated reverse motion
//     can override the boxes;
//   - a confidently negative fused agreement turns the WHOLE state by exactly 180 deg (yaw + pi, speed negated, the speed
//     cross-covariances negated; the anchor offset too) and negates the belief (hysteresis);
//   - the reverse-speed flip of limitStates is gone for vehicle tracks, and NERO_MOT_FLIP_DEFER's judged turn with it
//     (one mechanism: a vehicle track turns only by this belief);
//   - the two layers of a vehicle track follow the one with the stronger belief; the pedestrian layer of a pedestrian /
//     bicycle track follows its bicycle layer (field ped_follow).
// NERO_MOT_SIGN_BELIEF=1 = upstream's constants; or 11 fields:
//   tau,stationary_var,r_yaw_available,r_yaw_sign_unknown,dead_zone_deg,r_vel,vote_vel_par,vote_vel_var,flip_z,
//   flip_min_agreement,ped_follow[,range_ref,score_ref[,measured_v_cap]] (range_ref / score_ref > 0 weight each vote by the
//   box's range from the robot / its score; measured_v_cap (default 4 m/s): a speed measured by NERO_MOT_FLIP_DEFER's 0.2 s of
//   positions counts as certain evidence whose weight grows with the speed up to it)[,fwd_lock] (default 0.8 m/s: no turn makes a
//   track that moves forward at >= fwd_lock a reverser; a car really reversing keeps its heading because it is never turned)
// NERO_MOT_SIGN_DEBUG=1 logs every heading turn ([NERO_MOT_SIGN], also the car's own limitStates turns when the belief is
// off: research), =2 also every vote.

#ifndef AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__MODEL__NERO_SIGN_BELIEF_HPP_
#define AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__MODEL__NERO_SIGN_BELIEF_HPP_

#include <rclcpp/logging.hpp>
#include <rclcpp/time.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace autoware::multi_object_tracker
{

struct NeroSignBeliefParams
{
  bool enabled{false};
  double tau{3.0};                 // [s] evidence memory time constant
  double stationary_var{0.25};     // [-] variance of the uninformed agreement prior
  double r_yaw_available{0.5};     // [-] yaw-vote noise variance, sign known
  double r_yaw_sign_unknown{2.0};  // [-] yaw-vote noise variance, sign unknown
  double dead_zone{M_PI / 6.0};    // [rad] no-vote band around |yaw_diff| = 90 deg
  double r_vel{0.1};               // [-] velocity-evidence noise variance at the par speed
  double vote_vel_par{5.0 / 3.6};  // [m/s] speed where the velocity evidence saturates
  double vote_vel_var{1.0};        // [m^2/s^2] velocity variance where its certainty halves
  double flip_z{1.3};              // [-] posterior confidence gate (z-score) for flipping
  double flip_min_agreement{0.2};  // [-] fused-agreement magnitude floor for flipping
  double range_ref{0.0};           // [m] > 0: a vote's noise x max(range, 3 m) / range_ref (near views see the body; 0 = upstream)
  double score_ref{0.0};           // [-] > 0: a vote's noise x score_ref / max(score, 0.1) (0 = upstream)
  double measured_v_cap{4.0};      // [m/s] a FLIP_DEFER-measured speed's evidence grows with it up to here (field 14)
  double fwd_lock{0.8};            // [m/s] a turn may not make a track moving forward at >= this a reverser (field 15; 0 = off)
  bool ped_follow{false};          // the pedestrian layer follows the bicycle layer's sign (field 11; off by default: signbelief sb1 ping-pong)
  int debug{0};                    // NERO_MOT_SIGN_DEBUG
  NeroSignBeliefParams()
  {
    if (const char * env = std::getenv("NERO_MOT_SIGN_DEBUG"); env != nullptr && env[0] != '\0') {
      debug = std::atoi(env);
    }
    const char * env = std::getenv("NERO_MOT_SIGN_BELIEF");
    if (env == nullptr || env[0] == '\0' || std::string(env) == "0") return;
    if (std::string(env) == "1") {
      enabled = true;
    } else {
      double dz_deg = 30.0;
      int ped = 0;
      const int n = std::sscanf(
        env, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%d,%lf,%lf,%lf,%lf", &tau, &stationary_var, &r_yaw_available,
        &r_yaw_sign_unknown, &dz_deg, &r_vel, &vote_vel_par, &vote_vel_var, &flip_z, &flip_min_agreement, &ped,
        &range_ref, &score_ref, &measured_v_cap, &fwd_lock);
      enabled = n == 11 || n == 13 || n == 14 || n == 15;
      dead_zone = dz_deg * M_PI / 180.0;
      ped_follow = ped != 0;
    }
    if (enabled) {
      RCLCPP_INFO(
        rclcpp::get_logger("VehicleTracker"),
        "[NERO_MOT_SIGN_BELIEF] on: tau %.2f stationary_var %.3f r_yaw %.2f / %.2f dead_zone %.0f deg r_vel %.3f "
        "vel_par %.3f vel_var %.2f flip_z %.2f min_agreement %.2f ped_follow %d range_ref %.1f score_ref %.2f "
        "measured_v_cap %.2f fwd_lock %.2f",
        tau, stationary_var, r_yaw_available, r_yaw_sign_unknown, dead_zone * 180.0 / M_PI, r_vel, vote_vel_par,
        vote_vel_var, flip_z, flip_min_agreement, ped_follow ? 1 : 0, range_ref, score_ref, measured_v_cap, fwd_lock);
    } else {
      RCLCPP_WARN(
        rclcpp::get_logger("VehicleTracker"), "[NERO_MOT_SIGN_BELIEF] '%s' not understood: off (the car)", env);
    }
  }
};

inline const NeroSignBeliefParams & neroSignBeliefParams()
{
  static const NeroSignBeliefParams params;
  return params;
}

// Heading-sign belief (upstream OrientationSignBelief, the same arithmetic): detection-yaw votes accumulate in a scalar
// OU-Kalman agreement estimate (+1 = sign correct, -1 = wrong) that relaxes toward an uninformed prior over time; the
// tracked longitudinal velocity contributes memory-less at decision time, weighted by speed and velocity certainty.
class NeroSignBelief
{
public:
  NeroSignBelief() = default;
  // initial_sign: 2 = AVAILABLE, 1 = SIGN_UNKNOWN, 0 = UNAVAILABLE (types::OrientationAvailability's values)
  void init(const int initial_sign, const rclcpp::Time & time)
  {
    const auto & p = neroSignBeliefParams();
    variance_ = p.stationary_var;
    agreement_ = 0.0;
    last_update_time_ = time;
    // the seeding detection also seeds the tracker yaw, so it counts as one agreeing vote
    if (initial_sign == 2) {
      correct(1.0, p.r_yaw_available);
    } else if (initial_sign == 1) {
      correct(1.0, p.r_yaw_sign_unknown);
    }
    fused_agreement_ = agreement_;
    fused_variance_ = variance_;
    votes_ = 0;
  }

  // yaw_diff: normalized measurement yaw minus tracker yaw [rad]; returns +1 / -1 for a vote, 0 in the dead zone
  // range [m] (< 0 unknown) and score of the box weight the vote when range_ref / score_ref are set
  int vote(
    const rclcpp::Time & time, const double yaw_diff, const bool is_sign_known, const double range = -1.0,
    const double score = 1.0)
  {
    const auto & p = neroSignBeliefParams();
    predictTo(time);
    // near-perpendicular boxes carry axis ambiguity, not sign evidence
    if (std::abs(std::abs(yaw_diff) - M_PI_2) < p.dead_zone) return 0;
    const double z = std::abs(yaw_diff) < M_PI_2 ? 1.0 : -1.0;
    double r = is_sign_known ? p.r_yaw_available : p.r_yaw_sign_unknown;
    if (p.range_ref > 0.0 && range >= 0.0) r *= std::clamp(std::max(range, 3.0) / p.range_ref, 0.3, 4.0);
    if (p.score_ref > 0.0) r *= p.score_ref / std::max(score, 0.1);
    correct(z, r);
    ++votes_;
    return z > 0.0 ? 1 : -1;
  }

  // fuses the accumulated agreement with the instantaneous velocity-sign evidence; true = turn the track
  bool shouldFlip(const rclcpp::Time & time, const double vel_long, const double vel_var)
  {
    const auto & p = neroSignBeliefParams();
    predictTo(time);
    const double vel_ratio = vel_long / p.vote_vel_par;
    const double z_vel = std::clamp(vel_ratio, -1.0, 1.0);
    const double info_vel = std::min(vel_ratio * vel_ratio, 1.0) / p.r_vel * p.vote_vel_var / (p.vote_vel_var + vel_var);
    fused_variance_ = 1.0 / (1.0 / variance_ + info_vel);
    fused_agreement_ = (agreement_ / variance_ + z_vel * info_vel) * fused_variance_;
    return fused_agreement_ + p.flip_z * std::sqrt(fused_variance_) < 0.0 &&
           fused_agreement_ < -p.flip_min_agreement;
  }

  // NERO (signbelief): the same decision for a speed MEASURED from positions (NERO_MOT_FLIP_DEFER's 0.2 s judgement of a sudden
  // reverse speed): certain, and its weight grows with the speed up to v_cap (reversing at 5 m/s is far less plausible than at
  // 1.4 m/s; upstream's evidence saturates at vote_vel_par)
  bool shouldFlipMeasured(const rclcpp::Time & time, const double vel_long, const double v_cap)
  {
    const auto & p = neroSignBeliefParams();
    predictTo(time);
    const double ratio = std::min(std::abs(vel_long), std::max(v_cap, p.vote_vel_par)) / p.vote_vel_par;
    const double z_vel = vel_long > 0.0 ? std::min(ratio, 1.0) : -std::min(ratio, 1.0);
    const double info_vel = ratio * ratio / p.r_vel;
    fused_variance_ = 1.0 / (1.0 / variance_ + info_vel);
    fused_agreement_ = (agreement_ / variance_ + z_vel * info_vel) * fused_variance_;
    return fused_agreement_ + p.flip_z * std::sqrt(fused_variance_) < 0.0 &&
           fused_agreement_ < -p.flip_min_agreement;
  }

  // a 180 deg state flip turns every past disagreeing vote into an agreeing one
  void onFlipped()
  {
    agreement_ = -agreement_;
    fused_agreement_ = -fused_agreement_;
  }

  double agreement() const { return agreement_; }
  double variance() const { return variance_; }
  // yaw-vote agreement magnitude as a z-score
  double confidence() const { return std::abs(agreement_) / std::sqrt(variance_); }
  double fusedAgreement() const { return fused_agreement_; }
  double fusedVariance() const { return fused_variance_; }
  int votes() const { return votes_; }

private:
  // OU relaxation toward the (0, stationary_var) prior over the elapsed time; zero elapsed time leaves the state
  // untouched, so the boxes of one frame fuse independently of their order
  void predictTo(const rclcpp::Time & time)
  {
    const double dt = (time - last_update_time_).seconds();
    if (dt <= 0.0) return;
    const auto & p = neroSignBeliefParams();
    last_update_time_ = time;
    const double alpha = std::exp(-dt / p.tau);
    agreement_ *= alpha;
    variance_ = alpha * alpha * variance_ + (1.0 - alpha * alpha) * p.stationary_var;
  }

  // scalar Kalman update of the agreement estimate
  void correct(const double z, const double r)
  {
    const double gain = variance_ / (variance_ + r);
    agreement_ += gain * (z - agreement_);
    variance_ *= (1.0 - gain);
  }

  double agreement_{0.0};
  double variance_{0.25};
  rclcpp::Time last_update_time_{0, 0, RCL_ROS_TIME};
  double fused_agreement_{0.0};
  double fused_variance_{0.25};
  int votes_{0};
};

}  // namespace autoware::multi_object_tracker

#endif  // AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__MODEL__NERO_SIGN_BELIEF_HPP_
