// Copyright 2026 Nero (vstoplag). Licensed under the Apache License, Version 2.0.
//
// NERO_MOT_STAND_SPEED (vstoplag, 2026-11-04): the reported speed of a vehicle that stops or starts again.
// The vehicle tracker's constant-velocity Kalman filter (0.35 g process noise, positions only) read a car's stop ~0.4 s and
// its restart ~0.5 s after its own boxes did (+ 0.2 s detection latency in what planning gets). With the switch:
//  - (field 1 > 0 only, a research option) the vehicle models' longitudinal process noise is q_g [g]; 0 = the model's own
//    (the car's filter untouched: the shipped value; 0.7-1.0 g read stops sooner but held slow movers as standing);
//  - the PUBLISHED speed of a vehicle whose boxes stand is 0 (state untouched, as Apollo's MLF motion refiner): STAND when
//    the box end facing the robot and the box centre both stand (least-squares speed over the last 4 frames / >= 0.25 s
//    < 0.3 m/s) and the filter reads < 0.6 m/s; RELEASE when both are displaced the same way beyond k x their own spread
//    over the stand at 2 consecutive frames, or the filter agrees with such a displacement (>= 0.5 m/s at 2 frames), or both
//    moved > 1 m;
//  - after a stand of >= 0.8 s, a start seen on 3 growing, rigid frames beyond 3 sigma (>= 0.15 m) is published at the speed
//    those frames measure (2 d / t since the last standing frame) until the filter catches up (<= 1 s).
// Per frame = one detection stamp: its channels' boxes (a box the tracker applies twice counts once); the state before the
// frame is kept so a later channel of the same stamp re-evaluates the frame. Unset = the car (nothing here runs).
#ifndef AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__MODEL__NERO_STAND_SPEED_HPP_
#define AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__MODEL__NERO_STAND_SPEED_HPP_

#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <vector>

namespace autoware::multi_object_tracker
{

struct NeroStandSpeedParams
{
  bool enabled{false};
  double q_g{0.0};      // vehicle longitudinal process noise [g]; <= 0: the model's own (the car: 0.35)
  double k{2.0};        // release: displacement beyond k x the point's own spread over the stand
  bool boost{true};     // publish a measured start speed after a stand
  // fixed (vstoplag-3 offline design on the car's update sequence: bag1027 + five 10-29 trips)
  int n_in{4};
  double t_in{0.25};  // 4 frames = 3 intervals (frame stamps jitter around 0.1 s)
  double v_in{0.3};
  double v_fast_in{0.6};
  double v_kf_max{4.0};
  double t_ref{2.0};
  int n_in_re{7};
  double v_in_re{0.2};
  double e_min{0.05};
  double sig_max{0.12};
  double sig0{0.07};
  int n_out{2};
  double v_agree{0.5};
  double d_safety{1.0};
  double t_reanchor{1.0};
  double t_boost{1.0};
  double t_post{1.0};
  int n_boost{3};
  double g_min{0.0};
  double k_boost{3.0};
  double d_boost_min{0.15};
  double rigid_tol{0.15};
  double v_boost_fast{0.2};
  double t_stand_boost{0.8};
};

inline const NeroStandSpeedParams & neroStandSpeedParams()
{
  static const NeroStandSpeedParams p = [] {
    NeroStandSpeedParams q;
    const char * env = std::getenv("NERO_MOT_STAND_SPEED");
    if (env != nullptr && env[0] != '\0') {
      int boost = 1;
      q.enabled = std::sscanf(env, "%lf,%lf,%d", &q.q_g, &q.k, &boost) == 3 && q.k > 0.0;
      q.boost = boost != 0;
      if (q.enabled) {
        RCLCPP_INFO(
          rclcpp::get_logger("VehicleTracker"),
          "[NERO_MOT_STAND_SPEED] on: process noise %s%.2f g, release k %.2f, start boost %d",
          q.q_g > 0.0 ? "" : "the model's (field 0) ", q.q_g > 0.0 ? q.q_g : 0.35, q.k,
          q.boost ? 1 : 0);
      }
    }
    return q;
  }();
  return p;
}

class NeroStandSpeed
{
public:
  struct Box
  {
    double x, y, yaw, length, anchor_x;
  };

  // one measurement of the frame at time t (channel ch), after the filter's update: its heading and speed
  void measure(const double t, const unsigned ch, const Box & box, const double kf_yaw, const double kf_v)
  {
    if (!has_frame_ || std::fabs(t - frame_t_) > 1e-4) {
      before_ = cur_;  // commit the previous frame
      has_frame_ = true;
      frame_t_ = t;
      boxes_.clear();
      channels_.clear();
    }
    if (std::find(channels_.begin(), channels_.end(), ch) == channels_.end()) {
      channels_.push_back(ch);
      boxes_.push_back(box);
    }
    cur_ = before_;
    step(t, kf_yaw, kf_v);
  }

  bool valid() const { return has_frame_; }
  bool standing() const { return cur_.standing; }
  double output() const { return cur_.v_out; }

  // NERO_MOT_SIGN_BELIEF (signbelief_1106): the track's heading was turned by exactly 180 deg (its speed negated). Everything
  // here is kept along the heading: seen from the turned heading the same frames have front and rear swapped and every
  // displacement, start direction and output speed negated (the stand / release / start decisions continue unchanged).
  void onHeadingTurned()
  {
    turnState(before_);
    turnState(cur_);
  }

private:
  enum { F = 0, R = 1, C = 2 };

  static int swappedKey(const int k) { return k == F ? R : (k == R ? F : k); }

  template <class Frames>
  static void turnFrames(Frames & frames)
  {
    for (auto & f : frames) {
      std::swap(f.p[F], f.p[R]);
      f.key = swappedKey(f.key);
    }
  }
  struct Frame
  {
    double t;
    std::array<std::array<double, 2>, 3> p;  // front / rear end centre, box centre (map)
    int key;                                  // the point facing the robot
  };
  struct Hist
  {
    double t, dn, dc;
    int key;
  };
  struct State
  {
    std::deque<Frame> h, band, allst;
    bool standing{false};
    int pend{0};
    double last_d{0.0};
    int agree{0};
    bool has_out_since{false};
    double out_since{0.0};
    double t_dep{0.0}, t_st{0.0}, t_rel{-1e9};
    bool post{false};
    std::array<std::array<double, 2>, 3> post_an{};
    std::array<double, 3> post_sg{};
    double post_t_dep{0.0}, post_t_rel{0.0};
    std::vector<Hist> post_hist;
    bool boost{false};
    double boost_v{0.0}, boost_until{0.0}, boost_dir{1.0};
    double v_out{0.0};
  };

  static void turnState(State & s)
  {
    turnFrames(s.h);
    turnFrames(s.band);
    turnFrames(s.allst);
    s.last_d = -s.last_d;
    std::swap(s.post_an[F], s.post_an[R]);
    std::swap(s.post_sg[F], s.post_sg[R]);
    for (auto & q : s.post_hist) {
      q.dn = -q.dn;
      q.dc = -q.dc;
      q.key = swappedKey(q.key);
    }
    s.boost_dir = -s.boost_dir;
    s.v_out = -s.v_out;
  }

  State before_, cur_;
  bool has_frame_{false};
  double frame_t_{0.0};
  std::vector<Box> boxes_;
  std::vector<unsigned> channels_;

  static double proj(const std::array<double, 2> & a, const double ux, const double uy) { return a[0] * ux + a[1] * uy; }

  static double median(std::vector<double> v)
  {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
  }

  static double slope(const std::vector<double> & t, const std::vector<double> & y)
  {
    double mt = 0.0, my = 0.0;
    for (size_t i = 0; i < t.size(); ++i) {
      mt += t[i];
      my += y[i];
    }
    mt /= t.size();
    my /= t.size();
    double stt = 0.0, sty = 0.0;
    for (size_t i = 0; i < t.size(); ++i) {
      stt += (t[i] - mt) * (t[i] - mt);
      sty += (t[i] - mt) * (y[i] - my);
    }
    return stt > 1e-9 ? sty / stt : 0.0;
  }

  static std::array<double, 2> mean(const std::deque<Frame> & fr, const int key)
  {
    std::array<double, 2> m{0.0, 0.0};
    for (const auto & f : fr) {
      m[0] += f.p[key][0];
      m[1] += f.p[key][1];
    }
    m[0] /= fr.size();
    m[1] /= fr.size();
    return m;
  }

  static double sigma(const std::deque<Frame> & fr, const int key, const double ux, const double uy)
  {
    const size_t n0 = fr.size() > 30 ? fr.size() - 30 : 0;
    std::vector<double> q;
    for (size_t i = n0; i < fr.size(); ++i) q.push_back(proj(fr[i].p[key], ux, uy));
    const double m = median(q);
    for (auto & x : q) x = std::fabs(x - m);
    return 1.4826 * median(q);
  }

  void step(const double t, const double yaw, const double vfast)
  {
    const auto & P = neroStandSpeedParams();
    State & s = cur_;
    const double ux = std::cos(yaw), uy = std::sin(yaw);
    // the frame's points
    Frame fr;
    fr.t = t;
    fr.p = {};
    fr.key = C;
    for (size_t i = 0; i < boxes_.size(); ++i) {
      const auto & b = boxes_[i];
      double bx = std::cos(b.yaw), by = std::sin(b.yaw), ax = b.anchor_x;
      if (bx * ux + by * uy < 0.0) {
        bx = -bx;
        by = -by;
        ax = -ax;
      }
      const double h = 0.5 * b.length;
      fr.p[F][0] += b.x + h * bx;
      fr.p[F][1] += b.y + h * by;
      fr.p[R][0] += b.x - h * bx;
      fr.p[R][1] += b.y - h * by;
      fr.p[C][0] += b.x;
      fr.p[C][1] += b.y;
      if (i == 0) fr.key = std::fabs(ax) < 1e-6 ? C : (ax > 0.0 ? F : R);
    }
    for (auto & q : fr.p) {
      q[0] /= boxes_.size();
      q[1] /= boxes_.size();
    }
    const int key = fr.key;
    s.h.push_back(fr);
    if (s.h.size() > 40) s.h.pop_front();
    double out = vfast;
    if (!s.standing) {
      const bool re = t - s.t_rel < P.t_ref;
      const int n_in = re ? P.n_in_re : P.n_in;
      const double v_in = re ? P.v_in_re : P.v_in;
      if (static_cast<int>(s.h.size()) >= n_in) {
        const size_t a = s.h.size() - n_in;
        const double span = s.h.back().t - s.h[a].t;
        if (
          span >= P.t_in - 1e-6 && span <= 0.15 * n_in && std::fabs(vfast) < P.v_kf_max &&
          std::fabs(vfast) < P.v_fast_in) {
          std::vector<double> tw, yn, yc;
          for (size_t i = a; i < s.h.size(); ++i) {
            tw.push_back(s.h[i].t);
            yn.push_back(proj(s.h[i].p[key], ux, uy));
            yc.push_back(proj(s.h[i].p[C], ux, uy));
          }
          if (std::fabs(slope(tw, yn)) < v_in && std::fabs(slope(tw, yc)) < v_in) {
            s.standing = true;
            s.band.assign(s.h.begin() + a, s.h.end());
            s.allst = s.band;
            s.t_st = s.h[a].t;
            s.pend = 0;
            s.last_d = 0.0;
            s.agree = 0;
            s.has_out_since = false;
            s.t_dep = t;
            s.boost = false;
            s.post = false;
          }
        }
      }
    } else {
      const auto an = mean(s.band, key), ac = mean(s.band, C);
      double sn = P.sig0, sc = P.sig0;
      if (s.allst.size() >= 5) {
        sn = sigma(s.allst, key, ux, uy);
        sc = sigma(s.allst, C, ux, uy);
      }
      sn = std::min(std::max(sn, P.e_min), P.sig_max);
      sc = std::min(std::max(sc, P.e_min), P.sig_max);
      const double dn = (fr.p[key][0] - an[0]) * ux + (fr.p[key][1] - an[1]) * uy;
      const double dc = (fr.p[C][0] - ac[0]) * ux + (fr.p[C][1] - ac[1]) * uy;
      const double thr_n = P.k * sn, thr_c = P.k * sc;
      const bool same = (dn > 0.0) == (dc > 0.0);
      const bool in_band = std::fabs(dn) <= thr_n && std::fabs(dc) <= thr_c;
      const bool disp = same && std::fabs(dn) > thr_n && std::fabs(dc) > thr_c &&
                        std::fabs(dn) >= std::fabs(s.last_d) - 0.02;
      if (!disp) {
        s.allst.push_back(fr);
        if (s.allst.size() > 40) s.allst.pop_front();
      }
      if (in_band) {
        s.band.push_back(fr);
        if (s.band.size() > 30) s.band.pop_front();
        s.pend = 0;
        s.last_d = 0.0;
        s.has_out_since = false;
        s.t_dep = t;
      } else if (disp) {
        ++s.pend;
        s.last_d = dn;
      } else {
        s.pend = 0;
        s.last_d = 0.0;
        if (!s.has_out_since) {
          s.has_out_since = true;
          s.out_since = t;
        } else if (t - s.out_since >= P.t_reanchor) {
          s.band.clear();
          for (const auto & f : s.h)
            if (f.t >= s.out_since) s.band.push_back(f);
          s.allst = s.band;
          s.has_out_since = false;
        }
      }
      if (
        !in_band && same && std::fabs(dn) > P.e_min && std::fabs(dc) > P.e_min &&
        std::fabs(vfast) >= P.v_agree && (vfast > 0.0) == (dn > 0.0)) {
        ++s.agree;
      } else {
        s.agree = 0;
      }
      const bool safety = same && std::fabs(dn) > P.d_safety && std::fabs(dc) > P.d_safety;
      if (s.pend >= P.n_out || s.agree >= 2 || safety) {
        s.standing = false;
        s.t_rel = t;
        if (P.boost && t - s.t_st >= P.t_stand_boost) {
          s.post = true;
          for (int k = 0; k < 3; ++k) {
            s.post_an[k] = mean(s.band, k);
            const double g = s.allst.size() >= 5 ? sigma(s.allst, k, ux, uy) : P.sig0;
            s.post_sg[k] = std::min(std::max(g, P.e_min), P.sig_max);
          }
          s.post_t_dep = s.t_dep;
          s.post_t_rel = t;
          s.post_hist.clear();
        }
      }
    }
    if (!s.standing && s.post) {
      if (t - s.post_t_rel > P.t_post) {
        s.post = false;
      } else {
        const double dn2 = (fr.p[key][0] - s.post_an[key][0]) * ux + (fr.p[key][1] - s.post_an[key][1]) * uy;
        const double dc2 = (fr.p[C][0] - s.post_an[C][0]) * ux + (fr.p[C][1] - s.post_an[C][1]) * uy;
        s.post_hist.push_back({t, dn2, dc2, key});
        const int n = static_cast<int>(s.post_hist.size());
        if (!s.boost && n >= P.n_boost) {
          bool same_key = true, grow = true, rigid = true;
          const double sg = dn2 > 0.0 ? 1.0 : -1.0;
          for (int i = n - P.n_boost; i < n; ++i) {
            same_key = same_key && s.post_hist[i].key == key;
            rigid = rigid && std::fabs(s.post_hist[i].dn - s.post_hist[i].dc) < P.rigid_tol;
            if (i + 1 < n) grow = grow && sg * (s.post_hist[i + 1].dn - s.post_hist[i].dn) > P.g_min;
          }
          const bool big = sg * dn2 > std::max(P.k_boost * s.post_sg[key], P.d_boost_min) &&
                           sg * dc2 > P.k_boost * s.post_sg[C];
          if (same_key && grow && big && rigid && sg * vfast >= P.v_boost_fast) {
            s.boost = true;
            s.boost_dir = sg;
            s.boost_v = std::min(3.0, 2.0 * std::fabs(dn2) / (t - s.post_t_dep + 0.1));
            s.boost_until = t + P.t_boost;
            s.post = false;
          }
        }
      }
    }
    if (s.standing) {
      out = 0.0;
    } else if (s.boost && t <= s.boost_until) {
      if (vfast * s.boost_dir >= s.boost_v) {
        s.boost = false;
      } else {
        out = s.boost_dir * s.boost_v;
      }
    }
    s.v_out = out;
  }
};

}  // namespace autoware::multi_object_tracker

#endif  // AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__MODEL__NERO_STAND_SPEED_HPP_
