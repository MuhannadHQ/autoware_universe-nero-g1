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
#ifndef AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__NERO__NERO_STAND_SPEED_HPP_
#define AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__NERO__NERO_STAND_SPEED_HPP_

#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <string>
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
  // field 4 (camghost, 2026-11-10; absent / 0 = vstoplag_1104 as on the car): 1 = a frame's start evidence comes only from boxes of
  // channels that trust the box's extension AND orientation (CenterPoint / validator boxes). detection_by_tracker re-fits a box of the
  // TRACK's size and heading to the clusters (its channel trusts neither): its ends move with every re-fit heading (11-08 14:25: DBT
  // yaw 141 / 110 every frame -> a "rigid growing start" -> the boost published 2.8 m/s for a parked car). A frame with only such boxes
  // ("weak") never counts toward a release (pending / agree) or the boost; only the 1 m safety displacement still releases on it.
  bool trusted_evidence{false};
  // field 5 (camghost; absent / 0 = as on the car): the boost is re-verified on every later frame: it ends as soon as the near end
  // falls back more than this (m) from where it was when the boost fired (a re-fit / heading flip of a standing car's box steps forward
  // and back; a real start keeps its displacement). The first published speed of a start is unchanged.
  double reverify{0.0};
  // field 6 (camghost-2, 2026-11-10; absent / 0 = as on the car): the boost needs a fixed viewpoint. The robot's own motion may have
  // turned the view of the standing place (the stand's mean centre) by at most this many degrees between the stand's last frame and
  // the fire. CenterPoint's and detection_by_tracker's boxes of a parked car slide 0.4-0.7 m along it while the robot drives past
  // beside it (the visible part of the car changes): 12 of the 15 near-path phantoms only the boost made on the 10-27 field inputs.
  // Real starts were seen from a fixed view (40 of 41 must-keep starts < 3.5 deg). A held boost leaves the speed to the filter.
  double view_max_deg{0.0};
};

inline const NeroStandSpeedParams & neroStandSpeedParams()
{
  static const NeroStandSpeedParams p = [] {
    NeroStandSpeedParams q;
    const char * env = std::getenv("NERO_MOT_STAND_SPEED");
    if (env != nullptr && env[0] != '\0') {
      int boost = 1, trusted = 0;
      const int n =
        std::sscanf(env, "%lf,%lf,%d,%d,%lf,%lf", &q.q_g, &q.k, &boost, &trusted, &q.reverify, &q.view_max_deg);
      if (n < 5) q.reverify = 0.0;
      if (n < 6) q.view_max_deg = 0.0;
      q.enabled = n >= 3 && q.k > 0.0;
      q.boost = boost != 0;
      q.trusted_evidence = n >= 4 && trusted != 0;
      if (q.enabled) {
        RCLCPP_INFO(
          rclcpp::get_logger("VehicleTracker"),
          "[NERO_MOT_STAND_SPEED] on: process noise %s%.2f g, release k %.2f, start boost %d, trusted-box evidence %d, boost "
          "re-verify %.2f m, boost view gate %.1f deg",
          q.q_g > 0.0 ? "" : "the model's (field 0) ", q.q_g > 0.0 ? q.q_g : 0.35, q.k,
          q.boost ? 1 : 0, q.trusted_evidence ? 1 : 0, q.reverify, q.view_max_deg);
      }
    }
    return q;
  }();
  return p;
}

// research (NERO_MOT_STAND_DEBUG, camghost-3): the boost window's frames as " t:dn:dc" (relative to the release)
template <class H>
inline std::string neroPostHistStr(const H & h)
{
  std::string out;
  char b[64];
  const double t0 = h.empty() ? 0.0 : h.front().t;
  for (const auto & e : h) {
    std::snprintf(b, sizeof(b), " %.3f:%.3f:%.3f", e.t - t0, e.dn, e.dc);
    out += b;
  }
  return out;
}

class NeroStandSpeed
{
public:
  struct Box
  {
    double x, y, yaw, length, anchor_x;
    bool trusted{true};  // field 4: the channel trusts the box's extension and orientation
    bool has_ego{false};  // field 6: the robot's position when the box was measured (map)
    double ego_x{0.0}, ego_y{0.0};
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
    // research (NERO_MOT_STAND_DEBUG): the boxes' largest heading deviation from the track [rad], boxes used / trusted, length
    double dyaw{0.0};
    int n_used{0}, n_trusted{0};
    double len{0.0};
    bool has_ego{false};  // field 6
    double ego_x{0.0}, ego_y{0.0};
  };
  static bool debugOn()
  {
    static const bool on = [] {
      const char * e = std::getenv("NERO_MOT_STAND_DEBUG");
      return e != nullptr && std::atoi(e) == 1;
    }();
    return on;
  }
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
    std::array<double, 3> post_sg_raw{};  // research: the stand's measured spread before the floor / cap
    int post_n_allst{0};
    double post_stood{0.0};
    double post_t_dep{0.0}, post_t_rel{0.0};
    std::vector<Hist> post_hist;
    bool boost{false};
    double boost_v{0.0}, boost_until{0.0}, boost_dir{1.0};
    double boost_dn{0.0};  // field 5: the near end's displacement when the boost fired
    // field 6: the robot's position at the stand's last frame (t_dep), and as kept for the post-release window
    bool dep_ego{false}, post_dep_ego{false};
    double dep_ex{0.0}, dep_ey{0.0}, post_dep_ex{0.0}, post_dep_ey{0.0};
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
    std::swap(s.post_sg_raw[F], s.post_sg_raw[R]);
    for (auto & q : s.post_hist) {
      q.dn = -q.dn;
      q.dc = -q.dc;
      q.key = swappedKey(q.key);
    }
    s.boost_dir = -s.boost_dir;
    s.boost_dn = -s.boost_dn;
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
    // field 4: only the trusted boxes make the frame when there is one; otherwise the frame is weak (no start evidence)
    bool any_trusted = false;
    for (const auto & b : boxes_) any_trusted = any_trusted || b.trusted;
    const bool weak = P.trusted_evidence && !any_trusted;
    size_t n_used = 0;
    bool first_used = true;
    for (size_t i = 0; i < boxes_.size(); ++i) {
      const auto & b = boxes_[i];
      if (P.trusted_evidence && any_trusted && !b.trusted) continue;
      ++n_used;
      double bx = std::cos(b.yaw), by = std::sin(b.yaw), ax = b.anchor_x;
      if (bx * ux + by * uy < 0.0) {
        bx = -bx;
        by = -by;
        ax = -ax;
      }
      fr.dyaw = std::max(fr.dyaw, std::acos(std::min(1.0, std::max(-1.0, bx * ux + by * uy))));
      fr.n_trusted += b.trusted ? 1 : 0;
      fr.len += b.length;
      if (b.has_ego && !fr.has_ego) {
        fr.has_ego = true;
        fr.ego_x = b.ego_x;
        fr.ego_y = b.ego_y;
      }
      const double h = 0.5 * b.length;
      fr.p[F][0] += b.x + h * bx;
      fr.p[F][1] += b.y + h * by;
      fr.p[R][0] += b.x - h * bx;
      fr.p[R][1] += b.y - h * by;
      fr.p[C][0] += b.x;
      fr.p[C][1] += b.y;
      if (first_used) fr.key = std::fabs(ax) < 1e-6 ? C : (ax > 0.0 ? F : R);
      first_used = false;
    }
    for (auto & q : fr.p) {
      q[0] /= static_cast<double>(n_used);
      q[1] /= static_cast<double>(n_used);
    }
    fr.n_used = static_cast<int>(n_used);
    fr.len /= static_cast<double>(n_used);
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
            s.dep_ego = fr.has_ego;
            s.dep_ex = fr.ego_x;
            s.dep_ey = fr.ego_y;
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
      const bool disp = !weak && same && std::fabs(dn) > thr_n && std::fabs(dc) > thr_c &&
                        std::fabs(dn) >= std::fabs(s.last_d) - 0.02;
      if (weak) {
        // field 4: a weak frame neither enters the stand's band nor counts toward a release (the safety test below still runs)
      } else if (!disp) {
        s.allst.push_back(fr);
        if (s.allst.size() > 40) s.allst.pop_front();
      }
      if (weak) {
        // field 4: unchanged pending / agree / band
      } else if (in_band) {
        s.band.push_back(fr);
        if (s.band.size() > 30) s.band.pop_front();
        s.pend = 0;
        s.last_d = 0.0;
        s.has_out_since = false;
        s.t_dep = t;
        s.dep_ego = fr.has_ego;
        s.dep_ex = fr.ego_x;
        s.dep_ey = fr.ego_y;
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
      if (weak) {
        // field 4: no agreement counted on a weak frame (kept as it was)
      } else if (
        !in_band && same && std::fabs(dn) > P.e_min && std::fabs(dc) > P.e_min &&
        std::fabs(vfast) >= P.v_agree && (vfast > 0.0) == (dn > 0.0)) {
        ++s.agree;
      } else {
        s.agree = 0;
      }
      const bool safety = same && std::fabs(dn) > P.d_safety && std::fabs(dc) > P.d_safety;
      if (s.pend >= P.n_out || s.agree >= 2 || safety) {
        if (debugOn()) {
          RCLCPP_INFO(
            rclcpp::get_logger("VehicleTracker"),
            "[NERO_MOT_STAND_DBG] release t %.3f at (%.2f, %.2f) why %s dn %.2f dc %.2f thr %.2f/%.2f vf %.2f stood %.2f s dyaw %.0f "
            "used %d trusted %d len %.2f weak %d",
            t, fr.p[C][0], fr.p[C][1], s.pend >= P.n_out ? "pend" : (s.agree >= 2 ? "agree" : "safety"), dn, dc, thr_n, thr_c,
            vfast, t - s.t_st, fr.dyaw * 57.2958, fr.n_used, fr.n_trusted, fr.len, weak ? 1 : 0);
        }
        s.standing = false;
        s.t_rel = t;
        if (P.boost && t - s.t_st >= P.t_stand_boost) {
          s.post = true;
          for (int k = 0; k < 3; ++k) {
            s.post_an[k] = mean(s.band, k);
            const double g = s.allst.size() >= 5 ? sigma(s.allst, k, ux, uy) : P.sig0;
            s.post_sg_raw[k] = g;  // research (NERO_MOT_STAND_DEBUG): before the floor / cap
            s.post_sg[k] = std::min(std::max(g, P.e_min), P.sig_max);
          }
          s.post_n_allst = static_cast<int>(s.allst.size());
          s.post_stood = t - s.t_st;
          s.post_t_dep = s.t_dep;
          s.post_dep_ego = s.dep_ego;
          s.post_dep_ex = s.dep_ex;
          s.post_dep_ey = s.dep_ey;
          s.post_t_rel = t;
          s.post_hist.clear();
        }
      }
    }
    if (!s.standing && s.post) {
      if (t - s.post_t_rel > P.t_post) {
        s.post = false;
      } else if (weak) {
        // field 4: a weak frame adds no start evidence for the boost
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
          // field 6: the view of the standing place turned by the robot's own motion since the stand's last frame [deg]
          double view = 0.0;
          if (P.view_max_deg > 0.0 && s.post_dep_ego && fr.has_ego) {
            const double ax = s.post_an[C][0] - s.post_dep_ex, ay = s.post_an[C][1] - s.post_dep_ey;
            const double bx = s.post_an[C][0] - fr.ego_x, by = s.post_an[C][1] - fr.ego_y;
            view = std::fabs(std::atan2(ax * by - ay * bx, ax * bx + ay * by)) * 57.29578;
          }
          const bool view_ok = P.view_max_deg <= 0.0 || view <= P.view_max_deg;
          if (same_key && grow && big && rigid && sg * vfast >= P.v_boost_fast && !view_ok && debugOn()) {
            RCLCPP_INFO(
              rclcpp::get_logger("VehicleTracker"),
              "[NERO_MOT_STAND_DBG] boost held (view) t %.3f at (%.2f, %.2f) v %.2f dn2 %.2f dc2 %.2f vf %.2f view %.1f",
              t, fr.p[C][0], fr.p[C][1], std::min(3.0, 2.0 * std::fabs(dn2) / (t - s.post_t_dep + 0.1)), dn2, dc2, vfast, view);
          }
          if (same_key && grow && big && rigid && sg * vfast >= P.v_boost_fast && view_ok) {
            if (debugOn()) {
              double dmax = 0.0, lmin = 1e9, lmax = 0.0;
              int nt = 0;
              const int nh = static_cast<int>(s.h.size());
              for (int i = std::max(0, nh - P.n_boost); i < nh; ++i) {
                dmax = std::max(dmax, s.h[i].dyaw);
                lmin = std::min(lmin, s.h[i].len);
                lmax = std::max(lmax, s.h[i].len);
                nt += s.h[i].n_trusted > 0 ? 1 : 0;
              }
              RCLCPP_INFO(
                rclcpp::get_logger("VehicleTracker"),
                "[NERO_MOT_STAND_DBG] boost t %.3f at (%.2f, %.2f) v %.2f dn2 %.2f dc2 %.2f vf %.2f dyaw_max %.0f trusted_frames %d/%d "
                "len %.2f..%.2f view %.1f sgraw %.3f/%.3f sg %.3f/%.3f nst %d stood %.2f range %.1f tdep %.3f trel %.3f hist%s",
                t, fr.p[C][0], fr.p[C][1], std::min(3.0, 2.0 * std::fabs(dn2) / (t - s.post_t_dep + 0.1)), dn2, dc2, vfast,
                dmax * 57.2958, nt, P.n_boost, lmin, lmax, view, s.post_sg_raw[key], s.post_sg_raw[C], s.post_sg[key], s.post_sg[C],
                s.post_n_allst, s.post_stood,
                fr.has_ego ? std::hypot(fr.p[C][0] - fr.ego_x, fr.p[C][1] - fr.ego_y) : -1.0, s.post_t_dep, s.post_t_rel,
                neroPostHistStr(s.post_hist).c_str());
            }
            s.boost = true;
            s.boost_dn = dn2;
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
      bool fell_back = false;
      if (P.reverify > 0.0 && !weak) {
        // field 5: the near end relative to the stand's anchor, now vs when the boost fired
        const double dn_now = (fr.p[key][0] - s.post_an[key][0]) * ux + (fr.p[key][1] - s.post_an[key][1]) * uy;
        fell_back = s.boost_dir * dn_now < s.boost_dir * s.boost_dn - P.reverify;
        if (fell_back && debugOn()) {
          RCLCPP_INFO(
            rclcpp::get_logger("VehicleTracker"), "[NERO_MOT_STAND_DBG] boost end (fell back) t %.3f at (%.2f, %.2f) dn %.2f fired %.2f",
            t, fr.p[C][0], fr.p[C][1], dn_now, s.boost_dn);
        }
      }
      if (vfast * s.boost_dir >= s.boost_v || fell_back) {
        s.boost = false;
      } else {
        out = s.boost_dir * s.boost_v;
      }
    }
    s.v_out = out;
  }
};

}  // namespace autoware::multi_object_tracker

#endif  // AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__NERO__NERO_STAND_SPEED_HPP_
