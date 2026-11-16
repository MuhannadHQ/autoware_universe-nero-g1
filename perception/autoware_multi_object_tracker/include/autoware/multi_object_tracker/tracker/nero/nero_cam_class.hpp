// Copyright 2026 Nero (tracker-1). Licensed under the Apache License, Version 2.0.
//
// NERO_MOT_CLASS_CAM=<iou_min>,<t_win s>,<n_min>,<agree>,<t_stale s>,<debug>[,<iou_unk>,<n_unk>] (tracker-1, 2026-11-15, Ahmed 19:3x: "the camera's vehicle
// class becomes the witness inside fix 2"; only with NERO_MOT_STAND_HOLD + NERO_MOT_CLASS_HOLD; unset = the car: no subscription, nothing
// changes). camera0's YOLOX boxes (rois0) say CAR on 77 % of tracked-car frames and 99 % of tracked-truck frames, and over 27 tracks of
// the 11-15 drive changed class 4 times against the tracker's 19 (fixes/6_cameras, "Car class"; B1 0594: CAR in 21 of 21 frames while
// the tracker flipped to TRUCK in 5). The witness:
// - only for a STANDING track (published speed <= v_still) whose class the car publishes as CAR, TRUCK or BUS; and (Ahmed 20:4x:
//   "its fine if the waste container is a vehicle for now", fields 7-8, absent = off) for a standing UNKNOWN that camera0 sees clearly as a
//   vehicle (IoU >= iou_unk, >= n_unk votes): its CLASS becomes the camera's, its published outline stays the lidar's exactly (a class
//   never changes the box: no car-size box from a 1.9 m skip as 11-15 B2). It never relabels a PEDESTRIAN, a BICYCLE, a MOTORCYCLE or a
//   TRAILER (the person check is cam-1's);
// - the track's published box (after the standing hold) is projected into camera0's rectified image with the robot's pose at the
//   camera frame's stamp; the best-overlapping YOLOX box votes its class when it is CAR / TRUCK / BUS with IoU >= iou_min, and no vote
//   when the best-overlapping box is anything else;
// - its votes of the last t_win s decide when there are >= n_min of them and one class has >= agree of them: that class is published, and
//   it stays the track's published class while the track stands (until the camera decides another class, the track moves, or its class
//   leaves CAR / TRUCK / BUS): the witness's word is not taken back by the lidar class when the camera loses sight of a standing car
//   (without this the camera and the lidar hold took turns: 19:01 trip CAR <-> TRUCK 0.70 -> 0.91 flips per track-minute);
// - no camera, a camera older than t_stale s, no calibration or too few votes = the lidar-only class hold (NERO_MOT_CLASS_HOLD).
// Lidar, filter, association, boxes and speeds are untouched.
#ifndef AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__NERO__NERO_CAM_CLASS_HPP_
#define AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__NERO__NERO_CAM_CLASS_HPP_

#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <mutex>
#include <vector>

namespace autoware::multi_object_tracker::nero_cam_class
{
struct Params
{
  bool enabled{false};
  double iou_min{0.3};
  double t_win{1.5};
  int n_min{3};
  double agree{0.67};
  double t_stale{1.0};
  int debug{0};
  double iou_unk{0.0};  // 0 = UNKNOWN -> vehicle off
  int n_unk{5};
};

// NERO_MOT_CAM_LENS=<f_model px>,<f_true px>,<k1>,<k2>,<k3>,<k4>[,<dyaw deg>,<dpitch deg>] (cam-1, 2026-11-16; unset = the camera_info
// pinhole as tracker-1): camera0's rectified image is the fisheye remapped by camera0_rect with the model focal f_model (921.73, Kannala-
// Brandt k1..k4); the true lens is f_true (cam-1 fit on 15,170 standing vehicles, 11-08: 986 px; datasheet ~1000): a ray at theta from the
// optical axis lands at the rectified angle phi = thd^-1(f_true / f_model * thd(theta)), thd(t) = t (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8).
// With the pinhole, objects on the image sides were drawn up to 35 px toward the centre (the 11-15 camera evidence: 1.5-1.7 deg). dyaw /
// dpitch rotate the camera about base_link's z / y on top of the tf (the fit's balance: +0.3 / 0 deg).
struct Lens
{
  bool on{false};
  double s{1.0};
  double k[4]{0.0, 0.0, 0.0, 0.0};
  double dyaw{0.0}, dpitch{0.0};
  std::vector<double> tg, td;  // theta grid and thd(theta) for the inverse
};
inline const Lens & lens()
{
  static const Lens L = [] {
    Lens q;
    const char * e = std::getenv("NERO_MOT_CAM_LENS");
    if (e == nullptr || e[0] == '\0') return q;
    double fm = 0.0, ft = 0.0, dy = 0.0, dp = 0.0;
    const int n = std::sscanf(e, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &fm, &ft, &q.k[0], &q.k[1], &q.k[2], &q.k[3], &dy, &dp);
    if (n < 6 || fm <= 0.0 || ft <= 0.0) {
      RCLCPP_ERROR(rclcpp::get_logger("multi_object_tracker"), "[NERO_MOT_CAM_LENS] malformed '%s': the camera_info pinhole", e);
      return q;
    }
    q.s = ft / fm;
    q.dyaw = dy * M_PI / 180.0;
    q.dpitch = dp * M_PI / 180.0;
    for (int i = 0; i <= 2900; ++i) {
      const double t = 1.45 * i / 2900.0, t2 = t * t;
      q.tg.push_back(t);
      q.td.push_back(t * (1.0 + q.k[0] * t2 + q.k[1] * t2 * t2 + q.k[2] * t2 * t2 * t2 + q.k[3] * t2 * t2 * t2 * t2));
    }
    q.on = true;
    RCLCPP_INFO(
      rclcpp::get_logger("multi_object_tracker"),
      "[NERO_MOT_CAM_LENS] on: f_model %.2f, f_true %.2f (s %.4f), k %.5f %.5f %.5f %.5f, camera rotated yaw %+.2f / pitch %+.2f deg", fm, ft,
      q.s, q.k[0], q.k[1], q.k[2], q.k[3], dy, dp);
    return q;
  }();
  return L;
}
// optical-frame point -> rectified pixel (the lens model when on, else the pinhole); false = behind / too far off-axis
inline bool rectPixel(double qx, double qy, double qz, double fx, double fy, double cx, double cy, double * u, double * v)
{
  const auto & L = lens();
  if (!L.on) {
    if (qz <= 0.0) return false;
    *u = fx * qx / qz + cx;
    *v = fy * qy / qz + cy;
    return true;
  }
  if (qz <= 0.0) return false;
  const double rr = std::hypot(qx, qy);
  const double th = std::atan2(rr, qz);
  const double t2 = th * th;
  const double d = L.s * th * (1.0 + L.k[0] * t2 + L.k[1] * t2 * t2 + L.k[2] * t2 * t2 * t2 + L.k[3] * t2 * t2 * t2 * t2);
  if (rr < 1e-9) {
    if (qz <= 0.0) return false;
    *u = cx;
    *v = cy;
    return true;
  }
  const auto it = std::lower_bound(L.td.begin(), L.td.end(), d);
  double ph = L.tg.back();  // beyond the grid (> ~75 deg off-axis): clamped, far outside the 100 deg rectified view
  if (it != L.td.end()) {
    const size_t i = std::max<size_t>(1, static_cast<size_t>(it - L.td.begin()));
    const double w = (d - L.td[i - 1]) / std::max(1e-12, L.td[i] - L.td[i - 1]);
    ph = L.tg[i - 1] + w * (L.tg[i] - L.tg[i - 1]);
  }
  const double k = std::tan(ph) / std::max(rr, 1e-9);
  *u = cx + fx * k * qx;
  *v = cy + fy * k * qy;
  return true;
}

inline const Params & params()
{
  static const Params p = [] {
    Params q;
    const char * e = std::getenv("NERO_MOT_CLASS_CAM");
    if (e == nullptr || e[0] == '\0') return q;
    q.enabled = std::sscanf(
                  e, "%lf,%lf,%d,%lf,%lf,%d,%lf,%d", &q.iou_min, &q.t_win, &q.n_min, &q.agree, &q.t_stale, &q.debug, &q.iou_unk,
                  &q.n_unk) >= 5 &&
                q.iou_min > 0.0 && q.t_win > 0.0 && q.n_min >= 1 && q.agree > 0.5 && q.t_stale > 0.0;
    if (q.enabled) {
      RCLCPP_INFO(
        rclcpp::get_logger("multi_object_tracker"),
        "[NERO_MOT_CLASS_CAM] on: IoU >= %.2f, votes of the last %.2f s (>= %d, one class >= %.2f), camera stale after %.2f s, debug %d; "
        "standing UNKNOWN -> vehicle %s (IoU >= %.2f, >= %d votes; the outline is never changed)",
        q.iou_min, q.t_win, q.n_min, q.agree, q.t_stale, q.debug, q.iou_unk > 0.0 ? "on" : "off", q.iou_unk, q.n_unk);
    }
    return q;
  }();
  return p;
}

struct Roi
{
  double x0, y0, x1, y1;
  int label;
};
struct Ego
{
  double t, x, y, z, yaw;
};

struct State
{
  std::mutex mtx;
  bool cam_ok{false};
  bool intr_ok{false};
  std::array<double, 9> R{};  // base_link <- camera optical (row-major)
  std::array<double, 3> T{};
  double fx{0.0}, fy{0.0}, cx{0.0}, cy{0.0};
  int w{0}, h{0};
  double rois_t{-1.0};
  std::vector<Roi> rois;
  std::deque<Ego> ego;
};

inline State & state()
{
  static State s;
  return s;
}

// cam-1: the robot's pose AT a camera frame's stamp. The node sets a lookup (the tracker's odometry / tf at the stamp, never waiting);
// without it (or when it has nothing yet) the pose history is interpolated between the two poses bracketing the stamp (never
// extrapolated; none = the frame is skipped). tracker-1's nearest-history pose was ~0.1 s older than the frame (the history is stamped at
// lidar measurement times, which arrive after the camera frame): boxes off by up to ~90 px for near objects while driving.
using EgoLookup = std::function<bool(double, Ego *)>;
inline EgoLookup & egoLookup()
{
  static EgoLookup f;
  return f;
}
inline void setEgoLookup(EgoLookup f) { egoLookup() = std::move(f); }
inline bool egoAt(double t, const std::deque<Ego> & hist, Ego * e)
{
  if (egoLookup() && egoLookup()(t, e)) return true;
  for (size_t i = 1; i < hist.size(); ++i) {
    const auto & a = hist[i - 1];
    const auto & b = hist[i];
    if (a.t <= t && t <= b.t && b.t - a.t <= 0.25) {
      const double w = (b.t > a.t) ? (t - a.t) / (b.t - a.t) : 0.0;
      e->t = t;
      e->x = a.x + w * (b.x - a.x);
      e->y = a.y + w * (b.y - a.y);
      e->z = a.z + w * (b.z - a.z);
      e->yaw = a.yaw + w * std::atan2(std::sin(b.yaw - a.yaw), std::cos(b.yaw - a.yaw));
      return true;
    }
  }
  return false;
}

inline void setCamera(const std::array<double, 9> & R0, const std::array<double, 3> & T)
{
  // NERO_MOT_CAM_LENS dyaw / dpitch: R = Rz(dyaw) Ry(dpitch) R0 (base_link <- optical, row-major)
  std::array<double, 9> R = R0;
  const auto & L = lens();
  if (L.on && (L.dyaw != 0.0 || L.dpitch != 0.0)) {
    const double cy = std::cos(L.dyaw), sy = std::sin(L.dyaw), cp = std::cos(L.dpitch), sp = std::sin(L.dpitch);
    const double M[9] = {cy * cp, -sy, cy * sp, sy * cp, cy, sy * sp, -sp, 0.0, cp};
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) R[3 * r + c] = M[3 * r] * R0[c] + M[3 * r + 1] * R0[3 + c] + M[3 * r + 2] * R0[6 + c];
  }
  auto & s = state();
  std::lock_guard<std::mutex> lk(s.mtx);
  s.R = R;
  s.T = T;
  s.cam_ok = true;
}
inline bool cameraOk()
{
  auto & s = state();
  std::lock_guard<std::mutex> lk(s.mtx);
  return s.cam_ok;
}
inline void setIntrinsics(double fx, double fy, double cx, double cy, int w, int h)
{
  auto & s = state();
  std::lock_guard<std::mutex> lk(s.mtx);
  s.fx = fx;
  s.fy = fy;
  s.cx = cx;
  s.cy = cy;
  s.w = w;
  s.h = h;
  s.intr_ok = fx > 0.0 && fy > 0.0 && w > 0 && h > 0;
}
inline void setRois(double t, std::vector<Roi> && rois)
{
  auto & s = state();
  std::lock_guard<std::mutex> lk(s.mtx);
  s.rois = std::move(rois);
  s.rois_t = t;
}
inline void setEgo(double t, double x, double y, double z, double yaw)
{
  auto & s = state();
  std::lock_guard<std::mutex> lk(s.mtx);
  if (!s.ego.empty() && t <= s.ego.back().t) return;
  s.ego.push_back({t, x, y, z, yaw});
  while (s.ego.size() > 128) s.ego.pop_front();
}

// The camera's class vote for a box (map frame; x, y centre, z centre, yaw, length, width, height) in the newest camera frame newer than
// after_t. Returns the frame stamp in *frame_t (or -1: no new usable frame) and the vote: 1 / 2 / 3 = CAR / TRUCK / BUS, 0 = no vote.
inline int vote(
  double now, double after_t, double bx, double by, double bz, double byaw, double bl, double bw, double bh, double * frame_t,
  double * iou_out, double iou_need = -1.0)
{
  const auto & p = params();
  auto & s = state();
  std::vector<Roi> rois;
  std::array<double, 9> R;
  std::array<double, 3> T;
  double fx, fy, cx, cy, rt;
  int W, H;
  Ego e{};
  std::deque<Ego> hist;
  *frame_t = -1.0;
  *iou_out = 0.0;
  {
    std::lock_guard<std::mutex> lk(s.mtx);
    if (!s.cam_ok || !s.intr_ok || s.rois_t < 0.0 || s.rois_t <= after_t || now - s.rois_t > p.t_stale || s.ego.empty()) return 0;
    rois = s.rois;
    rt = s.rois_t;
    R = s.R;
    T = s.T;
    fx = s.fx;
    fy = s.fy;
    cx = s.cx;
    cy = s.cy;
    W = s.w;
    H = s.h;
    hist = s.ego;
  }
  if (!egoAt(rt, hist, &e)) return 0;  // the robot at the frame's stamp (cam-1)
  *frame_t = rt;
  const double ce = std::cos(e.yaw), se = std::sin(e.yaw);
  const double c = std::cos(byaw), sn = std::sin(byaw);
  double u0 = 1e30, v0 = 1e30, u1 = -1e30, v1 = -1e30;
  for (const double a : {-0.5 * bl, 0.5 * bl}) {
    for (const double b : {-0.5 * bw, 0.5 * bw}) {
      for (const double dz : {-0.5 * bh, 0.5 * bh}) {
        const double mx = bx + a * c - b * sn - e.x;
        const double my = by + a * sn + b * c - e.y;
        const double px = ce * mx + se * my;
        const double py = -se * mx + ce * my;
        const double pz = bz + dz - e.z;
        const double vx = px - T[0], vy = py - T[1], vz = pz - T[2];
        const double ox = R[0] * vx + R[3] * vy + R[6] * vz;
        const double oy = R[1] * vx + R[4] * vy + R[7] * vz;
        const double oz = R[2] * vx + R[5] * vy + R[8] * vz;
        if (oz < 0.5) return 0;  // a corner behind / at the camera: no vote
        double u = 0.0, v = 0.0;
        if (!rectPixel(ox, oy, oz, fx, fy, cx, cy, &u, &v)) return 0;
        u0 = std::min(u0, u);
        v0 = std::min(v0, v);
        u1 = std::max(u1, u);
        v1 = std::max(v1, v);
      }
    }
  }
  // the box must be in view (clipped to the image, at least half of its projected area inside)
  const double a_full = (u1 - u0) * (v1 - v0);
  u0 = std::max(u0, 0.0);
  v0 = std::max(v0, 0.0);
  u1 = std::min(u1, static_cast<double>(W));
  v1 = std::min(v1, static_cast<double>(H));
  if (u1 <= u0 || v1 <= v0 || a_full <= 0.0 || (u1 - u0) * (v1 - v0) < 0.5 * a_full) return 0;
  double best_iou = 0.0;
  int best_label = -1;
  for (const auto & r : rois) {
    const double iw = std::min(u1, r.x1) - std::max(u0, r.x0);
    const double ih = std::min(v1, r.y1) - std::max(v0, r.y0);
    if (iw <= 0.0 || ih <= 0.0) continue;
    const double inter = iw * ih;
    const double uni = (u1 - u0) * (v1 - v0) + (r.x1 - r.x0) * (r.y1 - r.y0) - inter;
    const double iou = uni > 0.0 ? inter / uni : 0.0;
    if (iou > best_iou) {
      best_iou = iou;
      best_label = r.label;
    }
  }
  *iou_out = best_iou;
  if (best_iou < (iou_need > 0.0 ? iou_need : p.iou_min)) return 0;
  return (best_label == 1 || best_label == 2 || best_label == 3) ? best_label : 0;
}
}  // namespace autoware::multi_object_tracker::nero_cam_class

#endif  // AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__NERO__NERO_CAM_CLASS_HPP_
