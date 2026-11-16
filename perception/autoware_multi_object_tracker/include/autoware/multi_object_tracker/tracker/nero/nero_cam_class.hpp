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

inline void setCamera(const std::array<double, 9> & R, const std::array<double, 3> & T)
{
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
    // the robot at the frame's stamp (nearest measurement pose; the poses come ~10 Hz)
    double best = 1e300;
    for (const auto & q : s.ego) {
      if (std::fabs(q.t - rt) < best) {
        best = std::fabs(q.t - rt);
        e = q;
      }
    }
    if (best > 0.15) return 0;
  }
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
        const double u = fx * ox / oz + cx, v = fy * oy / oz + cy;
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
