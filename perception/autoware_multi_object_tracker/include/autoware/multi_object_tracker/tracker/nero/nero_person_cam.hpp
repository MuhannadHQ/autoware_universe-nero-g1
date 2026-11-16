// Copyright 2026 Nero (cam-1). Licensed under the Apache License, Version 2.0.
//
// NERO_MOT_PERSON_CAM=<n_clear>,<n_win>,<t_win s>,<t_stand s>,<v_still m/s>,<x_min m>,<x_max m>,<edge px>,<occ_max>,<iou_p>,<ioa_p>,<pix_p>,
//   <iou_o>,<w_o>,<t_stale s>[,<debug>] (cam-1, 2026-11-15; 11-15 Al Falah drive Area 6 use 1, "the person check"; unset = the car: no mask subscription,
// nothing changes). A STANDING track the car publishes as PEDESTRIAN or BICYCLE that camera0 sees clearly and that shows NO person in
// camera0 is published as UNKNOWN; its published shape (outline / box) and everything else stay exactly the car's. 11-15 B8: CenterPoint
// called a green wheeled waste container PEDESTRIAN / BICYCLE (no container class); static avoidance asked a pedestrian's 1.7 m and swerved
// 1.5 m, 0.41 m over the line with two cars passing (takeover). camera0's YOLOX (seg16cls) had 0 person boxes and 0 person pixels on it
// (75 frames) while the same frames held 85 person boxes on real people; its person boxes were real people 694 of 707 times on 11-08.
// Rule v2 (per track, per new camera0 frame; boxes projected with NERO_MOT_CAM_LENS when set; v1's eye check found a real man at a car's
// corner relabelled for 0.2 s: box beside him (lens), no YOLOX person box, positive evidence by mask alone):
// - standing: published speed <= v_still for >= t_stand s (any faster publish restarts the clock, empties the window, ends a relabel);
// - seen: the box centre x_min..x_max m ahead of base_link, its projected 3D box fully inside the rectified image (>= edge px from every
//   border, every corner in front of the camera), at most occ_max of it covered by the projected box of a NEARER published object;
// - person: a camera PEDESTRIAN / BICYCLE / MOTORCYCLE box with IoU >= iou_p on the box, or with >= ioa_p of it inside the WIDE region
//   (the box widened by max(half its width, 20 px) each side and 20 % up), or person + bike mask pixels in the wide region >= pix_p of
//   the box's area; ANY person frame keeps the track's car class for the rest of its life (sticky);
// - clear: seen, no person, and a camera CAR / TRUCK / BUS box with IoU >= iou_o on the box that is no wider than w_o x the box (the camera
//   box is this object, not a car behind it: containers read CAR, Ahmed 11-15 20:4x "its fine if the waste container is a vehicle for now");
// - relabelled (latched for the stand): standing >= t_stand with >= n_clear clear frames among its last n_win seen frames within t_win s
//   (a camera slower than n_clear / t_win frames per second never relabels) and no person frame ever; kept while it stands (also out of
//   view or occluded); ended by a moving publish, a non-PEDESTRIAN / BICYCLE class, any person frame, or camera0's boxes older than
//   t_stale s. No camera, a stale / dead camera, no calibration, no mask = the car's class.
// Lidar, filter, association, boxes, shapes, speeds and the object set are untouched; obstacle_stop / AEB do not read the class difference
// (see cam-1 DELIVERY). With NERO_MOT_CAM_SPLIT the relabel reaches planning only (detection_by_tracker keeps the car's class).
#ifndef AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__NERO__NERO_PERSON_CAM_HPP_
#define AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__NERO__NERO_PERSON_CAM_HPP_

#include "autoware/multi_object_tracker/tracker/nero/nero_cam_class.hpp"

#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <vector>

namespace autoware::multi_object_tracker::nero_person_cam
{
struct Params
{
  bool enabled{false};
  int n_clear{10};
  int n_win{15};
  double t_win{2.5};
  double t_stand{2.0};
  double v_still{0.3};
  double x_min{4.0};
  double x_max{25.0};
  double edge{4.0};
  double occ_max{0.2};
  double iou_p{0.1};
  double ioa_p{0.3};
  double pix_p{0.02};
  double iou_o{0.3};
  double w_o{2.0};
  double t_stale{0.5};
  int debug{0};
};

inline const Params & params()
{
  static const Params p = [] {
    Params q;
    const char * e = std::getenv("NERO_MOT_PERSON_CAM");
    if (e == nullptr || e[0] == '\0') return q;
    const int n = std::sscanf(
      e, "%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%d", &q.n_clear, &q.n_win, &q.t_win, &q.t_stand, &q.v_still, &q.x_min,
      &q.x_max, &q.edge, &q.occ_max, &q.iou_p, &q.ioa_p, &q.pix_p, &q.iou_o, &q.w_o, &q.t_stale, &q.debug);
    q.enabled = n >= 15 && q.n_clear >= 1 && q.n_win >= q.n_clear && q.t_win > 0.0 && q.t_stand >= 0.0 && q.x_max > q.x_min &&
                q.t_stale > 0.0 && q.iou_p > 0.0 && q.ioa_p > 0.0 && q.pix_p > 0.0;
    if (q.enabled) {
      RCLCPP_INFO(
        rclcpp::get_logger("multi_object_tracker"),
        "[NERO_MOT_PERSON_CAM] on: a standing (<= %.2f m/s for %.1f s) PEDESTRIAN / BICYCLE camera0 sees %.1f-%.1f m ahead, fully in view "
        "(%.0f px), occluded <= %.2f, with >= %d clear of its last %d frames within %.1f s (person: box IoU >= %.2f or >= %.2f of it in the "
        "wide region, person pixels in the wide region >= %.3f of the box; other: a vehicle box IoU >= %.2f no wider than %.1f x), camera "
        "<= %.2f s old -> UNKNOWN (shape kept), latched while it stands; any person frame = never; debug %d",
        q.v_still, q.t_stand, q.x_min, q.x_max, q.edge, q.occ_max, q.n_clear, q.n_win, q.t_win, q.iou_p, q.ioa_p, q.pix_p, q.iou_o, q.w_o,
        q.t_stale, q.debug);
    } else {
      RCLCPP_ERROR(rclcpp::get_logger("multi_object_tracker"), "[NERO_MOT_PERSON_CAM] malformed '%s': off (the car)", e);
    }
    return q;
  }();
  return p;
}

// any camera0 consumer in the tracker (subscriptions, the robot's pose history)
inline bool anyCam() { return nero_cam_class::params().enabled || params().enabled; }

// camera0's 16-class mask (tensorrt_yolox ~/out/mask: run-length mono8, 1 byte class + 4 byte count per run), the newest frame
struct MaskState
{
  std::mutex mtx;
  double t{-1.0};
  int w{0}, h{0};
  std::vector<uint8_t> rle;
};
inline MaskState & maskState()
{
  static MaskState s;
  return s;
}
inline void setMask(double t, int w, int h, const std::vector<uint8_t> & data)
{
  auto & s = maskState();
  std::lock_guard<std::mutex> lk(s.mtx);
  s.t = t;
  s.w = w;
  s.h = h;
  s.rle = data;
}

// one camera frame decoded for the evaluation of every candidate of one publish
struct Frame
{
  bool ok{false};
  double t{-1.0};
  std::vector<nero_cam_class::Roi> rois;
  std::array<double, 9> R{};
  std::array<double, 3> T{};
  double fx{0.0}, fy{0.0}, cx{0.0}, cy{0.0};
  int W{0}, H{0};
  nero_cam_class::Ego ego{};
  std::vector<uint8_t> mask;  // W x H class indices (decoded), empty = none
};

// the newest frame whose boxes and mask share one stamp, fresh (<= t_stale) and newer than after_t; ok = false otherwise
inline Frame snapshot(double now)
{
  const auto & p = params();
  Frame f;
  auto & s = nero_cam_class::state();
  auto & m = maskState();
  std::vector<uint8_t> rle;
  std::deque<nero_cam_class::Ego> hist;
  int mw = 0, mh = 0;
  double mt = -1.0;
  {
    std::lock_guard<std::mutex> lk(m.mtx);
    rle = m.rle;
    mw = m.w;
    mh = m.h;
    mt = m.t;
  }
  {
    std::lock_guard<std::mutex> lk(s.mtx);
    if (!s.cam_ok || !s.intr_ok || s.rois_t < 0.0 || now - s.rois_t > p.t_stale || s.ego.empty()) return f;
    if (mt < 0.0 || std::fabs(mt - s.rois_t) > 1e-4) return f;  // the mask of the same YOLOX call only
    if (mw != s.w || mh != s.h) return f;
    f.rois = s.rois;
    f.t = s.rois_t;
    f.R = s.R;
    f.T = s.T;
    f.fx = s.fx;
    f.fy = s.fy;
    f.cx = s.cx;
    f.cy = s.cy;
    f.W = s.w;
    f.H = s.h;
    hist = s.ego;
  }
  if (!nero_cam_class::egoAt(f.t, hist, &f.ego)) return f;  // the robot at the frame's stamp
  // decode the run-length mask
  const size_t n_px = static_cast<size_t>(f.W) * static_cast<size_t>(f.H);
  f.mask.assign(n_px, 0);
  size_t pos = 0;
  for (size_t i = 0; i + 5 <= rle.size() && pos < n_px; i += 5) {
    int32_t cnt = 0;
    std::memcpy(&cnt, &rle[i + 1], sizeof(int32_t));
    if (cnt <= 0) continue;
    const size_t c = std::min(static_cast<size_t>(cnt), n_px - pos);
    std::fill(f.mask.begin() + static_cast<std::ptrdiff_t>(pos), f.mask.begin() + static_cast<std::ptrdiff_t>(pos + c), rle[i]);
    pos += c;
  }
  if (pos != n_px) return f;  // a short / broken mask: no frame
  f.ok = true;
  return f;
}

// a published object's projection into the frame: image box (unclipped), all corners in front, centre ahead of base_link, range
struct Proj
{
  bool front{false};
  double u0{0}, v0{0}, u1{0}, v1{0};
  double ahead{0}, range{0};
};
// corners: footprint points (object frame) x {bottom, top}; centre (map), yaw, height
inline Proj project(const Frame & f, double ox, double oy, double oz, double yaw, double h, const std::vector<std::array<double, 2>> & fp)
{
  Proj r;
  const double ce = std::cos(f.ego.yaw), se = std::sin(f.ego.yaw);
  const double c = std::cos(yaw), sn = std::sin(yaw);
  r.u0 = r.v0 = 1e30;
  r.u1 = r.v1 = -1e30;
  const double hh = 0.5 * std::max(h, 0.3);
  double sx = 0.0, sy = 0.0;
  for (const auto & q : fp) {
    for (const double dz : {-hh, hh}) {
      const double mx = ox + q[0] * c - q[1] * sn - f.ego.x;
      const double my = oy + q[0] * sn + q[1] * c - f.ego.y;
      const double px = ce * mx + se * my;
      const double py = -se * mx + ce * my;
      const double pz = oz + dz - f.ego.z;
      sx += px;
      sy += py;
      const double vx = px - f.T[0], vy = py - f.T[1], vz = pz - f.T[2];
      const double qx = f.R[0] * vx + f.R[3] * vy + f.R[6] * vz;
      const double qy = f.R[1] * vx + f.R[4] * vy + f.R[7] * vz;
      const double qz = f.R[2] * vx + f.R[5] * vy + f.R[8] * vz;
      if (qz < 0.5) return r;  // a corner behind / at the camera
      double u = 0.0, v = 0.0;
      if (!nero_cam_class::rectPixel(qx, qy, qz, f.fx, f.fy, f.cx, f.cy, &u, &v)) return r;
      r.u0 = std::min(r.u0, u);
      r.v0 = std::min(r.v0, v);
      r.u1 = std::max(r.u1, u);
      r.v1 = std::max(r.v1, v);
    }
  }
  const double n = 2.0 * static_cast<double>(fp.size());
  r.ahead = sx / n;
  r.range = std::hypot(sx / n - f.T[0], sy / n - f.T[1]);
  r.front = true;
  return r;
}

inline double area(double u0, double v0, double u1, double v1) { return std::max(0.0, u1 - u0) * std::max(0.0, v1 - v0); }
inline double inter(const Proj & a, double u0, double v0, double u1, double v1)
{
  return area(std::max(a.u0, u0), std::max(a.v0, v0), std::min(a.u1, u1), std::min(a.v1, v1));
}

// the frame's evidence for one candidate (its projection p, the nearer objects' projections for occlusion)
struct Evidence
{
  bool seen{false};
  bool person{false};
  bool other{false};
  double occ{0.0};
  double pix_person{0.0};  // person + bike pixels in the wide region / the box's area
  double iou_person{0.0};
  double iou_vehicle{0.0};
};
inline Evidence evaluate(const Frame & f, const Proj & p, const std::vector<Proj> & others)
{
  const auto & q = params();
  Evidence e;
  if (!p.front) return e;
  const double W = static_cast<double>(f.W), H = static_cast<double>(f.H);
  if (p.u0 < q.edge || p.v0 < q.edge || p.u1 > W - q.edge || p.v1 > H - q.edge) return e;  // fully in view
  if (p.ahead < q.x_min || p.ahead > q.x_max) return e;
  const double a = area(p.u0, p.v0, p.u1, p.v1);
  if (a < 1.0) return e;
  for (const auto & o : others) {
    if (!o.front || o.range >= p.range - 0.3) continue;  // only objects nearer to the camera can hide it
    const double ou0 = std::max(o.u0, 0.0), ov0 = std::max(o.v0, 0.0), ou1 = std::min(o.u1, W), ov1 = std::min(o.v1, H);
    e.occ = std::max(e.occ, inter(p, ou0, ov0, ou1, ov1) / a);
  }
  if (e.occ > q.occ_max) return e;
  e.seen = true;
  const double bw = p.u1 - p.u0;
  const double mw = std::max(0.5 * bw, 20.0);
  Proj wide = p;
  wide.u0 = std::max(0.0, p.u0 - mw);
  wide.u1 = std::min(W, p.u1 + mw);
  wide.v0 = std::max(0.0, p.v0 - 0.2 * (p.v1 - p.v0));
  for (const auto & r : f.rois) {
    const double ra = std::max(1e-6, (r.x1 - r.x0) * (r.y1 - r.y0));
    const double I = inter(p, r.x0, r.y0, r.x1, r.y1);
    const double iou = I > 0.0 ? I / (a + ra - I) : 0.0;
    if (r.label == 5 || r.label == 6 || r.label == 7) {  // MOTORCYCLE / BICYCLE / PEDESTRIAN
      e.iou_person = std::max(e.iou_person, iou);
      if (iou >= q.iou_p || inter(wide, r.x0, r.y0, r.x1, r.y1) / ra >= q.ioa_p) e.person = true;
    } else if (r.label == 1 || r.label == 2 || r.label == 3) {  // CAR / TRUCK / BUS: the object itself, not a car behind it
      e.iou_vehicle = std::max(e.iou_vehicle, iou);
      if (iou >= q.iou_o && (r.x1 - r.x0) <= q.w_o * bw) e.other = true;
    }
  }
  // person (6) + bike (8) pixels of seg16cls's mask in the wide region, counted against the box's own area
  const int u0 = std::max(0, static_cast<int>(std::floor(wide.u0))), v0 = std::max(0, static_cast<int>(std::floor(wide.v0)));
  const int u1 = std::min(f.W, static_cast<int>(std::ceil(wide.u1))), v1 = std::min(f.H, static_cast<int>(std::ceil(wide.v1)));
  size_t np = 0;
  for (int v = v0; v < v1; ++v) {
    const uint8_t * row = f.mask.data() + static_cast<size_t>(v) * static_cast<size_t>(f.W);
    for (int u = u0; u < u1; ++u) {
      if (row[u] == 6 || row[u] == 8) ++np;
    }
  }
  e.pix_person = static_cast<double>(np) / a;
  if (e.pix_person >= q.pix_p) e.person = true;
  return e;
}

// per-track state (kept by the tracker, written only by the processor's publication)
struct State
{
  double stand_t0{-1.0};
  double frame_t{-1.0};             // the newest camera frame taken
  std::deque<std::array<double, 3>> win;  // seen frames while standing: frame stamp, person (0 / 1), clear (0 / 1)
  bool person_ever{false};
  bool latched{false};  // relabelled for this stand
};
}  // namespace autoware::multi_object_tracker::nero_person_cam

#endif  // AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__NERO__NERO_PERSON_CAM_HPP_
