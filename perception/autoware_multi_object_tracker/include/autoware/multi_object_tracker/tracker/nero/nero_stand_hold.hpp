// Copyright 2026 Nero (tracker-1). Licensed under the Apache License, Version 2.0.
//
// NERO_MOT_STAND_HOLD=<v_still>,<v_rel>,<k_rel>,<r_disp>,<k_disp>,<v_disp>,<r_dir>,<k_dir>,<v_dir>,<g_min>,<gy_min>,<gs_min>,<pre s>,
//   <t_settle s>,<n0>,<t_back s>,<r_back m>,<r_twd m>,<k_twd>,<debug>   (tracker-1, 2026-11-15; unset = the car: nothing here runs, the published objects are byte-identical).
// The upstream tracker (trkup_1110, on the car since 11-11) turns and shifts the PUBLISHED box of a standing vehicle from frame to
// frame: on the 11-08 replays 7.7 % of its frames turn > 2 deg (old car tracker 0.4 %), the centre jumps 0.119 m p90 (0.079), and the
// union of a standing car's boxes reaches 0.51 m past its median box sideways (0.17). Research twin (per-update trace): 80 % of the
// > 2 deg turns come from detection_by_tracker's wheel-anchor refits (they move ONE end of the car to a cluster edge and so rotate the
// box about the other end), 60 % of the centre motion from the merged CenterPoint boxes' along-car jitter. Static avoidance keeps the
// union of a target's boxes (its envelope), so the wander became go-arounds (11-15: 11 of 15 captures). trkup's filter change
// (NERO_MOT_DBT_CENTRE) doubled phantoms, so this rule is OUTPUT ONLY:
// - a vehicle track (CAR / TRUCK / BUS / TRAILER, a bounding box >= 0.5 m long) whose published speed is <= v_still is held;
// - the held box comes from the track's OWN trusted boxes only (a measurement channel that trusts extent and orientation, a box,
//   the normal update path: CenterPoint through the merger), never from detection_by_tracker's refits: a running estimate with
//   gain max(1/(n + n0), g_min) for the centre, max(1/(n + n0), gy_min) for the heading (folded, the published sign kept) and
//   max(1/(n + n0), gs_min) for the size, started from the car's published box (worth n0 boxes) and the trusted boxes of the last <pre> s (the old 0.44 tracker: a running mean of CenterPoint headings, time
//   constant ~13 s; here ~5 s heading / ~2 s centre so a slow drift of the measurement is followed). No trusted box yet = as the car;
// - released (the box as the car publishes it) at once on the car's own motion reading: published speed > v_rel for k_rel publishes
//   in a row (k_rel 1 = the first publish the car calls it moving: its first published motion keeps the car's time and position), or
//   the car's box moved > r_dir the way its forward speed (> v_dir) points for k_dir publishes, or lies > r_disp from the held centre
//   for k_disp publishes at >= v_disp. A release after which the car published at most ONE moving speed (> v_rel) and its box is
//   back within r_back of the held one when it stands again within t_back resumes the estimate (the car's one-frame false speeds of
//   a standing car, ~1 % of its frames); any other release ends the estimate;
// - toward the robot: when the car's box CENTRE is closer to the robot (base_link), or for a centre ahead of the robot closer to the
//   robot's heading line, than the held centre by > r_twd on k_twd publishes in a row, the held centre moves to the car's centre (the
//   held box is never further from the robot than the car's by more than r_twd; heading and size stay held: a box swinging about one
//   end, the detection_by_tracker refit artefact, does not move its centre, so only translation toward the robot or its path counts);
// - a track the car published moving (> v_still on 2 publishes in a row) in the last t_settle s is not held (a car creeping in a
//   queue keeps the car's box); a track standing since its birth is held from its first publish;
// - the filter, association, overlap prune, speeds, classes, existence and every other class are untouched (publication only).
// NERO_MOT_CLASS_HOLD=<t_cls s>[,<debug>] (with NERO_MOT_STAND_HOLD): while a vehicle track stands (published speed <= v_still), a
// change of its published class between vehicle classes (CAR <-> TRUCK <-> BUS <-> TRAILER) is published only once the new class has
// lasted t_cls (11-08 replays: 8.1 flips per standing track-minute, 6.4 of them CAR <-> TRUCK lasting ~0.2 s; old car tracker 1.5); a
// change to or from any other class (pedestrian, bicycle, UNKNOWN, ...) is published at once, and a publish the car calls moving
// (> v_still) carries the car's class (2 such publishes in a row end the hold's state).
#ifndef AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__NERO__NERO_STAND_HOLD_HPP_
#define AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__NERO__NERO_STAND_HOLD_HPP_

#include "autoware/multi_object_tracker/object_model/classes.hpp"

#include <rclcpp/rclcpp.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <utility>
#include <vector>

namespace autoware::multi_object_tracker
{
struct NeroStandHoldParams
{
  bool enabled{false};
  double v_still{0.3};
  double v_rel{0.5};
  int k_rel{1};
  double r_disp{0.5};
  int k_disp{3};
  double v_disp{0.2};
  double r_dir{0.2};
  int k_dir{1};
  double v_dir{0.3};
  double g_min{0.05};
  double gy_min{0.02};
  double gs_min{0.05};
  double pre{0.5};
  double t_settle{1.0};
  double n0{3.0};
  double t_back{1.0};
  double r_back{0.3};
  double r_twd{0.3};
  int k_twd{2};
  int debug{0};
  // class hold
  bool cls_enabled{false};
  double t_cls{1.0};
  int cls_debug{0};
};

constexpr std::size_t kNeroStandHoldMaxBoxes = 40;  // trusted boxes kept per track (consumed at each publish)

inline const NeroStandHoldParams & neroStandHoldParams()
{
  static const NeroStandHoldParams p = [] {
    NeroStandHoldParams q;
    const char * env = std::getenv("NERO_MOT_STAND_HOLD");
    if (env != nullptr && env[0] != '\0') {
      q.enabled = std::sscanf(
                    env, "%lf,%lf,%d,%lf,%d,%lf,%lf,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%d,%d", &q.v_still, &q.v_rel,
                    &q.k_rel, &q.r_disp, &q.k_disp, &q.v_disp, &q.r_dir, &q.k_dir, &q.v_dir, &q.g_min, &q.gy_min, &q.gs_min, &q.pre,
                    &q.t_settle, &q.n0, &q.t_back, &q.r_back, &q.r_twd, &q.k_twd, &q.debug) >= 19 &&
                  q.k_twd >= 1 &&
                  q.n0 >= 0.0 &&
                  q.v_still >= 0.0 && q.v_rel >= q.v_still && q.k_rel >= 1 && q.k_disp >= 1 && q.k_dir >= 1 && q.g_min > 0.0 &&
                  q.gy_min > 0.0 && q.gs_min > 0.0;
      if (q.enabled) {
        RCLCPP_INFO(
          rclcpp::get_logger("multi_object_tracker"),
          "[NERO_MOT_STAND_HOLD] on: stand <= %.2f m/s; release v > %.2f x %d, displacement > %.2f m x %d at >= %.2f m/s, forward > %.2f m "
          "x %d at > %.2f m/s; gains >= %.3f / %.3f / %.3f, prior %.1f boxes, pre %.2f s, settle %.2f s after motion, resume %.2f s / "
          "%.2f m, toward the robot > %.2f m x %d, debug %d",
          q.v_still, q.v_rel, q.k_rel, q.r_disp, q.k_disp, q.v_disp, q.r_dir, q.k_dir, q.v_dir, q.g_min, q.gy_min, q.gs_min, q.n0, q.pre,
          q.t_settle, q.t_back, q.r_back, q.r_twd, q.k_twd, q.debug);
      }
    }
    const char * ce = std::getenv("NERO_MOT_CLASS_HOLD");
    if (q.enabled && ce != nullptr && ce[0] != '\0') {
      q.cls_enabled = std::sscanf(ce, "%lf,%d", &q.t_cls, &q.cls_debug) >= 1 && q.t_cls > 0.0;
      if (q.cls_enabled) {
        RCLCPP_INFO(
          rclcpp::get_logger("multi_object_tracker"), "[NERO_MOT_CLASS_HOLD] on: vehicle <-> vehicle class change after %.2f s, debug %d",
          q.t_cls, q.cls_debug);
      }
    }
    return q;
  }();
  return p;
}

// per-track state (kept by the tracker, written only by the processor's publication)
struct NeroStandHoldState
{
  bool held{false};
  bool suspended{false};  // released after a single moving reading: the estimate kept for a resume
  double t_rel{-1.0};
  int hi_since{0};        // publishes > v_rel since the release
  int mv_n{0};            // publishes in a row > v_still
  double t_moving{-1.0};  // the last publish of 2+ in a row the car published it moving (> v_still); < 0 = never
  double t0{-1.0};
  int hi_n{0};
  int disp_n{0};
  int dir_n{0};
  int twd_n{0};
  std::deque<std::array<double, 6>> cp;  // trusted boxes (map): t, x, y, yaw, length, width (recorded at the update)
  double cp_done{-1.0};                  // the newest trusted box the estimate has taken
  int n{0};                              // trusted boxes in the estimate (0 = none)
  std::array<double, 5> est{};           // x, y, yaw, length, width
  // class hold
  bool cls_init{false};
  std::vector<classes::Classification> cls_pub;
  classes::Label cls_cand{classes::Label::UNKNOWN};
  double cls_cand_t{-1.0};
  // NERO_MOT_CLASS_CAM: camera0's class votes (frame stamp, 1 / 2 / 3 = CAR / TRUCK / BUS) and the newest frame taken
  std::deque<std::pair<double, int>> cam_votes;
  double cam_t{-1.0};
  int cam_label{0};  // the camera's last decision for this stand (1 / 2 / 3 = CAR / TRUCK / BUS; 0 = none): kept while the track stands
};
}  // namespace autoware::multi_object_tracker

#endif  // AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__NERO__NERO_STAND_HOLD_HPP_
