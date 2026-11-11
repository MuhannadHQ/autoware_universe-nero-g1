// Copyright 2026 Nero (boxyaw). Licensed under the Apache License, Version 2.0.
//
// NERO_MOT_STILL_BOX=<max_dev deg>,<t_hold s>,<v_still m/s>,<t_birth s>,<v_birth m/s>,<max_grow m>,<debug>[,<min_length m>]
// (boxyaw, 2026-11-08; ported onto upstream by trkup-2 2026-11-10 without the REBIRTH birth branch; unset = upstream: nothing here runs, the published objects are byte-identical).
// A standing vehicle box (CAR / TRUCK / BUS / TRAILER) whose heading is a few degrees off swings its far corner across the
// real points: 11-08 #19, a truck parked parallel to the road half in the lane, its own points 1.0-1.08 m from the robot's
// path; CenterPoint's TRUCK boxes came 25.7-34.8 deg against the road's 25.5 (mean +4), the vehicle track published
// ~30 deg and its front corner reached 0.47 m from the path centre (inside obstacle_stop's 0.41 + 0.1) -> a hard stop
// from 20 km/h at 6 m. The points' own fit was right all along: detection_by_tracker's box (an L-shape fit of the
// track's cluster) came 23.3-27.6 deg with its inner edge at the points.
// With the switch, ONLY the PUBLISHED box of a still vehicle track (the filter's state untouched) is re-aligned with the
// measured points:
//  - still = published speed <= v_still, or a vehicle track born (NERO_MOT_REBIRTH) on a standing (not mover) UNKNOWN
//    track within t_birth of its birth at <= v_birth (its first boxes carry the UNKNOWN's false reveal speed);
//  - the points' heading = the median of: this track's point-fitted boxes (detection_by_tracker channel: extent and
//    heading not trusted by the tracker) of the last t_hold, and the longest edge (>= 0.4 x the box length) of a measured
//    outline of the last t_hold (this track's, else in the birth window its standing UNKNOWN predecessor's); each within
//    max_dev of the box's heading (either sign; the published sign is kept). The first point-fitted box of a newborn
//    track is fitted around its own wrong heading (#19: 31.7 deg), hence the median and the outline;
//  - the points = that outline only (the point-fitted box gives the heading; its far edges are a reference size);
//  - the published box = the box at that heading (centre and size as the track has them), united with the outline's
//    points when there is one: an edge moves out only to cover measured points, the box never gets smaller; a union that grows the box by
//    more than max_grow in length or width is not used (the box as today).
//  - never toward the robot: the re-aligned box's corners ahead of the robot may not come closer to the line along the
//    robot's heading than the box today (else the box as today): the switch only takes boxes back to their points, it
//    never adds a box's reach toward the robot's path;
//  - moving vehicles, pedestrians, bicycles and UNKNOWN objects are never touched.
#ifndef AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__NERO__NERO_STILL_BOX_HPP_
#define AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__NERO__NERO_STILL_BOX_HPP_

#include <rclcpp/rclcpp.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace autoware::multi_object_tracker
{
struct NeroStillBoxParams
{
  bool enabled{false};
  double max_dev{10.0 * M_PI / 180.0};  // [rad]
  double t_fresh{1.0};                  // [s] t_hold: how long a still object's points count
  double v_still{0.5};                  // [m/s]
  double t_birth{1.0};                  // [s]
  double v_birth{1.0};                  // [m/s]
  double max_grow{1.5};                 // [m]
  int debug{0};
  double min_edge_ratio{0.4};
  double min_length{0.0};  // [m] only boxes at least this long (field 8, optional; 0 = every vehicle box)
};

inline const NeroStillBoxParams & neroStillBoxParams()
{
  static const NeroStillBoxParams p = [] {
    NeroStillBoxParams q;
    const char * env = std::getenv("NERO_MOT_STILL_BOX");
    if (env != nullptr && env[0] != '\0') {
      double dev_deg = 10.0;
      q.enabled = std::sscanf(
                    env, "%lf,%lf,%lf,%lf,%lf,%lf,%d,%lf", &dev_deg, &q.t_fresh, &q.v_still, &q.t_birth, &q.v_birth,
                    &q.max_grow, &q.debug, &q.min_length) >= 7 &&
                  dev_deg > 0.0 && q.t_fresh > 0.0;
      q.max_dev = dev_deg * M_PI / 180.0;
      if (q.enabled) {
        RCLCPP_INFO(
          rclcpp::get_logger("multi_object_tracker"),
          "[NERO_MOT_STILL_BOX] on: max_dev %.1f deg, t_hold %.2f s, v_still %.2f m/s, birth %.2f s at <= %.2f m/s, "
          "max_grow %.2f m, debug %d, min_length %.1f m",
          dev_deg, q.t_fresh, q.v_still, q.t_birth, q.v_birth, q.max_grow, q.debug, q.min_length);
      }
    }
    return q;
  }();
  return p;
}
}  // namespace autoware::multi_object_tracker

#endif  // AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__NERO__NERO_STILL_BOX_HPP_
