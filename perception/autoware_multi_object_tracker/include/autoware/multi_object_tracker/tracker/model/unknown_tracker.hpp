// Copyright 2020 Tier IV, Inc.
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
//
//
// Author: v1.0 Yukihiro Saito
//

#ifndef AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__MODEL__UNKNOWN_TRACKER_HPP_
#define AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__MODEL__UNKNOWN_TRACKER_HPP_

#include "autoware/multi_object_tracker/object_model/object_model.hpp"
#include "autoware/multi_object_tracker/object_model/types.hpp"
#include "autoware/multi_object_tracker/tracker/model/tracker_base.hpp"
#include "autoware/multi_object_tracker/tracker/motion_model/cv_motion_model.hpp"

#include <autoware/kalman_filter/kalman_filter.hpp>

#include <array>
#include <deque>
#include <vector>

namespace autoware::multi_object_tracker
{

class UnknownTracker : public Tracker
{
private:
  rclcpp::Logger logger_;

  object_model::ObjectModel object_model_ = object_model::unknown;

  CVMotionModel motion_model_;
  using IDX = CVMotionModel::IDX;

  // bb1012 (2026-10-13) param-gated fixes, defaults = stock behaviour
  bool has_meas_{false};
  double last_meas_x_{0.0};
  double last_meas_y_{0.0};
  double last_meas_area_{0.0};
  rclcpp::Time last_meas_time_;
  void initializeState(
    const rclcpp::Time & time, const types::DynamicObject & object, const bool zero_velocity);
  static double footprintArea(const types::DynamicObject & object);

public:
  // NERO_MOT_UNKNOWN_MOVER (oncomingtrack): the track's own measured positions (one per measurement time) and the
  // last velocity they showed as a mover
  struct NeroPoint
  {
    double t;
    double x;
    double y;
    std::vector<std::array<double, 2>> outline;  // the measured footprint in the map frame
    double yaw;  // the measurement's yaw: a cluster carries the robot's heading at its time
    double area;  // the measured footprint's area
  };
  // a line test's points without copies (the window holds one point per measurement time: <= ~20 in 0.5 s)
  static constexpr int kNeroMaxPoints = 64;
  struct NeroView
  {
    std::array<const NeroPoint *, kNeroMaxPoints> p{};
    int n{0};
    void push(const NeroPoint * q)
    {
      if (n == kNeroMaxPoints) {  // keep the newest (never reached at 10-20 Hz in a 0.5 s window)
        for (int i = 1; i < n; ++i) p[i - 1] = p[i];
        --n;
      }
      p[n++] = q;
    }
    const NeroPoint & front() const { return *p[0]; }
    const NeroPoint & back() const { return *p[n - 1]; }
  };

private:
  std::deque<NeroPoint> nero_points_;
  bool nero_mover_{false};
  double nero_mover_vx_{0.0};
  double nero_mover_vy_{0.0};
  rclcpp::Time nero_mover_time_;
  int nero_misses_{0};     // consecutive measurement times the line test failed while a mover
  int nero_fragments_{0};  // consecutive measurement times taken as the mover's rear fragments
  double nero_fragment_t_{-1.0};
  double nero_miss_t_{-1.0};
  double nero_mover_start_{-1.0};  // [s] when the current mover's line first passed
  double nero_birth_t_{-1.0};      // [s] the track's first measurement
  bool nero_inherited_{false};     // its points went to a newer track
  double nero_mover_floor_{0.0};   // [m/s] the speed floor of the current mover's kind (crossmover: oncoming or crossing)
  int nero_cross_passes_{0};      // crossmover: consecutive measurement times whose start test passed
  double nero_cross_t_{-1.0};      // ... the last of them (-1: the last test failed)
  double nero_cross_first_t_{-1.0};  // crossmover c3: ... the first of them
  double nero_mover_since_{-1.0};    // crossmover c3: the first passing test of the run that started the current mover
  bool nero_mover_cross_{false};     // crossmover c3: the mover's last passing line was a crossing one
  void neroMoverStarts(const double t, const double vx, const double vy, const rclcpp::Time & time);
  void neroInitWithVelocity(
    const rclcpp::Time & time, const types::DynamicObject & object, const double vx, const double vy,
    const double sigma_v);
  static NeroPoint neroPoint(const rclcpp::Time & time, const types::DynamicObject & object);

public:
  // NERO_MOT_UNKNOWN_MOVER: the straight-line test over measured positions (stats for research logs)
  struct NeroLine
  {
    int n{0};
    double span{0.0};
    double vx{0.0};
    double vy{0.0};
    double rms{0.0};
    double disp{0.0};    // first -> last measured centre
    double v_near{0.0};  // speed along the line of the outline's rear-most and front-most points
    double v_far{0.0};
    double area{0.0};     // median footprint area of the measurements
    bool crossing{false};  // crossmover: the line runs across the robot's heading (not against it)
    double lead_step{0.0};  // crossmover: the leading end's largest single step / its travel over the window
    double last_step{0.0};  // crossmover: the leading end's newest step as a speed / the line speed
    bool ok{false};       // starts a mover: the centres on the line, travel, both outline ends moving with it
    bool ok_cont{false};  // keeps one: the centres on the line and the leading end moving with it (a car seen closer
                          // shows more of its side: its trailing end lags)
  };
  static NeroLine neroLineTest(const std::deque<NeroPoint> & points, const int min_n);
  static NeroLine neroLineTest(const NeroView & points, const int min_n);
  NeroLine nero_last_line_;  // research logs only

private:
  int neroCrossPasses(const double t, const NeroLine & line);  // crossmover

public:
  struct Options
  {
    // publish the footprint where it was last measured (+ velocity * time since that measurement)
    // instead of re-centring the latest measured footprint on the filtered centre
    bool measured_footprint{false};
    // a measurement whose centre jumps more than this from the prediction [m] while its footprint area changes by more
    // than shape_jump_area_ratio is a cluster fragment/merge flip, not motion: restart the state there with zero
    // velocity (0.0 = off)
    double shape_jump_reset_distance{0.0};
    double shape_jump_area_ratio{1.5};
  };
  static Options options;

  UnknownTracker(const rclcpp::Time & time, const types::DynamicObject & object);

  bool predict(const rclcpp::Time & time) override;
  bool measure(
    const types::DynamicObject & object, const rclcpp::Time & time,
    const types::InputChannel & channel_info) override;
  bool measureWithPose(const types::DynamicObject & object);
  bool measureWithShape(const types::DynamicObject & object);
  bool getTrackedObject(const rclcpp::Time & time, types::DynamicObject & object) const override;

  // NERO_MOT_UNKNOWN_MOVER
  static bool neroMoverEnabled();
  // crossmover: crossing movers on (NERO_MOT_UNKNOWN_MOVER's 7th / 8th fields) and the robot's position at the current
  // measurement time (for the crossing mover's log line only)
  static bool neroCrossEnabled();
  static void neroSetEgo(const double x, const double y);
  // a mover at <time> (its measured positions showed one within the hold): its velocity (map frame)
  bool neroMover(const rclcpp::Time & time, double & vx, double & vy) const;
  // the velocity of its last straight line (map frame)
  void neroMoverLine(double & vx, double & vy) const
  {
    vx = nero_mover_vx_;
    vy = nero_mover_vy_;
  }
  // a measurement at <time> at (x, y) continues this track's measured positions on a mover's straight line
  bool neroMoverContinues(const rclcpp::Time & time, const types::DynamicObject & measurement) const;
  // how long the mover at <time> has held its line [s] (< 0: no mover)
  double neroMoverAge(const rclcpp::Time & time) const;
  // crossmover c3: its measured positions passed a CROSSING mover's line within the mover hold before <time> (a mover
  // now, or one whose line just ended): the speed it is published with comes from that line, not from the stock filter
  bool neroMoverRecent(const rclcpp::Time & time) const;
  // crossmover c3: a crossing mover at <time>: how long its line has passed, from the first passing start test of the
  // run that started it (its <cross_tests> consecutive passes count) [s] (< 0: none)
  double neroCrossHeld(const rclcpp::Time & time) const;
  // crossmover c3: where it was last measured (false: never)
  bool neroLastMeasured(double & x, double & y) const
  {
    x = last_meas_x_;
    y = last_meas_y_;
    return has_meas_;
  }
  // a young track (born within kNeroInheritAge) the gates lost: the number of its measured positions that a new
  // track born from <newborn> at <time> continues on a straight line against the robot's heading (0: none)
  int neroInheritable(const rclcpp::Time & time, const UnknownTracker & newborn) const;
  // take those positions as this (new) track's history; a mover at once when they pass the start test
  void neroInheritFrom(UnknownTracker & from, const rclcpp::Time & time, const types::DynamicObject & measurement);
};

}  // namespace autoware::multi_object_tracker

#endif  // AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__MODEL__UNKNOWN_TRACKER_HPP_
