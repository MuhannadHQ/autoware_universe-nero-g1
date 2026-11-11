// Copyright 2020 TIER IV, Inc.
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

#ifndef AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__TRACKERS__POLYGON_TRACKER_HPP_
#define AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__TRACKERS__POLYGON_TRACKER_HPP_

#include "autoware/multi_object_tracker/configurations.hpp"
#include "autoware/multi_object_tracker/object_model/object_model.hpp"
#include "autoware/multi_object_tracker/tracker/motion_model/cv_motion_model.hpp"
#include "autoware/multi_object_tracker/tracker/motion_model/static_motion_model.hpp"
#include "autoware/multi_object_tracker/tracker/nero/nero_unknown_mover.hpp"
#include "autoware/multi_object_tracker/tracker/shape_model/polygon_shape_model.hpp"
#include "autoware/multi_object_tracker/tracker/trackers/tracker_base.hpp"
#include "autoware/multi_object_tracker/types.hpp"

#include <deque>

namespace autoware::multi_object_tracker
{

class PolygonTracker : public Tracker
{
private:
  rclcpp::Logger logger_;

  object_model::ObjectModel object_model_ = object_model::unknown;

  CVMotionModel motion_model_;
  StaticMotionModel static_motion_model_;

  bool enable_velocity_estimation_;
  // Per-label motion-output gate, evaluated at publish time against the track's current label.
  LabelBoolMap enable_motion_output_;

  geometry_msgs::msg::Pose last_pose_;

  PolygonShapeModel shape_model_;

  bool updateKinematics(const types::DynamicObject & object);

  // Continuously attenuate the published velocity by its direction-projected uncertainty
  // (mirrors VehicleTracker): zero when too uncertain, unchanged when large enough, rescaling
  // the 2D twist so the velocity direction is preserved.
  void suppressUncertainVelocity(types::DynamicObject & object) const;

  // [nero trkup] NERO_MOT_UNKNOWN_MOVER (our 0.44 UnknownTracker's mover rule; nero_unknown_mover.hpp). Used only with
  // the switch set and velocity estimation on; unset = upstream exactly.
  bool nero_on_{false};
  bool nero_has_meas_{false};
  double nero_last_meas_x_{0.0};
  double nero_last_meas_y_{0.0};
  rclcpp::Time nero_last_meas_time_;
  std::deque<nero::MoverPoint> nero_points_;
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
  double nero_mover_floor_{0.0};   // [m/s] the speed floor of the current mover's kind (oncoming or crossing)
  int nero_cross_passes_{0};       // crossmover: consecutive measurement times whose start test passed
  double nero_cross_t_{-1.0};      // ... the last of them (-1: the last test failed)
  double nero_cross_first_t_{-1.0};  // ... the first of them
  double nero_mover_since_{-1.0};    // the first passing test of the run that started the current mover
  bool nero_mover_cross_{false};     // the mover's last passing line was a crossing one
  nero::MoverLine nero_last_line_;
  bool neroMeasure(const types::DynamicObject & object, const rclcpp::Time & time);
  void neroMoverStarts(const double t, const double vx, const double vy, const rclcpp::Time & time);
  void neroInitWithVelocity(
    const rclcpp::Time & time, const types::DynamicObject & object, const double vx, const double vy,
    const double sigma_v);
  int neroCrossPasses(const double t, const nero::MoverLine & line);
  void neroPublish(const rclcpp::Time & time, types::DynamicObject & object) const;

public:
  // [nero trkup] NERO_MOT_UNKNOWN_MOVER: the rule acts on this track (switch set, velocity estimation on)
  bool neroOn() const { return nero_on_; }
  // a mover at <time> (its measured positions showed one within the hold): its velocity (map frame)
  bool neroMover(const rclcpp::Time & time, double & vx, double & vy) const;
  // the velocity of its last straight line (map frame)
  void neroMoverLine(double & vx, double & vy) const
  {
    vx = nero_mover_vx_;
    vy = nero_mover_vy_;
  }
  // a measurement at <time> continues this track's measured positions on a mover's straight line
  bool neroMoverContinues(const rclcpp::Time & time, const types::DynamicObject & measurement) const;
  // how long the mover at <time> has held its line [s] (< 0: no mover)
  double neroMoverAge(const rclcpp::Time & time) const;
  // crossmover: its measured positions passed a CROSSING mover's line within the mover hold before <time>
  bool neroMoverRecent(const rclcpp::Time & time) const;
  // crossmover: a crossing mover at <time>: how long its line has passed, from the first passing start test of the
  // run that started it [s] (< 0: none)
  double neroCrossHeld(const rclcpp::Time & time) const;
  // where it was last measured (false: never)
  bool neroLastMeasured(double & x, double & y) const
  {
    x = nero_last_meas_x_;
    y = nero_last_meas_y_;
    return nero_has_meas_;
  }
  // a young track the gates lost: the number of its measured positions that a new track born from <newborn> at
  // <time> continues on a mover's line (0: none)
  int neroInheritable(const rclcpp::Time & time, const PolygonTracker & newborn) const;
  // take those positions as this (new) track's history; a mover at once when they pass the start test
  void neroInheritFrom(
    PolygonTracker & from, const rclcpp::Time & time, const types::DynamicObject & measurement);

public:
  PolygonTracker(
    const rclcpp::Time & time, const types::DynamicObject & object,
    const PolygonTrackerConfig & config);

  bool predict(const rclcpp::Time & time) override;
  bool measure(
    const types::DynamicObject & object, const rclcpp::Time & time,
    const types::InputChannel & channel_info) override;
  bool getTrackedObject(
    const rclcpp::Time & time, types::DynamicObject & object,
    const bool to_publish = false) const override;

  bool getMotionState(
    const rclcpp::Time & time, geometry_msgs::msg::Pose & pose, std::array<double, 36> & pose_cov,
    geometry_msgs::msg::Twist & twist, std::array<double, 36> & twist_cov) const override;
  rclcpp::Time getStateTime() const override
  {
    return enable_velocity_estimation_ ? motion_model_.getLastPredictionTime()
                                       : static_motion_model_.getLastPredictionTime();
  }

  ShapeModelBase & getShapeModel() override { return shape_model_; }
  const ShapeModelBase & getShapeModel() const override { return shape_model_; }
  void assembleShapeTo(types::DynamicObject & output, bool /*to_publish*/) const override
  {
    shape_model_.exportTo(output);
  }
};

}  // namespace autoware::multi_object_tracker

#endif  // AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__TRACKERS__POLYGON_TRACKER_HPP_
