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

#ifndef AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__TRACKERS__VEHICLE_TRACKER_HPP_
#define AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__TRACKERS__VEHICLE_TRACKER_HPP_

#include "autoware/multi_object_tracker/object_model/object_model.hpp"
#include "autoware/multi_object_tracker/tracker/motion_model/bicycle_motion_model.hpp"
#include "autoware/multi_object_tracker/tracker/nero/nero_stand_speed.hpp"
#include "autoware/multi_object_tracker/tracker/shape_model/vehicle_shape_model.hpp"
#include "autoware/multi_object_tracker/tracker/trackers/tracker_base.hpp"
#include "autoware/multi_object_tracker/tracker/update/orientation_sign_belief.hpp"
#include "autoware/multi_object_tracker/tracker/update/vehicle_update_strategy.hpp"
#include "autoware/multi_object_tracker/types.hpp"

#include <optional>

namespace autoware::multi_object_tracker
{

class VehicleTracker : public Tracker
{
private:
  rclcpp::Logger logger_;

  object_model::ObjectModel object_model_;

  BicycleMotionModel motion_model_;
  using IDX = BicycleMotionModel::IDX;

  VehicleShapeModel shape_model_;

  // Tracks which end of the vehicle was used as anchor in the last conditioned update.
  // Consumed by setObjectShape() so UnstableShapeFilter commits the new length correctly.
  BicycleMotionModel::LengthUpdateAnchor shape_update_anchor_;

  // Interval [s] since the last measurement update.
  double time_since_correction_{0.0};

  // Accumulated heading-sign evidence from raw detection yaws.
  OrientationSignBelief sign_belief_;

  // Belief-driven 180° flip on the fused posterior: yaw votes decide the sign at low speed and
  // the instantaneous velocity sign decides above the par speed.
  void evaluateSignFlip(const rclcpp::Time & time);

  // EKF kinematic update — selects update variant based on data availability.
  bool updateKinematics(
    const types::DynamicObject & object, const types::InputChannel & channel_info);
  // Wheel-anchor EKF update (front or rear) plus z/height updates.
  // Also records the anchor in shape_update_anchor_. `prediction` supplies the tracked body center
  // used by the lateral (over-/under-wide polygon) anchor correction.
  bool updateWheelKinematics(
    const UpdateStrategy & strategy, const types::DynamicObject & measurement,
    const types::DynamicObject & prediction);

  // Relax the front/rear covariance asymmetry before an EKF pose update, scaled by
  // time_since_correction_.
  void applyAxleCovarianceBlend();

  // NERO_MOT_STAND_SPEED (vstoplag_1104, ported onto upstream): the published speed of a vehicle whose boxes stand / start
  // again; the ego position at the measurement time decides which box end faces the robot (0.44's anchor rule)
  NeroStandSpeed nero_stand_;

  // NERO_MOT_FLIP_DEFER (flipspeed_1030, ported; its turn part is upstream's sign belief #13050): a SUDDEN reverse speed
  // (a box that jumped back on a young or standing track) holds the track standing; 0.2 s of measured positions decide
  std::array<double, 8> nero_vh_t_{};
  std::array<double, 8> nero_vh_v_{};
  int nero_vh_n_{0};
  double nero_v_before_{0.0};
  double nero_v_frames_{0.0};
  bool nero_held_{false};
  bool nero_hold_state_{false};       // the held track's state is held standing (published when held, or it stood)
  mutable bool nero_published_{false};  // the track has been published (getTrackedObject to_publish)
  double nero_held_t0_{0.0};
  double nero_held_ux_{1.0};
  double nero_held_uy_{0.0};
  double nero_hold_pvv_{1.0};
  std::array<double, 32> nero_ht_{};
  std::array<double, 32> nero_ha_{};
  int nero_hn_{0};
  void neroDeferPre(const rclcpp::Time & time);
  void neroDeferPost(
    const types::DynamicObject & measurement, const rclcpp::Time & time, const types::InputChannel & channel_info);
  // NERO_MOT_NEWBORN_REV (trkup-2, from the newborn census): a young track's reverse speed is exported only once an update kept it
  int nero_rp_n_{0};          // updates from a spawning channel (the detections)
  double nero_rp_a_{0.0};     // the speed before / after the last such update
  double nero_rp_b_{0.0};
  double nero_rp_pre_{0.0};   // the speed before the current update
  bool nero_fd_rev_{false};   // FLIP_DEFER decided "it moves backward" (measured)
  std::optional<geometry_msgs::msg::Point> nero_ego_pos_;
  void neroStandMeasure(
    const types::DynamicObject & box, const rclcpp::Time & time, const types::InputChannel & channel_info);

public:
  VehicleTracker(
    const object_model::ObjectModel & object_model, const rclcpp::Time & time,
    const types::DynamicObject & object);

  bool predict(const rclcpp::Time & time) override;
  bool measure(
    const types::DynamicObject & object, const rclcpp::Time & time,
    const types::InputChannel & channel_info) override;

  bool conditionedUpdate(
    const types::DynamicObject & measurement, const types::DynamicObject & prediction,
    const rclcpp::Time & measurement_time, const types::InputChannel & channel_info) override;

  bool getTrackedObject(
    const rclcpp::Time & time, types::DynamicObject & object,
    const bool to_publish = false) const override;

  bool getMotionState(
    const rclcpp::Time & time, geometry_msgs::msg::Pose & pose, std::array<double, 36> & pose_cov,
    geometry_msgs::msg::Twist & twist, std::array<double, 36> & twist_cov) const override;
  rclcpp::Time getStateTime() const override { return motion_model_.getLastPredictionTime(); }

  // Heading state, sign-belief confidence, and 180° state flip; composite trackers use these
  // for heading-sign consensus across their layers.
  double getYawState() const { return motion_model_.getYawState(); }
  double signBeliefConfidence() const { return sign_belief_.confidence(); }
  void flipOrientationSign();

  ShapeModelBase & getShapeModel() override { return shape_model_; }
  const ShapeModelBase & getShapeModel() const override { return shape_model_; }
  void assembleShapeTo(types::DynamicObject & output, bool to_publish) const override;

  void setEgoPose(const std::optional<geometry_msgs::msg::Point> & ego_pos) override
  {
    nero_ego_pos_ = ego_pos;  // NERO_MOT_STAND_SPEED only
    Tracker::setEgoPose(ego_pos);
  }

  // Overridden because the committed shape also drives the motion-model length (and anchor).
  // mergeFootprintFrom() is handled by the base via getShapeModel().mergeFrom().
  void setObjectShape(const autoware_perception_msgs::msg::Shape & shape) override;

  // Clusters (trust_extension=false) have unreliable bbox orientation — always use conditioned.
  UpdatePath selectUpdatePath(
    bool trust_extension, bool has_significant_shape_change) const override
  {
    if (!trust_extension) return UpdatePath::CONDITIONED;
    return has_significant_shape_change ? UpdatePath::TRY_EXTENSION : UpdatePath::NORMAL;
  }

  // Vehicle boxes coast on partial updates; staleness on full measurements is meaningful.
  double getElapsedTimeFromFullMeasurement(const rclcpp::Time & current_time) const override
  {
    return elapsedSinceLastFullMeasurement(current_time);
  }
};

}  // namespace autoware::multi_object_tracker

#endif  // AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__TRACKERS__VEHICLE_TRACKER_HPP_
