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

#ifndef AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__MODEL__VEHICLE_TRACKER_HPP_
#define AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__MODEL__VEHICLE_TRACKER_HPP_

#include "autoware/multi_object_tracker/object_model/object_model.hpp"
#include "autoware/multi_object_tracker/object_model/types.hpp"
#include "autoware/multi_object_tracker/tracker/model/nero_sign_belief.hpp"
#include "autoware/multi_object_tracker/tracker/model/tracker_base.hpp"
#include "autoware/multi_object_tracker/tracker/motion_model/bicycle_motion_model.hpp"

#include <autoware/kalman_filter/kalman_filter.hpp>

#include <algorithm>
#include "autoware/multi_object_tracker/tracker/model/nero_stand_speed.hpp"

#include <array>
#include <optional>

namespace autoware::multi_object_tracker
{

class VehicleTracker : public Tracker
{
private:
  object_model::ObjectModel object_model_;
  rclcpp::Logger logger_;

  double velocity_deviation_threshold_;

  Eigen::Vector2d tracking_offset_;

  BicycleMotionModel motion_model_;
  using IDX = BicycleMotionModel::IDX;

  // NERO_MOT_SIGN_BELIEF (signbelief, upstream #13050): the heading-sign evidence of every box and the track's speed
  NeroSignBelief nero_sign_;
  void neroSignVote(
    const types::DynamicObject & object, const rclcpp::Time & time, const types::InputChannel & channel_info);
  void neroSignEvaluate(const rclcpp::Time & time);

  // NERO_MOT_FLIP_DEFER (flipspeed): after a sudden reverse speed the track is held as standing and the object turned
  // by 180 deg (as stock turns it) runs beside it on the same measurements until they show which one is right
  std::optional<BicycleMotionModel> nero_turned_;
  std::array<double, 8> nero_v_hist_t_{};         // the last updates: their time [s] and the speed before them
  std::array<double, 8> nero_v_hist_v_{};
  int nero_v_hist_n_{0};
  int nero_turned_count_{0};                      // measured positions since the sudden reverse speed
  std::array<double, 8> nero_turned_t_{};         // [s] after the first one
  std::array<double, 8> nero_turned_along_{};     // [m] along the turned heading
  double nero_turned_ux_{1.0};
  double nero_turned_uy_{0.0};
  double nero_turned_x_{0.0};
  double nero_turned_y_{0.0};
  rclcpp::Time nero_turned_time_;
  rclcpp::Time nero_now_;
  double nero_hold_pvv_{1.0};
  bool nero_published_{false};       // the track has been published
  bool nero_hold_published_{false};  // ... already when it was held
  bool nero_hold_own_still_{false};  // when held, its own speeds over >= t_judge never moved the turned way
  bool nero_show_turned_{false};     // published while judged without having been published before: shows the turned
  const char * nero_first_pub_{""};  // what the vehicle track at its place said at its first publication (log)
  std::array<uint8_t, 4> nero_log_uuid_{};
  const char * nero_log_model_{""};
  // NERO_MOT_STAND_SPEED (vstoplag): the published speed of a vehicle whose boxes stand / start again
  NeroStandSpeed nero_stand_;
  bool updateMotionModel(
    BicycleMotionModel & motion_model, const types::DynamicObject & object,
    const bool is_yaw_available) const;
  void neroAddTurnedPosition(const types::DynamicObject & object, const rclcpp::Time & time);

public:
  struct Options
  {
    // bb1012 (2026-10-13): lidar_centerpoint marks BICYCLE/MOTORCYCLE orientation UNAVAILABLE (only car-like classes get
    // SIGN_UNKNOWN), so a bicycle track keeps the yaw of its birth detection forever (an outlier birth yaw is never
    // corrected). true = update the bicycle-model yaw from BICYCLE/MOTORCYCLE measurements too, as front/back-ambiguous,
    // with the model yaw measurement noise. false = stock behaviour.
    bool use_unavailable_bicycle_yaw{false};
  };
  static Options options;

  VehicleTracker(
    const object_model::ObjectModel & object_model, const rclcpp::Time & time,
    const types::DynamicObject & object);

  bool predict(const rclcpp::Time & time) override;
  bool measure(
    const types::DynamicObject & object, const rclcpp::Time & time,
    const types::InputChannel & channel_info) override;
  bool measureWithPose(
    const types::DynamicObject & object, const types::InputChannel & channel_info);
  bool measureWithShape(const types::DynamicObject & object);
  bool getTrackedObject(const rclcpp::Time & time, types::DynamicObject & object) const override;
  void neroRefinePublished(types::DynamicObject & object) const override;

  // NERO_MOT_FLIP_DEFER (flipspeed): switch state and the names its log lines use
  static bool neroFlipDeferEnabled();
  // decide a judged turn once its positions span t_judge
  void neroDecideTurned();
  // a turn is judged and the track has not been published yet
  bool neroJudgedUnpublished() const override { return nero_turned_.has_value() && !nero_published_; }
  // the track is published from now on (with the vehicle track already at its place, if any)
  void neroOnPublished(
    const bool has_partner, const double partner_vx, const double partner_vy) override;
  void neroSetLogName(const unique_identifier_msgs::msg::UUID & uuid, const char * model)
  {
    std::copy(uuid.uuid.begin(), uuid.uuid.begin() + 4, nero_log_uuid_.begin());
    nero_log_model_ = model;
  }

  // NERO_MOT_SIGN_BELIEF (signbelief): the heading, the belief's confidence (layer consensus) and the exact 180 deg turn
  // of the whole state (motion model, anchor offset, belief)
  double neroYawState() const { return motion_model_.getStateElement(IDX::YAW); }
  double neroSpeedState() const { return motion_model_.getStateElement(IDX::VEL); }
  double neroSignConfidence() const { return nero_sign_.confidence(); }
  void neroFlipOrientationSign(const char * why);
};

}  // namespace autoware::multi_object_tracker

#endif  // AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__MODEL__VEHICLE_TRACKER_HPP_
