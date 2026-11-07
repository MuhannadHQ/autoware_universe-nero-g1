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

#include "autoware/multi_object_tracker/tracker/model/pedestrian_and_bicycle_tracker.hpp"

#include <autoware_utils/math/normalization.hpp>
#include <rclcpp/logging.hpp>

#include <cmath>

namespace autoware::multi_object_tracker
{
PedestrianAndBicycleTracker::PedestrianAndBicycleTracker(
  const rclcpp::Time & time, const types::DynamicObject & object)
: Tracker(time, object),
  pedestrian_tracker_(time, object),
  bicycle_tracker_(object_model::bicycle, time, object)
{
}

bool PedestrianAndBicycleTracker::predict(const rclcpp::Time & time)
{
  pedestrian_tracker_.predict(time);
  bicycle_tracker_.predict(time);
  return true;
}

bool PedestrianAndBicycleTracker::measure(
  const types::DynamicObject & object, const rclcpp::Time & time,
  const types::InputChannel & channel_info)
{
  if (neroSignBeliefParams().debug > 0) bicycle_tracker_.neroSetLogName(object_.uuid, "bic");
  pedestrian_tracker_.measure(object, time, channel_info);
  bicycle_tracker_.measure(object, time, channel_info);
  if (neroSignBeliefParams().enabled && neroSignBeliefParams().ped_follow) {
    // NERO_MOT_SIGN_BELIEF (upstream #13050): only the bicycle layer accumulates heading-sign evidence; the pedestrian
    // layer follows it
    if (
      std::fabs(autoware_utils::normalize_radian(
        bicycle_tracker_.neroYawState() - pedestrian_tracker_.neroYawState())) > M_PI_2) {
      pedestrian_tracker_.neroFlipOrientation();
      if (neroSignBeliefParams().debug > 0) {
        RCLCPP_INFO(
          rclcpp::get_logger("PedestrianAndBicycleTracker"),
          "[NERO_MOT_SIGN] %02x%02x%02x%02x ped turn (follows the bicycle layer): t %.2f yaw -> %.0f", object_.uuid.uuid[0],
          object_.uuid.uuid[1], object_.uuid.uuid[2], object_.uuid.uuid[3], time.seconds(),
          pedestrian_tracker_.neroYawState() * 180.0 / M_PI);
      }
    }
  }

  return true;
}

bool PedestrianAndBicycleTracker::getTrackedObject(
  const rclcpp::Time & time, types::DynamicObject & object) const
{
  using Label = autoware_perception_msgs::msg::ObjectClassification;
  const uint8_t label = getHighestProbLabel();

  if (label == Label::PEDESTRIAN) {
    pedestrian_tracker_.getTrackedObject(time, object);
  } else if (label == Label::BICYCLE || label == Label::MOTORCYCLE) {
    bicycle_tracker_.getTrackedObject(time, object);
  }
  object.uuid = object_.uuid;
  object.classification = object_.classification;
  return true;
}

}  // namespace autoware::multi_object_tracker
