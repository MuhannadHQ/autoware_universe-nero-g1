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

#include "autoware/multi_object_tracker/tracker/model/tracker_base.hpp"

#include "autoware/multi_object_tracker/tracker/model/nero_still_box.hpp"

#include <tf2/utils.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace
{
float updateProbability(
  const float & prior, const float & true_positive, const float & false_positive)
{
  constexpr float max_updated_probability = 0.999;
  constexpr float min_updated_probability = 0.100;
  const float probability =
    (prior * true_positive) / (prior * true_positive + (1 - prior) * false_positive);
  return std::clamp(probability, min_updated_probability, max_updated_probability);
}
float decayProbability(const float & prior, const float & delta_time)
{
  constexpr float minimum_probability = 0.001;
  const float decay_rate = log(0.5f) / 0.3f;  // half-life (50% decay) of 0.3s
  return std::max(prior * std::exp(decay_rate * delta_time), minimum_probability);
}
}  // namespace

namespace autoware::multi_object_tracker
{

Tracker::Tracker(const rclcpp::Time & time, const types::DynamicObject & detected_object)
: no_measurement_count_(0),
  total_no_measurement_count_(0),
  total_measurement_count_(1),
  last_update_with_measurement_time_(time),
  nero_last_full_update_time(time),
  object_(detected_object)
{
  // Generate random number
  std::mt19937 gen(std::random_device{}());
  std::independent_bits_engine<std::mt19937, 8, uint8_t> bit_eng(gen);
  unique_identifier_msgs::msg::UUID uuid_msg;
  std::generate(uuid_msg.uuid.begin(), uuid_msg.uuid.end(), bit_eng);
  object_.uuid = uuid_msg;

  // NERO_MOT_FAST_BIRTH: the first measured box
  nero_first_x = nero_last_x = detected_object.pose.position.x;
  nero_first_y = nero_last_y = detected_object.pose.position.y;
  nero_last_yaw = tf2::getYaw(detected_object.pose.orientation);
  nero_first_t = nero_last_t = time;
  if (neroStillBoxParams().enabled) neroStillBoxRecord(detected_object, time, nullptr);

  // Initialize existence probabilities
  existence_probabilities_.resize(types::max_channel_size, 0.001);
  total_existence_probability_ = 0.001;
}

void Tracker::initializeExistenceProbabilities(
  const uint & channel_index, const float & existence_probability)
{
  // The initial existence probability is modeled
  // since the incoming object's existence probability is not reliable
  // existence probability on each channel
  constexpr float initial_existence_probability = 0.1;
  existence_probabilities_[channel_index] = initial_existence_probability;

  // total existence probability
  constexpr float max_probability = 0.999;
  constexpr float min_probability = 0.100;
  total_existence_probability_ =
    std::max(std::min(existence_probability, max_probability), min_probability);
  nero_min_meas_prob = existence_probability;  // NERO_MOT_FAST_BIRTH: the spawning measurement
}

bool Tracker::updateWithMeasurement(
  const types::DynamicObject & object, const rclcpp::Time & measurement_time,
  const types::InputChannel & channel_info)
{
  // Update existence probability
  {
    no_measurement_count_ = 0;
    ++total_measurement_count_;

    // existence probability on each channel
    const float delta_time =
      std::abs((measurement_time - last_update_with_measurement_time_).seconds());
    constexpr float probability_true_detection = 0.9;
    constexpr float probability_false_detection = 0.2;

    // update measured channel probability without decay
    const uint & channel_index = channel_info.index;
    existence_probabilities_[channel_index] = updateProbability(
      existence_probabilities_[channel_index], probability_true_detection,
      probability_false_detection);

    // decay other channel probabilities
    for (size_t i = 0; i < existence_probabilities_.size(); ++i) {
      if (i != channel_index) {
        existence_probabilities_[i] = decayProbability(existence_probabilities_[i], delta_time);
      }
    }

    // update total existence probability
    const double existence_probability =
      channel_info.trust_existence_probability ? object.existence_probability : 0.6;
    total_existence_probability_ = updateProbability(
      total_existence_probability_, existence_probability, probability_false_detection);
  }

  // NERO_MOT_FAST_BIRTH: count each detection stamp once
  if (measurement_time != last_update_with_measurement_time_) ++nero_distinct_stamps;
  if (channel_info.trust_extension) {
    nero_last_x = object.pose.position.x;
    nero_last_y = object.pose.position.y;
    nero_last_yaw = tf2::getYaw(object.pose.orientation);
    nero_last_t = measurement_time;
  }
  if (channel_info.trust_existence_probability) {
    nero_min_meas_prob = std::min(nero_min_meas_prob, object.existence_probability);
  }
  last_update_with_measurement_time_ = measurement_time;
  if (channel_info.trust_extension) nero_last_full_update_time = measurement_time;

  // Update classification
  if (
    channel_info.trust_classification &&
    autoware::object_recognition_utils::getHighestProbLabel(object.classification) !=
      autoware_perception_msgs::msg::ObjectClassification::UNKNOWN) {
    updateClassification(object.classification);
  }

  if (neroStillBoxParams().enabled) neroStillBoxRecord(object, measurement_time, &channel_info);

  // Update object
  measure(object, measurement_time, channel_info);

  return true;
}

bool Tracker::updateWithoutMeasurement(const rclcpp::Time & now)
{
  // Update existence probability
  ++no_measurement_count_;
  ++total_no_measurement_count_;
  {
    // decay existence probability
    float const delta_time = (now - last_update_with_measurement_time_).seconds();
    for (float & existence_probability : existence_probabilities_) {
      existence_probability = decayProbability(existence_probability, delta_time);
    }
    total_existence_probability_ = decayProbability(total_existence_probability_, delta_time);
  }

  return true;
}

void Tracker::updateClassification(
  const std::vector<autoware_perception_msgs::msg::ObjectClassification> & classification)
{
  // classification algorithm:
  // 0. Normalize the input classification
  // 1-1. Update the matched classification probability with a gain (ratio of 0.05)
  // 1-2. If the label is not found, add it to the classification list
  // 2. Remove the class with probability < remove_threshold (0.001)
  // 3. Normalize tracking classification

  // Parameters
  // if the remove_threshold is too high (compare to the gain), the classification will be removed
  // immediately
  const float gain = 0.05;
  constexpr float remove_threshold = 0.001;

  // Normalization function
  auto normalizeProbabilities =
    [](std::vector<autoware_perception_msgs::msg::ObjectClassification> & classification) {
      float sum = 0.0;
      for (const auto & a_class : classification) {
        sum += a_class.probability;
      }
      for (auto & a_class : classification) {
        a_class.probability /= sum;
      }
    };

  // Normalize the input
  auto classification_input = classification;
  normalizeProbabilities(classification_input);

  auto & classification_ = object_.classification;

  // Update the matched classification probability with a gain
  for (const auto & new_class : classification_input) {
    bool found = false;
    for (auto & old_class : classification_) {
      if (new_class.label == old_class.label) {
        old_class.probability += new_class.probability * gain;
        found = true;
        break;
      }
    }
    // If the label is not found, add it to the classification list
    if (!found) {
      auto adding_class = new_class;
      adding_class.probability *= gain;
      classification_.push_back(adding_class);
    }
  }

  // If the probability is less than the threshold, remove the class
  classification_.erase(
    std::remove_if(
      classification_.begin(), classification_.end(),
      [remove_threshold](const auto & a_class) { return a_class.probability < remove_threshold; }),
    classification_.end());

  // Normalize tracking classification
  normalizeProbabilities(classification_);
}

void Tracker::limitObjectExtension(const object_model::ObjectModel object_model)
{
  auto & object_extension = object_.shape.dimensions;
  // set maximum and minimum size
  object_extension.x = std::clamp(
    object_extension.x, object_model.size_limit.length_min, object_model.size_limit.length_max);
  object_extension.y = std::clamp(
    object_extension.y, object_model.size_limit.width_min, object_model.size_limit.width_max);
  object_extension.z = std::clamp(
    object_extension.z, object_model.size_limit.height_min, object_model.size_limit.height_max);
}

}  // namespace autoware::multi_object_tracker

namespace autoware::multi_object_tracker
{
void Tracker::neroStillBoxRecord(
  const types::DynamicObject & object, const rclcpp::Time & time, const types::InputChannel * channel_info)
{
  // NERO_MOT_STILL_BOX (boxyaw): keep this measurement's evidence of the points (map frame)
  if (object.nero_has_ego) {
    nero_sb_has_ego = true;
    nero_sb_ego = {object.nero_ego_x, object.nero_ego_y, object.nero_ego_yaw};
  }
  const double x = object.pose.position.x;
  const double y = object.pose.position.y;
  const double yaw = tf2::getYaw(object.pose.orientation);
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  if (object.shape.type == autoware_perception_msgs::msg::Shape::POLYGON) {
    if (object.shape.footprint.points.size() < 3) return;
    nero_sb_outline.clear();
    for (const auto & p : object.shape.footprint.points) {
      nero_sb_outline.push_back({x + p.x * c - p.y * s, y + p.x * s + p.y * c});
    }
    nero_sb_outline_t = time.seconds();
    return;
  }
  if (
    channel_info == nullptr || channel_info->trust_extension || channel_info->trust_orientation ||
    object.shape.type != autoware_perception_msgs::msg::Shape::BOUNDING_BOX) {
    return;
  }
  nero_sb_boxes.push_back(
    {time.seconds(), x, y, yaw, object.shape.dimensions.x, object.shape.dimensions.y});
  if (nero_sb_boxes.size() > 5) nero_sb_boxes.pop_front();
}
}  // namespace autoware::multi_object_tracker
