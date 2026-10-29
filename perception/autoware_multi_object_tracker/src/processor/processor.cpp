// Copyright 2024 TIER IV, Inc.
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

#include "processor.hpp"

#include "autoware/multi_object_tracker/object_model/object_model.hpp"
#include "autoware/multi_object_tracker/object_model/shapes.hpp"
#include "autoware/multi_object_tracker/object_model/types.hpp"
#include "autoware/multi_object_tracker/tracker/tracker.hpp"

#include <autoware/object_recognition_utils/object_recognition_utils.hpp>
#include <autoware_utils/ros/msg_covariance.hpp>

#include <autoware_perception_msgs/msg/tracked_objects.hpp>

#include <tf2/utils.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace autoware::multi_object_tracker
{
using autoware_utils::ScopedTimeTrack;
using Label = autoware_perception_msgs::msg::ObjectClassification;
using LabelType = autoware_perception_msgs::msg::ObjectClassification::_label_type;

TrackerProcessor::TrackerProcessor(
  const TrackerProcessorConfig & config, const AssociatorConfig & associator_config,
  const std::vector<types::InputChannel> & channels_config)
: config_(config), channels_config_(channels_config)
{
  association_ = std::make_unique<DataAssociation>(associator_config);

  if (const char * env = std::getenv("NERO_MOT_REBIRTH"); env != nullptr && env[0] != '\0') {
    double v[4] = {2.0, 0.5, 3.0, 1.0};
    std::sscanf(env, "%lf,%lf,%lf,%lf", &v[0], &v[1], &v[2], &v[3]);
    nero_rebirth_.enabled = true;
    nero_rebirth_.min_count = std::max(1, static_cast<int>(v[0]));
    nero_rebirth_.margin = v[1];
    nero_rebirth_.max_dist = v[2];
    nero_rebirth_.max_pred_speed = v[3];
    RCLCPP_INFO(
      rclcpp::get_logger("multi_object_tracker"),
      "[NERO_MOT_REBIRTH] min_count %d margin %.2f max_dist %.2f max_pred_speed %.2f",
      nero_rebirth_.min_count, nero_rebirth_.margin, nero_rebirth_.max_dist,
      nero_rebirth_.max_pred_speed);
  }
  if (const char * env = std::getenv("NERO_MOT_DUP_FRESH"); env != nullptr && env[0] != '\0') {
    nero_dup_fresh_ = std::atof(env);
    RCLCPP_INFO(
      rclcpp::get_logger("multi_object_tracker"), "[NERO_MOT_DUP_FRESH] fresh %.2f s",
      nero_dup_fresh_);
  }
}

void TrackerProcessor::predict(const rclcpp::Time & time)
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  for (auto itr = list_tracker_.begin(); itr != list_tracker_.end(); ++itr) {
    (*itr)->predict(time);
  }
}

void TrackerProcessor::associate(
  const types::DynamicObjectList & detected_objects,
  std::unordered_map<int, int> & direct_assignment,
  std::unordered_map<int, int> & reverse_assignment) const
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  const auto & tracker_list = list_tracker_;
  // global nearest neighbor
  Eigen::MatrixXd score_matrix = association_->calcScoreMatrix(
    detected_objects, tracker_list);  // row : tracker, col : measurement
  association_->assign(score_matrix, direct_assignment, reverse_assignment);
}

void TrackerProcessor::update(
  const types::DynamicObjectList & detected_objects,
  const std::unordered_map<int, int> & direct_assignment)
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  int tracker_idx = 0;
  const auto & time = detected_objects.header.stamp;
  for (auto tracker_itr = list_tracker_.begin(); tracker_itr != list_tracker_.end();
       ++tracker_itr, ++tracker_idx) {
    if (direct_assignment.find(tracker_idx) != direct_assignment.end()) {
      // found
      const auto & associated_object =
        detected_objects.objects.at(direct_assignment.find(tracker_idx)->second);
      const types::InputChannel channel_info = channels_config_[associated_object.channel_index];
      (*(tracker_itr))->updateWithMeasurement(associated_object, time, channel_info);

    } else {
      // not found
      (*(tracker_itr))->updateWithoutMeasurement(time);
    }
  }
}

void TrackerProcessor::spawn(
  const types::DynamicObjectList & detected_objects,
  const std::unordered_map<int, int> & reverse_assignment)
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  const auto channel_config = channels_config_[detected_objects.channel_index];
  // If spawn is disabled, return
  if (!channel_config.is_spawn_enabled) {
    return;
  }

  // Spawn new trackers for the objects that are not associated
  const auto & time = detected_objects.header.stamp;
  for (size_t i = 0; i < detected_objects.objects.size(); ++i) {
    if (reverse_assignment.find(i) != reverse_assignment.end()) {  // found
      continue;
    }
    const auto & new_object = detected_objects.objects.at(i);
    std::shared_ptr<Tracker> predecessor;
    types::DynamicObject seeded_object;
    if (nero_rebirth_.enabled) {
      predecessor = neroFindPredecessor(new_object, channel_config, time, seeded_object);
    }
    std::shared_ptr<Tracker> tracker = createNewTracker(predecessor ? seeded_object : new_object, time);
    if (predecessor) {
      predecessor->nero_successor = tracker;
      tracker->nero_is_successor = true;
      tracker->nero_predecessor = predecessor;
      tracker->nero_predecessor_uuid = predecessor->getUUID();
      tracker->nero_predecessor_count = predecessor->getTotalMeasurementCount();
    }

    // Initialize existence probabilities
    if (channel_config.trust_existence_probability) {
      tracker->initializeExistenceProbabilities(
        new_object.channel_index, new_object.existence_probability);
    } else {
      tracker->initializeExistenceProbabilities(new_object.channel_index, 0.5);
    }

    // Update the tracker with the new object
    list_tracker_.push_back(tracker);
  }
}

std::shared_ptr<Tracker> TrackerProcessor::createNewTracker(
  const types::DynamicObject & object, const rclcpp::Time & time) const
{
  const LabelType label =
    autoware::object_recognition_utils::getHighestProbLabel(object.classification);
  if (config_.tracker_map.count(label) != 0) {
    const auto tracker = config_.tracker_map.at(label);
    if (tracker == "bicycle_tracker")
      return std::make_shared<VehicleTracker>(object_model::bicycle, time, object);
    if (tracker == "big_vehicle_tracker")
      return std::make_shared<VehicleTracker>(object_model::big_vehicle, time, object);
    if (tracker == "multi_vehicle_tracker")
      return std::make_shared<MultipleVehicleTracker>(time, object);
    if (tracker == "normal_vehicle_tracker")
      return std::make_shared<VehicleTracker>(object_model::normal_vehicle, time, object);
    if (tracker == "pass_through_tracker")
      return std::make_shared<PassThroughTracker>(time, object);
    if (tracker == "pedestrian_and_bicycle_tracker")
      return std::make_shared<PedestrianAndBicycleTracker>(time, object);
    if (tracker == "pedestrian_tracker") return std::make_shared<PedestrianTracker>(time, object);
  }
  return std::make_shared<UnknownTracker>(time, object);
}

void TrackerProcessor::prune(const rclcpp::Time & time)
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  // Check tracker lifetime: if the tracker is old, delete it
  removeOldTracker(time);
  // Check tracker overlap: if the tracker is overlapped, delete the one with lower IOU
  removeOverlappedTracker(time);
  if (nero_rebirth_.enabled) {
    neroHandOver(time);
  }
}

std::shared_ptr<Tracker> TrackerProcessor::neroFindPredecessor(
  const types::DynamicObject & measurement, const types::InputChannel & channel,
  const rclcpp::Time & time, types::DynamicObject & seeded_measurement) const
{
  // only a classification-trusted vehicle measurement continues an UNKNOWN track
  const auto label = autoware::object_recognition_utils::getHighestProbLabel(measurement.classification);
  if (
    !channel.trust_classification ||
    !autoware::object_recognition_utils::isCarLikeVehicle(label) ||
    measurement.shape.type != autoware_perception_msgs::msg::Shape::BOUNDING_BOX) {
    return nullptr;
  }
  const double myaw = tf2::getYaw(measurement.pose.orientation);
  const double half_l = 0.5 * measurement.shape.dimensions.x + nero_rebirth_.margin;
  const double half_w = 0.5 * measurement.shape.dimensions.y + nero_rebirth_.margin;
  std::shared_ptr<Tracker> best;
  types::DynamicObject best_object;
  for (const auto & candidate : list_tracker_) {
    if (!autoware::object_recognition_utils::isCarLikeVehicle(candidate->getHighestProbLabel())) {
      continue;
    }
    // a vehicle track already covers the place: the UNKNOWN there is its fragment, not a new object
    types::DynamicObject object;
    if (!candidate->getTrackedObject(time, object)) continue;
    const double dx = object.pose.position.x - measurement.pose.position.x;
    const double dy = object.pose.position.y - measurement.pose.position.y;
    if (
      std::abs(dx * std::cos(myaw) + dy * std::sin(myaw)) <= half_l &&
      std::abs(-dx * std::sin(myaw) + dy * std::cos(myaw)) <= half_w) {
      return nullptr;
    }
  }
  for (const auto & candidate : list_tracker_) {
    if (
      !candidate->nero_successor.expired() || candidate->getHighestProbLabel() != Label::UNKNOWN) {
      continue;
    }
    types::DynamicObject object;
    if (!candidate->getTrackedObject(time, object)) continue;
    if (
      std::hypot(object.twist.linear.x, object.twist.linear.y) > nero_rebirth_.max_pred_speed) {
      continue;  // a moving UNKNOWN keeps today's handling
    }
    // the UNKNOWN track's centre inside the vehicle box (+ margin): the same object
    if (!neroCovers(measurement, object)) continue;
    if (!best || candidate->getTotalMeasurementCount() > best->getTotalMeasurementCount()) {
      best = candidate;
      best_object = object;
    }
  }
  if (!best) return nullptr;
  // the vehicle track's motion starts from the UNKNOWN track's (what is published for the object today), not from
  // the centre motion of its first boxes
  using autoware_utils::xyzrpy_covariance_index::XYZRPY_COV_IDX;
  const double uyaw = tf2::getYaw(best_object.pose.orientation);
  const double vx = std::cos(uyaw) * best_object.twist.linear.x - std::sin(uyaw) * best_object.twist.linear.y;
  const double vy = std::sin(uyaw) * best_object.twist.linear.x + std::cos(uyaw) * best_object.twist.linear.y;
  seeded_measurement = measurement;
  seeded_measurement.kinematics.has_twist = true;
  seeded_measurement.twist.linear.x = vx * std::cos(myaw) + vy * std::sin(myaw);
  seeded_measurement.twist.linear.y = 0.0;
  seeded_measurement.kinematics.has_twist_covariance = true;
  seeded_measurement.twist_covariance[XYZRPY_COV_IDX::X_X] = std::max(
    0.25, best_object.twist_covariance[XYZRPY_COV_IDX::X_X] +
            best_object.twist_covariance[XYZRPY_COV_IDX::Y_Y]);
  return best;
}

bool TrackerProcessor::neroCovers(
  const types::DynamicObject & vehicle, const types::DynamicObject & unknown) const
{
  const double dx = unknown.pose.position.x - vehicle.pose.position.x;
  const double dy = unknown.pose.position.y - vehicle.pose.position.y;
  if (std::hypot(dx, dy) > nero_rebirth_.max_dist) return false;
  const double yaw = tf2::getYaw(vehicle.pose.orientation);
  return std::abs(dx * std::cos(yaw) + dy * std::sin(yaw)) <=
           0.5 * vehicle.shape.dimensions.x + nero_rebirth_.margin &&
         std::abs(-dx * std::sin(yaw) + dy * std::cos(yaw)) <=
           0.5 * vehicle.shape.dimensions.y + nero_rebirth_.margin;
}

bool TrackerProcessor::neroStockRemoves(
  const std::shared_ptr<Tracker> & vehicle_tracker, const std::shared_ptr<Tracker> & unknown_tracker,
  const rclcpp::Time & time) const
{
  // removeOverlappedTracker's rule for an UNKNOWN track overlapping a vehicle track
  types::DynamicObject vehicle;
  types::DynamicObject unknown;
  if (
    !vehicle_tracker->getTrackedObject(time, vehicle) ||
    !unknown_tracker->getTrackedObject(time, unknown)) {
    return false;
  }
  const double distance = std::hypot(
    vehicle.pose.position.x - unknown.pose.position.x,
    vehicle.pose.position.y - unknown.pose.position.y);
  if (
    distance > config_.max_dist_matrix(Label::UNKNOWN, vehicle_tracker->getHighestProbLabel())) {
    return false;
  }
  constexpr double min_union_iou_area = 1e-2;
  return shapes::get2dIoU(vehicle, unknown, min_union_iou_area) >
           config_.min_unknown_object_removal_iou &&
         vehicle_tracker->getElapsedTimeFromLastUpdate(time) <=
           unknown_tracker->getElapsedTimeFromLastUpdate(time);
}

bool TrackerProcessor::neroHidden(
  const std::shared_ptr<Tracker> & tracker, const rclcpp::Time & time) const
{
  (void)time;
  const auto successor = tracker->nero_successor.lock();
  if (!successor || !successor->nero_handed_over || tracker->nero_removed_at_count < 0) {
    return false;
  }
  // a stock track removed there would be published again after confident_count_threshold new measurements
  return tracker->getTotalMeasurementCount() - tracker->nero_removed_at_count <
         config_.confident_count_threshold.at(Label::UNKNOWN);
}

void TrackerProcessor::neroHandOver(const rclcpp::Time & time)
{
  for (const auto & tracker : list_tracker_) {
    // an UNKNOWN track whose vehicle successor died publishes the object under its uuid again
    if (
      tracker->nero_handed_over_to && tracker->nero_successor.expired() &&
      tracker->getUUID() != tracker->nero_own_uuid) {
      tracker->setUUID(tracker->nero_own_uuid);
    }
    if (!tracker->nero_is_successor) continue;
    const auto predecessor = tracker->nero_predecessor.lock();
    if (predecessor) {
      types::DynamicObject vehicle;
      types::DynamicObject unknown;
      if (
        tracker->getTrackedObject(time, vehicle) && predecessor->getTrackedObject(time, unknown) &&
        !neroCovers(vehicle, unknown)) {
        // the vehicle track left the UNKNOWN track's object: two objects from now on
        predecessor->nero_successor.reset();
        predecessor->nero_handed_over_to = false;
        predecessor->nero_removed_at_count = -1;
        tracker->nero_predecessor.reset();
        if (!tracker->nero_handed_over) tracker->nero_is_successor = false;
        continue;
      }
    }
    if (tracker->nero_handed_over) {
      if (predecessor && neroStockRemoves(tracker, predecessor, time)) {
        predecessor->nero_removed_at_count = predecessor->getTotalMeasurementCount();
      }
      continue;
    }
    if (predecessor) {
      tracker->nero_predecessor_count = predecessor->getTotalMeasurementCount();
    }
    if (!isConfidentTracker(tracker)) continue;
    // the vehicle track publishes the object under its uuid; the UNKNOWN track goes on under the vehicle track's
    // (never published) uuid, hidden where the stock overlap rule would have removed it (neroHidden)
    const auto vehicle_uuid = tracker->getUUID();
    tracker->setUUID(tracker->nero_predecessor_uuid);
    tracker->nero_handed_over = true;
    if (predecessor) {
      predecessor->nero_own_uuid = tracker->nero_predecessor_uuid;
      predecessor->setUUID(vehicle_uuid);
      predecessor->nero_handed_over_to = true;
      predecessor->nero_removed_at_count =
        neroStockRemoves(tracker, predecessor, time) ? predecessor->getTotalMeasurementCount() : -1;
    }
  }
}

void TrackerProcessor::removeOldTracker(const rclcpp::Time & time)
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  // Check elapsed time from last update
  for (auto itr = list_tracker_.begin(); itr != list_tracker_.end(); ++itr) {
    const bool is_old = config_.tracker_lifetime < (*itr)->getElapsedTimeFromLastUpdate(time);
    // If the tracker is old, delete it
    if (is_old) {
      auto erase_itr = itr;
      --itr;
      list_tracker_.erase(erase_itr);
    }
  }
}

// NERO flatos (NERO_MOT_SPAWN_PRUNE): removes the UNKNOWN trackers just spawned from a measurement whose centre lies
// inside a live moving vehicle track's box at that measurement's time (a piece of that vehicle)
std::vector<std::shared_ptr<Tracker>> TrackerProcessor::removeSpawnedUnknownOverlapped(
  const rclcpp::Time & time, const std::vector<std::shared_ptr<Tracker>> & spawned, const double max_age,
  const double min_vehicle_speed)
{
  std::vector<std::shared_ptr<Tracker>> removed;
  for (const auto & candidate : spawned) {
    if (candidate->getHighestProbLabel() != Label::UNKNOWN) continue;  // new vehicles etc.: today's handling
    types::DynamicObject object2;
    if (!candidate->getTrackedObject(time, object2)) continue;
    for (const auto & other : list_tracker_) {
      if (std::find(spawned.begin(), spawned.end(), other) != spawned.end()) continue;
      // a live vehicle track only (measured within max_age), not one coasting on its prediction
      if (!autoware::object_recognition_utils::isCarLikeVehicle(other->getHighestProbLabel())) continue;
      if (other->getElapsedTimeFromLastUpdate(time) > max_age) continue;
      types::DynamicObject object1;
      if (!other->getTrackedObject(time, object1)) continue;
      if (object1.shape.type != autoware_perception_msgs::msg::Shape::BOUNDING_BOX) continue;
      // a MOVING vehicle only: for a standing one the publish-time check sees the same geometry (no timing gap)
      if (std::hypot(object1.twist.linear.x, object1.twist.linear.y) < min_vehicle_speed) continue;
      // the new UNKNOWN's centre inside the vehicle's box at the same time = a piece of that vehicle (its rear band
      // under the crop ceiling, left outside the car track because UNKNOWN measurements never update a vehicle track
      // and prune()'s centre-distance gate (1 m) never reaches a vehicle's ends)
      const double yaw = tf2::getYaw(object1.pose.orientation);
      const double dx = object2.pose.position.x - object1.pose.position.x;
      const double dy = object2.pose.position.y - object1.pose.position.y;
      if (
        std::abs(dx * std::cos(yaw) + dy * std::sin(yaw)) <= 0.5 * object1.shape.dimensions.x &&
        std::abs(-dx * std::sin(yaw) + dy * std::cos(yaw)) <= 0.5 * object1.shape.dimensions.y) {
        removed.push_back(candidate);
        break;
      }
    }
  }
  for (const auto & tracker : removed) list_tracker_.remove(tracker);
  return removed;
}

// This function removes overlapped trackers based on distance and IoU criteria
void TrackerProcessor::removeOverlappedTracker(const rclcpp::Time & time)
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  // Create sorted list with non-UNKNOWN objects first, then by measurement count
  std::vector<std::shared_ptr<Tracker>> sorted_list_tracker(
    list_tracker_.begin(), list_tracker_.end());
  std::sort(
    sorted_list_tracker.begin(), sorted_list_tracker.end(),
    [&time](const std::shared_ptr<Tracker> & a, const std::shared_ptr<Tracker> & b) {
      bool a_unknown = (a->getHighestProbLabel() == Label::UNKNOWN);
      bool b_unknown = (b->getHighestProbLabel() == Label::UNKNOWN);
      if (a_unknown != b_unknown) {
        return b_unknown;  // Put non-UNKNOWN objects first
      }
      if (a->getTotalMeasurementCount() != b->getTotalMeasurementCount()) {
        return a->getTotalMeasurementCount() >
               b->getTotalMeasurementCount();  // Then sort by measurement count
      }
      return a->getElapsedTimeFromLastUpdate(time) <
             b->getElapsedTimeFromLastUpdate(time);  // Finally sort by elapsed time (smaller first)
    });

  /* Iterate through the list of trackers */
  for (size_t i = 0; i < sorted_list_tracker.size(); ++i) {
    types::DynamicObject object1;
    if (!sorted_list_tracker[i]->getTrackedObject(time, object1)) continue;
    // NERO_MOT_REBIRTH: an UNKNOWN track hidden behind its vehicle successor removes no other track
    if (nero_rebirth_.enabled && neroHidden(sorted_list_tracker[i], time)) continue;
    // Compare the current tracker with the remaining trackers
    for (size_t j = i + 1; j < sorted_list_tracker.size(); ++j) {
      types::DynamicObject object2;
      if (!sorted_list_tracker[j]->getTrackedObject(time, object2)) continue;

      // Calculate the distance between the two objects
      const double distance = std::hypot(
        object1.pose.position.x - object2.pose.position.x,
        object1.pose.position.y - object2.pose.position.y);
      const auto & label1 = sorted_list_tracker[i]->getHighestProbLabel();
      const auto & label2 = sorted_list_tracker[j]->getHighestProbLabel();
      const double max_dist_matrix_value = config_.max_dist_matrix(
        label2, label1);  // Get the maximum distance threshold for the labels

      // If the distance is too large, skip
      if (distance > max_dist_matrix_value) {
        continue;
      }

      // Check the Intersection over Union (IoU) between the two objects
      constexpr double min_union_iou_area = 1e-2;
      const auto iou = shapes::get2dIoU(object1, object2, min_union_iou_area);
      bool delete_candidate_tracker = false;

      // NERO_MOT_REBIRTH: an UNKNOWN track continued by a vehicle track stays until the vehicle track is published
      if (
        nero_rebirth_.enabled &&
        sorted_list_tracker[j]->nero_successor.lock() == sorted_list_tracker[i]) {
        continue;
      }

      // If both trackers are UNKNOWN, delete the younger tracker
      // If one side of the tracker is UNKNOWN, delete UNKNOWN objects
      if (label1 == Label::UNKNOWN || label2 == Label::UNKNOWN) {
        if (iou > config_.min_unknown_object_removal_iou) {
          if (label2 == Label::UNKNOWN) {
            delete_candidate_tracker = true;
          }
        }
      } else {  // If neither object is UNKNOWN, delete the younger tracker
        if (iou > config_.min_known_object_removal_iou) {
          /* erase only when prioritized one has a measurement */
          delete_candidate_tracker = true;
        }
      }

      if (delete_candidate_tracker) {
        /* erase only when prioritized one has later(or equal time) meas than the other's */
        // NERO_MOT_DUP_FRESH: or, for two vehicle tracks, when the prioritized one is confident and had a full (box)
        // measurement within <fresh> s, whichever input channel measured the other one last
        if (
          sorted_list_tracker[i]->getElapsedTimeFromLastUpdate(time) <=
            sorted_list_tracker[j]->getElapsedTimeFromLastUpdate(time) ||
          (nero_dup_fresh_ > 0.0 && label1 != Label::UNKNOWN && label2 != Label::UNKNOWN &&
           sorted_list_tracker[i]->neroElapsedFromFullUpdate(time) <= nero_dup_fresh_ &&
           isConfidentTracker(sorted_list_tracker[i]))) {
          // Remove from original list_tracker
          list_tracker_.remove(sorted_list_tracker[j]);
          // Remove from sorted list
          sorted_list_tracker.erase(sorted_list_tracker.begin() + j);
          --j;
        }
      }
    }
  }
}

bool TrackerProcessor::isConfidentTracker(const std::shared_ptr<Tracker> & tracker) const
{
  // Confidence is determined by counting the number of measurements.
  // If the number of measurements is equal to or greater than the threshold, the tracker is
  // considered confident.
  auto label = tracker->getHighestProbLabel();
  if (nero_rebirth_.enabled && tracker->nero_handed_over_to) {
    return true;
  }
  if (nero_rebirth_.enabled && tracker->nero_is_successor) {
    // NERO_MOT_REBIRTH: the object's measurements of both classes count
    if (tracker->nero_handed_over) return true;
    const auto predecessor = tracker->nero_predecessor.lock();
    const int predecessor_count =
      predecessor ? predecessor->getTotalMeasurementCount() : tracker->nero_predecessor_count;
    return tracker->getTotalMeasurementCount() >= nero_rebirth_.min_count &&
           tracker->getTotalMeasurementCount() + predecessor_count >=
             config_.confident_count_threshold.at(label);
  }
  return tracker->getTotalMeasurementCount() >= config_.confident_count_threshold.at(label);
}

void TrackerProcessor::getTrackedObjects(
  const rclcpp::Time & time, autoware_perception_msgs::msg::TrackedObjects & tracked_objects) const
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  tracked_objects.header.stamp = time;
  types::DynamicObject tracked_object;
  for (const auto & tracker : list_tracker_) {
    // Skip if the tracker is not confident
    if (!isConfidentTracker(tracker)) continue;
    // NERO_MOT_REBIRTH: the UNKNOWN track whose object its vehicle successor publishes
    if (nero_rebirth_.enabled && neroHidden(tracker, time)) continue;
    // Get the tracked object, extrapolated to the given time
    if (tracker->getTrackedObject(time, tracked_object)) {
      tracked_objects.objects.push_back(toTrackedObjectMsg(tracked_object));
    }
  }
}

void TrackerProcessor::getTentativeObjects(
  const rclcpp::Time & time,
  autoware_perception_msgs::msg::TrackedObjects & tentative_objects) const
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  tentative_objects.header.stamp = time;
  types::DynamicObject tracked_object;
  for (const auto & tracker : list_tracker_) {
    if (!isConfidentTracker(tracker)) {
      if (tracker->getTrackedObject(time, tracked_object)) {
        tentative_objects.objects.push_back(toTrackedObjectMsg(tracked_object));
      }
    }
  }
}

void TrackerProcessor::setTimeKeeper(std::shared_ptr<autoware_utils::TimeKeeper> time_keeper_ptr)
{
  time_keeper_ = std::move(time_keeper_ptr);
}

}  // namespace autoware::multi_object_tracker
