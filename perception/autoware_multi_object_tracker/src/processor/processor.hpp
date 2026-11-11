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

#ifndef PROCESSOR__PROCESSOR_HPP_
#define PROCESSOR__PROCESSOR_HPP_

#include "autoware/multi_object_tracker/association/adaptive_threshold_cache.hpp"
#include "autoware/multi_object_tracker/association/association_manager.hpp"
#include "autoware/multi_object_tracker/merger/tracker_overlap_manager.hpp"
#include "autoware/multi_object_tracker/tracker/trackers/polygon_tracker.hpp"
#include "autoware/multi_object_tracker/tracker/trackers/tracker_base.hpp"
#include "autoware/multi_object_tracker/types.hpp"

#include <autoware_utils_debug/time_keeper.hpp>
#include <rclcpp/rclcpp.hpp>

#include "autoware_perception_msgs/msg/detected_objects.hpp"
#include "autoware_perception_msgs/msg/tracked_objects.hpp"
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <list>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace autoware::multi_object_tracker
{
class TrackerProcessor
{
public:
  TrackerProcessor(
    const TrackerConfigs & tracker_configs, const TrackerCreationConfig & creation_config,
    const TrackerAssociationConfig & association_config,
    const TrackerOverlapManagerConfig & tracker_overlap_manager_config,
    const std::vector<types::InputChannel> & channels_config, const rclcpp::Logger & logger,
    rclcpp::Clock::SharedPtr clock);

  const std::list<std::shared_ptr<Tracker>> & getListTracker() const { return list_tracker_; }

  // Tracker processes
  void updateEgoPose(const std::optional<geometry_msgs::msg::PoseStamped> & ego_pose_stamped);
  void predictTrackers(const rclcpp::Time & time);
  types::AssociationResult associate(const types::DynamicObjectList & detected_objects) const;
  void update(const types::AssociatedObjects & associated_objects);
  void spawn(const types::AssociatedObjects & associated_objects);
  void prune(const rclcpp::Time & time);

  // Output processes
  void getTrackedObjects(
    const rclcpp::Time & time,
    autoware_perception_msgs::msg::TrackedObjects & tracked_objects) const;
  void getTentativeObjects(
    const rclcpp::Time & time,
    autoware_perception_msgs::msg::TrackedObjects & tentative_objects) const;
  void getMergedObjects(
    const rclcpp::Time & time, const geometry_msgs::msg::Transform & tf_base_to_world,
    autoware_perception_msgs::msg::DetectedObjects & merged_objects) const;

  void setTimeKeeper(std::shared_ptr<autoware_utils_debug::TimeKeeper> time_keeper_ptr);

private:
  const TrackerConfigs tracker_configs_;
  const TrackerCreationConfig creation_config_;
  const std::vector<types::InputChannel> & channels_config_;

  std::optional<geometry_msgs::msg::PoseStamped> ego_pose_;

  std::unique_ptr<AssociationManager> association_manager_;
  std::unique_ptr<TrackerOverlapManager> tracker_overlap_manager_;

  mutable rclcpp::Time last_prune_time_;

  std::list<std::shared_ptr<Tracker>> list_tracker_;
  std::optional<geometry_msgs::msg::Pose> getEgoPose() const;
  // NERO_MOT_STILL_BOX (boxyaw, ported): re-align a still vehicle's PUBLISHED box with its measured points
  void neroStillBox(
    const std::shared_ptr<Tracker> & tracker, const rclcpp::Time & time, types::DynamicObject & object) const;
  void neroStillBoxBirth(
    const std::shared_ptr<Tracker> & tracker, const types::DynamicObject & object, const rclcpp::Time & time) const;
  void removeOldTracker(const rclcpp::Time & time);
  std::shared_ptr<Tracker> createNewTracker(
    const types::DynamicObject & object, const rclcpp::Time & time) const;

  std::shared_ptr<autoware_utils_debug::TimeKeeper> time_keeper_;
  AdaptiveThresholdCache adaptive_threshold_cache_;

  rclcpp::Logger logger_;
  rclcpp::Clock::SharedPtr clock_;

  // [nero trkup] NERO_MOT_REBIRTH=<min_count>,<margin m>,<max_dist m>,<max_pred_speed m/s> (tbx_1029, ported from our
  // 0.44 TrackerProcessor; unset = upstream): a vehicle measurement that starts a new track on an UNKNOWN track's
  // object (the UNKNOWN's published centre inside its box + margin, standing or slower than max_pred_speed; with
  // NERO_MOT_UNKNOWN_MOVER also a mover along the box) continues that object: its motion starts from the UNKNOWN's
  // published one (a mover's: its line), it is confident with >= <min_count> measurements of its own once the UNKNOWN
  // is, and when published it takes the UNKNOWN's published uuid; the UNKNOWN goes on (hidden where the stock prune
  // would absorb it) and takes the uuid back if the vehicle track dies. The link ends when the UNKNOWN's centre leaves
  // the vehicle box (+ margin).
  struct NeroRebirth
  {
    bool enabled{false};
    int min_count{2};
    double margin{0.5};
    double max_dist{3.0};
    double max_pred_speed{1.0};
  };
  NeroRebirth nero_rebirth_;
  TrackerOverlapManagerConfig nero_overlap_config_;
  std::shared_ptr<Tracker> neroFindPredecessor(
    const types::DynamicObject & measurement, const types::InputChannel & channel,
    const rclcpp::Time & time, types::DynamicObject & seeded_measurement) const;
  bool neroCovers(const types::DynamicObject & vehicle, const types::DynamicObject & unknown) const;
  bool neroStockRemoves(
    const std::shared_ptr<Tracker> & vehicle, const std::shared_ptr<Tracker> & unknown,
    const rclcpp::Time & time) const;
  void neroHandOver(const rclcpp::Time & time);
  // [nero trkup] NERO_MOT_UNKNOWN_MOVER (oncomingtrack / crossmover; PolygonTracker holds the rule): an UNKNOWN
  // measurement no track took that continues a recently measured UNKNOWN track's mover line updates that track; a new
  // UNKNOWN track continuing a young track the gates lost takes over its positions; with NERO_MOT_REBIRTH a vehicle
  // measurement on a mover whose motion runs along the box continues it, heading the way it moves
  bool neroMoverLink(
    const types::DynamicObject & measurement, const types::InputChannel & channel, const rclcpp::Time & time);
  void neroMoverInherit(
    const std::shared_ptr<PolygonTracker> & newborn, const types::DynamicObject & measurement,
    const rclcpp::Time & time);
  bool neroMoverAlong(
    const std::shared_ptr<Tracker> & tracker, const rclcpp::Time & time, const double box_yaw) const;
  bool neroMoverMadeAlong(
    const std::shared_ptr<Tracker> & tracker, const types::DynamicObject & object, const rclcpp::Time & time,
    const double box_yaw) const;
  bool neroMoverMadeFast(
    const std::shared_ptr<Tracker> & tracker, const types::DynamicObject & object, const rclcpp::Time & time) const;
};

}  // namespace autoware::multi_object_tracker

#endif  // PROCESSOR__PROCESSOR_HPP_
