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

#include "autoware/multi_object_tracker/association/association.hpp"
#include "autoware/multi_object_tracker/object_model/types.hpp"
#include "autoware/multi_object_tracker/tracker/model/tracker_base.hpp"

#include <autoware_utils/system/time_keeper.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/utils.h>

#include "autoware_perception_msgs/msg/detected_objects.hpp"
#include "autoware_perception_msgs/msg/tracked_objects.hpp"

#include <algorithm>
#include <cmath>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace autoware::multi_object_tracker
{
class UnknownTracker;  // NERO_MOT_UNKNOWN_MOVER
using LabelType = autoware_perception_msgs::msg::ObjectClassification::_label_type;

struct TrackerProcessorConfig
{
  std::map<LabelType, std::string> tracker_map;
  float tracker_lifetime;                              // [s]
  float min_known_object_removal_iou;                  // ratio [0, 1]
  float min_unknown_object_removal_iou;                // ratio [0, 1]
  std::map<LabelType, int> confident_count_threshold;  // [count]
  Eigen::MatrixXd max_dist_matrix;
};

class TrackerProcessor
{
public:
  TrackerProcessor(
    const TrackerProcessorConfig & config, const AssociatorConfig & associator_config,
    const std::vector<types::InputChannel> & channels_config);

  const std::list<std::shared_ptr<Tracker>> & getListTracker() const { return list_tracker_; }
  // tracker processes
  void predict(const rclcpp::Time & time);
  void associate(
    const types::DynamicObjectList & detected_objects,
    std::unordered_map<int, int> & direct_assignment,
    std::unordered_map<int, int> & reverse_assignment) const;
  void update(
    const types::DynamicObjectList & detected_objects,
    const std::unordered_map<int, int> & direct_assignment);
  void spawn(
    const types::DynamicObjectList & detected_objects,
    const std::unordered_map<int, int> & reverse_assignment);
  void prune(const rclcpp::Time & time);
  // NERO flatos (NERO_MOT_SPAWN_PRUNE): an UNKNOWN tracker just spawned from a measurement whose centre lies inside a
  // live (measured within max_age) MOVING (>= min_vehicle_speed) vehicle track's box at that measurement's time is a
  // piece of that vehicle and is removed. Returns the removed trackers.
  std::vector<std::shared_ptr<Tracker>> removeSpawnedUnknownOverlapped(
    const rclcpp::Time & time, const std::vector<std::shared_ptr<Tracker>> & spawned, const double max_age,
    const double min_vehicle_speed);

  // output
  bool isConfidentTracker(const std::shared_ptr<Tracker> & tracker) const;
  void getTrackedObjects(
    const rclcpp::Time & time,
    autoware_perception_msgs::msg::TrackedObjects & tracked_objects) const;
  void getTentativeObjects(
    const rclcpp::Time & time,
    autoware_perception_msgs::msg::TrackedObjects & tentative_objects) const;

  void getExistenceProbabilities(std::vector<std::vector<float>> & existence_vectors) const;

  void setTimeKeeper(std::shared_ptr<autoware_utils::TimeKeeper> time_keeper_ptr);

private:
  const TrackerProcessorConfig config_;
  const std::vector<types::InputChannel> & channels_config_;

  std::unique_ptr<DataAssociation> association_;

  std::list<std::shared_ptr<Tracker>> list_tracker_;
  void removeOldTracker(const rclcpp::Time & time);
  void removeOverlappedTracker(const rclcpp::Time & time);
  std::shared_ptr<Tracker> createNewTracker(
    const types::DynamicObject & object, const rclcpp::Time & time) const;

  std::shared_ptr<autoware_utils::TimeKeeper> time_keeper_;

  // NERO_MOT_REBIRTH=<min_count>,<margin m>,<max_dist m>,<max_pred_speed m/s> (tbx, 2026-10-28; unset = stock): a
  // vehicle measurement that starts a new track on an UNKNOWN track's object (the UNKNOWN track's centre inside its
  // box + margin, standing or slower than max_pred_speed) continues that object: its motion starts from the UNKNOWN
  // track's velocity, the object's measurements of both classes count for confident_count_threshold (with at least
  // <min_count> vehicle measurements), and when published it takes the UNKNOWN track's uuid (the UNKNOWN track takes
  // the vehicle track's unpublished one). The UNKNOWN track stays (updated by the object's clusters) while the vehicle
  // track lives: hidden where the stock overlap rule would have removed it and until it has had as many measurements
  // since as a stock track needs to be published again (confident_count_threshold), a separate object otherwise; it takes the object's uuid back if the vehicle track dies (never a gap, no uuid churn). The link ends
  // when the UNKNOWN track's centre leaves the vehicle box (+ margin): two objects from then on (before the hand-over
  // the vehicle track goes on as a stock one)
  struct NeroRebirth
  {
    bool enabled{false};
    int min_count{2};
    double margin{0.5};
    double max_dist{3.0};
    double max_pred_speed{1.0};
  };
  NeroRebirth nero_rebirth_;
  std::shared_ptr<Tracker> neroFindPredecessor(
    const types::DynamicObject & measurement, const types::InputChannel & channel,
    const rclcpp::Time & time, types::DynamicObject & seeded_measurement) const;
  bool neroCovers(const types::DynamicObject & vehicle, const types::DynamicObject & unknown) const;
  // NERO_MOT_FLIP_DEFER (flipspeed): a published vehicle track at a judged track's place and its velocity
  bool neroFindPartner(
    const std::shared_ptr<Tracker> & tracker, const rclcpp::Time & time, double & vx, double & vy) const;
  bool neroStockRemoves(
    const std::shared_ptr<Tracker> & vehicle, const std::shared_ptr<Tracker> & unknown,
    const rclcpp::Time & time) const;
  void neroHandOver(const rclcpp::Time & time);
  bool neroHidden(const std::shared_ptr<Tracker> & tracker, const rclcpp::Time & time) const;

  // NERO_MOT_DUP_FRESH=<fresh s> (tbx, 2026-10-28; unset = stock; upstream's freshness tier, autoware_universe #12996 /
  // merger/survival_ranking.cpp, full_measure_stale_threshold 0.35 s): removeOverlappedTracker deletes a vehicle track
  // overlapping a prioritized vehicle track (IoU > min_known_object_removal_iou) also when the prioritized one is
  // confident and had a FULL measurement (a channel that trusts the box extent: not detection_by_tracker, which only
  // re-measures the track's own shape) within <fresh> s, not only when its last measurement is the newer one. With two input
  // channels a duplicate born from the other channel's detection otherwise survives every prune, gets published with
  // the speed of its first boxes' motion and nearer the path (10-26e #59: 279a07 on the parked 41db95, -2.82)
  double nero_dup_fresh_{0.0};

  // NERO_MOT_UNKNOWN_MOVER (oncomingtrack, 2026-10-31; unset = stock; UnknownTracker holds the rule): an UNKNOWN
  // measurement that no track took but that continues a recently measured UNKNOWN track's straight mover line updates
  // that track instead of starting one at speed 0; with NERO_MOT_REBIRTH, a vehicle measurement on a mover whose motion
  // runs along the vehicle box (within 30 deg, either way) continues it like a standing one, heading the way it moves
  bool neroMoverLink(
    const types::DynamicObject & measurement, const types::InputChannel & channel, const rclcpp::Time & time);
  void neroMoverInherit(
    const std::shared_ptr<UnknownTracker> & newborn, const types::DynamicObject & measurement,
    const rclcpp::Time & time);
  bool neroMoverAlong(
    const std::shared_ptr<Tracker> & tracker, const rclcpp::Time & time, const double box_yaw) const;
  // crossmover c3 (NERO_MOT_UNKNOWN_MOVER with the crossing fields): an UNKNOWN published fast because of a crossing
  // mover line within the last mover hold (a young mover, or one whose line just ended) is continued by a vehicle box
  // its motion runs along (within 30 deg, either way) like a slow one, and no vehicle track that is not published yet
  // removes it (else the object vanishes until that track is confident: 10-27 20:45:11.3, 13 m in the path).
  // Oncoming movers keep oncomingtrack's rules (a 0.1 s fragment line at a standing vehicle: 20:28:47.9)
  bool neroMoverMadeAlong(
    const std::shared_ptr<Tracker> & tracker, const types::DynamicObject & object, const rclcpp::Time & time,
    const double box_yaw) const;
  bool neroMoverMadeFast(
    const std::shared_ptr<Tracker> & tracker, const types::DynamicObject & object, const rclcpp::Time & time) const;
};

}  // namespace autoware::multi_object_tracker

#endif  // PROCESSOR__PROCESSOR_HPP_
