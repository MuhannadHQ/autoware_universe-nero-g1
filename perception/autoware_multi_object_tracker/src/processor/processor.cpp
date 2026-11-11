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

#include "autoware/multi_object_tracker/merger/detail/redundancy_check.hpp"
#include "autoware/multi_object_tracker/object_model/object_model.hpp"
#include "autoware/multi_object_tracker/tracker/nero/nero_unknown_mover.hpp"
#include "autoware/multi_object_tracker/tracker/tracker.hpp"
#include "autoware/multi_object_tracker/tracker/nero/nero_still_box.hpp"
#include "autoware/multi_object_tracker/tracker/trackers/static_tracker.hpp"
#include "autoware/multi_object_tracker/types.hpp"

#include <autoware_utils_geometry/msg/covariance.hpp>
#include <autoware_utils_math/normalization.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2/transform_datatypes.hpp>
#include <tf2/utils.hpp>

#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <autoware_utils_math/normalization.hpp>
#include <limits>
#include <tf2/utils.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace autoware::multi_object_tracker
{
using autoware_utils_debug::ScopedTimeTrack;

namespace
{
// [nero trkup] NERO_MOT_REBIRTH: the car's isCarLikeVehicle
bool neroCarLike(const classes::Label label)
{
  return label == classes::Label::CAR || label == classes::Label::TRUCK || label == classes::Label::BUS ||
         label == classes::Label::TRAILER;
}
}  // namespace

TrackerProcessor::TrackerProcessor(
  const TrackerConfigs & tracker_configs, const TrackerCreationConfig & creation_config,
  const TrackerAssociationConfig & association_config,
  const TrackerOverlapManagerConfig & tracker_overlap_manager_config,
  const std::vector<types::InputChannel> & channels_config, const rclcpp::Logger & logger,
  rclcpp::Clock::SharedPtr clock)
: tracker_configs_(tracker_configs),
  creation_config_(creation_config),
  channels_config_(channels_config),
  logger_(logger),
  clock_(std::move(clock))
{
  association_manager_ = std::make_unique<AssociationManager>(association_config, channels_config);
  tracker_overlap_manager_ =
    std::make_unique<TrackerOverlapManager>(tracker_overlap_manager_config);
  nero_overlap_config_ = tracker_overlap_manager_config;

  // [nero trkup] NERO_MOT_REBIRTH (unset = upstream)
  if (const char * env = std::getenv("NERO_MOT_REBIRTH"); env != nullptr && env[0] != '\0') {
    double v[4] = {2.0, 0.5, 3.0, 1.0};
    std::sscanf(env, "%lf,%lf,%lf,%lf", &v[0], &v[1], &v[2], &v[3]);
    nero_rebirth_.enabled = true;
    nero_rebirth_.min_count = std::max(1, static_cast<int>(v[0]));
    nero_rebirth_.margin = v[1];
    nero_rebirth_.max_dist = v[2];
    nero_rebirth_.max_pred_speed = v[3];
    RCLCPP_INFO(
      logger_, "[NERO_MOT_REBIRTH] min_count %d margin %.2f max_dist %.2f max_pred_speed %.2f",
      nero_rebirth_.min_count, nero_rebirth_.margin, nero_rebirth_.max_dist, nero_rebirth_.max_pred_speed);
  }
}

std::optional<geometry_msgs::msg::Pose> TrackerProcessor::getEgoPose() const
{
  return ego_pose_ ? std::make_optional(ego_pose_->pose) : std::nullopt;
}

void TrackerProcessor::updateEgoPose(
  const std::optional<geometry_msgs::msg::PoseStamped> & ego_pose_stamped)
{
  ego_pose_ = ego_pose_stamped;
}

void TrackerProcessor::predictTrackers(const rclcpp::Time & time)
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  for (auto itr = list_tracker_.begin(); itr != list_tracker_.end(); ++itr) {
    (*itr)->predict(time);
  }
}

types::AssociationResult TrackerProcessor::associate(
  const types::DynamicObjectList & detected_objects) const
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  return association_manager_->associate(detected_objects, list_tracker_, ego_pose_);
}

void TrackerProcessor::update(const types::AssociatedObjects & associated_objects)
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  const auto & detected_objects = associated_objects.objects;
  const auto & association_result = associated_objects.association;

  int tracker_idx = 0;
  const auto & time = detected_objects.header.stamp;
  // [nero trkup] crossmover: the robot's position at this measurement time (the crossing mover's log line only)
  if (nero::unknownMoverParams().cross && ego_pose_) {
    nero::setEgo(ego_pose_->pose.position.x, ego_pose_->pose.position.y);
  }
  for (auto tracker_itr = list_tracker_.begin(); tracker_itr != list_tracker_.end();
       ++tracker_itr, ++tracker_idx) {
    bool found = false;
    size_t measurement_idx = 0;
    unique_identifier_msgs::msg::UUID tracker_uuid = (*tracker_itr)->getUUID();

    if (association_result.tracker_to_measurement.count(tracker_uuid)) {
      unique_identifier_msgs::msg::UUID measurement_uuid =
        association_result.tracker_to_measurement.at(tracker_uuid);
      const auto idx = detected_objects.getObjectIndexByUuid(measurement_uuid);
      if (idx) {
        measurement_idx = *idx;
        found = true;
      }
    }

    if (found) {
      const auto & associated_object = detected_objects.objects.at(measurement_idx);
      const types::InputChannel channel_info = channels_config_[associated_object.channel_index];
      const bool has_significant_shape_change = association_result.wasShapeChanged(tracker_uuid);
      (*tracker_itr)
        ->setEgoPose(ego_pose_ ? std::make_optional(ego_pose_->pose.position) : std::nullopt);
      if (neroStillBoxParams().enabled && ego_pose_) {
        // NERO_MOT_STILL_BOX: the robot's pose at this measurement
        (*tracker_itr)->nero_sb_has_ego = true;
        (*tracker_itr)->nero_sb_ego = {
          ego_pose_->pose.position.x, ego_pose_->pose.position.y, tf2::getYaw(ego_pose_->pose.orientation)};
      }
      (*(tracker_itr))
        ->updateWithMeasurement(
          associated_object, time, channel_info, has_significant_shape_change);
      // [nero trkup] NERO_MOT_REBIRTH: a successor's <min_count> counts its classification-trusted vehicle measurements
      // (on the car only those reach a vehicle track; upstream also lets clusters and detection_by_tracker boxes in)
      if (
        (*tracker_itr)->nero_is_successor && channel_info.trust_classification &&
        neroCarLike(classes::getHighestProbLabel(associated_object.classification))) {
        ++(*tracker_itr)->nero_vehicle_count;
      }
    } else {
      (*(tracker_itr))->updateWithoutMeasurement(time);
    }
  }
}

void TrackerProcessor::spawn(const types::AssociatedObjects & associated_objects)
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  const auto & detected_objects = associated_objects.objects;
  const auto & association_result = associated_objects.association;

  const auto channel_config = channels_config_[detected_objects.channel_index];
  if (!channel_config.is_spawn_enabled) {
    return;
  }

  const auto & time = detected_objects.header.stamp;
  const bool nero_mover = nero::unknownMoverParams().enabled;
  for (size_t i = 0; i < detected_objects.objects.size(); ++i) {
    const auto & new_object = detected_objects.objects.at(i);
    if (association_result.measurement_to_tracker.count(new_object.uuid)) {
      continue;
    }
    // [nero trkup] NERO_MOT_UNKNOWN_MOVER: the measurement continues a mover's line: that track takes it
    if (nero_mover && neroMoverLink(new_object, channels_config_[new_object.channel_index], time)) {
      continue;
    }
    // [nero trkup] NERO_MOT_REBIRTH: a vehicle measurement on an UNKNOWN track's object continues that object
    std::shared_ptr<Tracker> predecessor;
    types::DynamicObject seeded_object;
    if (nero_rebirth_.enabled) {
      predecessor = neroFindPredecessor(new_object, channel_config, time, seeded_object);
    }
    std::shared_ptr<Tracker> tracker =
      createNewTracker(predecessor ? seeded_object : new_object, time);
    if (!tracker) continue;  // null combo: (shape, label) not accepted
    if (predecessor && nero_mover) {
      types::DynamicObject unknown;
      if (
        predecessor->getTrackedObject(time, unknown, true) &&
        std::hypot(unknown.twist.linear.x, unknown.twist.linear.y) > nero_rebirth_.max_pred_speed) {
        const auto u = tracker->getUUID().uuid;
        const auto p = predecessor->neroPubUUID().uuid;
        RCLCPP_INFO(
          logger_,
          "[NERO_MOT_UNKNOWN_MOVER] vehicle track %02x%02x%02x%02x continues moving UNKNOWN %02x%02x%02x%02x at "
          "(%.1f, %.1f), %.1f m/s",
          u[0], u[1], u[2], u[3], p[0], p[1], p[2], p[3], unknown.pose.position.x, unknown.pose.position.y,
          std::hypot(unknown.twist.linear.x, unknown.twist.linear.y));
      }
    }
    if (!predecessor && nero_mover) {
      // [nero trkup] NERO_MOT_UNKNOWN_MOVER: a new UNKNOWN track that continues a young track the gates lost takes over
      // its measured positions
      if (const auto polygon = std::dynamic_pointer_cast<PolygonTracker>(tracker)) {
        if (polygon->neroOn() && polygon->getHighestProbLabel() == classes::Label::UNKNOWN) {
          neroMoverInherit(polygon, new_object, time);
        }
      }
    }
    if (predecessor) {
      predecessor->nero_successor = tracker;
      tracker->nero_is_successor = true;
      tracker->nero_min_count = nero_rebirth_.min_count;
      tracker->nero_vehicle_count = 1;  // the vehicle measurement that started it
      tracker->nero_predecessor = predecessor;
      tracker->nero_predecessor_uuid = predecessor->neroPubUUID();
      tracker->nero_predecessor_confident =
        predecessor->isConfident(adaptive_threshold_cache_, getEgoPose(), std::nullopt);
    }

    if (channel_config.trust_existence_probability) {
      tracker->initializeExistenceProbabilities(
        new_object.channel_index, new_object.existence_probability);
    } else {
      tracker->initializeExistenceProbabilities(
        new_object.channel_index, types::default_existence_probability);
    }

    if (neroStillBoxParams().enabled) neroStillBoxBirth(tracker, new_object, time);
    list_tracker_.push_back(tracker);
  }
}

void TrackerProcessor::neroStillBoxBirth(
  const std::shared_ptr<Tracker> & tracker, const types::DynamicObject & object, const rclcpp::Time & time) const
{
  // NERO_MOT_STILL_BOX (birth branch on upstream): a new vehicle track whose box holds the last outline (measured within
  // t_birth) of a standing UNKNOWN track (published speed <= v_birth) takes that outline as its predecessor's
  const auto & p = neroStillBoxParams();
  const auto label = tracker->getHighestProbLabel();
  if (
    label != classes::Label::CAR && label != classes::Label::TRUCK && label != classes::Label::BUS &&
    label != classes::Label::TRAILER) {
    return;
  }
  const double t = time.seconds();
  const double psi = tf2::getYaw(object.pose.orientation);
  const double L = object.shape.dimensions.x;
  const double W = object.shape.dimensions.y;
  tracker->nero_sb_birth_t = t;
  double best = std::numeric_limits<double>::max();
  for (const auto & other : list_tracker_) {
    if (other->getHighestProbLabel() != classes::Label::UNKNOWN) continue;
    if (other->nero_sb_outline_t < 0.0 || t - other->nero_sb_outline_t > p.t_birth) continue;
    const auto & poly = other->nero_sb_outline;
    double cx = 0.0, cy = 0.0;
    for (const auto & q : poly) {
      cx += q[0];
      cy += q[1];
    }
    cx = cx / poly.size() - object.pose.position.x;
    cy = cy / poly.size() - object.pose.position.y;
    const double u = cx * std::cos(psi) + cy * std::sin(psi);
    const double w = -cx * std::sin(psi) + cy * std::cos(psi);
    if (std::fabs(u) > 0.5 * L + 0.5 || std::fabs(w) > 0.5 * W + 0.5) continue;
    geometry_msgs::msg::Pose pose;
    std::array<double, 36> cov{};
    geometry_msgs::msg::Twist twist;
    if (!other->getMotionState(time, pose, cov, twist, cov)) continue;
    if (std::hypot(twist.linear.x, twist.linear.y) > p.v_birth) continue;
    const double d = std::hypot(u, w);
    if (d < best) {
      best = d;
      tracker->nero_sb_pred_outline = poly;
      tracker->nero_sb_pred_outline_t = other->nero_sb_outline_t;
    }
  }
}

std::shared_ptr<Tracker> TrackerProcessor::createNewTracker(
  const types::DynamicObject & object, const rclcpp::Time & time) const
{
  const classes::Label label = classes::getHighestProbLabel(object.classification);
  const ShapeLabelKey key{types::toShapeType(object.shape.type), label};

  const auto tracker_type_opt = get_map_value_if_exists(creation_config_.shape_tracker_map, key);

  if (tracker_type_opt) {
    switch (tracker_type_opt->get()) {
      case types::TrackerType::MULTIPLE_VEHICLE:
        return std::make_shared<MultipleVehicleTracker>(time, object);
      case types::TrackerType::GENERAL_VEHICLE:
        return std::make_shared<VehicleTracker>(object_model::general_vehicle, time, object);
      case types::TrackerType::PEDESTRIAN_AND_BICYCLE:
        return std::make_shared<PedestrianAndBicycleTracker>(time, object);
      case types::TrackerType::NORMAL_VEHICLE:
        return std::make_shared<VehicleTracker>(object_model::normal_vehicle, time, object);
      case types::TrackerType::PEDESTRIAN:
        return std::make_shared<PedestrianTracker>(time, object);
      case types::TrackerType::BICYCLE:
        return std::make_shared<VehicleTracker>(object_model::bicycle, time, object);
      case types::TrackerType::BIG_VEHICLE:
        return std::make_shared<VehicleTracker>(object_model::big_vehicle, time, object);
      case types::TrackerType::STATIC:
        return std::make_shared<StaticTracker>(time, object, tracker_configs_.static_tracker);
      case types::TrackerType::POLYGON:
        return std::make_shared<PolygonTracker>(time, object, tracker_configs_.polygon_tracker);
      default:
        return std::make_shared<PolygonTracker>(time, object, tracker_configs_.polygon_tracker);
    }
  }

  if (creation_config_.explicit_null_combos.count(key)) {
    return nullptr;  // create: "null" — explicitly not accepted, silently skip
  }

  // implicitly omitted — not listed in tracker_assignment; error periodically
  RCLCPP_ERROR_THROTTLE(
    logger_, *clock_, 1000,
    "Received detection with unspecified tracker_assignment combination: shape=%s label=%s. "
    "Add an explicit entry (or create: \"null\") to suppress this error.",
    types::toString(key.first).c_str(), classes::toString(key.second).c_str());
  return nullptr;
}

void TrackerProcessor::prune(const rclcpp::Time & time)
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  // Minimum spacing between prune cycles; a call at an unchanged or earlier measurement time is
  // skipped.
  constexpr int64_t min_prune_interval_ns = 2 * 1000 * 1000;  // 2 ms
  if (time.nanoseconds() - last_prune_time_.nanoseconds() < min_prune_interval_ns) {
    // [nero trkup] NERO_MOT_REBIRTH: the hand-over runs after every message, as on the car (a frame's
    // detection_by_tracker message shares its stamp and can confirm a successor)
    if (nero_rebirth_.enabled) neroHandOver(time);
    return;
  }

  removeOldTracker(time);
  tracker_overlap_manager_->merge(list_tracker_, time, adaptive_threshold_cache_, getEgoPose());
  if (nero_rebirth_.enabled) {
    neroHandOver(time);  // [nero trkup] NERO_MOT_REBIRTH
  }

  last_prune_time_ = time;
}

void TrackerProcessor::neroStillBox(
  const std::shared_ptr<Tracker> & tracker, const rclcpp::Time & time, types::DynamicObject & object) const
{
  // NERO_MOT_STILL_BOX (boxyaw): nero_still_box.hpp
  const auto & p = neroStillBoxParams();
  const auto label = tracker->getHighestProbLabel();
  if (
    (label != classes::Label::CAR && label != classes::Label::TRUCK && label != classes::Label::BUS &&
     label != classes::Label::TRAILER) ||
    object.shape.type != autoware_perception_msgs::msg::Shape::BOUNDING_BOX) {
    return;
  }
  const double t = time.seconds();
  const double v = std::hypot(object.twist.linear.x, object.twist.linear.y);
  // the car's REBIRTH birth branch: on upstream the standing UNKNOWN track the vehicle track was born on (spawn: its last
  // outline inside the new box) gives its outline for the birth window
  const bool birth = tracker->nero_sb_pred_outline_t >= 0.0 && t - tracker->nero_sb_birth_t <= p.t_birth && v <= p.v_birth;
  if (v > p.v_still && !birth) return;  // a moving vehicle: as today

  const double psi = tf2::getYaw(object.pose.orientation);
  const double L = object.shape.dimensions.x;
  const double W = object.shape.dimensions.y;
  if (L < p.min_length) return;  // a box shorter than min_length: as today
  // the heading difference folded within +-90 deg (the box's sign is kept)
  const auto fold = [psi](const double a) {
    double d = autoware_utils_math::normalize_radian(a - psi);
    if (d > M_PI_2) d -= M_PI;
    if (d < -M_PI_2) d += M_PI;
    return d;
  };
  // the points' heading: the median of the point-fitted boxes of the last t_hold within max_dev of the box (the first
  // of them is fitted around a newborn track's own heading: 11-08 #19 31.7 deg against the road's 25.5), in the birth
  // window together with the standing predecessor's outline (its longest edge along the box)
  std::vector<double> devs;
  for (const auto & b : tracker->nero_sb_boxes) {
    const double d = fold(b[3]);
    if (t - b[0] <= p.t_fresh && std::fabs(d) <= p.max_dev) devs.push_back(d);
  }
  const std::array<double, 6> * box = nullptr;  // the newest of them
  for (const auto & b : tracker->nero_sb_boxes) {
    if (t - b[0] <= p.t_fresh && std::fabs(fold(b[3])) <= p.max_dev) box = &b;
  }
  const std::vector<std::array<double, 2>> * outline = nullptr;
  {
    // a measured outline: this track's own, else (birth) its standing predecessor's
    for (int owner = 0; owner < 2; ++owner) {
      const double poly_t = owner == 0 ? tracker->nero_sb_outline_t : (birth ? tracker->nero_sb_pred_outline_t : -1.0);
      if (poly_t < 0.0 || t - poly_t > p.t_fresh) continue;
      const auto & poly = owner == 0 ? tracker->nero_sb_outline : tracker->nero_sb_pred_outline;
      double best = p.min_edge_ratio * L;
      bool found = false;
      double d_best = 0.0;
      for (size_t i = 0; i < poly.size(); ++i) {
        const auto & a = poly[i];
        const auto & b = poly[(i + 1) % poly.size()];
        const double len = std::hypot(b[0] - a[0], b[1] - a[1]);
        const double d = fold(std::atan2(b[1] - a[1], b[0] - a[0]));
        if (len >= best && std::fabs(d) <= p.max_dev) {
          best = len;
          d_best = d;
          found = true;
        }
      }
      if (found) {
        devs.push_back(d_best);
        outline = &poly;
        break;
      }
    }
  }
  if (devs.empty()) return;
  std::sort(devs.begin(), devs.end());
  const size_t m = devs.size() / 2;
  const double dev = devs.size() % 2 == 1 ? devs[m] : 0.5 * (devs[m - 1] + devs[m]);
  const double phi = psi + dev;
  // the points: only a measured outline (its points as measured). The point-fitted box gives the heading only: its
  // edges away from the robot are detection_by_tracker's reference size, not points (c23 14:27:15.6: a CAR grown 1.5 m
  // toward the path by them)
  std::vector<std::array<double, 2>> pts;
  const char * src = box != nullptr ? (outline != nullptr ? "points boxes + outline" : "points boxes") : "outline";
  if (outline != nullptr) pts = *outline;
  // the points must be this box's object: their centroid inside the box + 0.5 m
  if (!pts.empty()) {
    double cx = 0.0;
    double cy = 0.0;
    for (const auto & q : pts) {
      cx += q[0];
      cy += q[1];
    }
    cx = cx / pts.size() - object.pose.position.x;
    cy = cy / pts.size() - object.pose.position.y;
    if (
      std::fabs(cx * std::cos(psi) + cy * std::sin(psi)) > 0.5 * L + 0.5 ||
      std::fabs(-cx * std::sin(psi) + cy * std::cos(psi)) > 0.5 * W + 0.5) {
      return;
    }
  }
  // the box at the points' heading (centre and size kept) united with the points, in that heading's frame
  const double c = std::cos(phi);
  const double s = std::sin(phi);
  const double x0 = object.pose.position.x;
  const double y0 = object.pose.position.y;
  double u0 = -0.5 * L;
  double u1 = 0.5 * L;
  double v0 = -0.5 * W;
  double v1 = 0.5 * W;
  for (const auto & q : pts) {
    const double dx = q[0] - x0;
    const double dy = q[1] - y0;
    const double u = dx * c + dy * s;
    const double w = -dx * s + dy * c;
    u0 = std::min(u0, u);
    u1 = std::max(u1, u);
    v0 = std::min(v0, w);
    v1 = std::max(v1, w);
  }
  const double new_l = u1 - u0;
  const double new_w = v1 - v0;
  if (new_l - L > p.max_grow || new_w - W > p.max_grow) {
    if (p.debug > 0) {
      const auto & id = object.uuid.uuid;
      RCLCPP_INFO(
        rclcpp::get_logger("multi_object_tracker"),
        "[NERO_MOT_STILL_BOX] %02x%02x%02x%02x t %.2f %s: union grows %.2fx%.2f -> %.2fx%.2f: kept as is", id[0], id[1],
        id[2], id[3], t, src, L, W, new_l, new_w);
    }
    return;
  }
  const double um = 0.5 * (u0 + u1);
  const double vm = 0.5 * (v0 + v1);
  // never toward the robot's line: the box's corners ahead of the robot may not come closer to the line along the robot's
  // heading than today's (c23 14:27:01.7: a parked CAR beside the standing robot would have come 0.58 -> 0.33 m from the
  // line by a heading the points did not confirm); a box whose points say it reaches further is left as today
  if (tracker->nero_sb_has_ego) {
    const double ex = tracker->nero_sb_ego[0];
    const double ey = tracker->nero_sb_ego[1];
    const double ec = std::cos(tracker->nero_sb_ego[2]);
    const double es = std::sin(tracker->nero_sb_ego[2]);
    const auto min_lat = [&](const double bx, const double by, const double byaw, const double bl, const double bw) {
      const double bc = std::cos(byaw);
      const double bs = std::sin(byaw);
      double ahead = std::numeric_limits<double>::max();
      double all = std::numeric_limits<double>::max();
      for (const double a : {-0.5 * bl, 0.5 * bl}) {
        for (const double b : {-0.5 * bw, 0.5 * bw}) {
          const double qx = bx + a * bc - b * bs - ex;
          const double qy = by + a * bs + b * bc - ey;
          const double lat = std::fabs(-qx * es + qy * ec);
          all = std::min(all, lat);
          if (qx * ec + qy * es > 0.0) ahead = std::min(ahead, lat);
        }
      }
      return ahead < std::numeric_limits<double>::max() ? ahead : all;
    };
    const double before = min_lat(x0, y0, psi, L, W);
    const double after = min_lat(x0 + um * c - vm * s, y0 + um * s + vm * c, phi, new_l, new_w);
    if (after < before - 0.02) {
      if (p.debug > 0) {
        const auto & id = object.uuid.uuid;
        RCLCPP_INFO(
          rclcpp::get_logger("multi_object_tracker"),
          "[NERO_MOT_STILL_BOX] %02x%02x%02x%02x t %.2f %s: yaw %.1f -> %.1f would bring it %.2f -> %.2f m from the robot's "
          "line: kept as is",
          id[0], id[1], id[2], id[3], t, src, psi * 180.0 / M_PI, phi * 180.0 / M_PI, before, after);
      }
      return;
    }
  }
  if (p.debug > 0) {
    const auto & id = object.uuid.uuid;
    RCLCPP_INFO(
      rclcpp::get_logger("multi_object_tracker"),
      "[NERO_MOT_STILL_BOX] %02x%02x%02x%02x t %.2f %s%s v %.2f: yaw %.1f -> %.1f, %.2fx%.2f -> %.2fx%.2f, centre +(%.2f, %.2f)",
      id[0], id[1], id[2], id[3], t, src, birth && v > p.v_still ? " (birth)" : "", v, psi * 180.0 / M_PI,
      phi * 180.0 / M_PI, L, W, new_l, new_w, um, vm);
  }
  const double nx = x0 + um * c - vm * s;
  const double ny = y0 + um * s + vm * c;
  {
    // a footprint the box carries (object frame) stays where it was in the map: re-expressed in the new box frame
    const double c0 = std::cos(psi), s0 = std::sin(psi);
    for (auto & q : object.shape.footprint.points) {
      const double mx = x0 + q.x * c0 - q.y * s0 - nx;
      const double my = y0 + q.x * s0 + q.y * c0 - ny;
      q.x = static_cast<float>(mx * c + my * s);
      q.y = static_cast<float>(-mx * s + my * c);
    }
  }
  object.pose.position.x = nx;
  object.pose.position.y = ny;
  // the vehicle models publish a yaw-only orientation (bicycle model)
  object.pose.orientation.x = 0.0;
  object.pose.orientation.y = 0.0;
  object.pose.orientation.z = std::sin(0.5 * phi);
  object.pose.orientation.w = std::cos(0.5 * phi);
  object.shape.dimensions.x = new_l;
  object.shape.dimensions.y = new_w;
}

void TrackerProcessor::removeOldTracker(const rclcpp::Time & time)
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  for (auto itr = list_tracker_.begin(); itr != list_tracker_.end(); ++itr) {
    if ((*itr)->isExpired(time, adaptive_threshold_cache_, getEgoPose())) {
      auto erase_itr = itr;
      --itr;
      list_tracker_.erase(erase_itr);
    }
  }
}

void TrackerProcessor::getTrackedObjects(
  const rclcpp::Time & time, autoware_perception_msgs::msg::TrackedObjects & tracked_objects) const
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  tracked_objects.header.stamp = time;
  types::DynamicObject tracked_object;
  for (const auto & tracker : list_tracker_) {
    if (!tracker->isConfident(adaptive_threshold_cache_, getEgoPose(), std::nullopt)) continue;
    // [nero trkup] NERO_MOT_UNKNOWN_CONFIRM=<n> (trkup-2 2026-11-10; unset = upstream): an UNKNOWN track is published only from its n-th
    // measurement, the car's confident_count_threshold (5). Upstream confirms after ~2, so 1-4-measurement flat ground flickers in the
    // robot's band were published (closed loop on the camera stack, 11-08 trips: obstacle_stop 33 -> 48 / 34 -> 51, every added stop a
    // newborn UNKNOWN 0.01-0.09 m tall living 0.0-0.4 s). Vehicles keep upstream's confirmation (realize windows: = the car).
    {
      static const int nero_unknown_confirm = [] {
        const char * e = std::getenv("NERO_MOT_UNKNOWN_CONFIRM");
        return e != nullptr ? std::atoi(e) : 0;
      }();
      if (
        nero_unknown_confirm > 0 && tracker->getHighestProbLabel() == classes::Label::UNKNOWN &&
        tracker->getTotalMeasurementCount() < nero_unknown_confirm) {
        // ... unless NERO_MOT_UNKNOWN_MOVER already measured it as a mover (>= 4 points on a straight line, both outline ends moving:
        // evidence no ground flicker has; w31 0b8b would otherwise be published moving one message later than the car)
        const auto poly = std::dynamic_pointer_cast<PolygonTracker>(tracker);
        double mvx = 0.0, mvy = 0.0;
        if (!(poly && poly->neroOn() && poly->neroMover(time, mvx, mvy))) continue;
      }
    }
    // [nero trkup] NERO_MOT_REBIRTH: the UNKNOWN track whose object its vehicle successor publishes
    if (nero_rebirth_.enabled && tracker->neroHidden()) continue;
    constexpr bool to_publish = true;
    if (tracker->getTrackedObject(time, tracked_object, to_publish)) {
      if (neroStillBoxParams().enabled) neroStillBox(tracker, time, tracked_object);
      tracked_object.existence_probability = tracker->getTotalExistenceProbability();
      tracked_object.classification = tracker->getClassification();
      if (tracker->neroHasPubUUID()) tracked_object.uuid = tracker->neroPubUUID();  // [nero trkup] REBIRTH
      tracked_objects.objects.push_back(types::toTrackedObjectMsg(tracked_object));
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
    if (tracker->isConfident(adaptive_threshold_cache_, getEgoPose(), std::nullopt)) continue;
    constexpr bool to_publish = false;
    if (tracker->getTrackedObject(time, tracked_object, to_publish)) {
      tentative_objects.objects.push_back(types::toTrackedObjectMsg(tracked_object));
    }
  }
}

void TrackerProcessor::getMergedObjects(
  const rclcpp::Time & time, const geometry_msgs::msg::Transform & tf_base_to_world,
  autoware_perception_msgs::msg::DetectedObjects & merged_objects) const
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  merged_objects.header.stamp = time;
  merged_objects.objects.clear();
  merged_objects.objects.reserve(list_tracker_.size());
  types::DynamicObject tracked_object;
  for (const auto & tracker : list_tracker_) {
    constexpr bool to_publish = false;
    if (tracker->getTrackedObject(time, tracked_object, to_publish)) {
      merged_objects.objects.push_back(types::toDetectedObjectMsg(tracked_object));
    }
  }

  // Transform poses from world frame to ego frame using the inverse of tf_base_to_world
  tf2::Transform tf2_base_to_world;
  tf2::fromMsg(tf_base_to_world, tf2_base_to_world);
  geometry_msgs::msg::TransformStamped ts;
  ts.transform = tf2::toMsg(tf2_base_to_world.inverse());

  for (auto & obj : merged_objects.objects) {
    tf2::doTransform(
      obj.kinematics.pose_with_covariance.pose, obj.kinematics.pose_with_covariance.pose, ts);
  }
}

void TrackerProcessor::setTimeKeeper(
  std::shared_ptr<autoware_utils_debug::TimeKeeper> time_keeper_ptr)
{
  time_keeper_ = std::move(time_keeper_ptr);
  association_manager_->setTimeKeeper(time_keeper_);
}

// ---------------------------------------------------------------------------------------------------------------------
// [nero trkup] NERO_MOT_REBIRTH + NERO_MOT_UNKNOWN_MOVER hand-over, ported from our 0.44 TrackerProcessor
// (/work/ahmed-boxyaw/src/wt/perception/autoware_multi_object_tracker/src/processor/processor.cpp). An UNKNOWN track's
// object is read as it is PUBLISHED (getTrackedObject(..., to_publish = true): what the car's UNKNOWN tracker returns);
// vehicle tracks as upstream reads them internally.
// ---------------------------------------------------------------------------------------------------------------------
std::shared_ptr<Tracker> TrackerProcessor::neroFindPredecessor(
  const types::DynamicObject & measurement, const types::InputChannel & channel,
  const rclcpp::Time & time, types::DynamicObject & seeded_measurement) const
{
  // only a classification-trusted vehicle measurement continues an UNKNOWN track
  const auto label = classes::getHighestProbLabel(measurement.classification);
  if (
    !channel.trust_classification || !neroCarLike(label) ||
    measurement.shape.type != autoware_perception_msgs::msg::Shape::BOUNDING_BOX) {
    return nullptr;
  }
  const double myaw = tf2::getYaw(measurement.pose.orientation);
  const double half_l = 0.5 * measurement.shape.dimensions.x + nero_rebirth_.margin;
  const double half_w = 0.5 * measurement.shape.dimensions.y + nero_rebirth_.margin;
  for (const auto & candidate : list_tracker_) {
    if (!neroCarLike(candidate->getHighestProbLabel())) continue;
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
  const bool mover_on = nero::unknownMoverParams().enabled;
  const bool cross_on = nero::unknownMoverParams().cross;
  std::shared_ptr<Tracker> best;
  types::DynamicObject best_object;
  for (const auto & candidate : list_tracker_) {
    if (
      !candidate->nero_successor.expired() ||
      candidate->getHighestProbLabel() != classes::Label::UNKNOWN) {
      continue;
    }
    types::DynamicObject object;
    if (!candidate->getTrackedObject(time, object, true)) continue;
    // the moving test reads the UNKNOWN's filter speed (what the car's test read: its UNKNOWN published its filter);
    // with NERO_MOT_UNKNOWN_MOVER an UNKNOWN that is no mover is published at 0 m/s, which would make every real
    // mover the rule does not catch look standing here (51377 j03_s5: the CAR seeded at 0 +- 0.5 m/s)
    types::DynamicObject filter_object;
    const double moving_speed =
      candidate->getTrackedObject(time, filter_object)
        ? std::hypot(filter_object.twist.linear.x, filter_object.twist.linear.y)
        : std::hypot(object.twist.linear.x, object.twist.linear.y);
    if (
      moving_speed > nero_rebirth_.max_pred_speed &&
      !(mover_on && neroMoverAlong(candidate, time, myaw)) &&
      !(cross_on && neroMoverMadeAlong(candidate, object, time, myaw))) {
      // a moving UNKNOWN keeps the stock handling (unless a mover along the box, or one the mover rule made fast along
      // it)
      continue;
    }
    // the UNKNOWN track's centre inside the vehicle box (+ margin): the same object
    if (!neroCovers(measurement, object)) continue;
    if (!best || candidate->getTotalMeasurementCount() > best->getTotalMeasurementCount()) {
      best = candidate;
      best_object = object;
    }
  }
  if (!best) return nullptr;
  // the vehicle track's motion starts from the UNKNOWN track's (what is published for the object), not from the centre
  // motion of its first boxes
  using autoware_utils_geometry::xyzrpy_covariance_index::XYZRPY_COV_IDX;
  const double uyaw = tf2::getYaw(best_object.pose.orientation);
  double vx = std::cos(uyaw) * best_object.twist.linear.x - std::sin(uyaw) * best_object.twist.linear.y;
  double vy = std::sin(uyaw) * best_object.twist.linear.x + std::cos(uyaw) * best_object.twist.linear.y;
  // a mover hands over the velocity its measured positions show (sigma 2 m/s: the vehicle track's own boxes take over
  // quickly)
  bool mover = false;
  if (mover_on && std::hypot(vx, vy) > nero_rebirth_.max_pred_speed) {
    const auto polygon = std::dynamic_pointer_cast<PolygonTracker>(best);
    if (polygon) {
      // crossmover c3: a crossing mover's line counts from the first of its consecutive start tests
      if (!cross_on || neroMoverAlong(best, time, myaw) || polygon->neroCrossHeld(time) >= 0.15) {
        polygon->neroMoverLine(vx, vy);
        mover = true;
      } else {
        // a crossing mover too young for the hand-over, or one whose line just ended: the vehicle track starts exactly
        // as the stock rule starts it (from its own box); only the object's uuid and its publication until that track
        // is confident come from the UNKNOWN
        seeded_measurement = measurement;
        return best;
      }
    }
  }
  seeded_measurement = measurement;
  // a mover's vehicle box heads the way it moves (the box's sign is not known)
  double syaw = myaw;
  if (mover && vx * std::cos(myaw) + vy * std::sin(myaw) < 0.0) {
    syaw = autoware_utils_math::normalize_radian(myaw + M_PI);
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, syaw);
    seeded_measurement.pose.orientation = tf2::toMsg(q);
    for (auto & pt : seeded_measurement.shape.footprint.points) {
      pt.x = -pt.x;
      pt.y = -pt.y;
    }
  }
  seeded_measurement.kinematics.has_twist = true;
  seeded_measurement.twist.linear.x = vx * std::cos(syaw) + vy * std::sin(syaw);
  seeded_measurement.twist.linear.y = 0.0;
  seeded_measurement.kinematics.has_twist_covariance = true;
  seeded_measurement.twist_covariance[XYZRPY_COV_IDX::X_X] = std::max(
    mover ? 4.0 : 0.25, best_object.twist_covariance[XYZRPY_COV_IDX::X_X] +
                          best_object.twist_covariance[XYZRPY_COV_IDX::Y_Y]);
  // upstream's VehicleTracker reads the lateral (slip) velocity variance from Y_Y when a twist covariance is given:
  // keep its own default there
  seeded_measurement.twist_covariance[XYZRPY_COV_IDX::Y_Y] =
    object_model::general_vehicle.bicycle_state.init_slip_angle_cov;
  return best;
}

bool TrackerProcessor::neroMoverMadeFast(
  const std::shared_ptr<Tracker> & tracker, const types::DynamicObject & object, const rclcpp::Time & time) const
{
  if (std::hypot(object.twist.linear.x, object.twist.linear.y) <= nero_rebirth_.max_pred_speed) return false;
  const auto polygon = std::dynamic_pointer_cast<PolygonTracker>(tracker);
  return polygon && polygon->neroMoverRecent(time);
}

bool TrackerProcessor::neroMoverMadeAlong(
  const std::shared_ptr<Tracker> & tracker, const types::DynamicObject & object, const rclcpp::Time & time,
  const double box_yaw) const
{
  if (!neroMoverMadeFast(tracker, object, time)) return false;
  // the published motion (map frame) within 30 deg of the box's axis, either way (as neroMoverAlong)
  const double uyaw = tf2::getYaw(object.pose.orientation);
  const double vx = std::cos(uyaw) * object.twist.linear.x - std::sin(uyaw) * object.twist.linear.y;
  const double vy = std::sin(uyaw) * object.twist.linear.x + std::cos(uyaw) * object.twist.linear.y;
  constexpr double max_angle = 30.0 * M_PI / 180.0;
  const double angle = std::abs(autoware_utils_math::normalize_radian(std::atan2(vy, vx) - box_yaw));
  return angle <= max_angle || angle >= M_PI - max_angle;
}

bool TrackerProcessor::neroMoverLink(
  const types::DynamicObject & measurement, const types::InputChannel & channel, const rclcpp::Time & time)
{
  if (classes::getHighestProbLabel(measurement.classification) != classes::Label::UNKNOWN) {
    return false;
  }
  std::shared_ptr<Tracker> best;
  for (const auto & candidate : list_tracker_) {
    if (candidate->getHighestProbLabel() != classes::Label::UNKNOWN) continue;
    // a track that took a measurement this cycle has its object
    if (candidate->getElapsedTimeFromLastUpdate(time) < 1e-3) continue;
    const auto polygon = std::dynamic_pointer_cast<PolygonTracker>(candidate);
    if (!polygon || !polygon->neroMoverContinues(time, measurement)) continue;
    if (!best || candidate->getTotalMeasurementCount() > best->getTotalMeasurementCount()) {
      best = candidate;
    }
  }
  if (!best) return false;
  best->setEgoPose(ego_pose_ ? std::make_optional(ego_pose_->pose.position) : std::nullopt);
  best->updateWithMeasurement(measurement, time, channel, false);
  return true;
}

void TrackerProcessor::neroMoverInherit(
  const std::shared_ptr<PolygonTracker> & newborn, const types::DynamicObject & measurement,
  const rclcpp::Time & time)
{
  std::shared_ptr<PolygonTracker> best;
  int best_n = 0;
  for (const auto & candidate : list_tracker_) {
    if (candidate->getHighestProbLabel() != classes::Label::UNKNOWN) continue;
    if (candidate->getElapsedTimeFromLastUpdate(time) < 1e-3) continue;  // measured this cycle
    const auto polygon = std::dynamic_pointer_cast<PolygonTracker>(candidate);
    if (!polygon) continue;
    const int n = polygon->neroInheritable(time, *newborn);
    if (n > best_n) {
      best = polygon;
      best_n = n;
    }
  }
  if (best) newborn->neroInheritFrom(*best, time, measurement);
}

bool TrackerProcessor::neroMoverAlong(
  const std::shared_ptr<Tracker> & tracker, const rclcpp::Time & time, const double box_yaw) const
{
  const auto polygon = std::dynamic_pointer_cast<PolygonTracker>(tracker);
  double vx = 0.0;
  double vy = 0.0;
  if (!polygon || !polygon->neroMover(time, vx, vy)) return false;
  // only a mover that has held its line >= 0.15 s (two measurement times after its start)
  constexpr double min_age = 0.15;
  if (polygon->neroMoverAge(time) < min_age) return false;
  // within 30 deg of the box's axis, either way
  constexpr double max_angle = 30.0 * M_PI / 180.0;
  const double angle = std::abs(autoware_utils_math::normalize_radian(std::atan2(vy, vx) - box_yaw));
  return angle <= max_angle || angle >= M_PI - max_angle;
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
  // the stock prune's rule for an UNKNOWN track overlapping a vehicle track (upstream's merger: a confident,
  // non-polygon vehicle absorbs a redundant UNKNOWN)
  types::DynamicObject vehicle;
  types::DynamicObject unknown;
  if (
    !vehicle_tracker->getTrackedObject(time, vehicle) ||
    !unknown_tracker->getTrackedObject(time, unknown)) {
    return false;
  }
  if (vehicle.shape.type == autoware_perception_msgs::msg::Shape::POLYGON) return false;
  if (!vehicle_tracker->isConfident(adaptive_threshold_cache_, getEgoPose(), time)) return false;
  return detail::isRedundant(
    vehicle, unknown, vehicle_tracker->getHighestProbLabel(), unknown_tracker->getHighestProbLabel(),
    vehicle_tracker->getKnownObjectProbability(), unknown_tracker->getKnownObjectProbability(),
    nero_overlap_config_);
}

void TrackerProcessor::neroHandOver(const rclcpp::Time & time)
{
  for (const auto & tracker : list_tracker_) {
    // an UNKNOWN track whose vehicle successor died publishes the object under its uuid again
    if (
      tracker->nero_handed_over_to && tracker->nero_successor.expired() &&
      tracker->neroPubUUID() != tracker->nero_own_uuid) {
      tracker->neroSetPubUUID(tracker->nero_own_uuid);
    }
    if (!tracker->nero_is_successor) continue;
    const auto predecessor = tracker->nero_predecessor.lock();
    if (predecessor) {
      types::DynamicObject vehicle;
      types::DynamicObject unknown;
      const bool unknown_ok =
        tracker->getTrackedObject(time, vehicle) && predecessor->getTrackedObject(time, unknown, true);
      bool covered = unknown_ok && neroCovers(vehicle, unknown);
      // crossmover c3: the link also holds while the cluster MEASURED last is inside the vehicle box
      if (unknown_ok && !covered && nero::unknownMoverParams().cross) {
        const auto polygon = std::dynamic_pointer_cast<PolygonTracker>(predecessor);
        double mx = 0.0;
        double my = 0.0;
        if (polygon && polygon->neroMoverRecent(time) && polygon->neroLastMeasured(mx, my)) {
          unknown.pose.position.x = mx;
          unknown.pose.position.y = my;
          covered = neroCovers(vehicle, unknown);
        }
      }
      if (unknown_ok && !covered) {
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
    if (predecessor && !tracker->nero_predecessor_confident) {
      tracker->nero_predecessor_confident =
        predecessor->isConfident(adaptive_threshold_cache_, getEgoPose(), std::nullopt);
    }
    if (!tracker->isConfident(adaptive_threshold_cache_, getEgoPose(), std::nullopt)) continue;
    // the vehicle track publishes the object under its uuid; the UNKNOWN track goes on under the vehicle track's
    // (never published) uuid, hidden where the stock prune would have absorbed it (neroHidden)
    const auto vehicle_uuid = tracker->neroPubUUID();
    tracker->neroSetPubUUID(tracker->nero_predecessor_uuid);
    tracker->nero_handed_over = true;
    if (predecessor) {
      predecessor->nero_own_uuid = tracker->nero_predecessor_uuid;
      predecessor->neroSetPubUUID(vehicle_uuid);
      predecessor->nero_handed_over_to = true;
      predecessor->nero_removed_at_count =
        neroStockRemoves(tracker, predecessor, time) ? predecessor->getTotalMeasurementCount() : -1;
    }
  }
}

}  // namespace autoware::multi_object_tracker
