// Copyright 2021 Tier IV, Inc.
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

#define EIGEN_MPL2_ONLY

#include "detection_by_tracker_node.hpp"

#include "autoware/object_recognition_utils/object_recognition_utils.hpp"
#include "autoware_utils/geometry/geometry.hpp"
#include "autoware_utils/math/unit_conversion.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using Label = autoware_perception_msgs::msg::ObjectClassification;
namespace
{
void setClusterInObjectWithFeature(
  const std_msgs::msg::Header & header, const pcl::PointCloud<pcl::PointXYZ> & cluster,
  tier4_perception_msgs::msg::DetectedObjectWithFeature & feature_object)
{
  sensor_msgs::msg::PointCloud2 ros_pointcloud;
  pcl::toROSMsg(cluster, ros_pointcloud);
  ros_pointcloud.header = header;
  feature_object.feature.cluster = ros_pointcloud;
}
autoware_perception_msgs::msg::Shape extendShape(
  const autoware_perception_msgs::msg::Shape & shape, const float scale)
{
  autoware_perception_msgs::msg::Shape output = shape;
  output.dimensions.x *= scale;
  output.dimensions.y *= scale;
  output.dimensions.z *= scale;
  for (auto & point : output.footprint.points) {
    point.x *= scale;
    point.y *= scale;
    point.z *= scale;
  }
  return output;
}

boost::optional<autoware::shape_estimation::ReferenceYawInfo> getReferenceYawInfo(
  const uint8_t label, const float yaw)
{
  const bool is_vehicle =
    Label::CAR == label || Label::TRUCK == label || Label::BUS == label || Label::TRAILER == label;
  if (is_vehicle) {
    return autoware::shape_estimation::ReferenceYawInfo{yaw, autoware_utils::deg2rad(30)};
  } else {
    return boost::none;
  }
}

boost::optional<autoware::shape_estimation::ReferenceShapeSizeInfo> getReferenceShapeSizeInfo(
  const uint8_t label, const autoware_perception_msgs::msg::Shape & shape)
{
  const bool is_vehicle =
    Label::CAR == label || Label::TRUCK == label || Label::BUS == label || Label::TRAILER == label;
  if (is_vehicle) {
    return autoware::shape_estimation::ReferenceShapeSizeInfo{
      shape, autoware::shape_estimation::ReferenceShapeSizeInfo::Mode::Min};
  } else {
    return boost::none;
  }
}
}  // namespace

namespace autoware::detection_by_tracker
{

DetectionByTracker::DetectionByTracker(const rclcpp::NodeOptions & node_options)
: rclcpp::Node("detection_by_tracker", node_options),
  tf_buffer_(this->get_clock()),
  tf_listener_(tf_buffer_)
{
  // Create publishers and subscribers
  trackers_sub_ = create_subscription<autoware_perception_msgs::msg::TrackedObjects>(
    "~/input/tracked_objects", rclcpp::QoS{1},
    std::bind(&TrackerHandler::onTrackedObjects, &tracker_handler_, std::placeholders::_1));
  initial_objects_sub_ =
    create_subscription<tier4_perception_msgs::msg::DetectedObjectsWithFeature>(
      "~/input/initial_objects", rclcpp::QoS{1},
      std::bind(&DetectionByTracker::onObjects, this, std::placeholders::_1));
  objects_pub_ =
    create_publisher<autoware_perception_msgs::msg::DetectedObjects>("~/output", rclcpp::QoS{1});

  // Set parameters
  tracker_ignore_.UNKNOWN = declare_parameter<bool>("tracker_ignore_label.UNKNOWN");
  tracker_ignore_.CAR = declare_parameter<bool>("tracker_ignore_label.CAR");
  tracker_ignore_.TRUCK = declare_parameter<bool>("tracker_ignore_label.TRUCK");
  tracker_ignore_.BUS = declare_parameter<bool>("tracker_ignore_label.BUS");
  tracker_ignore_.TRAILER = declare_parameter<bool>("tracker_ignore_label.TRAILER");
  tracker_ignore_.MOTORCYCLE = declare_parameter<bool>("tracker_ignore_label.MOTORCYCLE");
  tracker_ignore_.BICYCLE = declare_parameter<bool>("tracker_ignore_label.BICYCLE");
  tracker_ignore_.PEDESTRIAN = declare_parameter<bool>("tracker_ignore_label.PEDESTRIAN");

  // set maximum search setting for merger/divider
  setMaxSearchRange();

  shape_estimator_ = std::make_shared<autoware::shape_estimation::ShapeEstimator>(true, true);
  cluster_ = std::make_shared<autoware::euclidean_cluster::VoxelGridBasedEuclideanCluster>(
    false, 10, 10000, 0.7, 0.3, 0);
  debugger_ = std::make_shared<Debugger>(this);
  published_time_publisher_ = std::make_unique<autoware_utils::PublishedTimePublisher>(this);

  if (const char * env = std::getenv("NERO_DBT_VISIBLE_END"); env != nullptr && env[0] != '\0') {
    double v[2] = {0.3, 0.6};
    const int n = std::sscanf(env, "%lf,%lf", &v[0], &v[1]);
    if (n >= 1 && v[0] >= 0.0) {
      nero_ve_tol_ = v[0];
      nero_ve_sensor_x_ = v[1];
      RCLCPP_INFO(
        get_logger(),
        "[NERO_DBT_VISIBLE_END] on: a merged vehicle box's end facing the lidar stands at most %.2f m in front of the "
        "vehicle's visible end (first hits per bearing from x %.2f m in base_link); points of tracked pedestrians / "
        "bicycles / motorcycles are not a vehicle's",
        nero_ve_tol_, nero_ve_sensor_x_);
    } else {
      RCLCPP_ERROR(get_logger(), "[NERO_DBT_VISIBLE_END] malformed '%s': stock placement", env);
    }
  }
}

// NERO_DBT_VISIBLE_END: remove the points inside (box + 0.2 m) of any tracked pedestrian, bicycle or motorcycle
void DetectionByTracker::neroDropOthersPoints(
  const autoware_perception_msgs::msg::DetectedObjects & tracked_objects, pcl::PointCloud<pcl::PointXYZ> & cluster)
{
  using Label = autoware_perception_msgs::msg::ObjectClassification;
  constexpr double margin = 0.2;
  for (const auto & other : tracked_objects.objects) {
    const auto l = other.classification.front().label;
    if (l != Label::PEDESTRIAN && l != Label::BICYCLE && l != Label::MOTORCYCLE) continue;
    const auto & p = other.kinematics.pose_with_covariance.pose;
    const double yaw = tf2::getYaw(p.orientation);
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    double hl = 0.5 * other.shape.dimensions.x + margin;
    double hw = 0.5 * other.shape.dimensions.y + margin;
    if (other.shape.type == autoware_perception_msgs::msg::Shape::CYLINDER) hw = hl;
    const size_t before = cluster.points.size();
    cluster.points.erase(
      std::remove_if(
        cluster.points.begin(), cluster.points.end(),
        [&](const pcl::PointXYZ & q) {
          const double dx = q.x - p.position.x;
          const double dy = q.y - p.position.y;
          return std::abs(c * dx + s * dy) <= hl && std::abs(-s * dx + c * dy) <= hw;
        }),
      cluster.points.end());
    nero_ve_others_points_ += before - cluster.points.size();
  }
  cluster.width = static_cast<uint32_t>(cluster.points.size());
  cluster.height = 1;
}

namespace
{
constexpr int kNeroBins = 1440;  // 0.25 deg bearing bins
constexpr double kNeroBinRad = 2.0 * M_PI / kNeroBins;
inline int neroBin(double a)
{
  int b = static_cast<int>(std::floor((a + M_PI) / kNeroBinRad));
  return b < 0 ? 0 : (b >= kNeroBins ? kNeroBins - 1 : b);
}
}  // namespace

// NERO_DBT_VISIBLE_END: the nearest obstacle point per bearing (from the sensor) over every cluster of this frame
void DetectionByTracker::neroBuildFirstHits(const tier4_perception_msgs::msg::DetectedObjectsWithFeature & clusters)
{
  nero_fs_r1_.assign(kNeroBins, std::numeric_limits<float>::infinity());
  for (const auto & fo : clusters.feature_objects) {
    const auto & pc = fo.feature.cluster;
    if (pc.width * pc.height == 0) continue;
    sensor_msgs::PointCloud2ConstIterator<float> ix(pc, "x"), iy(pc, "y");
    for (; ix != ix.end(); ++ix, ++iy) {
      const double dx = *ix - nero_ve_sensor_x_;
      const double dy = *iy;
      const int b = neroBin(std::atan2(dy, dx));
      const float r = static_cast<float>(std::hypot(dx, dy));
      if (r < nero_fs_r1_[b]) nero_fs_r1_[b] = r;
    }
  }
}

// how far the sampled end edge (box frame offsets, box centre (cx, cy), axes (c, s)) stands in front of the surface
// the lidar hits right behind it: per sample the nearest hit within +-tol across its ray minus its range, counted only
// when that hit lies 0..3 m behind the sample (the vehicle's own visible end; a far hit = the sample is beside the
// object, not in front of it). The largest over the samples (<= 0: none).
double DetectionByTracker::neroSeenFreeDepth(
  double cx, double cy, double c, double s, const std::vector<std::pair<double, double>> & outline) const
{
  constexpr double kLocal = 3.0;
  double worst = -std::numeric_limits<double>::infinity();
  for (const auto & [u, w] : outline) {
    const double dx = cx + c * u - s * w - nero_ve_sensor_x_;
    const double dy = cy + s * u + c * w;
    const double r = std::hypot(dx, dy);
    if (r < 1.0) continue;
    const int b = neroBin(std::atan2(dy, dx));
    const int half = static_cast<int>(std::ceil(std::atan2(nero_ve_tol_, r) / kNeroBinRad));
    float r1 = std::numeric_limits<float>::infinity();
    for (int k = -half; k <= half; ++k) r1 = std::min(r1, nero_fs_r1_[(b + k + kNeroBins) % kNeroBins]);
    const double depth = static_cast<double>(r1) - r;
    if (depth > kLocal) continue;  // no hit, or only far behind: not in front of this vehicle's own end
    worst = std::max(worst, depth);
  }
  return worst;
}

// NERO_DBT_VISIBLE_END: the end of a vehicle box that faces the lidar may not stand more than tol in front of the
// vehicle's visible end (the surface the lidar hits on the same bearings, within 3 m behind it): the box (size and yaw
// kept) slides back along its length by the least amount (<= 1.5 m) that ends this. Boxes seen from the side (no end
// faces the lidar), their sides and far ends are never changed; no such slide within 1.5 m: stock placement.
// Returns true when the box moved.
bool DetectionByTracker::neroFitSeenSpace(autoware_perception_msgs::msg::DetectedObject & object)
{
  if (nero_fs_r1_.empty()) return false;
  auto & pose = object.kinematics.pose_with_covariance.pose;
  const double yaw = tf2::getYaw(pose.orientation);
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  const double hl = 0.5 * object.shape.dimensions.x;
  const double hw = 0.5 * object.shape.dimensions.y;
  const double x0 = pose.position.x;
  const double y0 = pose.position.y;
  const double su = c * (nero_ve_sensor_x_ - x0) + s * (0.0 - y0);
  if (std::abs(su) <= hl) return false;  // no end faces the lidar
  const double n = su > 0.0 ? 1.0 : -1.0;  // the end facing the lidar
  std::vector<std::pair<double, double>> edge;
  const double w_in = std::max(0.0, hw - 0.2);  // the end edge without its corners
  const int nw = std::max(2, static_cast<int>(std::ceil(2.0 * w_in / 0.2)));
  edge.reserve(nw + 1);
  for (int i = 0; i <= nw; ++i) edge.emplace_back(n * hl, -w_in + 2.0 * w_in * i / nw);
  ++nero_ve_boxes_;
  if (neroSeenFreeDepth(x0, y0, c, s, edge) <= nero_ve_tol_) return false;
  for (int i = 1; i <= 15; ++i) {
    const double d = -n * 0.1 * i;  // back along the length, away from the lidar
    if (neroSeenFreeDepth(x0 + c * d, y0 + s * d, c, s, edge) <= nero_ve_tol_) {
      pose.position.x += c * d;
      pose.position.y += s * d;
      ++nero_ve_moved_;
      nero_ve_moved_sum_ += std::abs(d);
      nero_ve_moved_max_ = std::max(nero_ve_moved_max_, std::abs(d));
      return true;
    }
  }
  ++nero_ve_unresolved_;
  return false;
}

void DetectionByTracker::setMaxSearchRange()
{
  using Label = autoware_perception_msgs::msg::ObjectClassification;
  // set max search distance for merger
  max_search_distance_for_merger_[Label::UNKNOWN] = 5.0;
  max_search_distance_for_merger_[Label::CAR] = 5.0;
  max_search_distance_for_merger_[Label::TRUCK] = 8.0;
  max_search_distance_for_merger_[Label::BUS] = 8.0;
  max_search_distance_for_merger_[Label::TRAILER] = 10.0;
  max_search_distance_for_merger_[Label::MOTORCYCLE] = 2.0;
  max_search_distance_for_merger_[Label::BICYCLE] = 1.0;
  max_search_distance_for_merger_[Label::PEDESTRIAN] = 1.0;

  // set max search distance for divider
  max_search_distance_for_divider_[Label::UNKNOWN] = 6.0;
  max_search_distance_for_divider_[Label::CAR] = 6.0;
  max_search_distance_for_divider_[Label::TRUCK] = 9.0;
  max_search_distance_for_divider_[Label::BUS] = 9.0;
  max_search_distance_for_divider_[Label::TRAILER] = 11.0;
  max_search_distance_for_divider_[Label::MOTORCYCLE] = 3.0;
  max_search_distance_for_divider_[Label::BICYCLE] = 2.0;
  max_search_distance_for_divider_[Label::PEDESTRIAN] = 2.0;
}

void DetectionByTracker::onObjects(
  const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr input_msg)
{
  debugger_->startMeasureProcessingTime();
  autoware_perception_msgs::msg::DetectedObjects detected_objects;
  detected_objects.header = input_msg->header;

  // get objects from tracking module
  autoware_perception_msgs::msg::DetectedObjects tracked_objects;
  {
    autoware_perception_msgs::msg::TrackedObjects objects, transformed_objects;
    const bool available_trackers =
      tracker_handler_.estimateTrackedObjects(input_msg->header.stamp, objects);
    if (
      !available_trackers ||
      !autoware::object_recognition_utils::transformObjects(
        objects, input_msg->header.frame_id, tf_buffer_, transformed_objects)) {
      objects_pub_->publish(detected_objects);
      published_time_publisher_->publish_if_subscribed(objects_pub_, detected_objects.header.stamp);
      return;
    }
    // to simplify post processes, convert tracked_objects to DetectedObjects message.
    tracked_objects = autoware::object_recognition_utils::toDetectedObjects(transformed_objects);
  }
  debugger_->publishInitialObjects(*input_msg);
  debugger_->publishTrackedObjects(tracked_objects);

  if (nero_ve_tol_ >= 0.0) neroBuildFirstHits(*input_msg);

  // merge over segmented objects
  tier4_perception_msgs::msg::DetectedObjectsWithFeature merged_objects;
  autoware_perception_msgs::msg::DetectedObjects no_found_tracked_objects;
  mergeOverSegmentedObjects(tracked_objects, *input_msg, no_found_tracked_objects, merged_objects);
  debugger_->publishMergedObjects(merged_objects);

  // divide under segmented objects
  tier4_perception_msgs::msg::DetectedObjectsWithFeature divided_objects;
  autoware_perception_msgs::msg::DetectedObjects temp_no_found_tracked_objects;
  divideUnderSegmentedObjects(
    no_found_tracked_objects, *input_msg, temp_no_found_tracked_objects, divided_objects);
  debugger_->publishDividedObjects(divided_objects);

  // merge under/over segmented objects to build output objects
  for (const auto & merged_object : merged_objects.feature_objects) {
    detected_objects.objects.push_back(merged_object.object);
  }
  for (const auto & divided_object : divided_objects.feature_objects) {
    detected_objects.objects.push_back(divided_object.object);
  }

  objects_pub_->publish(detected_objects);
  published_time_publisher_->publish_if_subscribed(objects_pub_, detected_objects.header.stamp);
  debugger_->publishProcessingTime();
  if (nero_ve_tol_ >= 0.0) {
    const rclcpp::Time now(input_msg->header.stamp, RCL_ROS_TIME);
    if (nero_ve_last_log_.nanoseconds() == 0) nero_ve_last_log_ = now;
    if ((now - nero_ve_last_log_).seconds() >= 10.0) {
      RCLCPP_INFO(
        get_logger(),
        "[NERO_DBT_VISIBLE_END] merged vehicle boxes since start %lu, slid back to the visible end %lu (mean %.2f "
        "m, max %.2f m), not resolvable (stock) %lu, points of tracked pedestrians / bicycles left out %lu",
        static_cast<unsigned long>(nero_ve_boxes_), static_cast<unsigned long>(nero_ve_moved_),
        nero_ve_moved_ > 0 ? nero_ve_moved_sum_ / nero_ve_moved_ : 0.0, nero_ve_moved_max_,
        static_cast<unsigned long>(nero_ve_unresolved_), static_cast<unsigned long>(nero_ve_others_points_));
      nero_ve_last_log_ = now;
    }
  }
}

void DetectionByTracker::divideUnderSegmentedObjects(
  const autoware_perception_msgs::msg::DetectedObjects & tracked_objects,
  const tier4_perception_msgs::msg::DetectedObjectsWithFeature & in_cluster_objects,
  autoware_perception_msgs::msg::DetectedObjects & out_no_found_tracked_objects,
  tier4_perception_msgs::msg::DetectedObjectsWithFeature & out_objects)
{
  constexpr float recall_min_threshold = 0.4;
  constexpr float precision_max_threshold = 0.5;
  constexpr float min_score_threshold = 0.4;

  out_objects.header = in_cluster_objects.header;
  out_no_found_tracked_objects.header = tracked_objects.header;

  for (const auto & tracked_object : tracked_objects.objects) {
    const auto & label = tracked_object.classification.front().label;
    if (tracker_ignore_.isIgnore(label)) continue;

    // change search range according to label type
    const float max_search_range = max_search_distance_for_divider_[label];

    std::optional<tier4_perception_msgs::msg::DetectedObjectWithFeature>
      highest_score_divided_object = std::nullopt;
    float highest_score = 0.0;

    for (const auto & initial_object : in_cluster_objects.feature_objects) {
      // search near object
      const float distance = autoware_utils::calc_distance2d(
        tracked_object.kinematics.pose_with_covariance.pose,
        initial_object.object.kinematics.pose_with_covariance.pose);
      if (max_search_range < distance) {
        continue;
      }
      // detect under segmented cluster
      const float recall =
        autoware::object_recognition_utils::get2dRecall(initial_object.object, tracked_object);
      const float precision =
        autoware::object_recognition_utils::get2dPrecision(initial_object.object, tracked_object);
      const bool is_under_segmented =
        (recall_min_threshold < recall && precision < precision_max_threshold);
      if (!is_under_segmented) {
        continue;
      }
      // optimize clustering
      tier4_perception_msgs::msg::DetectedObjectWithFeature divided_object;
      float score = optimizeUnderSegmentedObject(
        tracked_object, initial_object.feature.cluster, divided_object);
      if (score < min_score_threshold) {
        continue;
      }

      if (highest_score < score) {
        highest_score = score;
        highest_score_divided_object = divided_object;
      }
    }
    if (highest_score_divided_object) {  // found
      out_objects.feature_objects.push_back(highest_score_divided_object.value());
    } else {  // not found
      out_no_found_tracked_objects.objects.push_back(tracked_object);
    }
  }
}

float DetectionByTracker::optimizeUnderSegmentedObject(
  const autoware_perception_msgs::msg::DetectedObject & target_object,
  const sensor_msgs::msg::PointCloud2 & under_segmented_cluster,
  tier4_perception_msgs::msg::DetectedObjectWithFeature & output)
{
  constexpr float iter_rate = 0.8;
  constexpr int iter_max_count = 5;
  constexpr float initial_cluster_range = 0.7;
  float cluster_range = initial_cluster_range;
  constexpr float initial_voxel_size = initial_cluster_range / 2.0f;
  float voxel_size = initial_voxel_size;

  const auto & label = target_object.classification.front().label;

  // initialize clustering parameters
  autoware::euclidean_cluster::VoxelGridBasedEuclideanCluster cluster(
    false, 4, 10000, initial_cluster_range, initial_voxel_size, 0);

  // convert to pcl
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cluster(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(under_segmented_cluster, *pcl_cluster);

  // iterate to find best fit divided object
  float highest_iou = 0.0;
  tier4_perception_msgs::msg::DetectedObjectWithFeature highest_iou_object;
  boost::optional<geometry_msgs::msg::Pose> ref_pose = boost::none;
  for (int iter_count = 0; iter_count < iter_max_count;
       ++iter_count, cluster_range *= iter_rate, voxel_size *= iter_rate) {
    // divide under segmented cluster
    std::vector<pcl::PointCloud<pcl::PointXYZ>> divided_clusters;
    cluster.setTolerance(cluster_range);
    cluster.setVoxelLeafSize(voxel_size);
    cluster.cluster(pcl_cluster, divided_clusters);

    // find highest iou object in divided clusters
    float highest_iou_in_current_iter = 0.0f;
    tier4_perception_msgs::msg::DetectedObjectWithFeature highest_iou_object_in_current_iter;
    highest_iou_object_in_current_iter.object.classification = target_object.classification;
    for (const auto & divided_cluster : divided_clusters) {
      bool is_shape_estimated = shape_estimator_->estimateShapeAndPose(
        label, divided_cluster,
        getReferenceYawInfo(
          label, tf2::getYaw(target_object.kinematics.pose_with_covariance.pose.orientation)),
        getReferenceShapeSizeInfo(label, target_object.shape), ref_pose,
        highest_iou_object_in_current_iter.object.shape,
        highest_iou_object_in_current_iter.object.kinematics.pose_with_covariance.pose);
      if (!is_shape_estimated) {
        continue;
      }
      const float iou = autoware::object_recognition_utils::get2dIoU(
        highest_iou_object_in_current_iter.object, target_object);
      if (highest_iou_in_current_iter < iou) {
        highest_iou_in_current_iter = iou;
        setClusterInObjectWithFeature(
          under_segmented_cluster.header, divided_cluster, highest_iou_object_in_current_iter);
      }
    }

    // finish iteration when current score is under previous score
    if (highest_iou_in_current_iter < highest_iou) {
      break;
    }

    // copy for next iteration
    highest_iou = highest_iou_in_current_iter;
    highest_iou_object = highest_iou_object_in_current_iter;
  }

  // build output
  highest_iou_object.object.classification = target_object.classification;
  highest_iou_object.object.existence_probability =
    autoware::object_recognition_utils::get2dIoU(target_object, highest_iou_object.object);

  output = highest_iou_object;
  return highest_iou;
}

void DetectionByTracker::mergeOverSegmentedObjects(
  const autoware_perception_msgs::msg::DetectedObjects & tracked_objects,
  const tier4_perception_msgs::msg::DetectedObjectsWithFeature & in_cluster_objects,
  autoware_perception_msgs::msg::DetectedObjects & out_no_found_tracked_objects,
  tier4_perception_msgs::msg::DetectedObjectsWithFeature & out_objects)
{
  constexpr float precision_threshold = 0.5;
  out_objects.header = in_cluster_objects.header;
  out_no_found_tracked_objects.header = tracked_objects.header;

  for (const auto & tracked_object : tracked_objects.objects) {
    const auto & label = tracked_object.classification.front().label;
    if (tracker_ignore_.isIgnore(label)) continue;

    // change search range according to label type
    const float max_search_range = max_search_distance_for_merger_[label];

    // extend shape
    autoware_perception_msgs::msg::DetectedObject extended_tracked_object = tracked_object;
    extended_tracked_object.shape = extendShape(tracked_object.shape, /*scale*/ 1.1);

    pcl::PointCloud<pcl::PointXYZ> pcl_merged_cluster;
    for (const auto & initial_object : in_cluster_objects.feature_objects) {
      const float distance = autoware_utils::calc_distance2d(
        tracked_object.kinematics.pose_with_covariance.pose,
        initial_object.object.kinematics.pose_with_covariance.pose);

      if (max_search_range < distance) {
        continue;
      }

      // If there is an initial object in the tracker, it will be merged.
      const float precision = autoware::object_recognition_utils::get2dPrecision(
        initial_object.object, extended_tracked_object);
      if (precision < precision_threshold) {
        continue;
      }
      pcl::PointCloud<pcl::PointXYZ> pcl_cluster;
      pcl::fromROSMsg(initial_object.feature.cluster, pcl_cluster);
      pcl_merged_cluster += pcl_cluster;
    }

    // NERO_DBT_VISIBLE_END: points a tracked pedestrian / bicycle / motorcycle explains are not this vehicle's
    if (nero_ve_tol_ >= 0.0 && autoware::object_recognition_utils::isCarLikeVehicle(label)) {
      neroDropOthersPoints(tracked_objects, pcl_merged_cluster);
    }
    if (pcl_merged_cluster.points.empty()) {  // if clusters aren't found
      out_no_found_tracked_objects.objects.push_back(tracked_object);
      continue;
    }

    // build output clusters
    tier4_perception_msgs::msg::DetectedObjectWithFeature feature_object;
    feature_object.object.classification = tracked_object.classification;

    bool is_shape_estimated = shape_estimator_->estimateShapeAndPose(
      label, pcl_merged_cluster,
      getReferenceYawInfo(
        label, tf2::getYaw(tracked_object.kinematics.pose_with_covariance.pose.orientation)),
      getReferenceShapeSizeInfo(label, tracked_object.shape),
      tracked_object.kinematics.pose_with_covariance.pose, feature_object.object.shape,
      feature_object.object.kinematics.pose_with_covariance.pose);
    if (!is_shape_estimated) {
      out_no_found_tracked_objects.objects.push_back(tracked_object);
      continue;
    }

    if (nero_ve_tol_ >= 0.0 && autoware::object_recognition_utils::isCarLikeVehicle(label)) {
      neroFitSeenSpace(feature_object.object);
    }

    feature_object.object.existence_probability =
      autoware::object_recognition_utils::get2dIoU(tracked_object, feature_object.object);
    setClusterInObjectWithFeature(in_cluster_objects.header, pcl_merged_cluster, feature_object);
    out_objects.feature_objects.push_back(feature_object);
  }
}
}  // namespace autoware::detection_by_tracker

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::detection_by_tracker::DetectionByTracker)
