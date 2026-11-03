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

#ifndef DETECTION_BY_TRACKER_NODE_HPP_
#define DETECTION_BY_TRACKER_NODE_HPP_

#include "autoware/euclidean_cluster/euclidean_cluster.hpp"
#include "autoware/euclidean_cluster/utils.hpp"
#include "autoware/euclidean_cluster/voxel_grid_based_euclidean_cluster.hpp"
#include "autoware/shape_estimation/shape_estimator.hpp"
#include "autoware_utils/ros/published_time_publisher.hpp"
#include "debugger/debugger.hpp"
#include "tracker/tracker_handler.hpp"
#include "utils/utils.hpp"

#include <rclcpp/rclcpp.hpp>

#include "autoware_perception_msgs/msg/detected_objects.hpp"
#include "autoware_perception_msgs/msg/tracked_objects.hpp"
#include "tier4_perception_msgs/msg/detected_objects_with_feature.hpp"
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <tf2/LinearMath/Transform.h>
#include <tf2/convert.h>
#include <tf2/transform_datatypes.h>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <deque>
#include <map>
#include <memory>
#include <vector>

namespace autoware::detection_by_tracker
{

class DetectionByTracker : public rclcpp::Node
{
public:
  explicit DetectionByTracker(const rclcpp::NodeOptions & node_options);

private:
  rclcpp::Publisher<autoware_perception_msgs::msg::DetectedObjects>::SharedPtr objects_pub_;
  rclcpp::Subscription<autoware_perception_msgs::msg::TrackedObjects>::SharedPtr trackers_sub_;
  rclcpp::Subscription<tier4_perception_msgs::msg::DetectedObjectsWithFeature>::SharedPtr
    initial_objects_sub_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  TrackerHandler tracker_handler_;
  std::shared_ptr<autoware::shape_estimation::ShapeEstimator> shape_estimator_;
  std::shared_ptr<autoware::euclidean_cluster::EuclideanClusterInterface> cluster_;
  std::shared_ptr<Debugger> debugger_;

  // NERO_DBT_VISIBLE_END=<tol m>[,<sensor x m>] (parkghost 2026-11-02): a merged vehicle box's end facing the lidar may
  // not stand in front of the vehicle's visible end (the surface the lidar hits right behind it), and a vehicle's cluster
  // holds no points of a tracked pedestrian / bicycle / motorcycle (the reference-pose corrector extends a partly seen
  // vehicle toward the tracker's own prediction: the box followed the prediction and a standing vehicle's track ran off)
  double nero_ve_tol_{-1.0};
  double nero_ve_sensor_x_{0.6};
  uint64_t nero_ve_boxes_{0};
  uint64_t nero_ve_moved_{0};
  uint64_t nero_ve_unresolved_{0};
  double nero_ve_moved_sum_{0.0};
  double nero_ve_moved_max_{0.0};
  uint64_t nero_ve_others_points_{0};
  std::vector<float> nero_fs_r1_;  // nearest hit range per bearing bin of this frame's clusters (inf = no hit)
  void neroDropOthersPoints(
    const autoware_perception_msgs::msg::DetectedObjects & tracked_objects, pcl::PointCloud<pcl::PointXYZ> & cluster);
  rclcpp::Time nero_ve_last_log_{0, 0, RCL_ROS_TIME};
  void neroBuildFirstHits(const tier4_perception_msgs::msg::DetectedObjectsWithFeature & clusters);
  double neroSeenFreeDepth(
    double cx, double cy, double c, double s, const std::vector<std::pair<double, double>> & outline) const;
  bool neroFitSeenSpace(autoware_perception_msgs::msg::DetectedObject & object);
  std::map<uint8_t, int> max_search_distance_for_merger_;
  std::map<uint8_t, int> max_search_distance_for_divider_;

  detection_by_tracker::utils::TrackerIgnoreLabel tracker_ignore_;

  std::unique_ptr<autoware_utils::PublishedTimePublisher> published_time_publisher_;

  void setMaxSearchRange();

  void onObjects(
    const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr input_msg);

  void divideUnderSegmentedObjects(
    const autoware_perception_msgs::msg::DetectedObjects & tracked_objects,
    const tier4_perception_msgs::msg::DetectedObjectsWithFeature & in_cluster_objects,
    autoware_perception_msgs::msg::DetectedObjects & out_no_found_tracked_objects,
    tier4_perception_msgs::msg::DetectedObjectsWithFeature & out_objects);

  float optimizeUnderSegmentedObject(
    const autoware_perception_msgs::msg::DetectedObject & target_object,
    const sensor_msgs::msg::PointCloud2 & under_segmented_cluster,
    tier4_perception_msgs::msg::DetectedObjectWithFeature & output);

  void mergeOverSegmentedObjects(
    const autoware_perception_msgs::msg::DetectedObjects & tracked_objects,
    const tier4_perception_msgs::msg::DetectedObjectsWithFeature & in_cluster_objects,
    autoware_perception_msgs::msg::DetectedObjects & out_no_found_tracked_objects,
    tier4_perception_msgs::msg::DetectedObjectsWithFeature & out_objects);
};
}  // namespace autoware::detection_by_tracker

#endif  // DETECTION_BY_TRACKER_NODE_HPP_
