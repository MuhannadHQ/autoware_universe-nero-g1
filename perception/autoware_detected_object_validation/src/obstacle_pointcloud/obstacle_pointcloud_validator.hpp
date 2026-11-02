// Copyright 2022 Tier IV, Inc.
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

// NOLINTNEXTLINE(whitespace/line_length)
#ifndef OBSTACLE_POINTCLOUD__OBSTACLE_POINTCLOUD_VALIDATOR_HPP_
// NOLINTNEXTLINE(whitespace/line_length)
#define OBSTACLE_POINTCLOUD__OBSTACLE_POINTCLOUD_VALIDATOR_HPP_

#include "autoware_utils/ros/debug_publisher.hpp"
#include "autoware_utils/ros/published_time_publisher.hpp"
#include "debugger.hpp"

#include <rclcpp/rclcpp.hpp>

#include "autoware_perception_msgs/msg/detected_objects.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/pcl_search.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <array>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace autoware::detected_object_validation
{
namespace obstacle_pointcloud
{

struct PointsNumThresholdParam
{
  std::vector<int64_t> min_points_num;
  std::vector<int64_t> max_points_num;
  std::vector<double> min_points_and_distance_ratio;
};

class Validator
{
private:
  PointsNumThresholdParam points_num_threshold_param_;

protected:
  pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_pointcloud_;

public:
  explicit Validator(const PointsNumThresholdParam & points_num_threshold_param);
  inline pcl::PointCloud<pcl::PointXYZ>::Ptr getDebugPointCloudWithinObject() const
  {
    return cropped_pointcloud_;
  }

  virtual bool setKdtreeInputCloud(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & input_pointcloud) = 0;
  virtual bool validate_object(
    const autoware_perception_msgs::msg::DetectedObject & transformed_object) = 0;
  virtual std::optional<float> getMaxRadius(
    const autoware_perception_msgs::msg::DetectedObject & object) = 0;
  size_t getThresholdPointCloud(const autoware_perception_msgs::msg::DetectedObject & object);
  virtual pcl::PointCloud<pcl::PointXYZ>::Ptr getDebugNeighborPointCloud() = 0;

  virtual ~Validator() = default;
};

class Validator2D : public Validator
{
private:
  pcl::PointCloud<pcl::PointXY>::Ptr obstacle_pointcloud_;
  pcl::PointCloud<pcl::PointXY>::Ptr neighbor_pointcloud_;
  pcl::search::Search<pcl::PointXY>::Ptr kdtree_;

public:
  explicit Validator2D(PointsNumThresholdParam & points_num_threshold_param);

  static pcl::PointCloud<pcl::PointXYZ>::Ptr convertToXYZ(
    const pcl::PointCloud<pcl::PointXY>::Ptr & pointcloud_xy);
  inline pcl::PointCloud<pcl::PointXYZ>::Ptr getDebugNeighborPointCloud() override
  {
    return convertToXYZ(neighbor_pointcloud_);
  }

  bool setKdtreeInputCloud(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & input_cloud) override;
  bool validate_object(
    const autoware_perception_msgs::msg::DetectedObject & transformed_object) override;
  std::optional<float> getMaxRadius(
    const autoware_perception_msgs::msg::DetectedObject & object) override;
  std::optional<size_t> getPointCloudWithinObject(
    const autoware_perception_msgs::msg::DetectedObject & object,
    const pcl::PointCloud<pcl::PointXY>::Ptr pointcloud);
};
class Validator3D : public Validator
{
private:
  pcl::PointCloud<pcl::PointXYZ>::Ptr obstacle_pointcloud_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr neighbor_pointcloud_;
  pcl::search::Search<pcl::PointXYZ>::Ptr kdtree_;

public:
  explicit Validator3D(PointsNumThresholdParam & points_num_threshold_param);
  inline pcl::PointCloud<pcl::PointXYZ>::Ptr getDebugNeighborPointCloud() override
  {
    return neighbor_pointcloud_;
  }
  bool setKdtreeInputCloud(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & input_cloud) override;
  bool validate_object(
    const autoware_perception_msgs::msg::DetectedObject & transformed_object) override;
  std::optional<float> getMaxRadius(
    const autoware_perception_msgs::msg::DetectedObject & object) override;
  std::optional<size_t> getPointCloudWithinObject(
    const autoware_perception_msgs::msg::DetectedObject & object,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr neighbor_pointcloud);
};

class ObstaclePointCloudBasedValidator : public rclcpp::Node
{
public:
  explicit ObstaclePointCloudBasedValidator(const rclcpp::NodeOptions & node_options);

private:
  rclcpp::Publisher<autoware_perception_msgs::msg::DetectedObjects>::SharedPtr objects_pub_;
  message_filters::Subscriber<autoware_perception_msgs::msg::DetectedObjects> objects_sub_;
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> obstacle_pointcloud_sub_;
  std::unique_ptr<autoware_utils::DebugPublisher> debug_publisher_{nullptr};
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  typedef message_filters::sync_policies::ApproximateTime<
    autoware_perception_msgs::msg::DetectedObjects, sensor_msgs::msg::PointCloud2>
    SyncPolicy;
  typedef message_filters::Synchronizer<SyncPolicy> Sync;
  Sync sync_;
  PointsNumThresholdParam points_num_threshold_param_;
  double validate_max_distance_sq_;  // maximum object distance to validate, squared [m^2]

  std::shared_ptr<Debugger> debugger_;
  bool using_2d_validator_;
  std::unique_ptr<Validator> validator_;
  std::unique_ptr<autoware_utils::PublishedTimePublisher> published_time_publisher_;

  // [nero 2026-11-01 VALRANGE] NERO_VAL_BODY=<lo>,<hi>,<min_points>,<min_spread>[,<k_r2>,<min_visible>[,<min_vz>,
  // <vz_range>]]: a vehicle box
  // (CAR / TRUCK / BUS / TRAILER) that fails the obstacle-cloud count is still validated when the RAW lidar cloud (the
  // concatenated cloud, ground included) holds >= max(<min_points>, <k_r2> / range^2) points in its footprint between
  // <lo> and <hi> m over the box's own bottom (its z - height / 2: the vehicle's lower body over its own road), spread
  // >= <min_spread> m along or across the box (p10-p90), and >= <min_visible> of them within 0.4 m of a box face that
  // faces the sensor (a solid body returns from its visible faces; a box in front of a wall / under a canopy does not),
  // and, within <vz_range> m, spread >= <min_vz> m in height (p10-p90): a vehicle's lower body spans several lidar rings
  // there, a rail / bar under a car-park canopy one or two.
  // [nero 2026-11-01 VALRANGE v5] fields 9-10 <persist_frames>,<persist_radius_m>: a lower-body pass validates the box
  // only if one of the previous <persist_frames> frames (<= persist_frames x 0.1 + 0.05 s back) had a vehicle box within
  // <persist_radius_m> m of it in the MAP frame that was kept by the obstacle-cloud count or passed the lower-body test
  // (a 1-2 frame CenterPoint box on a kerb ring / shrub / cabinet never validates; a real vehicle needs two lower-body
  // passes in a row at most). No map tf at the objects' stamp = no lower-body validation that frame (counted).
  // [nero 2026-11-01 VALRANGE v6] fields 11-12 <min_body_ext_m>,<ext_from_range_m>: from that range on, the raw points
  // in the footprint from <lo> over the box bottom to the box top must span >= <min_body_ext_m> in height (p10-p90): a
  // vehicle's body spans its lower body to its roof, a thin bar / barrier layer over the road (t201518 20:30:14, 44 CAR
  // boxes at 23-62 m) or a flat kerb / ground ring only its own thickness (the height test of fields 7-8 ends at 25 m).
  // [nero 2026-11-01 VALRANGE v7] fields 13-15 <low_from_m>,<low_to_m>,<low_min_points>: from field 12's range on, a
  // vehicle must also show its LOWER body: >= <low_min_points> raw points in the footprint <low_from_m> to <low_to_m> over
  // the box bottom (bumper / sills / wheels). The t201518 bar's extent read 0.54-0.72 m in 9 of 44 frames (its rings at
  // +0.5..+1.5 m) while its lowest return was +0.5..+0.9 m over the box bottom with the road visible under it; real
  // vehicles at 30-60 m return from +0.1..+0.4 m.
  // [nero 2026-11-01 VALRANGE v8] fields 16-17 <far_min_points>,<far_from_m> (needs fields 13-15 on, far_from_m >= field
  // 12's range): from that range the lower-body rescue needs >= <far_min_points> points in the <lo>-<hi> slab instead of
  // <min_points>. Far oncoming cars (w31 10-27, 51-62 m) show 6-9 slab points with their lower body visible (low 7-10);
  // the lower-body test, not the count, separates bars / branches over the road there.
  // Unset / malformed = the stock validator exactly (no raw cloud subscription).
  struct NeroBody
  {
    bool on{false};
    double lo{0.25};
    double hi{1.0};
    int min_points{10};
    double min_spread{0.8};
    double k_r2{0.0};
    double min_visible{0.0};
    double min_vz{0.0};
    double vz_range{0.0};
    int persist_frames{0};  // 0 = no persistence test (v4)
    double persist_radius{0.0};
    double min_ext{0.0};  // v6: 0 = no body-extent test (v5)
    double ext_range{0.0};
    double low_lo{0.0};  // v7: the lower-body band over the box bottom; low_min 0 = no lower-body test (v6)
    double low_hi{0.0};
    int low_min{0};
    int far_min{0};  // v8: from far_range the slab count needed; 0 = <min_points> at every range (v7)
    double far_range{0.0};
  };
  NeroBody nero_body_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr nero_raw_sub_;
  std::mutex nero_raw_mutex_;
  std::deque<sensor_msgs::msg::PointCloud2::ConstSharedPtr> nero_raw_;
  // [range band][kept, dropped, validated by the lower body, no raw cloud, lower-body pass without a vehicle there in
  // the previous frames, lower-body pass without the map tf]
  std::array<std::array<uint64_t, 6>, 6> nero_count_{};
  rclcpp::Time nero_last_log_{0, 0, RCL_ROS_TIME};
  std::vector<int> nero_cell_start_, nero_cell_items_, nero_cell_fill_;  // candidate grid (reused per frame)
  bool nero_check_{false};  // NERO_VAL_BODY_CHECK=1: also the full loop, count frames whose lists differ
  uint64_t nero_check_frames_{0}, nero_check_bad_{0};
  // the last persist_frames frames: stamp + map-frame centres of the vehicle boxes kept by the count or passing the
  // lower-body test (fixed slots, vectors reused)
  struct NeroFrame
  {
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
    std::vector<std::array<double, 2>> centres;
  };
  std::vector<NeroFrame> nero_ring_;
  size_t nero_ring_head_{0}, nero_ring_size_{0};
  void neroBodyEvidence(
    const autoware_perception_msgs::msg::DetectedObjects & transformed_objects,
    const std::vector<size_t> & candidates, std::vector<char> & body,
    const builtin_interfaces::msg::Time & stamp, const std::string & frame_id);

private:
  void onObjectsAndObstaclePointCloud(
    const autoware_perception_msgs::msg::DetectedObjects::ConstSharedPtr & input_objects,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & input_obstacle_pointcloud);
};

}  // namespace obstacle_pointcloud
}  // namespace autoware::detected_object_validation

// NOLINTNEXTLINE(whitespace/line_length)
#endif  // OBSTACLE_POINTCLOUD__OBSTACLE_POINTCLOUD_VALIDATOR_HPP_
