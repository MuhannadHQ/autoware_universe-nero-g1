// Copyright 2025 TIER IV, Inc.
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

#pragma once

#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ROS includes
#include "autoware/point_types/types.hpp"

#include <diagnostic_updater/diagnostic_updater.hpp>
#include <managed_transform_buffer/managed_transform_buffer.hpp>
#include <point_cloud_msg_wrapper/point_cloud_msg_wrapper.hpp>

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <message_filters/pass_through.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>

namespace autoware::pointcloud_preprocessor
{
using autoware::point_types::PointXYZIRC;
using point_cloud_msg_wrapper::PointCloud2Modifier;

struct ConcatenatedCloudResult
{
  sensor_msgs::msg::PointCloud2::SharedPtr concatenate_cloud_ptr{nullptr};
  std::optional<std::unordered_map<std::string, sensor_msgs::msg::PointCloud2::SharedPtr>>
    topic_to_transformed_cloud_map;
  std::unordered_map<std::string, double> topic_to_original_stamp_map;
};

class CombineCloudHandler
{
private:
  rclcpp::Node & node_;

  std::string output_frame_;
  bool is_motion_compensated_;
  bool publish_synchronized_pointcloud_;
  bool keep_input_frame_in_synchronized_pointcloud_;
  std::unique_ptr<managed_transform_buffer::ManagedTransformBuffer> managed_tf_buffer_{nullptr};

  std::deque<geometry_msgs::msg::TwistStamped> twist_queue_;

  // [nero 2026-11-05 jolt] NERO_LIDAR_MOTION_3D: bring each cloud from its own reference time (header stamp + its first
  // point's time_stamp: the distortion corrector's reference) to the oldest stamp in 3D, with the IMU's angular velocity
  // (roll / pitch / yaw rate) and the twist's speed, instead of speed + yaw rate between header stamps. Off = upstream.
  bool nero_imu_3d_{false};
  std::deque<geometry_msgs::msg::Vector3Stamped> nero_imu_queue_;  // angular velocity in the output frame
  bool nero_imu_rotation_exists_{false};
  Eigen::Matrix3d nero_imu_rotation_{Eigen::Matrix3d::Identity()};
  bool nero_compute_transform_3d(
    const rclcpp::Time & old_stamp, const rclcpp::Time & new_stamp, Eigen::Matrix4f & transform);
  static rclcpp::Time nero_reference_time(const sensor_msgs::msg::PointCloud2 & cloud);
  // [nero 2026-11-05 jolt-6] inputs whose cloud lies at its HEADER stamp although its first point's time_stamp is later: the
  // front E1R's time_stamp starts 13.3 ms after its stamp in every frame since 11-03, while its corrected cloud lines up with
  // the top lidar at the stamp (0..+5 ms; the Airys at stamp + first point). Empty = every input at stamp + first point.
  std::unordered_set<std::string> nero_header_reference_topics_;

  /// @brief RclcppTimeHash structure defines a custom hash function for the rclcpp::Time type by
  /// using its nanoseconds representation as the hash value.
  struct RclcppTimeHash
  {
    std::size_t operator()(const rclcpp::Time & t) const
    {
      return std::hash<int64_t>()(t.nanoseconds());
    }
  };

  static void convert_to_xyzirc_cloud(
    const sensor_msgs::msg::PointCloud2::SharedPtr & input_cloud,
    sensor_msgs::msg::PointCloud2::SharedPtr & xyzirc_cloud);

  void correct_pointcloud_motion(
    const std::shared_ptr<sensor_msgs::msg::PointCloud2> & transformed_cloud_ptr,
    const std::vector<rclcpp::Time> & pc_stamps,
    std::unordered_map<rclcpp::Time, Eigen::Matrix4f, RclcppTimeHash> & transform_memo,
    std::shared_ptr<sensor_msgs::msg::PointCloud2> transformed_delay_compensated_cloud_ptr);

public:
  CombineCloudHandler(
    rclcpp::Node & node, std::string output_frame, bool is_motion_compensated,
    bool publish_synchronized_pointcloud, bool keep_input_frame_in_synchronized_pointcloud);
  void process_twist(
    const geometry_msgs::msg::TwistWithCovarianceStamped::ConstSharedPtr & twist_msg);
  void process_odometry(const nav_msgs::msg::Odometry::ConstSharedPtr & odometry_msg);
  void nero_enable_imu_3d() { nero_imu_3d_ = true; }
  void nero_set_header_reference_topics(const std::vector<std::string> & topics)
  {
    nero_header_reference_topics_ = std::unordered_set<std::string>(topics.begin(), topics.end());
  }
  void nero_process_imu(const sensor_msgs::msg::Imu::ConstSharedPtr & imu_msg);

  ConcatenatedCloudResult combine_pointclouds(
    std::unordered_map<std::string, sensor_msgs::msg::PointCloud2::SharedPtr> & topic_to_cloud_map);

  Eigen::Matrix4f compute_transform_to_adjust_for_old_timestamp(
    const rclcpp::Time & old_stamp, const rclcpp::Time & new_stamp);

  std::deque<geometry_msgs::msg::TwistStamped> get_twist_queue();
};

}  // namespace autoware::pointcloud_preprocessor
