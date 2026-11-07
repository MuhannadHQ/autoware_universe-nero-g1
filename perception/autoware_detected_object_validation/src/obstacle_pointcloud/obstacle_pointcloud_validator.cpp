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

#define EIGEN_MPL2_ONLY

#include "obstacle_pointcloud_validator.hpp"

#include <autoware/object_recognition_utils/object_recognition_utils.hpp>
#include <autoware_utils/geometry/boost_polygon_utils.hpp>

#include <boost/geometry.hpp>

#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2/exceptions.h>
#include <tf2/utils.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>

namespace autoware::detected_object_validation
{
namespace obstacle_pointcloud
{
namespace bg = boost::geometry;
using Shape = autoware_perception_msgs::msg::Shape;
using Polygon2d = autoware_utils::Polygon2d;

Validator::Validator(const PointsNumThresholdParam & points_num_threshold_param)
{
  points_num_threshold_param_.min_points_num = points_num_threshold_param.min_points_num;
  points_num_threshold_param_.max_points_num = points_num_threshold_param.max_points_num;
  points_num_threshold_param_.min_points_and_distance_ratio =
    points_num_threshold_param.min_points_and_distance_ratio;
}

size_t Validator::getThresholdPointCloud(
  const autoware_perception_msgs::msg::DetectedObject & object)
{
  const auto object_label_id = object.classification.front().label;
  const auto object_distance = std::hypot(
    object.kinematics.pose_with_covariance.pose.position.x,
    object.kinematics.pose_with_covariance.pose.position.y);
  size_t threshold_pc = std::clamp(
    static_cast<size_t>(std::lround(
      points_num_threshold_param_.min_points_and_distance_ratio.at(object_label_id) /
      object_distance)),
    static_cast<size_t>(points_num_threshold_param_.min_points_num.at(object_label_id)),
    static_cast<size_t>(points_num_threshold_param_.max_points_num.at(object_label_id)));
  return threshold_pc;
}

Validator2D::Validator2D(PointsNumThresholdParam & points_num_threshold_param)
: Validator(points_num_threshold_param)
{
}

bool Validator2D::setKdtreeInputCloud(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & input_cloud)
{
  obstacle_pointcloud_.reset(new pcl::PointCloud<pcl::PointXY>);
  pcl::fromROSMsg(*input_cloud, *obstacle_pointcloud_);
  if (obstacle_pointcloud_->empty()) {
    return false;
  }

  kdtree_ = pcl::make_shared<pcl::search::KdTree<pcl::PointXY>>(false);
  kdtree_->setInputCloud(obstacle_pointcloud_);

  return true;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr Validator2D::convertToXYZ(
  const pcl::PointCloud<pcl::PointXY>::Ptr & pointcloud_xy)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
  pointcloud_xyz->reserve(pointcloud_xy->size());
  for (const auto & point : *pointcloud_xy) {
    pointcloud_xyz->push_back(pcl::PointXYZ(point.x, point.y, 0.0));
  }
  return pointcloud_xyz;
}

std::optional<size_t> Validator2D::getPointCloudWithinObject(
  const autoware_perception_msgs::msg::DetectedObject & object,
  const pcl::PointCloud<pcl::PointXY>::Ptr pointcloud)
{
  std::vector<pcl::Vertices> vertices_array;
  pcl::Vertices vertices;
  Polygon2d poly2d =
    autoware_utils::to_polygon2d(object.kinematics.pose_with_covariance.pose, object.shape);
  if (bg::is_empty(poly2d)) return std::nullopt;

  pcl::PointCloud<pcl::PointXYZ>::Ptr poly3d(new pcl::PointCloud<pcl::PointXYZ>);

  for (size_t i = 0; i < poly2d.outer().size(); ++i) {
    vertices.vertices.emplace_back(i);
    vertices_array.emplace_back(vertices);
    poly3d->emplace_back(poly2d.outer().at(i).x(), poly2d.outer().at(i).y(), 0.0);
  }
  cropped_pointcloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::CropHull<pcl::PointXYZ> cropper;  // don't be implemented PointXY by PCL
  cropper.setInputCloud(convertToXYZ(pointcloud));
  cropper.setDim(2);
  cropper.setHullIndices(vertices_array);
  cropper.setHullCloud(poly3d);
  cropper.setCropOutside(true);
  cropper.filter(*cropped_pointcloud_);
  return cropped_pointcloud_->size();
}

bool Validator2D::validate_object(
  const autoware_perception_msgs::msg::DetectedObject & transformed_object)
{
  // get neighbor_pointcloud of object
  neighbor_pointcloud_.reset(new pcl::PointCloud<pcl::PointXY>);
  std::vector<int> indices;
  std::vector<float> distances;
  const auto search_radius = getMaxRadius(transformed_object);
  if (!search_radius) {
    return false;
  }
  kdtree_->radiusSearch(
    pcl::PointXY(
      transformed_object.kinematics.pose_with_covariance.pose.position.x,
      transformed_object.kinematics.pose_with_covariance.pose.position.y),
    search_radius.value(), indices, distances);
  for (const auto & index : indices) {
    neighbor_pointcloud_->push_back(obstacle_pointcloud_->at(index));
  }
  const auto num = getPointCloudWithinObject(transformed_object, neighbor_pointcloud_);
  if (!num) return true;

  size_t threshold_pointcloud_num = getThresholdPointCloud(transformed_object);
  if (num.value() > threshold_pointcloud_num) {
    return true;
  }
  return false;  // remove object
}

std::optional<float> Validator2D::getMaxRadius(
  const autoware_perception_msgs::msg::DetectedObject & object)
{
  if (object.shape.type == Shape::BOUNDING_BOX || object.shape.type == Shape::CYLINDER) {
    return std::hypot(object.shape.dimensions.x * 0.5f, object.shape.dimensions.y * 0.5f);
  } else if (object.shape.type == Shape::POLYGON) {
    float max_dist = 0.0;
    for (const auto & point : object.shape.footprint.points) {
      const float dist = std::hypot(point.x, point.y);
      max_dist = max_dist < dist ? dist : max_dist;
    }
    return max_dist;
  } else {
    return std::nullopt;
  }
}

Validator3D::Validator3D(PointsNumThresholdParam & points_num_threshold_param)
: Validator(points_num_threshold_param)
{
}
bool Validator3D::setKdtreeInputCloud(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & input_cloud)
{
  obstacle_pointcloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*input_cloud, *obstacle_pointcloud_);
  if (obstacle_pointcloud_->empty()) {
    return false;
  }
  // setup kdtree_
  kdtree_ = pcl::make_shared<pcl::search::KdTree<pcl::PointXYZ>>(false);
  kdtree_->setInputCloud(obstacle_pointcloud_);
  return true;
}
std::optional<float> Validator3D::getMaxRadius(
  const autoware_perception_msgs::msg::DetectedObject & object)
{
  if (object.shape.type == Shape::BOUNDING_BOX || object.shape.type == Shape::CYLINDER) {
    auto square_radius = (object.shape.dimensions.x * 0.5f) * (object.shape.dimensions.x * 0.5f) +
                         (object.shape.dimensions.y * 0.5f) * (object.shape.dimensions.y * 0.5f) +
                         (object.shape.dimensions.z * 0.5f) * (object.shape.dimensions.z * 0.5f);
    return std::sqrt(square_radius);
  } else if (object.shape.type == Shape::POLYGON) {
    float max_dist = 0.0;
    for (const auto & point : object.shape.footprint.points) {
      const float dist = std::hypot(point.x, point.y);
      max_dist = max_dist < dist ? dist : max_dist;
    }
    return std::hypot(max_dist, object.shape.dimensions.z * 0.5f);
  } else {
    return std::nullopt;
  }
}

std::optional<size_t> Validator3D::getPointCloudWithinObject(
  const autoware_perception_msgs::msg::DetectedObject & object,
  const pcl::PointCloud<pcl::PointXYZ>::Ptr neighbor_pointcloud)
{
  std::vector<pcl::Vertices> vertices_array;
  pcl::Vertices vertices;
  auto const & object_position = object.kinematics.pose_with_covariance.pose.position;
  auto const object_height = object.shape.dimensions.x;
  auto z_min = object_position.z - object_height / 2.0f;
  auto z_max = object_position.z + object_height / 2.0f;
  Polygon2d poly2d =
    autoware_utils::to_polygon2d(object.kinematics.pose_with_covariance.pose, object.shape);
  if (bg::is_empty(poly2d)) return std::nullopt;

  pcl::PointCloud<pcl::PointXYZ>::Ptr poly3d(new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < poly2d.outer().size(); ++i) {
    vertices.vertices.emplace_back(i);
    vertices_array.emplace_back(vertices);
    poly3d->emplace_back(poly2d.outer().at(i).x(), poly2d.outer().at(i).y(), 0.0);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_pointcloud_2d(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::CropHull<pcl::PointXYZ> cropper;
  cropper.setInputCloud(neighbor_pointcloud);
  cropper.setDim(2);
  cropper.setHullIndices(vertices_array);
  cropper.setHullCloud(poly3d);
  cropper.setCropOutside(true);
  cropper.filter(*cropped_pointcloud_2d);

  cropped_pointcloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  cropped_pointcloud_->reserve(cropped_pointcloud_2d->size());
  for (const auto & point : *cropped_pointcloud_2d) {
    if (point.z > z_min && point.z < z_max) {
      cropped_pointcloud_->push_back(point);
    }
  }
  return cropped_pointcloud_->size();
}

bool Validator3D::validate_object(
  const autoware_perception_msgs::msg::DetectedObject & transformed_object)
{
  neighbor_pointcloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<int> indices;
  std::vector<float> distances;
  const auto search_radius = getMaxRadius(transformed_object);
  if (!search_radius) {
    return false;
  }
  kdtree_->radiusSearch(
    pcl::PointXYZ(
      transformed_object.kinematics.pose_with_covariance.pose.position.x,
      transformed_object.kinematics.pose_with_covariance.pose.position.y,
      transformed_object.kinematics.pose_with_covariance.pose.position.z),
    search_radius.value(), indices, distances);
  for (const auto & index : indices) {
    neighbor_pointcloud_->push_back(obstacle_pointcloud_->at(index));
  }

  const auto num = getPointCloudWithinObject(transformed_object, neighbor_pointcloud_);
  if (!num) return true;

  size_t threshold_pointcloud_num = getThresholdPointCloud(transformed_object);
  if (num.value() > threshold_pointcloud_num) {
    return true;
  }
  return false;  // remove object
}

ObstaclePointCloudBasedValidator::ObstaclePointCloudBasedValidator(
  const rclcpp::NodeOptions & node_options)
: rclcpp::Node("obstacle_pointcloud_based_validator", node_options),
  objects_sub_(this, "~/input/detected_objects", rclcpp::QoS{1}.get_rmw_qos_profile()),
  obstacle_pointcloud_sub_(
    this, "~/input/obstacle_pointcloud",
    rclcpp::SensorDataQoS{}.keep_last(1).get_rmw_qos_profile()),
  tf_buffer_(get_clock()),
  tf_listener_(tf_buffer_),
  sync_(SyncPolicy(10), objects_sub_, obstacle_pointcloud_sub_)
{
  points_num_threshold_param_.min_points_num =
    declare_parameter<std::vector<int64_t>>("min_points_num");
  points_num_threshold_param_.max_points_num =
    declare_parameter<std::vector<int64_t>>("max_points_num");
  points_num_threshold_param_.min_points_and_distance_ratio =
    declare_parameter<std::vector<double>>("min_points_and_distance_ratio");
  const double validate_max_distance = declare_parameter<double>("validate_max_distance_m");
  validate_max_distance_sq_ = validate_max_distance * validate_max_distance;

  using_2d_validator_ = declare_parameter<bool>("using_2d_validator");

  using std::placeholders::_1;
  using std::placeholders::_2;

  sync_.registerCallback(
    std::bind(&ObstaclePointCloudBasedValidator::onObjectsAndObstaclePointCloud, this, _1, _2));
  if (using_2d_validator_) {
    validator_ = std::make_unique<Validator2D>(points_num_threshold_param_);
  } else {
    validator_ = std::make_unique<Validator3D>(points_num_threshold_param_);
  }

  objects_pub_ = create_publisher<autoware_perception_msgs::msg::DetectedObjects>(
    "~/output/objects", rclcpp::QoS{1});
  debug_publisher_ =
    std::make_unique<autoware_utils::DebugPublisher>(this, "obstacle_pointcloud_based_validator");

  const bool enable_debugger = declare_parameter<bool>("enable_debugger");
  if (enable_debugger) debugger_ = std::make_shared<Debugger>(this);
  published_time_publisher_ = std::make_unique<autoware_utils::PublishedTimePublisher>(this);

  // [nero 2026-11-01 VALRANGE] NERO_VAL_BODY=<lo>,<hi>,<min_points>,<min_spread>[,<k_r2>,<min_visible>[,<min_vz>,
  // <vz_range>]] (see the header)
  if (const char * env = std::getenv("NERO_VAL_BODY"); env && *env) {
    std::vector<double> v;
    std::stringstream ss(env);
    std::string tok;
    bool ok = true;
    while (std::getline(ss, tok, ',')) {
      try {
        size_t used = 0;
        v.push_back(std::stod(tok, &used));
        if (used != tok.size()) ok = false;
      } catch (...) {
        ok = false;
      }
    }
    if (
      ok &&
      (v.size() == 4 || v.size() == 6 || v.size() == 8 || v.size() == 10 || v.size() == 12 ||
       v.size() == 15 || v.size() == 17) &&
      v[0] >= 0.0 &&
      v[1] > v[0] && v[2] >= 1.0 && v[3] >= 0.0 &&
      (v.size() == 4 || (v[4] >= 0.0 && v[5] >= 0.0 && v[5] <= 1.0)) &&
      (v.size() < 8 || (v[6] >= 0.0 && v[7] >= 0.0)) &&
      (v.size() < 10 ||
       (v[8] >= 0.0 && v[8] <= 20.0 && v[8] == std::floor(v[8]) && v[9] > 0.0)) &&
      (v.size() < 12 || (v[10] >= 0.0 && v[11] >= 0.0)) &&
      (v.size() < 15 || (v[13] > v[12] && v[14] >= 0.0)) &&
      (v.size() < 17 || (v[14] >= 1.0 && v[15] >= 1.0 && v[16] >= v[11]))) {
      nero_body_.on = true;
      nero_body_.lo = v[0];
      nero_body_.hi = v[1];
      nero_body_.min_points = static_cast<int>(v[2]);
      nero_body_.min_spread = v[3];
      if (v.size() >= 6) {
        nero_body_.k_r2 = v[4];
        nero_body_.min_visible = v[5];
      }
      if (v.size() >= 8) {
        nero_body_.min_vz = v[6];
        nero_body_.vz_range = v[7];
      }
      if (v.size() >= 10) {
        nero_body_.persist_frames = static_cast<int>(v[8]);
        nero_body_.persist_radius = v[9];
        nero_ring_.resize(static_cast<size_t>(nero_body_.persist_frames));
      }
      if (v.size() >= 12) {
        nero_body_.min_ext = v[10];
        nero_body_.ext_range = v[11];
      }
      if (v.size() >= 15) {
        nero_body_.low_lo = v[12];
        nero_body_.low_hi = v[13];
        nero_body_.low_min = static_cast<int>(v[14]);
      }
      if (v.size() == 17) {
        nero_body_.far_min = static_cast<int>(v[15]);
        nero_body_.far_range = v[16];
      }
      if (const char * chk = std::getenv("NERO_VAL_BODY_CHECK"); chk && std::string(chk) == "1") {
        nero_check_ = true;
      }
      nero_raw_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "~/input/raw_pointcloud", rclcpp::SensorDataQoS{}.keep_last(3),
        [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
          std::lock_guard<std::mutex> lock(nero_raw_mutex_);
          nero_raw_.push_back(msg);
          while (nero_raw_.size() > 6) nero_raw_.pop_front();
        });
      RCLCPP_INFO(
        get_logger(),
        "[NERO_VAL_BODY] on: a vehicle box failing the obstacle-cloud count is validated by >= max(%d, "
        "%.0f/r^2) raw points %.2f-%.2f m over its own bottom spread >= %.2f m, >= %.2f of them on faces "
        "facing the sensor, height spread >= %.2f m within %.0f m, a vehicle within %.1f m (map) kept or "
        "passing in one of the previous %d frames (0 = not required), from %.0f m the points from %.2f m "
        "over the bottom to the box top spanning >= %.2f m in height (0 = not required) and >= %d of them "
        "%.2f-%.2f m over the bottom (the lower body; 0 = not required); from %.0f m >= %d slab points (0 = "
        "<min_points> everywhere) (raw cloud %s)",
        nero_body_.min_points, nero_body_.k_r2, nero_body_.lo, nero_body_.hi, nero_body_.min_spread,
        nero_body_.min_visible, nero_body_.min_vz, nero_body_.vz_range, nero_body_.persist_radius,
        nero_body_.persist_frames, nero_body_.ext_range, nero_body_.lo, nero_body_.min_ext,
        nero_body_.low_min, nero_body_.low_lo, nero_body_.low_hi, nero_body_.far_range, nero_body_.far_min,
        nero_raw_sub_->get_topic_name());
    } else {
      RCLCPP_ERROR(
        get_logger(), "[NERO_VAL_BODY] '%s' is not <lo>,<hi>,<min_points>,<min_spread>[,<k_r2>,<min_visible>[,<min_vz>,"
        "<vz_range>[,<persist_frames>,<persist_radius_m>[,<min_body_ext_m>,<ext_from_range_m>[,<low_from_m>,<low_to_m>,"
        "<low_min_points>[,<far_min_points>,<far_from_m>]]]]]]: stock validator",
        env);
    }
  }
  // [nero 2026-11-04 REALIZE] NERO_VAL_MOVER (see the header)
  if (const char * env = std::getenv("NERO_VAL_MOVER"); env && *env) {
    std::vector<double> v;
    std::stringstream ss(env);
    std::string tok;
    bool ok = true;
    while (std::getline(ss, tok, ',')) {
      try {
        size_t used = 0;
        v.push_back(std::stod(tok, &used));
        if (used != tok.size()) ok = false;
      } catch (...) {
        ok = false;
      }
    }
    if (
      ok && (v.size() == 7 || v.size() == 8 || v.size() == 12 || v.size() == 14 || v.size() == 16 || v.size() == 17) && v[0] >= 2.0 && v[0] <= 20.0 &&
      v[0] == std::floor(v[0]) && v[1] > 0.0 && v[1] <= 2.0 && v[2] > 0.0 && v[3] > 0.0 && v[4] > 0.0 &&
      v[4] <= 90.0 && v[5] >= 0.0 && v[6] >= 0.0 && v[6] == std::floor(v[6]) && (v[6] == 0.0 || nero_body_.on) &&
      (v.size() == 7 || v[7] >= 0.0) &&
      (v.size() < 16 || (v[14] > 0.0 && v[14] <= 1.0 && v[15] >= 0.0 && v[15] <= 10.0 && v[12] > 0.0)) &&
      (v.size() < 17 || (v[16] > 0.0 && v[16] <= 10.0)) &&
      (v.size() < 14 || (v[12] >= 0.0 && v[12] == std::floor(v[12]) && v[13] >= 0.0 && v[13] <= 2.0 &&
                         (v[12] == 0.0 || nero_body_.on))) &&
      (v.size() < 12 || (v[8] >= 0.0 && v[8] == std::floor(v[8]) && v[9] >= 0.0 && v[10] >= 0.0 &&
                         v[10] <= 1.0 && v[11] > 0.0 && ((v[8] == 0.0 && v[9] == 0.0) || nero_body_.on)))) {
      nero_mv_.on = true;
      nero_mv_.n = static_cast<int>(v[0]);
      nero_mv_.win = v[1];
      nero_mv_.rms = v[2];
      nero_mv_.vmin = v[3];
      nero_mv_.yaw = v[4];
      nero_mv_.travel = v[5];
      nero_mv_.min_slab = static_cast<int>(v[6]);
      if (v.size() >= 8) nero_mv_.min_range = v[7];
      if (v.size() >= 12) {
        nero_mv_.min_body = static_cast<int>(v[8]);
        nero_mv_.k_r2_body = v[9];
        nero_mv_.min_score = v[10];
        nero_mv_.v_first = v[11];
      }
      if (v.size() >= 14) {
        nero_mv_.min_road_body = static_cast<int>(v[12]);
        nero_mv_.road_h = v[13];
      }
      if (v.size() >= 16) {
        nero_mv_.max_stay = v[14];
        nero_mv_.twin_m = v[15];
      }
      if (v.size() == 17) nero_mv_.twin_lat = v[16];
      if (const char * dbg = std::getenv("NERO_VAL_MOVER_DEBUG"); dbg && std::string(dbg) == "1") {
        nero_mv_.debug = true;
      }
      RCLCPP_INFO(
        get_logger(),
        "[NERO_VAL_MOVER] on: a dropped vehicle box is validated when >= %d of its CenterPoint boxes within %.2f s "
        "lie on a line (map frame, rms <= %.2f m) at >= %.1f m/s over >= %.1f m, its heading within %.0f deg of the "
        "motion, >= %d lower-body slab points (0 = not required), from %.0f m; >= max(%d, %.0f/r^2) raw body points "
        "(0 = not required), the boxes' mean existence >= %.2f, first step within 1.0 m + %.1f m/s x the gap; >= %d raw "
        "points > %.2f m over the road (0 = not required); moving body: stay share < %.2f (1 = off), no kept vehicle box within "
        "%.1f m (0 = off) and across it within %.1f m (0 = any)",
        nero_mv_.n, nero_mv_.win, nero_mv_.rms, nero_mv_.vmin, nero_mv_.travel, nero_mv_.yaw, nero_mv_.min_slab,
        nero_mv_.min_range, nero_mv_.min_body, nero_mv_.k_r2_body, nero_mv_.min_score, nero_mv_.v_first,
        nero_mv_.min_road_body, nero_mv_.road_h, nero_mv_.max_stay, nero_mv_.twin_m, nero_mv_.twin_lat);
    } else {
      RCLCPP_ERROR(
        get_logger(),
        "[NERO_VAL_MOVER] '%s' is not <n>,<win_s>,<rms_m>,<vmin_mps>,<yaw_deg>,<min_travel_m>,<min_slab>[,<min_range_m>"
        "[,<min_body>,<k_r2_body>,<min_score>,<v_first_mps>[,<min_road_body>,<road_h_m>[,<max_stay>,<twin_m>[,<twin_lat_m>]]]]] (raw point tests "
        "need NERO_VAL_BODY): "
        "no mover validation",
        env);
    }
  }
  // [nero 2026-11-04 REALIZE-7] NERO_VAL_PED (see the header)
  if (const char * env = std::getenv("NERO_VAL_PED"); env && *env) {
    std::vector<double> v;
    std::stringstream ss(env);
    std::string tok;
    bool ok = true;
    while (std::getline(ss, tok, ',')) {
      try {
        size_t used = 0;
        v.push_back(std::stod(tok, &used));
        if (used != tok.size() || !std::isfinite(v.back())) ok = false;
      } catch (...) {
        ok = false;
      }
    }
    auto count_ok = [](const double x, const double hi) { return x >= 1.0 && x <= hi && x == std::floor(x); };
    if (!nero_body_.on) {
      RCLCPP_ERROR(
        get_logger(), "[NERO_VAL_PED] '%s': needs NERO_VAL_BODY on (its raw cloud subscription): no people validation",
        env);
    } else if (
      ok && v.size() == 16 && count_ok(v[0], 1e6) && v[1] >= 0.0 && v[2] >= 0.0 && count_ok(v[3], 100.0) && v[4] > 0.0 &&
      v[4] <= 5.0 && v[5] >= 0.0 && v[6] > 0.0 && v[7] >= 0.0 && count_ok(v[8], 100.0) && v[9] > 0.0 && v[9] <= 5.0 &&
      v[10] > 0.0 && v[11] >= 0.0 && v[12] > v[11] && v[12] <= 10.0 && v[13] >= 0.15 && v[13] <= 3.0 && v[14] >= 0.0 &&
      v[15] >= 0.0) {
      nero_ped_.on = true;
      nero_ped_.min_body = static_cast<int>(v[0]);
      nero_ped_.k_r2 = v[1];
      nero_ped_.min_ext = v[2];
      nero_ped_.n_move = static_cast<int>(v[3]);
      nero_ped_.w_move = v[4];
      nero_ped_.v_move = v[5];
      nero_ped_.rms = v[6];
      nero_ped_.min_travel = v[7];
      nero_ped_.n_stand = static_cast<int>(v[8]);
      nero_ped_.w_stand = v[9];
      nero_ped_.d_stand = v[10];
      nero_ped_.over_lo = v[11];
      nero_ped_.over_hi = v[12];
      nero_ped_.iso_pad = v[13];
      nero_ped_.iso_abs = v[14];
      nero_ped_.iso_rel = v[15];
      if (const char * dbg = std::getenv("NERO_VAL_PED_DEBUG"); dbg && std::string(dbg) == "1") {
        nero_ped_.debug = true;
      }
      RCLCPP_INFO(
        get_logger(),
        "[NERO_VAL_PED] on: a dropped PEDESTRIAN / BICYCLE box is validated when >= max(%d, %.0f/r^2) raw points in its "
        "footprint + 0.15 m 0.3-1.9 m over its bottom span >= %.2f m in height, and its chain of people boxes (map frame) "
        "moves (>= %d boxes within %.2f s on a line, rms <= %.2f m, >= %.2f m/s, travel >= %.2f m) or stands (>= %d boxes "
        "within %.2f s pairwise <= %.2f m, no raw point %.2f-%.2f m over its bottom in the footprint + 0.15 m, footprint + "
        "%.2f m adds <= max(%.1f, %.2f x body) body points)",
        nero_ped_.min_body, nero_ped_.k_r2, nero_ped_.min_ext, nero_ped_.n_move, nero_ped_.w_move, nero_ped_.rms,
        nero_ped_.v_move, nero_ped_.min_travel, nero_ped_.n_stand, nero_ped_.w_stand, nero_ped_.d_stand, nero_ped_.over_lo,
        nero_ped_.over_hi, nero_ped_.iso_pad, nero_ped_.iso_abs, nero_ped_.iso_rel);
    } else {
      RCLCPP_ERROR(
        get_logger(),
        "[NERO_VAL_PED] '%s' is not <min_body>,<k_r2>,<min_ext_m>,<n_move>,<w_move_s>,<v_move_mps>,<rms_m>,<min_travel_m>,"
        "<n_stand>,<w_stand_s>,<d_stand_m>,<over_lo_m>,<over_hi_m>,<iso_pad_m>,<iso_abs>,<iso_rel> (counts integers >= 1, "
        "windows 0-5 s, over_lo < over_hi, iso_pad >= 0.15 m): no people validation",
        env);
    }
  }
}

namespace
{
// [realize-6 research] NERO_VAL_MOVER_DUMP=<file>: one line per line-passing box (geometry, decision, raw points around it); debug only
FILE * neroMvDump()
{
  static FILE * f = [] {
    const char * p = std::getenv("NERO_VAL_MOVER_DUMP");
    return (p && *p) ? std::fopen(p, "a") : static_cast<FILE *>(nullptr);
  }();
  return f;
}
// numpy-style linear percentile of a sorted vector
double neroPercentile(const std::vector<float> & sorted, const double q)
{
  if (sorted.empty()) return 0.0;
  const double pos = q * static_cast<double>(sorted.size() - 1);
  const size_t i = static_cast<size_t>(pos);
  if (i + 1 >= sorted.size()) return sorted.back();
  const double f = pos - static_cast<double>(i);
  return sorted[i] + f * (sorted[i + 1] - sorted[i]);
}
size_t neroBand(const double range)
{
  if (range < 20.0) return 0;
  return std::min<size_t>(5, static_cast<size_t>(range / 10.0) - 1);
}
bool neroIsVehicle(const autoware_perception_msgs::msg::DetectedObject & object)
{
  using Label = autoware_perception_msgs::msg::ObjectClassification;
  if (object.classification.empty()) return false;
  const auto label = object.classification.front().label;
  return label == Label::CAR || label == Label::TRUCK || label == Label::BUS ||
         label == Label::TRAILER;
}
}  // namespace

void ObstaclePointCloudBasedValidator::neroBodyEvidence(
  const autoware_perception_msgs::msg::DetectedObjects & transformed_objects,
  const std::vector<size_t> & candidates, std::vector<char> & body,
  const builtin_interfaces::msg::Time & stamp, const std::string & frame_id, std::vector<int> * slab)
{
  sensor_msgs::msg::PointCloud2::ConstSharedPtr raw;
  {
    std::lock_guard<std::mutex> lock(nero_raw_mutex_);
    const rclcpp::Time t_obj(stamp);
    double best = 0.05;
    for (const auto & c : nero_raw_) {
      const double dt = std::abs((rclcpp::Time(c->header.stamp) - t_obj).seconds());
      if (dt <= best) {
        best = dt;
        raw = c;
      }
    }
  }
  struct Cand
  {
    size_t idx;
    double cx, cy, c, s, hl, hw, r, zlo, zhi, ztop, range, su, sv;
    double zbot, zl0, zl1;  // v7: the lowest z taken; the lower-body band (zl0, zl1] (empty when the test is off)
    std::vector<float> u, v, w;
    std::vector<float> b;  // v6: heights of the points from zlo to the box top (only when the extent test applies)
    size_t visible{0};
    size_t nlow{0};  // v7: points in the lower-body band
  };
  std::vector<Cand> cands;
  double zmin = 1e9, zmax = -1e9;
  for (const auto i : candidates) {
    const auto & o = transformed_objects.objects.at(i);
    const auto & p = o.kinematics.pose_with_covariance.pose;
    const double yaw = tf2::getYaw(p.orientation);
    Cand cd;
    cd.idx = i;
    cd.cx = p.position.x;
    cd.cy = p.position.y;
    cd.c = std::cos(yaw);
    cd.s = std::sin(yaw);
    cd.hl = o.shape.dimensions.x * 0.5;
    cd.hw = o.shape.dimensions.y * 0.5;
    cd.r = std::hypot(cd.hl, cd.hw);
    const double bottom = p.position.z - o.shape.dimensions.z * 0.5;
    cd.zlo = bottom + nero_body_.lo;
    cd.zhi = bottom + nero_body_.hi;
    cd.range = std::hypot(p.position.x, p.position.y);
    // v6: beyond <ext_from_range_m> the points up to the box top are gathered too (the body's height extent)
    cd.ztop = (nero_body_.min_ext > 0.0 && cd.range >= nero_body_.ext_range)
                ? std::max(cd.zhi, bottom + o.shape.dimensions.z)
                : cd.zhi;
    // v7: from the same range a vehicle must also show its lower body: >= <low_min_points> points <low_from_m> to <low_to_m>
    // over the box bottom (a bar / barrier arm / branch over the road has none there, the road is visible under it)
    const bool low_on = nero_body_.low_min > 0 && cd.range >= nero_body_.ext_range;
    cd.zl0 = low_on ? bottom + nero_body_.low_lo : cd.zlo;
    cd.zl1 = low_on ? bottom + nero_body_.low_hi : cd.zlo;
    cd.zbot = std::min(cd.zlo, cd.zl0);
    // the sensor (the cloud frame's origin) in the box frame: which faces it sees
    cd.su = cd.c * (-cd.cx) + cd.s * (-cd.cy);
    cd.sv = -cd.s * (-cd.cx) + cd.c * (-cd.cy);
    zmin = std::min(zmin, cd.zbot);
    zmax = std::max(zmax, cd.ztop);
    cands.push_back(std::move(cd));
  }
  if (!raw || raw->header.frame_id != frame_id || raw->width * raw->height == 0) {
    for (const auto & cd : cands) nero_count_[neroBand(cd.range)][3]++;
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "[NERO_VAL_BODY] no raw cloud for the objects' stamp in frame '%s' (%s): stock verdicts",
      frame_id.c_str(), raw ? raw->header.frame_id.c_str() : "none");
    return;
  }
  // one point's test against one candidate (the lower-body slab of its footprint)
  auto take = [](Cand & cd, const float x, const float y, const float z) {
    if (z <= cd.zbot || z > cd.ztop) return;
    const double dx = x - cd.cx, dy = y - cd.cy;
    if (std::abs(dx) > cd.r || std::abs(dy) > cd.r) return;
    const double u = cd.c * dx + cd.s * dy;
    const double v = -cd.s * dx + cd.c * dy;
    if (std::abs(u) > cd.hl || std::abs(v) > cd.hw) return;
    if (z > cd.zl0 && z <= cd.zl1) cd.nlow++;
    if (z <= cd.zlo) return;
    if (cd.ztop > cd.zhi) cd.b.push_back(z);
    if (z > cd.zhi) return;
    cd.u.push_back(static_cast<float>(u));
    cd.v.push_back(static_cast<float>(v));
    cd.w.push_back(z);
    if (
      (cd.su > cd.hl && u > cd.hl - 0.4) || (cd.su < -cd.hl && u < -cd.hl + 0.4) ||
      (cd.sv > cd.hw && v > cd.hw - 0.4) || (cd.sv < -cd.hw && v < -cd.hw + 0.4)) {
      cd.visible++;
    }
  };
  // the cloud's float32 x / y / z (read in place: no iterator per field)
  int ox = -1, oy = -1, oz = -1;
  for (const auto & f : raw->fields) {
    if (f.datatype != sensor_msgs::msg::PointField::FLOAT32 || f.count != 1) continue;
    if (f.name == "x") ox = static_cast<int>(f.offset);
    if (f.name == "y") oy = static_cast<int>(f.offset);
    if (f.name == "z") oz = static_cast<int>(f.offset);
  }
  if (ox < 0 || oy < 0 || oz < 0 || raw->is_bigendian) {
    for (const auto & cd : cands) nero_count_[neroBand(cd.range)][3]++;
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "[NERO_VAL_BODY] raw cloud without float32 x / y / z: stock verdicts");
    return;
  }
  // a 2 m grid over the candidates' footprints: each candidate is listed in every cell its test square
  // (centre +- r) touches plus one cell around it (no rounding edge), so every point the full loop would take for
  // a candidate is tested against it, in the same point order -> the same numbers (NERO_VAL_BODY_CHECK=1 proves it)
  constexpr double kCell = 2.0;
  double gx0 = 1e9, gy0 = 1e9, gx1 = -1e9, gy1 = -1e9;
  for (const auto & cd : cands) {
    gx0 = std::min(gx0, cd.cx - cd.r - kCell);
    gy0 = std::min(gy0, cd.cy - cd.r - kCell);
    gx1 = std::max(gx1, cd.cx + cd.r + kCell);
    gy1 = std::max(gy1, cd.cy + cd.r + kCell);
  }
  const int nx = static_cast<int>(std::floor((gx1 - gx0) / kCell)) + 1;
  const int ny = static_cast<int>(std::floor((gy1 - gy0) / kCell)) + 1;
  nero_cell_start_.assign(static_cast<size_t>(nx) * ny + 1, 0);
  auto cell_span = [&](const Cand & cd, int & ix0, int & ix1, int & iy0, int & iy1) {
    ix0 = std::max(0, static_cast<int>(std::floor((cd.cx - cd.r - gx0) / kCell)) - 1);
    ix1 = std::min(nx - 1, static_cast<int>(std::floor((cd.cx + cd.r - gx0) / kCell)) + 1);
    iy0 = std::max(0, static_cast<int>(std::floor((cd.cy - cd.r - gy0) / kCell)) - 1);
    iy1 = std::min(ny - 1, static_cast<int>(std::floor((cd.cy + cd.r - gy0) / kCell)) + 1);
  };
  for (const auto & cd : cands) {
    int ix0, ix1, iy0, iy1;
    cell_span(cd, ix0, ix1, iy0, iy1);
    for (int j = iy0; j <= iy1; ++j)
      for (int i = ix0; i <= ix1; ++i) nero_cell_start_[static_cast<size_t>(j) * nx + i + 1]++;
  }
  for (size_t k = 1; k < nero_cell_start_.size(); ++k) nero_cell_start_[k] += nero_cell_start_[k - 1];
  nero_cell_items_.resize(nero_cell_start_.back());
  nero_cell_fill_.assign(nero_cell_start_.begin(), nero_cell_start_.end() - 1);
  for (size_t k = 0; k < cands.size(); ++k) {
    int ix0, ix1, iy0, iy1;
    cell_span(cands[k], ix0, ix1, iy0, iy1);
    for (int j = iy0; j <= iy1; ++j)
      for (int i = ix0; i <= ix1; ++i)
        nero_cell_items_[nero_cell_fill_[static_cast<size_t>(j) * nx + i]++] = static_cast<int>(k);
  }
  const uint8_t * data = raw->data.data();
  for (uint32_t row = 0; row < raw->height; ++row) {
    const uint8_t * q = data + static_cast<size_t>(row) * raw->row_step;
    for (uint32_t col = 0; col < raw->width; ++col, q += raw->point_step) {
      float x, y, z;
      std::memcpy(&z, q + oz, sizeof(float));
      if (!(z > zmin && z <= zmax)) continue;
      std::memcpy(&x, q + ox, sizeof(float));
      std::memcpy(&y, q + oy, sizeof(float));
      const double fx = std::floor((x - gx0) / kCell), fy = std::floor((y - gy0) / kCell);
      if (!(fx >= 0.0 && fx < nx && fy >= 0.0 && fy < ny)) continue;
      const size_t cell = static_cast<size_t>(fy) * nx + static_cast<size_t>(fx);
      for (int e = nero_cell_start_[cell]; e < nero_cell_start_[cell + 1]; ++e) {
        take(cands[nero_cell_items_[e]], x, y, z);
      }
    }
  }
  if (nero_check_) {
    // NERO_VAL_BODY_CHECK=1: the full loop over every candidate as well; the per-candidate point lists must be equal
    std::vector<Cand> full = cands;
    for (auto & cd : full) {
      cd.u.clear();
      cd.v.clear();
      cd.w.clear();
      cd.b.clear();
      cd.visible = 0;
      cd.nlow = 0;
    }
    sensor_msgs::PointCloud2ConstIterator<float> ix(*raw, "x"), iy(*raw, "y"), iz(*raw, "z");
    for (; ix != ix.end(); ++ix, ++iy, ++iz) {
      const float z = *iz;
      if (!(z > zmin && z <= zmax)) continue;
      for (auto & cd : full) take(cd, *ix, *iy, z);
    }
    bool same = true;
    for (size_t k = 0; k < cands.size(); ++k) {
      same = same && full[k].u == cands[k].u && full[k].v == cands[k].v && full[k].w == cands[k].w && full[k].b == cands[k].b &&
             full[k].visible == cands[k].visible && full[k].nlow == cands[k].nlow;
    }
    nero_check_frames_++;
    if (!same) nero_check_bad_++;
  }
  for (auto & cd : cands) {
    // v8: from far_range (the lower-body test on there) a smaller slab count
    const int min_pts = (nero_body_.far_min > 0 && cd.range >= nero_body_.far_range) ? nero_body_.far_min
                                                                                       : nero_body_.min_points;
    if (slab) slab->at(cd.idx) = static_cast<int>(cd.u.size());  // NERO_VAL_MOVER's raw support (before any test)
    const double need =
      std::max(static_cast<double>(min_pts), nero_body_.k_r2 / std::max(1.0, cd.range * cd.range));
    if (static_cast<double>(cd.u.size()) < need) continue;
    if (
      static_cast<double>(cd.visible) <
      nero_body_.min_visible * static_cast<double>(cd.u.size())) {
      continue;
    }
    std::sort(cd.u.begin(), cd.u.end());
    std::sort(cd.v.begin(), cd.v.end());
    const double spread = std::max(
      neroPercentile(cd.u, 0.9) - neroPercentile(cd.u, 0.1),
      neroPercentile(cd.v, 0.9) - neroPercentile(cd.v, 0.1));
    if (spread < nero_body_.min_spread) continue;
    if (nero_body_.min_vz > 0.0 && cd.range < nero_body_.vz_range) {
      std::sort(cd.w.begin(), cd.w.end());
      if (neroPercentile(cd.w, 0.9) - neroPercentile(cd.w, 0.1) < nero_body_.min_vz) continue;
    }
    if (cd.ztop > cd.zhi) {
      // v6: a vehicle's points span its body (lower body to roof); a thin bar / kerb ring layer only its thickness
      if (cd.b.empty()) continue;
      std::sort(cd.b.begin(), cd.b.end());
      if (neroPercentile(cd.b, 0.9) - neroPercentile(cd.b, 0.1) < nero_body_.min_ext) continue;
    }
    if (cd.zl1 > cd.zl0 && cd.nlow < static_cast<size_t>(nero_body_.low_min)) continue;  // v7: no lower body
    body.at(cd.idx) = 1;
  }
}
void ObstaclePointCloudBasedValidator::neroMoverPass(
  const autoware_perception_msgs::msg::DetectedObjects & transformed_objects, std::vector<char> & keep,
  const std::vector<int> & slab, const builtin_interfaces::msg::Time & stamp, const std::string & frame_id)
{
  // [nero 2026-11-04 REALIZE] NERO_VAL_MOVER (see the header): chain this frame's vehicle boxes (map frame) onto the
  // chains of the previous frames, then validate a dropped vehicle box whose chain moves on a line
  const rclcpp::Time t_obj(stamp);
  const double t = t_obj.seconds();
  tf2::Transform map_from_cloud;
  try {
    const auto tf = tf_buffer_.lookupTransform("map", frame_id, t_obj, rclcpp::Duration::from_seconds(0.0));
    tf2::fromMsg(tf.transform, map_from_cloud);
  } catch (const tf2::TransformException &) {
    return;  // no map tf at this stamp: no mover validation this frame (the chains wait; a gap > 0.35 s ends them)
  }
  const double map_yaw = tf2::getYaw(map_from_cloud.getRotation());
  struct Box
  {
    size_t idx;
    double x, y, yaw, range, prob;
  };
  std::vector<Box> boxes;
  for (size_t i = 0; i < transformed_objects.objects.size(); ++i) {
    const auto & o = transformed_objects.objects.at(i);
    if (!neroIsVehicle(o)) continue;
    const auto & p = o.kinematics.pose_with_covariance.pose;
    const tf2::Vector3 m = map_from_cloud * tf2::Vector3(p.position.x, p.position.y, p.position.z);
    boxes.push_back(Box{
      i, m.x(), m.y(), tf2::getYaw(p.orientation) + map_yaw, std::hypot(p.position.x, p.position.y),
      static_cast<double>(o.existence_probability)});
  }
  // drop chains with a gap > 0.35 s
  nero_mv_chains_.erase(
    std::remove_if(
      nero_mv_chains_.begin(), nero_mv_chains_.end(),
      [&](const NeroMvChain & c) { return t - c.last_t > 0.35 || t - c.last_t <= 0.0; }),
    nero_mv_chains_.end());
  // greedy association by distance to each chain's predicted centre
  struct Pair
  {
    double d;
    size_t c, b;
  };
  std::vector<Pair> pairs;
  for (size_t c = 0; c < nero_mv_chains_.size(); ++c) {
    const auto & P = nero_mv_chains_[c].pts;
    const auto & L = P.back();
    double vx = 0.0, vy = 0.0;
    if (P.size() >= 2) {
      const auto & Q = P[P.size() - 2];
      const double dt0 = L[0] - Q[0];
      if (dt0 > 1e-3) {
        vx = (L[1] - Q[1]) / dt0;
        vy = (L[2] - Q[2]) / dt0;
      }
    }
    const double dt = t - L[0];
    const double px = L[1] + vx * dt, py = L[2] + vy * dt;
    const double gate = 1.0 + (P.size() >= 2 ? 2.5 : nero_mv_.v_first) * dt;
    for (size_t b = 0; b < boxes.size(); ++b) {
      const double d = std::hypot(boxes[b].x - px, boxes[b].y - py);
      if (d <= gate) pairs.push_back(Pair{d, c, b});
    }
  }
  std::sort(pairs.begin(), pairs.end(), [](const Pair & a, const Pair & b) { return a.d < b.d; });
  std::vector<int> chain_of(boxes.size(), -1);
  std::vector<char> used(nero_mv_chains_.size(), 0);
  for (const auto & pr : pairs) {
    if (used[pr.c] || chain_of[pr.b] >= 0) continue;
    used[pr.c] = 1;
    chain_of[pr.b] = static_cast<int>(pr.c);
  }
  const size_t keep_pts = static_cast<size_t>(nero_mv_.n) + 4;
  for (size_t b = 0; b < boxes.size(); ++b) {
    if (chain_of[b] < 0) {
      nero_mv_chains_.push_back(NeroMvChain{});
      chain_of[b] = static_cast<int>(nero_mv_chains_.size() - 1);
    }
    auto & ch = nero_mv_chains_[static_cast<size_t>(chain_of[b])];
    ch.pts.push_back({t, boxes[b].x, boxes[b].y, boxes[b].prob});
    if (ch.pts.size() > keep_pts) ch.pts.erase(ch.pts.begin());
    ch.last_t = t;
  }
  // the test for the dropped vehicle boxes
  struct Pass
  {
    size_t idx{0};
    double x{0.0}, y{0.0}, range{0.0}, v{0.0}, rms{0.0}, travel{0.0}, dyaw{0.0};
    int slab{0};
    double score{0.0};
    int body{0};
    // v3: road level (2nd lowest raw z around the footprint), the box bottom, points > road_h over the road, their top
    double road{std::numeric_limits<double>::quiet_NaN()}, bottom{0.0}, ztop{-1e9};
    double zlo1{1e9}, zlo2{1e9};
    int nring{0}, body_road{0};
    std::vector<float> zb;
    std::vector<float> fx, fy, fz;  // v4: footprint points (cloud frame) for the stay test
    double myaw{0.0};
    double stay{-1.0};
    std::vector<float> dp;  // research dump (NERO_VAL_MOVER_DUMP): x, y, z of every raw point in the footprint + 1.5 m
    double cx{0.0}, cy{0.0}, yaw{0.0}, l{0.0}, w{0.0}, h{0.0};
  };
  std::vector<Pass> passing;
  for (size_t b = 0; b < boxes.size(); ++b) {
    const auto & bx = boxes[b];
    if (keep[bx.idx] || bx.range < nero_mv_.min_range) continue;
    const auto & P = nero_mv_chains_[static_cast<size_t>(chain_of[b])].pts;
    if (static_cast<int>(P.size()) < nero_mv_.n) continue;
    const size_t i0 = P.size() - static_cast<size_t>(nero_mv_.n);
    const double t0 = P[i0][0];
    if (t - t0 > nero_mv_.win + 1e-3) continue;
    // least-squares line x(t), y(t) over the last n centres
    double st = 0.0, sx = 0.0, sy = 0.0, stt = 0.0, stx = 0.0, sty = 0.0;
    const double k = static_cast<double>(nero_mv_.n);
    for (size_t i = i0; i < P.size(); ++i) {
      const double u = P[i][0] - t0;
      st += u;
      sx += P[i][1];
      sy += P[i][2];
      stt += u * u;
      stx += u * P[i][1];
      sty += u * P[i][2];
    }
    const double den = k * stt - st * st;
    if (den <= 1e-9) continue;
    const double ax = (k * stx - st * sx) / den, bx0 = (sx - ax * st) / k;
    const double ay = (k * sty - st * sy) / den, by0 = (sy - ay * st) / k;
    double se = 0.0;
    for (size_t i = i0; i < P.size(); ++i) {
      const double u = P[i][0] - t0;
      const double ex = bx0 + ax * u - P[i][1], ey = by0 + ay * u - P[i][2];
      se += ex * ex + ey * ey;
    }
    const double rms = std::sqrt(se / k);
    const double v = std::hypot(ax, ay);
    const double travel = v * (t - t0);
    if (rms > nero_mv_.rms || v < nero_mv_.vmin || travel < nero_mv_.travel) continue;
    double dyaw = std::abs(std::remainder(bx.yaw - std::atan2(ay, ax), M_PI)) * 180.0 / M_PI;
    if (dyaw > nero_mv_.yaw) continue;
    const int s = slab.empty() ? 0 : slab.at(bx.idx);
    if (s < nero_mv_.min_slab) continue;
    double score = 0.0;
    for (size_t i = i0; i < P.size(); ++i) score += P[i][3];
    score /= k;
    if (score < nero_mv_.min_score) continue;
    Pass ps;
    ps.idx = bx.idx;
    ps.x = bx.x;
    ps.y = bx.y;
    ps.range = bx.range;
    ps.v = v;
    ps.rms = rms;
    ps.travel = travel;
    ps.dyaw = dyaw;
    ps.slab = s;
    ps.score = score;
    ps.body = 0;
    ps.myaw = bx.yaw;
    passing.push_back(std::move(ps));
  }
  // v2: raw body support of the line-passing boxes (one pass over the raw cloud, a bounding-box reject first)
  // the road features are logged with the debug switch only where the body test runs anyway (no decision changes)
  const bool need_road =
    nero_mv_.min_road_body > 0 || (nero_mv_.debug && (nero_mv_.min_body > 0 || nero_mv_.k_r2_body > 0.0)) ||
    (nero_mv_.debug && neroMvDump() != nullptr && (nero_mv_.min_body > 0 || nero_mv_.k_r2_body > 0.0));
  const bool need_body = nero_mv_.min_body > 0 || nero_mv_.k_r2_body > 0.0 || need_road;
  constexpr double kRing = 1.5;
  if (need_body && !passing.empty()) {
    sensor_msgs::msg::PointCloud2::ConstSharedPtr raw;
    {
      std::lock_guard<std::mutex> lock(nero_raw_mutex_);
      double best = 0.05;
      for (const auto & c : nero_raw_) {
        const double dtc = std::abs((rclcpp::Time(c->header.stamp) - t_obj).seconds());
        if (dtc <= best) {
          best = dtc;
          raw = c;
        }
      }
    }
    int ox = -1, oy = -1, oz = -1;
    if (raw && raw->header.frame_id == frame_id && !raw->is_bigendian) {
      for (const auto & f : raw->fields) {
        if (f.datatype != sensor_msgs::msg::PointField::FLOAT32 || f.count != 1) continue;
        if (f.name == "x") ox = static_cast<int>(f.offset);
        if (f.name == "y") oy = static_cast<int>(f.offset);
        if (f.name == "z") oz = static_cast<int>(f.offset);
      }
    }
    if (ox < 0 || oy < 0 || oz < 0) {
      passing.clear();  // no raw cloud: no mover validation this frame
    } else {
      struct Foot
      {
        double cx, cy, c, s, hl, hw, z0, z1, r, zr0;
      };
      std::vector<Foot> F;
      double bx0 = 1e9, by0 = 1e9, bx1 = -1e9, by1 = -1e9, bz0 = 1e9, bz1 = -1e9;
      for (const auto & ps : passing) {
        const auto & o = transformed_objects.objects.at(ps.idx);
        const auto & p = o.kinematics.pose_with_covariance.pose;
        const double yaw = tf2::getYaw(p.orientation);
        const double bottom = p.position.z - 0.5 * o.shape.dimensions.z;
        Foot f{p.position.x, p.position.y, std::cos(yaw), std::sin(yaw), 0.5 * std::max(0.5, o.shape.dimensions.x),
               0.5 * std::max(0.5, o.shape.dimensions.y), bottom + 0.25, bottom + o.shape.dimensions.z + 0.5, 0.0,
               bottom - 1.5};
        f.r = std::hypot(f.hl, f.hw) + (need_road ? kRing : 0.0);
        passing[F.size()].bottom = bottom;
        passing[F.size()].cx = f.cx;
        passing[F.size()].cy = f.cy;
        passing[F.size()].yaw = yaw;
        passing[F.size()].l = 2.0 * f.hl;
        passing[F.size()].w = 2.0 * f.hw;
        passing[F.size()].h = o.shape.dimensions.z;
        bx0 = std::min(bx0, f.cx - f.r);
        bx1 = std::max(bx1, f.cx + f.r);
        by0 = std::min(by0, f.cy - f.r);
        by1 = std::max(by1, f.cy + f.r);
        bz0 = std::min(bz0, need_road ? f.zr0 : f.z0);
        bz1 = std::max(bz1, f.z1);
        F.push_back(f);
      }
      const uint8_t * data = raw->data.data();
      for (uint32_t row = 0; row < raw->height; ++row) {
        const uint8_t * q = data + static_cast<size_t>(row) * raw->row_step;
        for (uint32_t col = 0; col < raw->width; ++col, q += raw->point_step) {
          float x, y, z;
          std::memcpy(&x, q + ox, sizeof(float));
          if (x < bx0 || x > bx1) continue;
          std::memcpy(&y, q + oy, sizeof(float));
          if (y < by0 || y > by1) continue;
          std::memcpy(&z, q + oz, sizeof(float));
          if (z <= bz0 || z > bz1) continue;
          for (size_t j = 0; j < F.size(); ++j) {
            const auto & f = F[j];
            if (z <= (need_road ? f.zr0 : f.z0) || z > f.z1) continue;
            const double dx = x - f.cx, dy = y - f.cy;
            if (std::abs(dx) > f.r || std::abs(dy) > f.r) continue;
            const double u = std::abs(f.c * dx + f.s * dy), w = std::abs(-f.s * dx + f.c * dy);
            const bool in_fp = u <= f.hl && w <= f.hw;
            if (in_fp && z > f.z0) passing[j].body++;
            if (!need_road) continue;
            if (u <= f.hl + kRing && w <= f.hw + kRing) {
              auto & ps = passing[j];
              ps.nring++;
              if (neroMvDump()) {
                ps.dp.push_back(x);
                ps.dp.push_back(y);
                ps.dp.push_back(z);
              }
              if (z < ps.zlo1) {
                ps.zlo2 = ps.zlo1;
                ps.zlo1 = z;
              } else if (z < ps.zlo2) {
                ps.zlo2 = z;
              }
            }
            if (in_fp) {
              passing[j].zb.push_back(z);
              if (nero_mv_.max_stay < 1.0) {
                passing[j].fx.push_back(x);
                passing[j].fy.push_back(y);
                passing[j].fz.push_back(z);
              }
            }
          }
        }
      }
    }
  }
  if (need_road) {
    for (auto & ps : passing) {
      if (ps.nring >= 2) ps.road = ps.zlo2;
      if (!std::isfinite(ps.road)) continue;
      for (const float z : ps.zb) {
        if (z > ps.road + nero_mv_.road_h) {
          ps.body_road++;
          ps.ztop = std::max(ps.ztop, static_cast<double>(z));
        }
      }
    }
  }
  // v4 stay: the standing body points NOW vs the standing raw points ~0.3 s earlier (both in the map frame)
  if (need_road && nero_mv_.max_stay < 1.0 && !passing.empty()) {
    sensor_msgs::msg::PointCloud2::ConstSharedPtr old;
    {
      std::lock_guard<std::mutex> lock(nero_raw_mutex_);
      double best = 1e9;
      for (const auto & cc : nero_raw_) {
        const double age = (t_obj - rclcpp::Time(cc->header.stamp)).seconds();
        if (age >= 0.15 && age <= 0.5 && std::abs(age - 0.3) < best) {
          best = std::abs(age - 0.3);
          old = cc;
        }
      }
    }
    tf2::Transform map_from_old;
    bool have_old = false;
    if (old && old->header.frame_id == frame_id && !old->is_bigendian) {
      try {
        const auto tfo = tf_buffer_.lookupTransform("map", frame_id, old->header.stamp, rclcpp::Duration::from_seconds(0.0));
        tf2::fromMsg(tfo.transform, map_from_old);
        have_old = true;
      } catch (const tf2::TransformException &) {
      }
    }
    int qx = -1, qy = -1, qz = -1;
    if (have_old) {
      for (const auto & fl : old->fields) {
        if (fl.datatype != sensor_msgs::msg::PointField::FLOAT32 || fl.count != 1) continue;
        if (fl.name == "x") qx = static_cast<int>(fl.offset);
        if (fl.name == "y") qy = static_cast<int>(fl.offset);
        if (fl.name == "z") qz = static_cast<int>(fl.offset);
      }
    }
    if (have_old && qx >= 0 && qy >= 0 && qz >= 0) {
      // each box's region in the OLD cloud's frame: its centre (map) moved into the old frame, radius half-diagonal + 0.5 m
      const tf2::Transform old_from_map = map_from_old.inverse();
      struct Reg
      {
        double ox, oy, r;
        std::vector<float> qx, qy;  // standing old points near the box, map frame
      };
      std::vector<Reg> R(passing.size());
      for (size_t j = 0; j < passing.size(); ++j) {
        // at the ego's own map height (the map z is the real altitude: z = 0 would shift the region by altitude x pitch)
        const tf2::Vector3 oc = old_from_map * tf2::Vector3(passing[j].x, passing[j].y, map_from_old.getOrigin().z());
        R[j].ox = oc.x();
        R[j].oy = oc.y();
        R[j].r = std::hypot(passing[j].l, passing[j].w) / 2.0 + 0.5;
      }
      const uint8_t * od = old->data.data();
      for (uint32_t row = 0; row < old->height; ++row) {
        const uint8_t * q = od + static_cast<size_t>(row) * old->row_step;
        for (uint32_t col = 0; col < old->width; ++col, q += old->point_step) {
          float x, y, z;
          std::memcpy(&x, q + qx, sizeof(float));
          std::memcpy(&y, q + qy, sizeof(float));
          for (size_t j = 0; j < passing.size(); ++j) {
            auto & rg = R[j];
            if (std::abs(x - rg.ox) > rg.r || std::abs(y - rg.oy) > rg.r) continue;
            const auto & ps = passing[j];
            if (!std::isfinite(ps.road)) continue;
            std::memcpy(&z, q + qz, sizeof(float));
            if (z <= ps.road + nero_mv_.road_h) continue;
            const tf2::Vector3 m = map_from_old * tf2::Vector3(x, y, z);
            rg.qx.push_back(static_cast<float>(m.x()));
            rg.qy.push_back(static_cast<float>(m.y()));
          }
        }
      }
      for (size_t j = 0; j < passing.size(); ++j) {
        auto & ps = passing[j];
        if (!std::isfinite(ps.road)) continue;
        int nb = 0, nst = 0;
        for (size_t i = 0; i < ps.fz.size(); ++i) {
          if (ps.fz[i] <= ps.road + nero_mv_.road_h) continue;
          ++nb;
          const tf2::Vector3 m = map_from_cloud * tf2::Vector3(ps.fx[i], ps.fy[i], ps.fz[i]);
          for (size_t k = 0; k < R[j].qx.size(); ++k) {
            const double dx = m.x() - R[j].qx[k], dy = m.y() - R[j].qy[k];
            if (dx * dx + dy * dy < 0.0625) {
              ++nst;
              break;
            }
          }
        }
        if (nb > 0) ps.stay = static_cast<double>(nst) / nb;
      }
    }
  }
  if (FILE * df = neroMvDump(); df && nero_mv_.debug) {
    for (const auto & ps : passing) {
      const double need = std::max(
        static_cast<double>(nero_mv_.min_body), nero_mv_.k_r2_body / std::max(1.0, ps.range * ps.range));
      const int dec = (need_body && static_cast<double>(ps.body) < need) ? 2
                      : (nero_mv_.min_road_body > 0 && ps.body_road < nero_mv_.min_road_body) ? 3 : 1;
      std::fprintf(
        df, "P %.3f %.2f %.2f %.1f %.2f %.2f %d %d %.3f %.3f %.4f %.2f %.2f %.2f %.3f %.3f %d %zu", t, ps.x, ps.y, ps.range,
        ps.v, ps.score, ps.body, dec, ps.cx, ps.cy, ps.yaw, ps.l, ps.w, ps.h, ps.bottom, ps.road, ps.body_road,
        ps.dp.size() / 3);
      for (const float v : ps.dp) std::fprintf(df, " %.3f", v);
      std::fprintf(df, "\n");
      // the raw cloud ~0.3 s earlier (nero_raw_ keeps 6), points within a generous disc around the box (base_link of THAT cloud;
      // the reader moves both clouds to the map with the window's EKF pose): "the place it left is empty" offline
      sensor_msgs::msg::PointCloud2::ConstSharedPtr old;
      {
        std::lock_guard<std::mutex> lock(nero_raw_mutex_);
        double best = 1e9;
        for (const auto & cc : nero_raw_) {
          const double age = (t_obj - rclcpp::Time(cc->header.stamp)).seconds();
          if (age >= 0.15 && std::abs(age - 0.3) < best) {
            best = std::abs(age - 0.3);
            old = cc;
          }
        }
      }
      if (old && old->header.frame_id == frame_id && !old->is_bigendian) {
        int qx = -1, qy = -1, qz = -1;
        for (const auto & fl : old->fields) {
          if (fl.datatype != sensor_msgs::msg::PointField::FLOAT32 || fl.count != 1) continue;
          if (fl.name == "x") qx = static_cast<int>(fl.offset);
          if (fl.name == "y") qy = static_cast<int>(fl.offset);
          if (fl.name == "z") qz = static_cast<int>(fl.offset);
        }
        if (qx >= 0 && qy >= 0 && qz >= 0) {
          const double rad = std::hypot(ps.l, ps.w) / 2.0 + std::abs(ps.v) * 0.4 + 6.0;
          std::vector<float> op;
          const uint8_t * od = old->data.data();
          for (uint32_t row = 0; row < old->height; ++row) {
            const uint8_t * q = od + static_cast<size_t>(row) * old->row_step;
            for (uint32_t col = 0; col < old->width; ++col, q += old->point_step) {
              float x, y, z;
              std::memcpy(&x, q + qx, sizeof(float));
              std::memcpy(&y, q + qy, sizeof(float));
              if (std::abs(x - ps.cx) > rad || std::abs(y - ps.cy) > rad) continue;
              std::memcpy(&z, q + qz, sizeof(float));
              if (z < ps.bottom - 1.5 || z > ps.bottom + ps.h + 0.5) continue;
              op.push_back(x);
              op.push_back(y);
              op.push_back(z);
            }
          }
          std::fprintf(df, "O %.3f %zu", rclcpp::Time(old->header.stamp).seconds(), op.size() / 3);
          for (const float v : op) std::fprintf(df, " %.3f", v);
          std::fprintf(df, "\n");
        }
      }
    }
    std::fflush(df);
  }
  for (const auto & ps : passing) {
    if (need_body) {
      const double need = std::max(
        static_cast<double>(nero_mv_.min_body), nero_mv_.k_r2_body / std::max(1.0, ps.range * ps.range));
      if (static_cast<double>(ps.body) < need) continue;
    }
    if (nero_mv_.min_road_body > 0 && ps.body_road < nero_mv_.min_road_body) {
      if (nero_mv_.debug) {
        RCLCPP_INFO(
          get_logger(),
          "[NERO_VAL_MOVER] reject_road t %.3f map (%.2f, %.2f) range %.1f v %.2f score %.2f body %d road %.2f bottom %.2f "
          "nring %d body_road %d htop %.2f",
          t, ps.x, ps.y, ps.range, ps.v, ps.score, ps.body, ps.road, ps.bottom, ps.nring, ps.body_road, ps.ztop - ps.road);
      }
      continue;
    }
    if (nero_mv_.max_stay < 1.0 && ps.stay >= nero_mv_.max_stay) {
      if (nero_mv_.debug) {
        RCLCPP_INFO(
          get_logger(), "[NERO_VAL_MOVER] reject_stay t %.3f map (%.2f, %.2f) range %.1f v %.2f body_road %d stay %.2f", t, ps.x,
          ps.y, ps.range, ps.v, ps.body_road, ps.stay);
      }
      continue;
    }
    if (nero_mv_.twin_m > 0.0) {
      bool twin = false;
      for (const auto & ob : boxes) {
        if (ob.idx == ps.idx || !keep[ob.idx]) continue;
        const double d = std::hypot(ob.x - ps.x, ob.y - ps.y);
        double dy = std::fmod(std::abs(ob.yaw - ps.myaw), M_PI);
        dy = std::min(dy, M_PI - dy);
        // realize-8 field 17: a twin is a second box on the SAME vehicle (overlapping across the kept box's heading);
        // boxes 2.5-4.5 m to its side were all cars in the next lane (cL: 72 of 72 twin rejects)
        const double lat = std::abs(-(ps.x - ob.x) * std::sin(ob.yaw) + (ps.y - ob.y) * std::cos(ob.yaw));
        if (
          d > 0.5 && d < nero_mv_.twin_m && dy < 30.0 * M_PI / 180.0 &&
          (nero_mv_.twin_lat <= 0.0 || lat < nero_mv_.twin_lat)) {
          twin = true;
          break;
        }
      }
      if (twin) {
        if (nero_mv_.debug) {
          RCLCPP_INFO(
            get_logger(), "[NERO_VAL_MOVER] reject_twin t %.3f map (%.2f, %.2f) range %.1f v %.2f", t, ps.x, ps.y, ps.range, ps.v);
        }
        continue;
      }
    }
    keep[ps.idx] = 1;
    nero_mv_count_[neroBand(ps.range)]++;
    if (nero_mv_.debug) {
      RCLCPP_INFO(
        get_logger(),
        "[NERO_VAL_MOVER] pass t %.3f map (%.2f, %.2f) range %.1f v %.2f rms %.3f travel %.2f dyaw %.1f slab %d "
        "score %.2f body %d road %.2f bottom %.2f nring %d body_road %d htop %.2f stay %.2f",
        t, ps.x, ps.y, ps.range, ps.v, ps.rms, ps.travel, ps.dyaw, ps.slab, ps.score, ps.body, ps.road, ps.bottom,
        ps.nring, ps.body_road, ps.ztop - ps.road, ps.stay);
    }
  }
}
void ObstaclePointCloudBasedValidator::neroPedPass(
  const autoware_perception_msgs::msg::DetectedObjects & transformed_objects, std::vector<char> & keep,
  const builtin_interfaces::msg::Time & stamp, const std::string & frame_id)
{
  // [nero 2026-11-04 REALIZE-7] NERO_VAL_PED (see the header): chain this frame's people boxes (map frame) onto the chains
  // of the previous frames (r5/valped common.py chains()), then rule R5 for each dropped people box
  using Label = autoware_perception_msgs::msg::ObjectClassification;
  const rclcpp::Time t_obj(stamp);
  const double t = t_obj.seconds();
  // chains whose last box is > 0.35 s old end (and every chain when the stamp went back)
  nero_ped_chains_.erase(
    std::remove_if(
      nero_ped_chains_.begin(), nero_ped_chains_.end(),
      [&](const NeroPedChain & c) { return t - c.last_t > 0.35 || t - c.last_t <= 0.0; }),
    nero_ped_chains_.end());
  std::vector<size_t> people;
  for (size_t i = 0; i < transformed_objects.objects.size(); ++i) {
    const auto & o = transformed_objects.objects.at(i);
    if (o.classification.empty()) continue;
    const auto label = o.classification.front().label;
    if (label == Label::PEDESTRIAN || label == Label::BICYCLE) people.push_back(i);
  }
  if (people.empty()) return;
  tf2::Transform map_from_cloud;
  try {
    const auto tf = tf_buffer_.lookupTransform("map", frame_id, t_obj, rclcpp::Duration::from_seconds(0.0));
    tf2::fromMsg(tf.transform, map_from_cloud);
  } catch (const tf2::TransformException &) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000, "[NERO_VAL_PED] no map tf at the objects' stamp: no people validation this frame");
    return;
  }
  struct Box
  {
    size_t idx;
    double x, y;
  };
  std::vector<Box> boxes;
  for (const auto i : people) {
    const auto & p = transformed_objects.objects.at(i).kinematics.pose_with_covariance.pose;
    const tf2::Vector3 m = map_from_cloud * tf2::Vector3(p.position.x, p.position.y, p.position.z);
    boxes.push_back(Box{i, m.x(), m.y()});
  }
  // greedy association by distance to each chain's predicted centre (ties: chain, then box order, as common.py)
  struct Pair
  {
    double d;
    size_t c, b;
  };
  std::vector<Pair> pairs;
  for (size_t c = 0; c < nero_ped_chains_.size(); ++c) {
    const auto & P = nero_ped_chains_[c].pts;
    const auto & L = P.back();
    double vx = 0.0, vy = 0.0;
    if (P.size() >= 2) {
      const auto & Q = P[P.size() - 2];
      const double dt0 = L[0] - Q[0];
      if (dt0 > 1e-3) {
        vx = (L[1] - Q[1]) / dt0;
        vy = (L[2] - Q[2]) / dt0;
      }
    }
    const double dt = t - L[0];
    const double px = L[1] + vx * dt, py = L[2] + vy * dt;
    const double gate = 1.0 + 2.5 * dt;
    for (size_t b = 0; b < boxes.size(); ++b) {
      const double d = std::hypot(boxes[b].x - px, boxes[b].y - py);
      if (d <= gate) pairs.push_back(Pair{d, c, b});
    }
  }
  std::sort(pairs.begin(), pairs.end(), [](const Pair & a, const Pair & b) {
    return a.d < b.d || (a.d == b.d && (a.c < b.c || (a.c == b.c && a.b < b.b)));
  });
  std::vector<int> chain_of(boxes.size(), -1);
  std::vector<char> used(nero_ped_chains_.size(), 0);
  for (const auto & pr : pairs) {
    if (used[pr.c] || chain_of[pr.b] >= 0) continue;
    used[pr.c] = 1;
    chain_of[pr.b] = static_cast<int>(pr.c);
  }
  // history: the longer test window + a margin, at least 2 s (the last two boxes always stay: the prediction)
  const double hist = std::max(2.0, std::max(nero_ped_.w_move, nero_ped_.w_stand) + 0.1);
  for (size_t b = 0; b < boxes.size(); ++b) {
    if (chain_of[b] < 0) {
      nero_ped_chains_.push_back(NeroPedChain{});
      chain_of[b] = static_cast<int>(nero_ped_chains_.size() - 1);
    }
    auto & ch = nero_ped_chains_[static_cast<size_t>(chain_of[b])];
    ch.pts.push_back({t, boxes[b].x, boxes[b].y});
    ch.total++;
    ch.last_t = t;
    size_t old = 0;
    while (old + 2 < ch.pts.size() && ch.pts[old][0] < t - hist) ++old;
    if (old > 0) ch.pts.erase(ch.pts.begin(), ch.pts.begin() + static_cast<std::ptrdiff_t>(old));
  }
  // the dropped people boxes (the car's verdict after every other rule)
  std::vector<size_t> cand;
  for (size_t b = 0; b < boxes.size(); ++b) {
    if (!keep[boxes[b].idx]) cand.push_back(b);
  }
  if (cand.empty()) return;
  sensor_msgs::msg::PointCloud2::ConstSharedPtr raw;
  {
    std::lock_guard<std::mutex> lock(nero_raw_mutex_);
    double best = 0.05;
    for (const auto & c : nero_raw_) {
      const double dtc = std::abs((rclcpp::Time(c->header.stamp) - t_obj).seconds());
      if (dtc <= best) {
        best = dtc;
        raw = c;
      }
    }
  }
  int ox = -1, oy = -1, oz = -1;
  if (raw && raw->header.frame_id == frame_id && !raw->is_bigendian && raw->width * raw->height > 0) {
    for (const auto & f : raw->fields) {
      if (f.datatype != sensor_msgs::msg::PointField::FLOAT32 || f.count != 1) continue;
      if (f.name == "x") ox = static_cast<int>(f.offset);
      if (f.name == "y") oy = static_cast<int>(f.offset);
      if (f.name == "z") oz = static_cast<int>(f.offset);
    }
  }
  if (ox < 0 || oy < 0 || oz < 0) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "[NERO_VAL_PED] no raw cloud (float32 x / y / z) for the objects' stamp in frame '%s' (%s): no people validation "
      "this frame",
      frame_id.c_str(), raw ? raw->header.frame_id.c_str() : "none");
    return;
  }
  // raw body features per candidate (common.py body_feats): box frame u along its yaw, v across, z over its bottom
  constexpr double kPad = 0.15, kLo = 0.3, kHi = 1.9;
  struct Feat
  {
    size_t b;
    double cx, cy, c, s, hl, hw, bottom, r, range;
    int nb{0}, nb_w{0}, over{0};
    double zmin{1e9}, zmax{-1e9};
  };
  std::vector<Feat> F;
  const double pad_max = std::max(kPad, nero_ped_.iso_pad);
  const double zr0 = std::min(kLo, nero_ped_.over_lo), zr1 = std::max(kHi, nero_ped_.over_hi);
  double zmin = 1e9, zmax = -1e9;
  for (const auto b : cand) {
    const auto & o = transformed_objects.objects.at(boxes[b].idx);
    const auto & p = o.kinematics.pose_with_covariance.pose;
    const double yaw = tf2::getYaw(p.orientation);
    Feat f;
    f.b = b;
    f.cx = p.position.x;
    f.cy = p.position.y;
    f.c = std::cos(yaw);
    f.s = std::sin(yaw);
    f.hl = 0.5 * o.shape.dimensions.x;
    f.hw = 0.5 * o.shape.dimensions.y;
    f.bottom = p.position.z - 0.5 * o.shape.dimensions.z;
    f.r = std::hypot(f.hl + pad_max, f.hw + pad_max);
    f.range = std::hypot(p.position.x, p.position.y);
    zmin = std::min(zmin, f.bottom + zr0);
    zmax = std::max(zmax, f.bottom + zr1);
    F.push_back(f);
  }
  // a 2 m grid over the candidates' squares (as neroBodyEvidence): each candidate listed in every cell its square touches
  // plus one around it
  constexpr double kCell = 2.0;
  double gx0 = 1e9, gy0 = 1e9, gx1 = -1e9, gy1 = -1e9;
  for (const auto & f : F) {
    gx0 = std::min(gx0, f.cx - f.r - kCell);
    gy0 = std::min(gy0, f.cy - f.r - kCell);
    gx1 = std::max(gx1, f.cx + f.r + kCell);
    gy1 = std::max(gy1, f.cy + f.r + kCell);
  }
  const int nx = static_cast<int>(std::floor((gx1 - gx0) / kCell)) + 1;
  const int ny = static_cast<int>(std::floor((gy1 - gy0) / kCell)) + 1;
  std::vector<int> cell_start(static_cast<size_t>(nx) * ny + 1, 0);
  auto cell_span = [&](const Feat & f, int & ix0, int & ix1, int & iy0, int & iy1) {
    ix0 = std::max(0, static_cast<int>(std::floor((f.cx - f.r - gx0) / kCell)) - 1);
    ix1 = std::min(nx - 1, static_cast<int>(std::floor((f.cx + f.r - gx0) / kCell)) + 1);
    iy0 = std::max(0, static_cast<int>(std::floor((f.cy - f.r - gy0) / kCell)) - 1);
    iy1 = std::min(ny - 1, static_cast<int>(std::floor((f.cy + f.r - gy0) / kCell)) + 1);
  };
  for (const auto & f : F) {
    int ix0, ix1, iy0, iy1;
    cell_span(f, ix0, ix1, iy0, iy1);
    for (int j = iy0; j <= iy1; ++j)
      for (int i = ix0; i <= ix1; ++i) cell_start[static_cast<size_t>(j) * nx + i + 1]++;
  }
  for (size_t k = 1; k < cell_start.size(); ++k) cell_start[k] += cell_start[k - 1];
  std::vector<int> cell_items(static_cast<size_t>(cell_start.back()));
  std::vector<int> cell_fill(cell_start.begin(), cell_start.end() - 1);
  for (size_t k = 0; k < F.size(); ++k) {
    int ix0, ix1, iy0, iy1;
    cell_span(F[k], ix0, ix1, iy0, iy1);
    for (int j = iy0; j <= iy1; ++j)
      for (int i = ix0; i <= ix1; ++i) cell_items[cell_fill[static_cast<size_t>(j) * nx + i]++] = static_cast<int>(k);
  }
  const uint8_t * data = raw->data.data();
  for (uint32_t row = 0; row < raw->height; ++row) {
    const uint8_t * q = data + static_cast<size_t>(row) * raw->row_step;
    for (uint32_t col = 0; col < raw->width; ++col, q += raw->point_step) {
      float x, y, z;
      std::memcpy(&z, q + oz, sizeof(float));
      if (!(z > zmin && z <= zmax)) continue;
      std::memcpy(&x, q + ox, sizeof(float));
      std::memcpy(&y, q + oy, sizeof(float));
      const double fx = std::floor((x - gx0) / kCell), fy = std::floor((y - gy0) / kCell);
      if (!(fx >= 0.0 && fx < nx && fy >= 0.0 && fy < ny)) continue;
      const size_t cell = static_cast<size_t>(fy) * nx + static_cast<size_t>(fx);
      for (int e = cell_start[cell]; e < cell_start[cell + 1]; ++e) {
        auto & f = F[static_cast<size_t>(cell_items[static_cast<size_t>(e)])];
        const double zr = z - f.bottom;
        const bool body_z = zr > kLo && zr <= kHi;
        const bool over_z = zr > nero_ped_.over_lo && zr <= nero_ped_.over_hi;
        if (!body_z && !over_z) continue;
        const double dx = x - f.cx, dy = y - f.cy;
        if (std::abs(dx) > f.r || std::abs(dy) > f.r) continue;
        const double u = std::abs(f.c * dx + f.s * dy), w = std::abs(-f.s * dx + f.c * dy);
        const bool in_fp = u <= f.hl + kPad && w <= f.hw + kPad;
        if (body_z) {
          if (in_fp) {
            f.nb++;
            f.zmin = std::min(f.zmin, zr);
            f.zmax = std::max(f.zmax, zr);
          }
          if (u <= f.hl + nero_ped_.iso_pad && w <= f.hw + nero_ped_.iso_pad) f.nb_w++;
        }
        if (over_z && in_fp) f.over++;
      }
    }
  }
  // rule R5 per candidate (r5/valped evaluate.py): R1 AND (moving OR (standing AND over == 0 AND iso))
  for (const auto & f : F) {
    const auto & bx = boxes[f.b];
    const auto & ch = nero_ped_chains_[static_cast<size_t>(chain_of[f.b])];
    const auto & P = ch.pts;  // time-ascending, its last entry this frame's box
    const double ext = f.nb > 0 ? f.zmax - f.zmin : 0.0;
    const double need = std::max(static_cast<double>(nero_ped_.min_body), nero_ped_.k_r2 / std::max(1.0, f.range * f.range));
    const bool r1 = static_cast<double>(f.nb) >= need && ext >= nero_ped_.min_ext;
    const bool iso =
      static_cast<double>(f.nb_w - f.nb) <= std::max(nero_ped_.iso_abs, nero_ped_.iso_rel * static_cast<double>(f.nb));
    // moving (common.py line_test): least-squares lines x(t), y(t) over the chain's boxes within the last w_move s
    size_t i0 = P.size();
    while (i0 > 0 && t - P[i0 - 1][0] <= nero_ped_.w_move + 1e-3) --i0;
    const size_t n_mv = P.size() - i0;
    bool mov = false;
    double v = 0.0, rms = 9.0;  // no line (< 2 boxes): common.py's sentinels
    if (n_mv >= 2 && P.back()[0] - P[i0][0] > 1e-3) {
      const double t0 = P[i0][0];
      double st = 0.0, sx = 0.0, sy = 0.0, stt = 0.0, stx = 0.0, sty = 0.0;
      const double k = static_cast<double>(n_mv);
      for (size_t i = i0; i < P.size(); ++i) {
        const double u = P[i][0] - t0;
        st += u;
        sx += P[i][1];
        sy += P[i][2];
        stt += u * u;
        stx += u * P[i][1];
        sty += u * P[i][2];
      }
      const double den = k * stt - st * st;
      if (den > 1e-12) {
        const double ax = (k * stx - st * sx) / den, bx0 = (sx - ax * st) / k;
        const double ay = (k * sty - st * sy) / den, by0 = (sy - ay * st) / k;
        double se = 0.0;
        for (size_t i = i0; i < P.size(); ++i) {
          const double u = P[i][0] - t0;
          const double ex = bx0 + ax * u - P[i][1], ey = by0 + ay * u - P[i][2];
          se += ex * ex + ey * ey;
        }
        rms = std::sqrt(se / k);
        v = std::hypot(ax, ay);
        const double travel = v * (P.back()[0] - t0);
        mov = static_cast<int>(n_mv) >= nero_ped_.n_move && rms <= nero_ped_.rms && v >= nero_ped_.v_move &&
              travel >= nero_ped_.min_travel;
      }
    }
    // standing (common.py standing_test): >= n_stand boxes within the last w_stand s, pairwise within d_stand
    size_t j0 = P.size();
    while (j0 > 0 && t - P[j0 - 1][0] <= nero_ped_.w_stand + 1e-3) --j0;
    bool stand = false;
    if (static_cast<int>(P.size() - j0) >= nero_ped_.n_stand) {
      const double d2 = nero_ped_.d_stand * nero_ped_.d_stand;
      stand = true;
      for (size_t a = j0; a < P.size() && stand; ++a) {
        for (size_t c = a + 1; c < P.size(); ++c) {
          const double dx = P[a][1] - P[c][1], dy = P[a][2] - P[c][2];
          if (dx * dx + dy * dy > d2) {
            stand = false;
            break;
          }
        }
      }
    }
    const bool pass = r1 && (mov || (stand && f.over == 0 && iso));
    if (pass) {
      keep[bx.idx] = 1;
      nero_ped_count_[neroBand(f.range)]++;
    }
    if (nero_ped_.debug) {
      RCLCPP_INFO(
        get_logger(),
        "[NERO_VAL_PED] %s t %.3f map (%.2f, %.2f) range %.1f lbl %d nb %d ext %.2f over %d nb_w %d chain %zu mov %d v %.2f "
        "rms %.3f stand %d",
        pass ? "keep" : "reject", t, bx.x, bx.y, f.range,
        static_cast<int>(transformed_objects.objects.at(bx.idx).classification.front().label), f.nb, ext, f.over, f.nb_w,
        ch.total, mov ? 1 : 0, v, rms, stand ? 1 : 0);
    }
  }
}

void ObstaclePointCloudBasedValidator::onObjectsAndObstaclePointCloud(
  const autoware_perception_msgs::msg::DetectedObjects::ConstSharedPtr & input_objects,
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & input_obstacle_pointcloud)
{
  autoware_perception_msgs::msg::DetectedObjects output, removed_objects;
  output.header = input_objects->header;
  removed_objects.header = input_objects->header;

  // Transform to pointcloud frame
  autoware_perception_msgs::msg::DetectedObjects transformed_objects;
  if (!autoware::object_recognition_utils::transformObjects(
        *input_objects, input_obstacle_pointcloud->header.frame_id, tf_buffer_,
        transformed_objects)) {
    // objects_pub_->publish(*input_objects);
    return;
  }
  bool validation_is_ready = true;
  if (!validator_->setKdtreeInputCloud(input_obstacle_pointcloud)) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5,
      "obstacle pointcloud is empty! Can not validate objects.");
    validation_is_ready = false;
  }

  // [nero 2026-11-01 VALRANGE] verdicts first, then (switch on) the lower-body evidence for the vehicle boxes the
  // obstacle cloud failed; the output keeps the input order exactly as before
  std::vector<char> keep(transformed_objects.objects.size(), 0);
  std::vector<size_t> nero_candidates, nero_kept_vehicles;
  for (size_t i = 0; i < transformed_objects.objects.size(); ++i) {
    const auto & transformed_object = transformed_objects.objects.at(i);
    // check object distance
    const double distance_sq =
      transformed_object.kinematics.pose_with_covariance.pose.position.x *
        transformed_object.kinematics.pose_with_covariance.pose.position.x +
      transformed_object.kinematics.pose_with_covariance.pose.position.y *
        transformed_object.kinematics.pose_with_covariance.pose.position.y;
    if (distance_sq > validate_max_distance_sq_) {
      // pass to output
      keep[i] = 1;
      continue;
    }

    const auto validated =
      validation_is_ready ? validator_->validate_object(transformed_object) : false;
    if (debugger_) {
      debugger_->addNeighborPointcloud(validator_->getDebugNeighborPointCloud());
      debugger_->addPointcloudWithinPolygon(validator_->getDebugPointCloudWithinObject());
    }
    keep[i] = validated ? 1 : 0;
    if (nero_body_.on && neroIsVehicle(transformed_object)) {
      nero_count_[neroBand(std::sqrt(distance_sq))][validated ? 0 : 1]++;
      if (!validated) {
        nero_candidates.push_back(i);
      } else if (nero_body_.persist_frames > 0) {
        nero_kept_vehicles.push_back(i);
      }
    }
  }
  std::vector<int> nero_slab(nero_mv_.on ? transformed_objects.objects.size() : 0, 0);
  if (nero_body_.on) {
    const auto t_start = std::chrono::steady_clock::now();
    std::vector<char> body(transformed_objects.objects.size(), 0);
    if (!nero_candidates.empty()) {
      neroBodyEvidence(
        transformed_objects, nero_candidates, body, input_objects->header.stamp,
        input_obstacle_pointcloud->header.frame_id, nero_mv_.on ? &nero_slab : nullptr);
    }
    auto range_of = [&](const size_t i) {
      const auto & q = transformed_objects.objects.at(i).kinematics.pose_with_covariance.pose.position;
      return std::hypot(q.x, q.y);
    };
    if (nero_body_.persist_frames <= 0) {
      for (const auto i : nero_candidates) {
        if (!body[i]) continue;
        keep[i] = 1;
        nero_count_[neroBand(range_of(i))][2]++;
      }
    } else {
      // v5 persistence: the map frame at the objects' stamp (zero timeout); without it no lower-body validation
      const rclcpp::Time t_obj(input_objects->header.stamp);
      tf2::Transform map_from_cloud;
      bool tf_ok = false;
      if (!nero_candidates.empty() || !nero_kept_vehicles.empty()) {
        try {
          const auto tf = tf_buffer_.lookupTransform(
            "map", input_obstacle_pointcloud->header.frame_id, t_obj,
            rclcpp::Duration::from_seconds(0.0));
          tf2::fromMsg(tf.transform, map_from_cloud);
          tf_ok = true;
        } catch (const tf2::TransformException &) {
          tf_ok = false;
        }
      }
      auto to_map = [&](const size_t i) {
        const auto & q = transformed_objects.objects.at(i).kinematics.pose_with_covariance.pose.position;
        const tf2::Vector3 m = map_from_cloud * tf2::Vector3(q.x, q.y, q.z);
        return std::array<double, 2>{m.x(), m.y()};
      };
      const size_t n_slots = nero_ring_.size();
      const double max_age = 0.1 * nero_body_.persist_frames + 0.05;
      for (const auto i : nero_candidates) {
        if (!body[i]) continue;
        const size_t band = neroBand(range_of(i));
        if (!tf_ok) {
          nero_count_[band][5]++;
          continue;
        }
        const auto c = to_map(i);
        bool seen = false;
        for (size_t k = 0; k < nero_ring_size_ && !seen; ++k) {
          const auto & f = nero_ring_[(nero_ring_head_ + n_slots - 1 - k) % n_slots];  // newest first
          const double age = (t_obj - f.stamp).seconds();
          if (age <= 0.0 || age > max_age) continue;
          for (const auto & q : f.centres) {
            if (std::hypot(q[0] - c[0], q[1] - c[1]) <= nero_body_.persist_radius) {
              seen = true;
              break;
            }
          }
        }
        if (seen) {
          keep[i] = 1;
          nero_count_[band][2]++;
        } else {
          nero_count_[band][4]++;
        }
      }
      // this frame's ok vehicles (kept by the count, or passing the lower-body test before the persistence gate)
      NeroFrame & f = nero_ring_[nero_ring_head_];
      f.stamp = t_obj;
      f.centres.clear();
      if (tf_ok) {
        for (const auto i : nero_kept_vehicles) f.centres.push_back(to_map(i));
        for (const auto i : nero_candidates) {
          if (body[i]) f.centres.push_back(to_map(i));
        }
      }
      nero_ring_head_ = (nero_ring_head_ + 1) % n_slots;
      nero_ring_size_ = std::min(nero_ring_size_ + 1, n_slots);
    }
    if (!nero_candidates.empty()) {
      debug_publisher_->publish<autoware_internal_debug_msgs::msg::Float64Stamped>(
        "debug/nero_body_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start)
          .count());
    }
  }
  if (nero_mv_.on) {
    const auto t_mv = std::chrono::steady_clock::now();
    neroMoverPass(
      transformed_objects, keep, nero_slab, input_objects->header.stamp,
      input_obstacle_pointcloud->header.frame_id);
    debug_publisher_->publish<autoware_internal_debug_msgs::msg::Float64Stamped>(
      "debug/nero_mover_ms",
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_mv).count());
  }
  if (nero_ped_.on) {
    // [nero 2026-11-04 REALIZE-7] NERO_VAL_PED: after every other rule (only the people boxes still dropped)
    const auto t_ped = std::chrono::steady_clock::now();
    neroPedPass(
      transformed_objects, keep, input_objects->header.stamp, input_obstacle_pointcloud->header.frame_id);
    debug_publisher_->publish<autoware_internal_debug_msgs::msg::Float64Stamped>(
      "debug/nero_ped_ms",
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ped).count());
  }
  if (nero_body_.on && (this->get_clock()->now() - nero_last_log_).seconds() > 10.0) {
    nero_last_log_ = this->get_clock()->now();
    std::stringstream ss;
    const char * bands[6] = {"<20", "20-30", "30-40", "40-50", "50-60", "60-70"};
    for (size_t b = 0; b < 6; ++b) {
      ss << " " << bands[b] << "m " << nero_count_[b][0] << "/" << nero_count_[b][1] << "/"
         << nero_count_[b][2] << "/" << nero_count_[b][3];
      if (nero_body_.persist_frames > 0) ss << "/" << nero_count_[b][4] << "/" << nero_count_[b][5];
      if (nero_mv_.on) ss << " mover " << nero_mv_count_[b];
      if (nero_ped_.on) ss << " ped " << nero_ped_count_[b];
    }
    RCLCPP_INFO(
      get_logger(),
      "[NERO_VAL_BODY] vehicle boxes since start, per range: kept by obstacle points / dropped by "
      "them / of those validated by the lower body / no raw cloud%s:%s",
      nero_body_.persist_frames > 0
        ? " / lower-body pass without a vehicle there in the previous frames / without the map tf"
        : "",
      ss.str().c_str());
    if (nero_check_) {
      RCLCPP_INFO(
        get_logger(), "[NERO_VAL_BODY] check: %lu frames, %lu with a different point list (grid vs full loop)",
        static_cast<unsigned long>(nero_check_frames_), static_cast<unsigned long>(nero_check_bad_));
    }
  }
  for (size_t i = 0; i < transformed_objects.objects.size(); ++i) {
    const auto & object = input_objects->objects.at(i);
    if (keep[i]) {
      output.objects.push_back(object);
    } else {
      removed_objects.objects.push_back(object);
    }
  }

  objects_pub_->publish(output);
  published_time_publisher_->publish_if_subscribed(objects_pub_, output.header.stamp);
  if (debugger_) {
    debugger_->publishRemovedObjects(removed_objects);
    debugger_->publishNeighborPointcloud(input_obstacle_pointcloud->header);
    debugger_->publishPointcloudWithinPolygon(input_obstacle_pointcloud->header);
  }

  // Publish processing time info
  const double pipeline_latency =
    std::chrono::duration<double, std::milli>(
      std::chrono::nanoseconds((this->get_clock()->now() - output.header.stamp).nanoseconds()))
      .count();
  debug_publisher_->publish<autoware_internal_debug_msgs::msg::Float64Stamped>(
    "debug/pipeline_latency_ms", pipeline_latency);
}

}  // namespace obstacle_pointcloud
}  // namespace autoware::detected_object_validation

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(
  autoware::detected_object_validation::obstacle_pointcloud::ObstaclePointCloudBasedValidator)
