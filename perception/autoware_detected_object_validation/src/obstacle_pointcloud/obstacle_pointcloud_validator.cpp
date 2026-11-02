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
}

namespace
{
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
  const builtin_interfaces::msg::Time & stamp, const std::string & frame_id)
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
  if (nero_body_.on) {
    const auto t_start = std::chrono::steady_clock::now();
    std::vector<char> body(transformed_objects.objects.size(), 0);
    if (!nero_candidates.empty()) {
      neroBodyEvidence(
        transformed_objects, nero_candidates, body, input_objects->header.stamp,
        input_obstacle_pointcloud->header.frame_id);
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
  if (nero_body_.on && (this->get_clock()->now() - nero_last_log_).seconds() > 10.0) {
    nero_last_log_ = this->get_clock()->now();
    std::stringstream ss;
    const char * bands[6] = {"<20", "20-30", "30-40", "40-50", "50-60", "60-70"};
    for (size_t b = 0; b < 6; ++b) {
      ss << " " << bands[b] << "m " << nero_count_[b][0] << "/" << nero_count_[b][1] << "/"
         << nero_count_[b][2] << "/" << nero_count_[b][3];
      if (nero_body_.persist_frames > 0) ss << "/" << nero_count_[b][4] << "/" << nero_count_[b][5];
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
