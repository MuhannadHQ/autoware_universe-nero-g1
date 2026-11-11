// Copyright 2018 Autoware Foundation. All rights reserved.
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

#include "shape_estimation_node.hpp"

#include "autoware/shape_estimation/shape_estimator.hpp"
#include "autoware/shape_estimation/model/nero_chi_shape.hpp"
#include "autoware_utils/math/unit_conversion.hpp"

#include "autoware_perception_msgs/msg/object_classification.hpp"

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif

#include <memory>
#include <string>

namespace autoware::shape_estimation
{

using Label = autoware_perception_msgs::msg::ObjectClassification;

ShapeEstimationNode::ShapeEstimationNode(const rclcpp::NodeOptions & node_options)
: Node("shape_estimation", node_options)
{
  using std::placeholders::_1;
  sub_ = create_subscription<DetectedObjectsWithFeature>(
    "input", rclcpp::QoS{1}, std::bind(&ShapeEstimationNode::callback, this, _1));

  pub_ = create_publisher<DetectedObjectsWithFeature>("objects", rclcpp::QoS{1});
  bool use_corrector = declare_parameter<bool>("use_corrector");
  bool use_filter = declare_parameter<bool>("use_filter");
  use_vehicle_reference_yaw_ = declare_parameter<bool>("use_vehicle_reference_yaw");
  use_vehicle_reference_shape_size_ = declare_parameter<bool>("use_vehicle_reference_shape_size");
  bool use_boost_bbox_optimizer = declare_parameter<bool>("use_boost_bbox_optimizer");
  fix_filtered_objects_label_to_unknown_ =
    declare_parameter<bool>("fix_filtered_objects_label_to_unknown");
  RCLCPP_INFO(this->get_logger(), "using boost shape estimation : %d", use_boost_bbox_optimizer);
  estimator_ =
    std::make_unique<ShapeEstimator>(use_corrector, use_filter, use_boost_bbox_optimizer);

#ifdef USE_CUDA
  use_ml_shape_estimation_ = declare_parameter<bool>("model_params.use_ml_shape_estimator");
  if (use_ml_shape_estimation_) {
    std::string model_path = declare_parameter<std::string>("model_path");
    min_points_ = declare_parameter<int>("model_params.minimum_points");
    std::string precision = declare_parameter<std::string>("model_params.precision");
    int batch_size = declare_parameter<int>("model_params.batch_size");
    tensorrt_shape_estimator_ =
      std::make_unique<TrtShapeEstimator>(model_path, precision, batch_size);
    if (this->declare_parameter("model_params.build_only", false)) {
      RCLCPP_INFO(this->get_logger(), "TensorRT engine is built and shutdown node.");
      rclcpp::shutdown();
    }
  }
#endif

  // NERO_SHAPE_UNKNOWN_CHI field 7 (wallbox-4): the chi-shape only near the planned trajectory; unset = no subscription (the car)
  if (model::neroChiCorridorWanted()) {
    nero_path_sub_ = create_subscription<autoware_planning_msgs::msg::Trajectory>(
      "/planning/scenario_planning/trajectory", rclcpp::QoS{1},
      [this](const autoware_planning_msgs::msg::Trajectory::ConstSharedPtr msg) {
        std::vector<std::array<double, 2>> p;
        p.reserve(msg->points.size());
        for (const auto & q : msg->points) p.push_back({q.pose.position.x, q.pose.position.y});
        std::lock_guard<std::mutex> lk(nero_path_mtx_);
        nero_path_map_.swap(p);
        nero_path_t_ = rclcpp::Time(msg->header.stamp).seconds();
      });
    nero_kin_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/localization/kinematic_state", rclcpp::QoS{10}, [this](const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
        const double t = rclcpp::Time(msg->header.stamp).seconds();
        std::lock_guard<std::mutex> lk(nero_path_mtx_);
        if (!nero_kin_.empty() && t <= nero_kin_.back()[0]) nero_kin_.clear();  // time went back (a new replay)
        nero_kin_.push_back(
          {t, msg->pose.pose.position.x, msg->pose.pose.position.y, tf2::getYaw(msg->pose.pose.orientation)});
        while (nero_kin_.size() > 2 && nero_kin_.front()[0] < t - 2.0) nero_kin_.pop_front();
      });
  }

  processing_time_publisher_ =
    std::make_unique<autoware_utils::DebugPublisher>(this, "shape_estimation");
  stop_watch_ptr_ = std::make_unique<autoware_utils::StopWatch<std::chrono::milliseconds>>();
  stop_watch_ptr_->tic("cyclic_time");
  stop_watch_ptr_->tic("processing_time");
  published_time_publisher_ = std::make_unique<autoware_utils::PublishedTimePublisher>(this);
}

static autoware_perception_msgs::msg::ObjectClassification::_label_type get_label(
  const autoware_perception_msgs::msg::DetectedObject::_classification_type & classification)
{
  if (classification.empty()) {
    return Label::UNKNOWN;
  }
  return classification.front().label;
}

static bool label_is_vehicle(
  const autoware_perception_msgs::msg::ObjectClassification::_label_type & label)
{
  return Label::CAR == label || Label::TRUCK == label || Label::BUS == label ||
         Label::TRAILER == label;
}

void ShapeEstimationNode::callback(const DetectedObjectsWithFeature::ConstSharedPtr input_msg)
{
  stop_watch_ptr_->toc("processing_time", true);
  // Guard
  if (pub_->get_subscription_count() < 1) {
    return;
  }

  // Create output msg
  DetectedObjectsWithFeature output_msg;
  output_msg.header = input_msg->header;

  // NERO_SHAPE_UNKNOWN_CHI field 7: the planned trajectory in this message's frame (a fresh plan, <= 1 s; else every cluster)
  model::neroChiSetCorridor(nullptr);
  if (nero_path_sub_) {
    std::vector<std::array<double, 2>> pm;
    double tp = -1.0;
    bool ok = false;
    double ex = 0.0, ey = 0.0, eyaw = 0.0;
    const double ts = rclcpp::Time(input_msg->header.stamp).seconds();
    {
      std::lock_guard<std::mutex> lk(nero_path_mtx_);
      pm = nero_path_map_;
      tp = nero_path_t_;
      // the robot (base_link) at the clusters' stamp: interpolated between the two kinematic states around it (<= 0.2 s away)
      for (std::size_t i = 1; i < nero_kin_.size(); ++i) {
        const auto & a = nero_kin_[i - 1];
        const auto & b = nero_kin_[i];
        if (a[0] <= ts && ts <= b[0] && b[0] - a[0] <= 0.2) {
          const double w = b[0] > a[0] ? (ts - a[0]) / (b[0] - a[0]) : 0.0;
          ex = a[1] + w * (b[1] - a[1]);
          ey = a[2] + w * (b[2] - a[2]);
          eyaw = a[3] + w * std::remainder(b[3] - a[3], 2.0 * M_PI);
          ok = true;
          break;
        }
      }
      if (!ok && !nero_kin_.empty() && std::abs(nero_kin_.back()[0] - ts) <= 0.1) {  // the newest state just before the stamp
        ex = nero_kin_.back()[1];
        ey = nero_kin_.back()[2];
        eyaw = nero_kin_.back()[3];
        ok = true;
      }
    }
    if (ok && pm.size() >= 2 && std::abs(ts - tp) <= 1.0 && input_msg->header.frame_id == "base_link") {
      const double c = std::cos(eyaw);
      const double s = std::sin(eyaw);
      nero_path_local_.clear();
      for (const auto & q : pm) {
        const double dx = q[0] - ex;
        const double dy = q[1] - ey;
        nero_path_local_.push_back({c * dx + s * dy, -s * dx + c * dy});
      }
      model::neroChiSetCorridor(&nero_path_local_);
      static const bool dbg = std::getenv("NERO_SHAPE_UNKNOWN_CHI_DEBUG") != nullptr;
      if (dbg) {
        double dmin = 1e9;
        for (const auto & q : nero_path_local_) dmin = std::min(dmin, std::hypot(q[0], q[1]));
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000, "[NERO_CHI_COR] stamp %.3f path_t %.3f ego (%.2f %.2f %.3f) path n %zu nearest %.2f",
          ts, tp, ex, ey, eyaw, nero_path_local_.size(), dmin);
      }
    }
  }

  // Create ml model input batch
  DetectedObjectsWithFeature input_trt_batch;

  // Estimate shape for each object and pack msg
  for (const auto & feature_object : input_msg->feature_objects) {
    const auto & object = feature_object.object;
    const auto label = get_label(object.classification);
    const auto is_vehicle = label_is_vehicle(label);
    const auto & feature = feature_object.feature;
    // convert ros to pcl
    pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(feature.cluster, *cluster);

    // check cluster data
    if (cluster->empty()) {
      continue;
    }

#ifdef USE_CUDA
    // If ml based shape estimation is enabled, add object to input batch and continue
    if (is_vehicle && use_ml_shape_estimation_ && cluster->size() > min_points_) {
      input_trt_batch.feature_objects.push_back(feature_object);
      continue;
    }
#endif

    // estimate shape and pose
    autoware_perception_msgs::msg::Shape shape;
    geometry_msgs::msg::Pose pose;
    boost::optional<ReferenceYawInfo> ref_yaw_info = boost::none;
    boost::optional<ReferenceShapeSizeInfo> ref_shape_size_info = boost::none;
    boost::optional<geometry_msgs::msg::Pose> ref_pose = boost::none;
    if (use_vehicle_reference_yaw_ && is_vehicle) {
      ref_yaw_info = ReferenceYawInfo{
        static_cast<float>(tf2::getYaw(object.kinematics.pose_with_covariance.pose.orientation)),
        autoware_utils::deg2rad(10)};
    }
    if (use_vehicle_reference_shape_size_ && is_vehicle) {
      ref_shape_size_info = ReferenceShapeSizeInfo{object.shape, ReferenceShapeSizeInfo::Mode::Min};
    }
    const bool estimated_success = estimator_->estimateShapeAndPose(
      label, *cluster, ref_yaw_info, ref_shape_size_info, ref_pose, shape, pose);

    // If the shape estimation fails, change to Unknown object.
    if (!fix_filtered_objects_label_to_unknown_ && !estimated_success) {
      continue;
    }
    output_msg.feature_objects.push_back(feature_object);
    if (!estimated_success) {
      output_msg.feature_objects.back().object.classification.front().label = Label::UNKNOWN;
    }

    output_msg.feature_objects.back().object.shape = shape;
    output_msg.feature_objects.back().object.kinematics.pose_with_covariance.pose = pose;
  }

#ifdef USE_CUDA
  DetectedObjectsWithFeature output_model;
  if (use_ml_shape_estimation_ && !input_trt_batch.feature_objects.empty()) {
    tensorrt_shape_estimator_->inference(input_trt_batch, output_model);
  }
  for (auto & feature_object : output_model.feature_objects) {
    output_msg.feature_objects.push_back(feature_object);
  }
#endif

  // Publish
  pub_->publish(output_msg);
  published_time_publisher_->publish_if_subscribed(pub_, output_msg.header.stamp);
  processing_time_publisher_->publish<autoware_internal_debug_msgs::msg::Float64Stamped>(
    "debug/cyclic_time_ms", stop_watch_ptr_->toc("cyclic_time", true));
  processing_time_publisher_->publish<autoware_internal_debug_msgs::msg::Float64Stamped>(
    "debug/processing_time_ms", stop_watch_ptr_->toc("processing_time", true));
}
}  // namespace autoware::shape_estimation

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::shape_estimation::ShapeEstimationNode)
