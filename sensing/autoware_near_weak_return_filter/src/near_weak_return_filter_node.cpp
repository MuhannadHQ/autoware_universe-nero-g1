// Copyright 2026 Nero (Ahmed Alqurashi)
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

#include "autoware/near_weak_return_filter/near_weak_return_filter.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <algorithm>
#include <chrono>
#include <memory>

namespace autoware::near_weak_return_filter
{
using sensor_msgs::msg::PointCloud2;

// [nero 2026-10-28 AEBD] Loaded into the front E1R's container only when NERO_E1R_NEAR_WEAK is set (sensor kit
// launch); without it the E1R path is exactly the car's. A cloud without x / y / z / intensity / distance is passed
// through unchanged; a cloud with nothing to drop is republished byte for byte.
class NearWeakReturnFilterComponent : public rclcpp::Node
{
public:
  explicit NearWeakReturnFilterComponent(const rclcpp::NodeOptions & options)
  : Node("near_weak_return_filter", options)
  {
    params_.max_distance = declare_parameter<double>("max_distance");
    params_.max_intensity = static_cast<int>(declare_parameter<int64_t>("max_intensity"));
    params_.neighbor_radius_ratio = declare_parameter<double>("neighbor_radius_ratio");
    params_.max_neighbors = static_cast<int>(declare_parameter<int64_t>("max_neighbors"));
    const bool publish_removed = declare_parameter<bool>("publish_removed", false);

    pub_ = create_publisher<PointCloud2>("output", rclcpp::SensorDataQoS().keep_last(5));
    if (publish_removed) {
      pub_removed_ =
        create_publisher<PointCloud2>("~/debug/removed/pointcloud", rclcpp::SensorDataQoS().keep_last(5));
    }
    sub_ = create_subscription<PointCloud2>(
      "input", rclcpp::SensorDataQoS().keep_last(5),
      [this](PointCloud2::ConstSharedPtr msg) { on_cloud(msg); });
    RCLCPP_INFO(
      get_logger(),
      "near weak return filter: drop if distance < %.2f m, intensity <= %d, <= %d neighbours within %.3f x range",
      params_.max_distance, params_.max_intensity, params_.max_neighbors, params_.neighbor_radius_ratio);
  }

private:
  void on_cloud(const PointCloud2::ConstSharedPtr & msg)
  {
    const auto t0 = std::chrono::steady_clock::now();
    const auto drop = find_dropped(*msg, params_);
    if (!drop) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 10000,
        "cloud lacks x/y/z/intensity/distance (float32 x/y/z/distance): passed through unchanged");
      pub_->publish(*msg);
      return;
    }
    const size_t n_drop = static_cast<size_t>(std::count(drop->begin(), drop->end(), true));
    if (n_drop == 0) {
      pub_->publish(*msg);
    } else {
      auto out = std::make_unique<PointCloud2>(remove_points(*msg, *drop));
      pub_->publish(std::move(out));
      if (pub_removed_) pub_removed_->publish(remove_points(*msg, *drop, true));
      total_dropped_ += n_drop;
      ++frames_with_drops_;
    }
    const double ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    max_ms_ = std::max(max_ms_, ms);
    if (n_drop > 0) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "dropped %zu weak isolated near returns in this frame (%zu in %zu frames so far, max %.2f ms)",
        n_drop, total_dropped_, frames_with_drops_, max_ms_);
    }
  }

  Params params_;
  size_t total_dropped_{0};
  size_t frames_with_drops_{0};
  double max_ms_{0.0};
  rclcpp::Publisher<PointCloud2>::SharedPtr pub_;
  rclcpp::Publisher<PointCloud2>::SharedPtr pub_removed_;
  rclcpp::Subscription<PointCloud2>::SharedPtr sub_;
};

}  // namespace autoware::near_weak_return_filter

RCLCPP_COMPONENTS_REGISTER_NODE(autoware::near_weak_return_filter::NearWeakReturnFilterComponent)
