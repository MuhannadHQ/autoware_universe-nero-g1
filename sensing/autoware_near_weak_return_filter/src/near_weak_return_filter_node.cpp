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
#include <std_msgs/msg/header.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <optional>

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
    // [nero 2026-11-05 AEBD2] the dust stage: off unless dust.enabled (NERO_E1R_NEAR_DUST); off = the component as before
    DustParams dp;
    dp.enabled = declare_parameter<bool>("dust.enabled", false);
    dp.max_distance = declare_parameter<double>("dust.max_distance", dp.max_distance);
    dp.max_intensity = static_cast<int>(declare_parameter<int64_t>("dust.max_intensity", dp.max_intensity));
    dp.min_z = declare_parameter<double>("dust.min_z", dp.min_z);
    dp.max_z = declare_parameter<double>("dust.max_z", dp.max_z);
    dp.min_x = declare_parameter<double>("dust.min_x", dp.min_x);
    dp.max_abs_y = declare_parameter<double>("dust.max_abs_y", dp.max_abs_y);
    dp.angle_radius = declare_parameter<double>("dust.angle_radius", dp.angle_radius);
    dp.range_tol_abs = declare_parameter<double>("dust.range_tol_abs", dp.range_tol_abs);
    dp.range_tol_rel = declare_parameter<double>("dust.range_tol_rel", dp.range_tol_rel);
    dp.surface_core = static_cast<int>(declare_parameter<int64_t>("dust.surface_core", dp.surface_core));
    dp.surface_min = static_cast<int>(declare_parameter<int64_t>("dust.surface_min", dp.surface_min));
    dp.line_core = static_cast<int>(declare_parameter<int64_t>("dust.line_core", dp.line_core));
    dp.line_min = static_cast<int>(declare_parameter<int64_t>("dust.line_min", dp.line_min));
    dp.line_extent = declare_parameter<double>("dust.line_extent", dp.line_extent);
    dp.segment_range = declare_parameter<double>("dust.segment_range", dp.segment_range);
    dp.segment_margin = declare_parameter<double>("dust.segment_margin", dp.segment_margin);
    dp.local_radius = declare_parameter<double>("dust.local_radius", dp.local_radius);
    dp.local_share = declare_parameter<double>("dust.local_share", dp.local_share);
    dp.planar_max = declare_parameter<double>("dust.planar_max", dp.planar_max);
    dp.planar_min_points = static_cast<int>(declare_parameter<int64_t>("dust.planar_min_points", dp.planar_min_points));
    dp.planar_subsets = declare_parameter<bool>("dust.planar_subsets", dp.planar_subsets);
    dp.frame_min = static_cast<int>(declare_parameter<int64_t>("dust.frame_min", dp.frame_min));
    dp.hold_sec = declare_parameter<double>("dust.hold_sec", dp.hold_sec);
    dp.gate_max_intensity = static_cast<int>(declare_parameter<int64_t>("dust.gate_max_intensity", dp.gate_max_intensity));
    dp.gate_weak_share = declare_parameter<double>("dust.gate_weak_share", dp.gate_weak_share);
    dp.hold_min = static_cast<int>(declare_parameter<int64_t>("dust.hold_min", dp.hold_min));
    // [nero 2026-11-06 aebdust_1106] the gate's own volume, and follower mode (a side lidar gated by the front E1R)
    dp.gate_max_abs_y = declare_parameter<double>("dust.gate_max_abs_y", dp.gate_max_abs_y);
    dp.follow = declare_parameter<bool>("dust.follow", dp.follow);
    dp.follow_sec = declare_parameter<double>("dust.follow_sec", dp.follow_sec);
    if (dp.enabled) dust_ = std::make_unique<DustStage>(dp);
    if (dust_ && dp.follow) {
      // the leading stage's open frames (the front E1R's ~/dust_gate, remapped by the launch)
      gate_sub_ = create_subscription<std_msgs::msg::Header>(
        "dust_gate_in", rclcpp::QoS(10),
        [this](std_msgs::msg::Header::ConstSharedPtr h) { dust_->external_gate_open(rclcpp::Time(h->stamp).seconds()); });
    } else if (dust_) {
      // every frame this stage's gate is open: the cloud's header (stamp), for the side lidars' followers
      gate_pub_ = create_publisher<std_msgs::msg::Header>("~/dust_gate", rclcpp::QoS(10));
    }

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
    if (dust_) {
      RCLCPP_INFO(
        get_logger(),
        "[NERO_E1R_NEAR_DUST] dust stage on: distance < %.2f m, x > %.2f, |y| <= %.2f, %.2f < z <= %.2f, intensity <= %d, off surfaces (core %d, >= %d) and "
        "lines (core %d, >= %d or >= %.2f m), dust share >= %.2f in 3x3x3 voxels of %.2f m (not a plane: < %.3f), burst >= %d at intensity <= %d and >= %.2f of them (hold %.1f s while >= %d)",
        dp.max_distance, dp.min_x, dp.max_abs_y, dp.min_z, dp.max_z, dp.max_intensity, dp.surface_core, dp.surface_min, dp.line_core, dp.line_min, dp.line_extent,
        dp.local_share, dp.local_radius, dp.planar_max, dp.frame_min, dp.gate_max_intensity, dp.gate_weak_share, dp.hold_sec, dp.hold_min);
      if (dp.follow) {
        RCLCPP_INFO(
          get_logger(), "[NERO_AIRY_NEAR_DUST] follower: acts only within %.2f s of an open frame of the front E1R's dust gate (no gate of its own)",
          dp.follow_sec);
      } else {
        RCLCPP_INFO(get_logger(), "[NERO_E1R_NEAR_DUST] the frame gate counts returns with |y| <= %.2f; open frames published on ~/dust_gate",
          dp.gate_max_abs_y < 0.0 ? dp.max_abs_y : std::min(dp.gate_max_abs_y, dp.max_abs_y));
      }
    }
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
    if (!dust_) {
      if (n_drop == 0) {
        pub_->publish(*msg);
      } else {
        auto out = std::make_unique<PointCloud2>(remove_points(*msg, *drop));
        pub_->publish(std::move(out));
        if (pub_removed_) pub_removed_->publish(remove_points(*msg, *drop, true));
        total_dropped_ += n_drop;
        ++frames_with_drops_;
      }
    } else {
      // [nero 2026-11-05 AEBD2] the dust stage on what the weak-return stage keeps (aebdust_1106: no copy of the input
      // when the first stage drops nothing; same output)
      std::optional<PointCloud2> kept_storage;
      if (n_drop > 0) kept_storage = remove_points(*msg, *drop);
      const PointCloud2 & kept = kept_storage ? *kept_storage : *msg;
      DustStats st;
      const double stamp = rclcpp::Time(msg->header.stamp).seconds();
      const auto dust = dust_->process(kept, stamp, st);
      const size_t n_dust = dust ? static_cast<size_t>(std::count(dust->begin(), dust->end(), true)) : 0;
      if (gate_pub_ && st.gate_open) gate_pub_->publish(msg->header);
      if (n_dust == 0) {
        if (n_drop == 0) {
          pub_->publish(*msg);
        } else {
          pub_->publish(std::move(*kept_storage));
        }
      } else {
        pub_->publish(remove_points(kept, *dust));
        if (pub_removed_) pub_removed_->publish(remove_points(kept, *dust, true));
        dust_dropped_ += n_dust;
        ++dust_frames_;
      }
      if (n_drop > 0) {
        if (pub_removed_ && n_dust == 0) pub_removed_->publish(remove_points(*msg, *drop, true));
        total_dropped_ += n_drop;
        ++frames_with_drops_;
      }
      if (n_dust > 0) {
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "%s dust frame: dropped %zu dust returns (%zu candidates; %zu in %zu frames so far, max %.2f ms)",
          dust_->params().follow ? "[NERO_AIRY_NEAR_DUST]" : "[NERO_E1R_NEAR_DUST]", n_dust, st.candidates, dust_dropped_,
          dust_frames_, max_ms_);
      }
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
  std::unique_ptr<DustStage> dust_;
  size_t dust_dropped_{0};
  size_t dust_frames_{0};
  size_t total_dropped_{0};
  size_t frames_with_drops_{0};
  double max_ms_{0.0};
  rclcpp::Publisher<PointCloud2>::SharedPtr pub_;
  rclcpp::Publisher<PointCloud2>::SharedPtr pub_removed_;
  rclcpp::Subscription<PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr gate_pub_;
  rclcpp::Subscription<std_msgs::msg::Header>::SharedPtr gate_sub_;
};

}  // namespace autoware::near_weak_return_filter

RCLCPP_COMPONENTS_REGISTER_NODE(autoware::near_weak_return_filter::NearWeakReturnFilterComponent)
