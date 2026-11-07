// Copyright 2021 Tier IV, Inc. All rights reserved.
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

#include "node.hpp"

#include "grid_ground_filter.hpp"
#include "nero_band_road.hpp"
#include "nero_ground_level.hpp"

#include <autoware_utils/geometry/geometry.hpp>
#include <autoware_utils/math/normalization.hpp>
#include <autoware_utils/math/unit_conversion.hpp>
#include <autoware_vehicle_info_utils/vehicle_info_utils.hpp>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace autoware::ground_segmentation
{
namespace
{
// NERO_GROUND_BAND_ROAD (rampground, QUEUE PITCHG): per node instance, kept out of the class layout.
struct NeroBandRoadState
{
  nero::BandRoad band_road;
  bool debug = false;
  std::vector<uint8_t> ng;
  std::vector<float> xyz;
  size_t frames = 0, points = 0;
  float r_min = 0.0f, r_max = 0.0f, drop20 = 0.0f;
  std::chrono::steady_clock::time_point last_log{};
};
std::mutex g_nero_band_road_mutex;
std::unordered_map<const void *, std::unique_ptr<NeroBandRoadState>> g_nero_band_road;

NeroBandRoadState * neroBandRoad(const void * node)
{
  std::lock_guard<std::mutex> lock(g_nero_band_road_mutex);
  const auto it = g_nero_band_road.find(node);
  return it == g_nero_band_road.end() ? nullptr : it->second.get();
}

// NERO_GROUND_LEVEL (jolt, QUEUE PTH): per node instance, kept out of the class layout (nero_ground_level.hpp).
struct NeroGroundLevelState
{
  nero::GroundLevel level;
  float ceiling = 0.52f;
  float x_front = 1.25f;  // the vehicle's front (vehicle_info max_longitudinal_offset)
  size_t frames = 0, levelled = 0, cut = 0;
  float max_pitch = 0.0f, max_roll = 0.0f;
  double ms_fit = 0.0, ms_lev = 0.0, ms_cls = 0.0;
  std::shared_ptr<sensor_msgs::msg::PointCloud2> buf;  // reused every cloud (no per-cloud allocation)
  std::vector<size_t> src;
  std::vector<float> bxyz;  // base_link x, y, z of the kept points (for NERO_GROUND_BAND_ROAD: the car's frame)
  std::chrono::steady_clock::time_point last_log{};
};
std::mutex g_nero_ground_level_mutex;
std::unordered_map<const void *, std::unique_ptr<NeroGroundLevelState>> g_nero_ground_level;

NeroGroundLevelState * neroGroundLevel(const void * node)
{
  std::lock_guard<std::mutex> lock(g_nero_ground_level_mutex);
  const auto it = g_nero_ground_level.find(node);
  return it == g_nero_ground_level.end() ? nullptr : it->second.get();
}

// take out of the obstacle indices the points the band's road-referenced ceiling leaves out (nero_band_road.hpp):
// n points (x, y, z at `stride` floats); the indices are byte offsets of `step`-byte points of the filtered cloud
void neroBandRoadOn(
  NeroBandRoadState & nb, const float * xyz, size_t stride, size_t n, size_t step, const builtin_interfaces::msg::Time & stamp,
  pcl::PointIndices & no_ground_indices, const rclcpp::Logger & logger)
{
  nb.ng.assign(n, 0);
  for (const auto i : no_ground_indices.indices) nb.ng[static_cast<size_t>(i) / step] = 1;
  const auto st = nb.band_road.apply(xyz, stride, n, nb.ng);
  if (st.removed) {
    size_t k = 0;
    for (const auto i : no_ground_indices.indices)
      if (nb.ng[static_cast<size_t>(i) / step]) no_ground_indices.indices[k++] = i;
    no_ground_indices.indices.resize(k);
    if (nb.debug) {
      RCLCPP_INFO(
        logger,
        "[NERO_GROUND_BAND_ROAD] frame %d.%09u: %zu obstacle points above road + %.2f m out (range %.1f-%.1f m, "
        "lowest %.2f m over the road; road 20 m ahead at base z %+.2f m)",
        stamp.sec, stamp.nanosec, st.removed, nb.band_road.band, st.r_min, st.r_max,
        st.h_min, st.drop20);
    }
    if (!nb.frames || st.r_min < nb.r_min) nb.r_min = st.r_min;
    if (!nb.frames || st.r_max > nb.r_max) nb.r_max = st.r_max;
    if (!nb.frames || st.drop20 < nb.drop20) nb.drop20 = st.drop20;
    ++nb.frames;
    nb.points += st.removed;
  }
  const auto now = std::chrono::steady_clock::now();
  if (nb.frames && now - nb.last_log >= std::chrono::seconds(1)) {
    RCLCPP_INFO(
      logger,
      "[NERO_GROUND_BAND_ROAD] %zu frames / %zu obstacle points above road + %.2f m left out since the last line "
      "(range %.1f-%.1f m; lowest road 20 m ahead at base z %+.2f m)",
      nb.frames, nb.points, nb.band_road.band, nb.r_min, nb.r_max, nb.drop20);
    nb.frames = nb.points = 0;
    nb.last_log = now;
  }
}

void neroApplyBandRoad(
  NeroBandRoadState & nb, const sensor_msgs::msg::PointCloud2::ConstSharedPtr & input,
  pcl::PointIndices & no_ground_indices, const rclcpp::Logger & logger)
{
  const size_t step = input->point_step;
  const size_t n = static_cast<size_t>(input->width) * input->height;
  if (!n || !step || input->data.size() < n * step) return;
  int ox = -1, oy = -1, oz = -1;
  bool f32 = true;
  for (const auto & f : input->fields) {
    if (f.name == "x" || f.name == "y" || f.name == "z") {
      (f.name == "x" ? ox : f.name == "y" ? oy : oz) = static_cast<int>(f.offset);
      f32 = f32 && f.datatype == sensor_msgs::msg::PointField::FLOAT32;
    }
  }
  if (ox < 0 || oy < 0 || oz < 0 || !f32) return;
  const float * xyz = nullptr;
  size_t stride = 3;
  if (ox == 0 && oy == 4 && oz == 8 && step % 4 == 0) {
    xyz = reinterpret_cast<const float *>(input->data.data());
    stride = step / 4;
  } else {
    nb.xyz.resize(3 * n);
    for (size_t i = 0; i < n; ++i) {
      std::memcpy(&nb.xyz[3 * i], &input->data[i * step + ox], 4);
      std::memcpy(&nb.xyz[3 * i + 1], &input->data[i * step + oy], 4);
      std::memcpy(&nb.xyz[3 * i + 2], &input->data[i * step + oz], 4);
    }
    xyz = nb.xyz.data();
  }
  neroBandRoadOn(nb, xyz, stride, n, step, input->header.stamp, no_ground_indices, logger);
}
}  // namespace

using autoware::pointcloud_preprocessor::get_param;
using autoware::vehicle_info_utils::VehicleInfoUtils;
using autoware_utils::calc_distance3d;
using autoware_utils::deg2rad;
using autoware_utils::normalize_degree;
using autoware_utils::normalize_radian;
using autoware_utils::ScopedTimeTrack;

ScanGroundFilterComponent::ScanGroundFilterComponent(const rclcpp::NodeOptions & options)
: autoware::pointcloud_preprocessor::Filter("ScanGroundFilter", options)
{
  // set initial parameters
  {
    // modes
    elevation_grid_mode_ = declare_parameter<bool>("elevation_grid_mode");

    // common parameters
    radial_divider_angle_rad_ =
      static_cast<float>(deg2rad(declare_parameter<double>("radial_divider_angle_deg")));
    radial_dividers_num_ = std::ceil(2.0 * M_PI / radial_divider_angle_rad_);

    // common thresholds
    global_slope_max_angle_rad_ =
      static_cast<float>(deg2rad(declare_parameter<double>("global_slope_max_angle_deg")));
    local_slope_max_angle_rad_ =
      static_cast<float>(deg2rad(declare_parameter<double>("local_slope_max_angle_deg")));
    global_slope_max_ratio_ = std::tan(global_slope_max_angle_rad_);
    local_slope_max_ratio_ = std::tan(local_slope_max_angle_rad_);
    split_points_distance_tolerance_ =
      static_cast<float>(declare_parameter<double>("split_points_distance_tolerance"));

    // vehicle info
    vehicle_info_ = VehicleInfoUtils(*this).getVehicleInfo();

    // non-grid parameters
    use_virtual_ground_point_ = declare_parameter<bool>("use_virtual_ground_point");
    split_height_distance_ = static_cast<float>(declare_parameter<double>("split_height_distance"));

    // grid mode parameters
    use_recheck_ground_cluster_ = declare_parameter<bool>("use_recheck_ground_cluster");
    use_lowest_point_ = declare_parameter<bool>("use_lowest_point");
    detection_range_z_max_ = static_cast<float>(declare_parameter<double>("detection_range_z_max"));
    low_priority_region_x_ = static_cast<float>(declare_parameter<double>("low_priority_region_x"));
    center_pcl_shift_ = static_cast<float>(declare_parameter<double>("center_pcl_shift"));
    non_ground_height_threshold_ =
      static_cast<float>(declare_parameter<double>("non_ground_height_threshold"));

    // grid parameters
    grid_size_m_ = static_cast<float>(declare_parameter<double>("grid_size_m"));
    grid_mode_switch_radius_ =
      static_cast<float>(declare_parameter<double>("grid_mode_switch_radius"));
    gnd_grid_buffer_size_ = declare_parameter<int>("gnd_grid_buffer_size");
    virtual_lidar_z_ = vehicle_info_.vehicle_height_m;

    // initialize grid filter
    {
      GridGroundFilterParameter param;
      param.global_slope_max_angle_rad = global_slope_max_angle_rad_;
      param.local_slope_max_angle_rad = local_slope_max_angle_rad_;
      param.radial_divider_angle_rad = radial_divider_angle_rad_;

      param.use_recheck_ground_cluster = use_recheck_ground_cluster_;
      param.use_lowest_point = use_lowest_point_;
      param.detection_range_z_max = detection_range_z_max_;
      param.non_ground_height_threshold = non_ground_height_threshold_;

      param.grid_size_m = grid_size_m_;
      param.grid_mode_switch_radius = grid_mode_switch_radius_;
      param.gnd_grid_buffer_size = gnd_grid_buffer_size_;
      param.virtual_lidar_x = vehicle_info_.wheel_base_m / 2.0f + center_pcl_shift_;
      param.virtual_lidar_y = 0.0f;
      param.virtual_lidar_z = virtual_lidar_z_;

      grid_ground_filter_ptr_ = std::make_unique<GridGroundFilter>(param);
    }
  }

  using std::placeholders::_1;
  set_param_res_ = this->add_on_set_parameters_callback(
    std::bind(&ScanGroundFilterComponent::onParameter, this, _1));

  // initialize debug tool
  {
    using autoware_utils::DebugPublisher;
    using autoware_utils::StopWatch;
    stop_watch_ptr_ = std::make_unique<StopWatch<std::chrono::milliseconds>>();
    debug_publisher_ptr_ = std::make_unique<DebugPublisher>(this, "scan_ground_filter");
    stop_watch_ptr_->tic("cyclic_time");
    stop_watch_ptr_->tic("processing_time");

    bool use_time_keeper = declare_parameter<bool>("publish_processing_time_detail");
    if (use_time_keeper) {
      detailed_processing_time_publisher_ =
        this->create_publisher<autoware_utils::ProcessingTimeDetail>(
          "~/debug/processing_time_detail_ms", 1);
      auto time_keeper = autoware_utils::TimeKeeper(detailed_processing_time_publisher_);
      time_keeper_ = std::make_shared<autoware_utils::TimeKeeper>(time_keeper);

      // set time keeper to grid
      grid_ground_filter_ptr_->setTimeKeeper(time_keeper_);
    }
  }

  // NERO_GROUND_LEVEL (jolt, QUEUE PTH): classify and cut the band's ceiling in each cloud's road frame (unset = the car)
  {
    // a node created where an earlier one lived must not inherit its state (unit tests create several)
    std::lock_guard<std::mutex> lock(g_nero_ground_level_mutex);
    g_nero_ground_level.erase(this);
  }
  if (const char * sw = std::getenv("NERO_GROUND_LEVEL")) {
    auto gl = std::make_unique<NeroGroundLevelState>();
    if (elevation_grid_mode_ && gl->level.parse(sw)) {
      gl->ceiling = gl->level.ceiling > 0.0f ? gl->level.ceiling : static_cast<float>(vehicle_info_.vehicle_height_m);
      gl->x_front = static_cast<float>(vehicle_info_.max_longitudinal_offset_m);
      RCLCPP_INFO(
        get_logger(),
        "[NERO_GROUND_LEVEL] on: ground + band ceiling %.2f m over the road (road line from %.0f-%.0f m ahead, |y| < %.1f, "
        "tilt <= %.1f deg; the robot's own plane up to its front %.2f m; behind it not levelled), output in base_link",
        gl->ceiling, gl->level.x0, gl->level.x1, gl->level.yh, gl->level.max_tilt_deg, gl->x_front);
      std::lock_guard<std::mutex> lock(g_nero_ground_level_mutex);
      g_nero_ground_level[this] = std::move(gl);
    } else {
      RCLCPP_ERROR(get_logger(), "[NERO_GROUND_LEVEL] '%s' invalid or not elevation_grid_mode: switch off", sw);
    }
  }

  // NERO_GROUND_BAND_ROAD (rampground, QUEUE PITCHG): the obstacle band's ceiling from the road ahead (unset = the car)
  if (const char * sw = std::getenv("NERO_GROUND_BAND_ROAD")) {
    auto nb = std::make_unique<NeroBandRoadState>();
    if (elevation_grid_mode_ && nb->band_road.parse(sw)) {
      const char * dbg = std::getenv("NERO_GROUND_BAND_ROAD_DEBUG");
      nb->debug = dbg && std::atoi(dbg) > 0;
      const auto & b = nb->band_road;
      RCLCPP_INFO(
        get_logger(),
        "[NERO_GROUND_BAND_ROAD] on: band %.2f + margin %.2f over the road, support %.2f (ground %.2f..), gate %.2f m / "
        "%.1f deg, ahead within %.0f m and %.0f deg",
        b.band, b.margin, b.support_h, b.ground_low, b.gate, b.gate_deg, b.rmax, b.cone_deg);
      std::lock_guard<std::mutex> lock(g_nero_band_road_mutex);
      g_nero_band_road[this] = std::move(nb);
    } else {
      RCLCPP_ERROR(
        get_logger(), "[NERO_GROUND_BAND_ROAD] '%s' invalid or not elevation_grid_mode: switch off", sw);
    }
  }
}

void ScanGroundFilterComponent::convertPointcloud(
  const PointCloud2ConstPtr & in_cloud,
  std::vector<PointCloudVector> & out_radial_ordered_points) const
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  out_radial_ordered_points.resize(radial_dividers_num_);
  PointData current_point;

  const auto inv_radial_divider_angle_rad = 1.0f / radial_divider_angle_rad_;

  const size_t in_cloud_data_size = in_cloud->data.size();
  const size_t in_cloud_point_step = in_cloud->point_step;

  {  // grouping pointcloud by its azimuth angle
    std::unique_ptr<ScopedTimeTrack> inner_st_ptr;
    if (time_keeper_)
      inner_st_ptr = std::make_unique<ScopedTimeTrack>("azimuth_angle_grouping", *time_keeper_);

    pcl::PointXYZ input_point;
    for (size_t data_index = 0; data_index + in_cloud_point_step <= in_cloud_data_size;
         data_index += in_cloud_point_step) {
      // Get Point
      data_accessor_.getPoint(in_cloud, data_index, input_point);

      // determine the azimuth angle group
      auto radius{static_cast<float>(std::hypot(input_point.x, input_point.y))};
      auto theta{normalize_radian(std::atan2(input_point.x, input_point.y), 0.0)};
      auto radial_div{static_cast<size_t>(std::floor(theta * inv_radial_divider_angle_rad))};

      current_point.radius = radius;
      current_point.point_state = PointLabel::INIT;
      current_point.data_index = data_index;

      // store the point in the corresponding radial division
      out_radial_ordered_points[radial_div].emplace_back(current_point);
    }
  }

  {  // sorting pointcloud by distance, on each azimuth angle group
    std::unique_ptr<ScopedTimeTrack> inner_st_ptr;
    if (time_keeper_) inner_st_ptr = std::make_unique<ScopedTimeTrack>("sort", *time_keeper_);

    for (size_t i = 0; i < radial_dividers_num_; ++i) {
      std::sort(
        out_radial_ordered_points[i].begin(), out_radial_ordered_points[i].end(),
        [](const PointData & a, const PointData & b) { return a.radius < b.radius; });
    }
  }
}

void ScanGroundFilterComponent::calcVirtualGroundOrigin(pcl::PointXYZ & point) const
{
  point.x = vehicle_info_.wheel_base_m;
  point.y = 0;
  point.z = 0;
}

void ScanGroundFilterComponent::classifyPointCloud(
  const PointCloud2ConstPtr & in_cloud,
  const std::vector<PointCloudVector> & in_radial_ordered_clouds,
  pcl::PointIndices & out_no_ground_indices) const
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  out_no_ground_indices.indices.clear();

  const pcl::PointXYZ init_ground_point(0, 0, 0);
  pcl::PointXYZ virtual_ground_point(0, 0, 0);
  calcVirtualGroundOrigin(virtual_ground_point);

  // run the classification algorithm for each ray (azimuth division)
  for (size_t i = 0; i < in_radial_ordered_clouds.size(); ++i) {
    float prev_gnd_radius = 0.0f;
    float prev_gnd_slope = 0.0f;
    PointsCentroid ground_cluster, non_ground_cluster;
    PointLabel point_label_curr = PointLabel::INIT;

    pcl::PointXYZ prev_gnd_point(0, 0, 0), point_curr, point_prev;

    // iterate over the points in the ray
    for (size_t j = 0; j < in_radial_ordered_clouds[i].size(); ++j) {
      float points_distance = 0.0f;
      const float local_slope_max_angle = local_slope_max_angle_rad_;

      // set the previous point
      point_prev = point_curr;
      PointLabel point_label_prev = point_label_curr;

      // set the current point
      const PointData & pd = in_radial_ordered_clouds[i][j];
      point_label_curr = pd.point_state;

      data_accessor_.getPoint(in_cloud, pd.data_index, point_curr);
      if (j == 0) {
        bool is_front_side = (point_curr.x > virtual_ground_point.x);
        if (use_virtual_ground_point_ && is_front_side) {
          prev_gnd_point = virtual_ground_point;
        } else {
          prev_gnd_point = init_ground_point;
        }
        prev_gnd_radius = std::hypot(prev_gnd_point.x, prev_gnd_point.y);
        prev_gnd_slope = 0.0f;
        ground_cluster.initialize();
        non_ground_cluster.initialize();
        points_distance = calc_distance3d(point_curr, prev_gnd_point);
      } else {
        points_distance = calc_distance3d(point_curr, point_prev);
      }

      float radius_distance_from_gnd = pd.radius - prev_gnd_radius;
      float height_from_gnd = point_curr.z - prev_gnd_point.z;
      float height_from_obj = point_curr.z - non_ground_cluster.getAverageHeight();
      bool calculate_slope = true;
      bool is_point_close_to_prev =
        (points_distance <
         (pd.radius * radial_divider_angle_rad_ + split_points_distance_tolerance_));

      float global_slope_ratio = point_curr.z / pd.radius;
      // check points which is far enough from previous point
      if (global_slope_ratio > global_slope_max_ratio_) {
        point_label_curr = PointLabel::NON_GROUND;
        calculate_slope = false;
      } else if (
        (point_label_prev == PointLabel::NON_GROUND) &&
        (std::abs(height_from_obj) >= split_height_distance_)) {
        calculate_slope = true;
      } else if (is_point_close_to_prev && std::abs(height_from_gnd) < split_height_distance_) {
        // close to the previous point, set point follow label
        point_label_curr = PointLabel::POINT_FOLLOW;
        calculate_slope = false;
      }
      if (is_point_close_to_prev) {
        height_from_gnd = point_curr.z - ground_cluster.getAverageHeight();
        radius_distance_from_gnd = pd.radius - ground_cluster.getAverageRadius();
      }
      if (calculate_slope) {
        // far from the previous point
        auto local_slope = std::atan2(height_from_gnd, radius_distance_from_gnd);
        if (local_slope - prev_gnd_slope > local_slope_max_angle) {
          // the point is outside of the local slope threshold
          point_label_curr = PointLabel::NON_GROUND;
        } else {
          point_label_curr = PointLabel::GROUND;
        }
      }

      if (point_label_curr == PointLabel::GROUND) {
        ground_cluster.initialize();
        non_ground_cluster.initialize();
      }
      if (point_label_curr == PointLabel::NON_GROUND) {
        out_no_ground_indices.indices.push_back(pd.data_index);
      } else if (  // NOLINT
        (point_label_prev == PointLabel::NON_GROUND) &&
        (point_label_curr == PointLabel::POINT_FOLLOW)) {
        point_label_curr = PointLabel::NON_GROUND;
        out_no_ground_indices.indices.push_back(pd.data_index);
      } else if (  // NOLINT
        (point_label_prev == PointLabel::GROUND) &&
        (point_label_curr == PointLabel::POINT_FOLLOW)) {
        point_label_curr = PointLabel::GROUND;
      } else {
      }

      // update the ground state
      if (point_label_curr == PointLabel::GROUND) {
        prev_gnd_radius = pd.radius;
        prev_gnd_point = pcl::PointXYZ(point_curr.x, point_curr.y, point_curr.z);
        ground_cluster.addPoint(pd.radius, point_curr.z);
        prev_gnd_slope = ground_cluster.getAverageSlope();
      }
      // update the non ground state
      if (point_label_curr == PointLabel::NON_GROUND) {
        non_ground_cluster.addPoint(pd.radius, point_curr.z);
      }
    }
  }
}

void ScanGroundFilterComponent::extractObjectPoints(
  const PointCloud2ConstPtr & in_cloud_ptr, const pcl::PointIndices & in_indices,
  PointCloud2 & out_object_cloud) const
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  size_t output_data_size = 0;

  for (const auto & idx : in_indices.indices) {
    std::memcpy(
      &out_object_cloud.data[output_data_size], &in_cloud_ptr->data[idx],
      in_cloud_ptr->point_step * sizeof(uint8_t));
    output_data_size += in_cloud_ptr->point_step;
  }
}

void ScanGroundFilterComponent::faster_filter(
  const PointCloud2ConstPtr & input, [[maybe_unused]] const IndicesPtr & indices,
  PointCloud2 & output,
  [[maybe_unused]] const autoware::pointcloud_preprocessor::TransformInfo & transform_info)
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  std::scoped_lock lock(mutex_);
  if (stop_watch_ptr_) stop_watch_ptr_->toc("processing_time", true);

  if (!data_accessor_.isInitialized()) {
    data_accessor_.setField(input);
    grid_ground_filter_ptr_->setDataAccessor(input);
  }

  pcl::PointIndices no_ground_indices;

  if (elevation_grid_mode_) {
    auto * gl = neroGroundLevel(this);
    const auto & fl = input->fields;
    if (
      gl && input->point_step % 4 == 0 && fl.size() >= 3 && fl[0].name == "x" && fl[0].offset == 0 && fl[1].name == "y" &&
      fl[1].offset == 4 && fl[2].name == "z" && fl[2].offset == 8) {
      // NERO_GROUND_LEVEL: the points under the ceiling, rotated into this cloud's road frame (in base_link up to 3 m ahead, blended
      // to the road frame by 6 m: jg9), are classified (on a level body = the car's input exactly); the kept points are taken from the
      // input (base_link)
      const size_t n = input->width * input->height, ps = input->point_step, stride = ps / 4;
      const auto c0 = std::chrono::steady_clock::now();
      gl->level.fit(reinterpret_cast<const float *>(input->data.data()), n, stride);
      const auto c1 = std::chrono::steady_clock::now();
      if (!gl->buf) gl->buf = std::make_shared<PointCloud2>();
      auto & lev = *gl->buf;
      lev.header = input->header; lev.fields = input->fields; lev.is_bigendian = input->is_bigendian;
      lev.point_step = input->point_step; lev.is_dense = input->is_dense; lev.height = 1;
      if (lev.data.size() < input->data.size()) lev.data.resize(input->data.size());
      auto & src = gl->src; src.clear(); src.reserve(n);
      size_t m = 0, over = 0;
      const auto * in = input->data.data();
      auto * out = lev.data.data();
      const auto & L = gl->level;
      const float ceil_z = gl->ceiling;
      // NERO_GROUND_BAND_ROAD (rampground, on the car) runs on the kept points in base_link, as on the car (jolt-5): its gate
      // ("the road ahead lies below the body's plane") and its road profile are base_link rules; in the road frame its gate never
      // opens, and the car's ceiling kept under jolt-4's rule let the foliage it removes on the car back in (w31 20:48:03.9)
      auto * nbr = neroBandRoad(this);
      auto & bx = gl->bxyz;
      if (nbr && bx.size() < 3 * n) bx.resize(3 * n);  // never shrinks: no per-cloud allocation / zeroing
      float * bp = nbr ? bx.data() : nullptr;
      if (L.ok) {
        const float x_front = gl->x_front;
        for (size_t k = 0; k < n; ++k) {
          float q[3], z1 = 0.0f;
          std::memcpy(q, in + k * ps, 12);
          if (!(q[0] > 0.0f)) {  // behind the robot: not levelled, the car's band (jolt-3: a road rising ahead lifted it)
            if (!(q[2] < ceil_z)) { ++over; continue; }
            std::memcpy(out + m * ps, in + k * ps, ps);
            src.push_back(k * ps); ++m;
            if (bp) std::memcpy(bp + 3 * (m - 1), q, 12);
            continue;
          }
          const float z = L.road_z(q, z1);
          // over the road the point stands on (jolt-3) AND over the car's own base_link ceiling (jolt-4): the levelled band only ever
          // adds to the car's. Where the road falls away (a crown toward a wall, a dip, the body nose-up) the ceiling over the local
          // road sits lower in base_link and cut wall / car parts the car keeps (lidarinsert s1_087 recess: a low object's track lost)
          if (!(z - L.ref_z(q, x_front) < ceil_z) && !(q[2] < ceil_z)) { ++over; continue; }
          if (bp) std::memcpy(bp + 3 * m, q, 12);  // base_link, before the rotation
          // jolt-8 (jg9): classified in base_link near the robot, in the road frame from xn1 ahead (nero_ground_level.hpp)
          float c[3];
          L.classify_xyz(q, z, z1, c);
          std::memcpy(out + m * ps, in + k * ps, ps);
          std::memcpy(out + m * ps, c, 12);
          src.push_back(k * ps); ++m;
        }
      } else {
        // no trusted road plane: the base_link band exactly as the car's crop box (z < vehicle height)
        for (size_t k = 0; k < n; ++k) {
          float z;
          std::memcpy(&z, in + k * ps + 8, 4);
          if (!(z < ceil_z)) { ++over; continue; }
          std::memcpy(out + m * ps, in + k * ps, ps);
          src.push_back(k * ps); ++m;
          if (bp) std::memcpy(bp + 3 * (m - 1), in + k * ps, 12);
        }
      }
      lev.width = static_cast<uint32_t>(m); lev.row_step = static_cast<uint32_t>(m * ps);
      // the grid filter walks data.size(): give it exactly m points (shrinking keeps the capacity)
      lev.data.resize(m * ps);
      const auto c2 = std::chrono::steady_clock::now();
      PointCloud2ConstPtr lev_c = gl->buf;
      grid_ground_filter_ptr_->process(lev_c, no_ground_indices);
      if (nbr) neroBandRoadOn(*nbr, bx.data(), 3, m, ps, input->header.stamp, no_ground_indices, get_logger());
      for (auto & ix : no_ground_indices.indices) ix = static_cast<int>(src[static_cast<size_t>(ix) / ps]);
      const auto c3 = std::chrono::steady_clock::now();
      gl->ms_fit += std::chrono::duration<double, std::milli>(c1 - c0).count();
      gl->ms_lev += std::chrono::duration<double, std::milli>(c2 - c1).count();
      gl->ms_cls += std::chrono::duration<double, std::milli>(c3 - c2).count();
      gl->frames++; gl->levelled += gl->level.ok ? 1 : 0; gl->cut += over;
      if (gl->level.ok) {  // trusted fits only (an untrusted one keeps its too-steep angles for nothing)
        gl->max_pitch = std::max(gl->max_pitch, std::abs(gl->level.pitch)); gl->max_roll = std::max(gl->max_roll, std::abs(gl->level.roll));
      }
      const auto now = std::chrono::steady_clock::now();
      if (now - gl->last_log > std::chrono::seconds(30)) {
        RCLCPP_INFO(
          get_logger(), "[NERO_GROUND_LEVEL] %zu clouds (%zu levelled, max |pitch| %.1f |roll| %.1f deg), %zu points over the ceiling; "
          "mean ms fit %.2f level %.2f classify %.2f",
          gl->frames, gl->levelled, gl->max_pitch * 57.2958f, gl->max_roll * 57.2958f, gl->cut, gl->ms_fit / std::max<size_t>(1, gl->frames),
          gl->ms_lev / std::max<size_t>(1, gl->frames), gl->ms_cls / std::max<size_t>(1, gl->frames));
        gl->frames = gl->levelled = gl->cut = 0; gl->max_pitch = gl->max_roll = 0.0f; gl->ms_fit = gl->ms_lev = gl->ms_cls = 0.0;
        gl->last_log = now;
      }
    } else {
      grid_ground_filter_ptr_->process(input, no_ground_indices);
      if (auto * nb = neroBandRoad(this)) neroApplyBandRoad(*nb, input, no_ground_indices, get_logger());
    }
  } else {
    std::vector<PointCloudVector> radial_ordered_points;
    convertPointcloud(input, radial_ordered_points);
    classifyPointCloud(input, radial_ordered_points, no_ground_indices);
  }
  output.row_step = no_ground_indices.indices.size() * input->point_step;
  output.data.resize(output.row_step);
  output.width = no_ground_indices.indices.size();
  output.fields = input->fields;
  output.is_dense = true;
  output.height = input->height;
  output.is_bigendian = input->is_bigendian;
  output.point_step = input->point_step;
  output.header = input->header;

  extractObjectPoints(input, no_ground_indices, output);
  if (debug_publisher_ptr_ && stop_watch_ptr_) {
    const double cyclic_time_ms = stop_watch_ptr_->toc("cyclic_time", true);
    const double processing_time_ms = stop_watch_ptr_->toc("processing_time", true);
    debug_publisher_ptr_->publish<autoware_internal_debug_msgs::msg::Float64Stamped>(
      "debug/cyclic_time_ms", cyclic_time_ms);
    debug_publisher_ptr_->publish<autoware_internal_debug_msgs::msg::Float64Stamped>(
      "debug/processing_time_ms", processing_time_ms);
  }
}

// TODO(taisa1): Temporary Implementation: Delete this function definition when all the filter
// nodes conform to new API.
void ScanGroundFilterComponent::filter(
  const PointCloud2ConstPtr & input, [[maybe_unused]] const IndicesPtr & indices,
  PointCloud2 & output)
{
  (void)input;
  (void)indices;
  (void)output;
}

rcl_interfaces::msg::SetParametersResult ScanGroundFilterComponent::onParameter(
  const std::vector<rclcpp::Parameter> & param)
{
  if (get_param(param, "grid_size_m", grid_size_m_)) {
    // grid_ptr_->initialize(grid_size_m_, radial_divider_angle_rad_, grid_mode_switch_radius_);
  }
  if (get_param(param, "grid_mode_switch_radius", grid_mode_switch_radius_)) {
    // grid_ptr_->initialize(grid_size_m_, radial_divider_angle_rad_, grid_mode_switch_radius_);
  }
  double global_slope_max_angle_deg{get_parameter("global_slope_max_angle_deg").as_double()};
  if (get_param(param, "global_slope_max_angle_deg", global_slope_max_angle_deg)) {
    global_slope_max_angle_rad_ = deg2rad(global_slope_max_angle_deg);
    global_slope_max_ratio_ = std::tan(global_slope_max_angle_rad_);
    RCLCPP_DEBUG(
      get_logger(), "Setting global_slope_max_angle_rad to: %f.", global_slope_max_angle_rad_);
    RCLCPP_DEBUG(get_logger(), "Setting global_slope_max_ratio to: %f.", global_slope_max_ratio_);
  }
  double local_slope_max_angle_deg{get_parameter("local_slope_max_angle_deg").as_double()};
  if (get_param(param, "local_slope_max_angle_deg", local_slope_max_angle_deg)) {
    local_slope_max_angle_rad_ = deg2rad(local_slope_max_angle_deg);
    local_slope_max_ratio_ = std::tan(local_slope_max_angle_rad_);
    RCLCPP_DEBUG(
      get_logger(), "Setting local_slope_max_angle_rad to: %f.", local_slope_max_angle_rad_);
    RCLCPP_DEBUG(get_logger(), "Setting local_slope_max_ratio to: %f.", local_slope_max_ratio_);
  }
  double radial_divider_angle_deg{get_parameter("radial_divider_angle_deg").as_double()};
  if (get_param(param, "radial_divider_angle_deg", radial_divider_angle_deg)) {
    radial_divider_angle_rad_ = deg2rad(radial_divider_angle_deg);
    radial_dividers_num_ = std::ceil(2.0 * M_PI / radial_divider_angle_rad_);
    // grid_ptr_->initialize(grid_size_m_, radial_divider_angle_rad_, grid_mode_switch_radius_);
    RCLCPP_DEBUG(
      get_logger(), "Setting radial_divider_angle_rad to: %f.", radial_divider_angle_rad_);
    RCLCPP_DEBUG(get_logger(), "Setting radial_dividers_num to: %zu.", radial_dividers_num_);
  }
  if (get_param(param, "split_points_distance_tolerance", split_points_distance_tolerance_)) {
    RCLCPP_DEBUG(
      get_logger(), "Setting split_points_distance_tolerance to: %f.",
      split_points_distance_tolerance_);
  }
  if (get_param(param, "split_height_distance", split_height_distance_)) {
    RCLCPP_DEBUG(get_logger(), "Setting split_height_distance to: %f.", split_height_distance_);
  }
  if (get_param(param, "use_virtual_ground_point", use_virtual_ground_point_)) {
    RCLCPP_DEBUG_STREAM(
      get_logger(),
      "Setting use_virtual_ground_point to: " << std::boolalpha << use_virtual_ground_point_);
  }
  if (get_param(param, "use_recheck_ground_cluster", use_recheck_ground_cluster_)) {
    RCLCPP_DEBUG_STREAM(
      get_logger(),
      "Setting use_recheck_ground_cluster to: " << std::boolalpha << use_recheck_ground_cluster_);
  }
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";

  return result;
}

}  // namespace autoware::ground_segmentation

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::ground_segmentation::ScanGroundFilterComponent)
