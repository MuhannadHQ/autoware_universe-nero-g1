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

#ifndef AUTOWARE__NEAR_WEAK_RETURN_FILTER__NEAR_WEAK_RETURN_FILTER_HPP_
#define AUTOWARE__NEAR_WEAK_RETURN_FILTER__NEAR_WEAK_RETURN_FILTER_HPP_

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <cstddef>
#include <optional>
#include <vector>

namespace autoware::near_weak_return_filter
{

// [nero 2026-10-28 AEBD] Weak returns of the front E1R 0.4-0.75 m in front of the bumper (dust or near-range noise)
// made AEB emergency-stop the robot at 11.6 km/h (10-27 20:47:49.7). A point is dropped only if ALL hold:
//  - its own `distance` field (range from the sensor) is below max_distance,
//  - its intensity is at most max_intensity,
//  - at most max_neighbors other points lie within neighbor_radius_ratio x its range (3D): it is not part of a
//    surface. A real surface, black ones included, returns in every pixel it covers, so its returns have
//    neighbours; dust returns are scattered. The idea (sparse AND weak = noise) is upstream's polar voxel noise
//    filter (autoware_universe #12496, simple mode), here per point and only near the sensor.
struct Params
{
  double max_distance{1.5};
  int max_intensity{3};
  double neighbor_radius_ratio{0.07};
  int max_neighbors{10};
};

// true = drop. std::nullopt when the cloud lacks x / y / z / intensity / distance (the caller then passes it through).
std::optional<std::vector<bool>> find_dropped(
  const sensor_msgs::msg::PointCloud2 & cloud, const Params & params);

// The input with the dropped points removed; every kept point's bytes and the order are unchanged.
sensor_msgs::msg::PointCloud2 remove_points(
  const sensor_msgs::msg::PointCloud2 & cloud, const std::vector<bool> & drop, bool keep_dropped = false);

}  // namespace autoware::near_weak_return_filter

#endif  // AUTOWARE__NEAR_WEAK_RETURN_FILTER__NEAR_WEAK_RETURN_FILTER_HPP_
