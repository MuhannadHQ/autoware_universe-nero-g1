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
#include <cstdint>
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

// [nero 2026-11-05 AEBD2] Dense dust in front of the bumper survives the stage above (its core has > max_neighbors
// neighbours by design) and still made AEB emergency-stop the robot (10-29 19:00:42 / 19:01:00 / 19:03:01, 11-01
// 16:20, 11-04 14:46 / 18:21). This second stage runs on what the first one keeps. In the E1R's range image
// (its own azimuth / elevation fields) a real surface, black or bright, is a connected patch: neighbouring pixels
// return at consistent ranges. Dust is volumetric: neighbouring pixels hit different particles at random ranges,
// so its returns form only tiny, scattered patches (10-29: median 2-5 returns, the largest 164; real objects near
// the E1R: segments of thousands). A return is DUST when ALL hold:
//  - near, in front, above the road plane and at most max_z high (base_link: x > min_x, |y| <= max_abs_y,
//    min_z < z <= max_z = where AEB and the obstacle crop look), distance < max_distance, intensity <= max_intensity,
//  - not on a structure: not in a 'surface' segment (connected returns with >= surface_core range-consistent
//    neighbours, >= surface_min of them; a touching return joins it) and not in a 'line' segment (>= line_core
//    consistent neighbours, i.e. a row of the road at grazing incidence, a rope, a cable) of >= line_min returns or
//    spread >= line_extent m (std along its main axis),
//  - its neighbourhood is dust: in its local_radius voxel and the 26 around it, such returns are >= local_share of
//    all returns (a real object's edge returns sit next to its surface and are kept), and neither all returns there
//    nor the dust-like ones alone lie on a plane (smallest / largest eigenvalue of their spread < planar_max: the
//    road at grazing incidence, which the range image breaks into short rows, and a dark surface with dropouts are
//    planes there; a dust cloud is a volume),
//  - the frame is a dust frame. A burst opens it: >= frame_min such returns at intensity <= gate_max_intensity that
//    are also >= gate_weak_share of all such returns (dust is weak; a car's grazing side with its reflectors and tyres
//    breaks up in the range image too, but is mostly bright: 11-03 17:56:10.7, weak share 0.24). It stays a dust
//    frame for hold_sec while the frame still has >= hold_min weak ones (the bright part of the same cloud).
// Anything else passes untouched; outside dust frames nothing is ever removed.
// [nero 2026-11-06 AEBD2, aebdust_1106] The same cloud also reaches beyond that box: E1R dust at |y| 1.3-2.0 (11-03
// 14:42:27) and the side Airys' view of it (11-04 14:46:32 / 18:21:19, the left Airy) still made obstacle_stop stop the
// robot for 0.7-1.4 s. (1) The box is wider (max_abs_y) while the frame gate still counts only its proven volume
// (gate_max_abs_y). (2) A side lidar runs the same tests in FOLLOWER mode: it has no gate of its own (it sees too little
// of the cloud to tell dust from a burst of anything else) and acts only while the front E1R's dust gate is open
// (an E1R frame within follow_sec of its cloud, the E1R's gate published by its own stage); else nothing is removed.
struct DustParams
{
  bool enabled{false};
  double max_distance{2.5};
  int max_intensity{255};
  double min_z{0.08};            // [m] base_link: the road and anything at it are never candidates
  double max_z{0.6};             // [m] base_link: above AEB's and the obstacle crop's ceiling (0.52) nothing is touched
  double min_x{1.0};             // [m] base_link
  double max_abs_y{2.0};         // [m] base_link (aebdust_1105: 1.3)
  double gate_max_abs_y{1.3};    // [m] base_link: the frame gate counts only returns with |y| <= this (< 0: max_abs_y)
  bool follow{false};            // follower mode (a side lidar): no own gate, acts only while the external gate is open
  double follow_sec{0.3};        // [s] follower: the external gate is open for a cloud stamped within this of an open frame
  double angle_radius{0.015};    // [rad] azimuth / elevation neighbourhood (~1.5 E1R pixels)
  double range_tol_abs{0.02};    // [m] consistent range: |d1 - d2| < abs + rel * min(d1, d2)
  double range_tol_rel{0.02};
  int surface_core{3};
  int surface_min{20};
  int line_core{2};
  int line_min{200};
  double line_extent{0.12};      // [m]
  double segment_range{3.5};     // [m] returns used for the segments (structures reaching beyond max_distance)
  double segment_margin{-1.0};   // [m] if >= 0: only returns within this of the dust box (x, y, z) go into the segments (a side
                                 // lidar's ground beside the robot is most of its near returns); < 0: all within segment_range
  double local_radius{0.1};      // [m] voxel size; the neighbourhood is the 3 x 3 x 3 voxels around
  double local_share{0.5};
  double planar_max{0.02};
  int planar_min_points{8};      // a plane needs at least this many returns (all returns / the dust-like / weak ones)
  bool planar_subsets{true};     // also test the dust-like / weak returns alone for a plane (a dark surface with dropouts)
  int frame_min{60};
  int gate_max_intensity{7};
  double gate_weak_share{0.5};
  int hold_min{3};               // aebdust_1105: 10 (the tail of a cloud with 3-9 weak returns left kept 9-15 dust returns)
  double hold_sec{1.0};
};

struct DustStats
{
  size_t candidates{0};     // near, not on a structure
  size_t dominant{0};       // ... in a dust neighbourhood
  size_t gate_count{0};     // ... at intensity <= gate_max_intensity, |y| <= gate_max_abs_y (the frame gate's count)
  size_t gate_dominant{0};  // dominant ones with |y| <= gate_max_abs_y (the gate's weak share is taken of these)
  bool gate_open{false};
};

// Offline diagnostics only (find_dust's optional output): why each return of the cloud was kept or not.
enum DustReason : uint8_t {
  kOutside = 0,       // not in the dust volume
  kDust = 1,          // dust-like (dropped when the gate is open)
  kSurface = 2,       // on a surface segment of the range image
  kLine = 3,          // on a line segment
  kBright = 4,        // intensity above max_intensity
  kNotDominant = 5,   // its neighbourhood is mostly other returns
  kPlane = 6,         // its neighbourhood is a plane
  kEarlyOut = 7,      // in the volume, frame returned early (too few weak returns for any gate)
};

// true = dust (independent of the frame gate: the caller drops them only when the gate is open).
// std::nullopt when the cloud lacks x / y / z / intensity / distance / azimuth / elevation as float32 (+ intensity).
// reasons (optional, offline tools): per return of the cloud a DustReason.
std::optional<std::vector<bool>> find_dust(
  const sensor_msgs::msg::PointCloud2 & cloud, const DustParams & params, DustStats & stats,
  std::vector<uint8_t> * reasons = nullptr);

// The frame gate with its hold (or, in follower mode, the external gate), shared by the node and the offline tools.
class DustStage
{
public:
  explicit DustStage(const DustParams & params) : params_(params) {}
  // drop mask for this frame (all false while the gate is closed); std::nullopt as find_dust. In follower mode a closed
  // external gate returns an all-false mask without looking at the cloud.
  std::optional<std::vector<bool>> process(
    const sensor_msgs::msg::PointCloud2 & cloud, double stamp_sec, DustStats & stats,
    std::vector<uint8_t> * reasons = nullptr);
  // follower mode: an open frame of the external gate (the front E1R's stage), by its cloud stamp
  void external_gate_open(double stamp_sec);
  // follower mode: is the external gate open for a cloud stamped stamp_sec
  bool external_open_at(double stamp_sec) const;
  const DustParams & params() const { return params_; }

private:
  DustParams params_;
  double last_dust_frame_{-1e18};
  double last_external_open_{-1e18};
};

// The input with the dropped points removed; every kept point's bytes and the order are unchanged.
sensor_msgs::msg::PointCloud2 remove_points(
  const sensor_msgs::msg::PointCloud2 & cloud, const std::vector<bool> & drop, bool keep_dropped = false);

}  // namespace autoware::near_weak_return_filter

#endif  // AUTOWARE__NEAR_WEAK_RETURN_FILTER__NEAR_WEAK_RETURN_FILTER_HPP_
