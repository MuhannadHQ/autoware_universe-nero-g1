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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>

namespace autoware::near_weak_return_filter
{
namespace
{
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

std::optional<PointField> field(const PointCloud2 & cloud, const std::string & name)
{
  for (const auto & f : cloud.fields) {
    if (f.name == name && f.count >= 1) return f;
  }
  return std::nullopt;
}

float read_float(const uint8_t * p, const PointField & f)
{
  float v;
  std::memcpy(&v, p + f.offset, sizeof(float));
  return v;
}

// intensity as a number, whatever its stored type
std::optional<double> read_number(const uint8_t * p, const PointField & f)
{
  const uint8_t * q = p + f.offset;
  switch (f.datatype) {
    case PointField::UINT8:
      return static_cast<double>(*q);
    case PointField::INT8:
      return static_cast<double>(*reinterpret_cast<const int8_t *>(q));
    case PointField::UINT16: {
      uint16_t v;
      std::memcpy(&v, q, sizeof(v));
      return v;
    }
    case PointField::INT16: {
      int16_t v;
      std::memcpy(&v, q, sizeof(v));
      return v;
    }
    case PointField::UINT32: {
      uint32_t v;
      std::memcpy(&v, q, sizeof(v));
      return v;
    }
    case PointField::INT32: {
      int32_t v;
      std::memcpy(&v, q, sizeof(v));
      return v;
    }
    case PointField::FLOAT32: {
      float v;
      std::memcpy(&v, q, sizeof(v));
      return v;
    }
    case PointField::FLOAT64: {
      double v;
      std::memcpy(&v, q, sizeof(v));
      return v;
    }
    default:
      return std::nullopt;
  }
}

int64_t cell_key(int64_t ix, int64_t iy, int64_t iz)
{
  constexpr int64_t off = 1 << 20;
  return ((ix + off) << 42) | ((iy + off) << 21) | (iz + off);
}
}  // namespace

std::optional<std::vector<bool>> find_dropped(const PointCloud2 & cloud, const Params & params)
{
  const auto fx = field(cloud, "x");
  const auto fy = field(cloud, "y");
  const auto fz = field(cloud, "z");
  const auto fi = field(cloud, "intensity");
  const auto fd = field(cloud, "distance");
  if (!fx || !fy || !fz || !fi || !fd) return std::nullopt;
  for (const auto & f : {*fx, *fy, *fz, *fd}) {
    if (f.datatype != PointField::FLOAT32) return std::nullopt;
  }
  switch (fi->datatype) {
    case PointField::INT8:
    case PointField::UINT8:
    case PointField::INT16:
    case PointField::UINT16:
    case PointField::INT32:
    case PointField::UINT32:
    case PointField::FLOAT32:
    case PointField::FLOAT64:
      break;
    default:
      return std::nullopt;
  }
  const size_t n = static_cast<size_t>(cloud.width) * cloud.height;
  if (cloud.point_step == 0 || cloud.data.size() < n * cloud.point_step) return std::nullopt;

  std::vector<bool> drop(n, false);
  const double r_max = params.max_distance;
  const double cell = params.neighbor_radius_ratio * r_max;  // the largest neighbour radius
  if (n == 0 || r_max <= 0.0 || cell <= 0.0 || params.max_neighbors < 0) return drop;

  // candidates: near and weak
  std::vector<uint32_t> cand;
  for (size_t i = 0; i < n; ++i) {
    const uint8_t * p = cloud.data.data() + i * cloud.point_step;
    const float d = read_float(p, *fd);
    if (!(d < r_max)) continue;  // also skips NaN
    const auto inten = read_number(p, *fi);
    if (!inten || !(*inten <= params.max_intensity)) continue;
    cand.push_back(static_cast<uint32_t>(i));
  }
  if (cand.empty()) return drop;

  // neighbour pool: every point whose range can put it within a candidate's radius (any intensity)
  struct P
  {
    float x, y, z;
    uint32_t idx;
  };
  std::unordered_map<int64_t, std::vector<P>> grid;
  const double pool_r = r_max + cell;
  auto idx_of = [cell](float v) { return static_cast<int64_t>(std::floor(v / cell)); };
  for (size_t i = 0; i < n; ++i) {
    const uint8_t * p = cloud.data.data() + i * cloud.point_step;
    const float d = read_float(p, *fd);
    if (!(d < pool_r)) continue;
    const float x = read_float(p, *fx), y = read_float(p, *fy), z = read_float(p, *fz);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
    grid[cell_key(idx_of(x), idx_of(y), idx_of(z))].push_back({x, y, z, static_cast<uint32_t>(i)});
  }

  for (const auto i : cand) {
    const uint8_t * p = cloud.data.data() + static_cast<size_t>(i) * cloud.point_step;
    const float x = read_float(p, *fx), y = read_float(p, *fy), z = read_float(p, *fz);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
    const double rho = params.neighbor_radius_ratio * read_float(p, *fd);
    const double rho2 = rho * rho;
    const int64_t cx = idx_of(x), cy = idx_of(y), cz = idx_of(z);
    int count = 0;
    for (int64_t ix = cx - 1; ix <= cx + 1 && count <= params.max_neighbors; ++ix) {
      for (int64_t iy = cy - 1; iy <= cy + 1 && count <= params.max_neighbors; ++iy) {
        for (int64_t iz = cz - 1; iz <= cz + 1 && count <= params.max_neighbors; ++iz) {
          const auto it = grid.find(cell_key(ix, iy, iz));
          if (it == grid.end()) continue;
          for (const auto & q : it->second) {
            if (q.idx == i) continue;
            const double dx = q.x - x, dy = q.y - y, dz = q.z - z;
            if (dx * dx + dy * dy + dz * dz <= rho2 && ++count > params.max_neighbors) break;
          }
        }
      }
    }
    if (count <= params.max_neighbors) drop[i] = true;
  }
  return drop;
}

PointCloud2 remove_points(const PointCloud2 & cloud, const std::vector<bool> & drop, bool keep_dropped)
{
  PointCloud2 out;
  out.header = cloud.header;
  out.fields = cloud.fields;
  out.is_bigendian = cloud.is_bigendian;
  out.point_step = cloud.point_step;
  out.is_dense = cloud.is_dense;
  out.height = 1;
  const size_t n = static_cast<size_t>(cloud.width) * cloud.height;
  size_t kept = 0;
  for (size_t i = 0; i < n; ++i) kept += (drop[i] == keep_dropped) ? 1 : 0;
  out.data.resize(kept * cloud.point_step);
  size_t o = 0;
  for (size_t i = 0; i < n; ++i) {
    if (drop[i] != keep_dropped) continue;
    std::memcpy(out.data.data() + o * cloud.point_step, cloud.data.data() + i * cloud.point_step, cloud.point_step);
    ++o;
  }
  out.width = static_cast<uint32_t>(kept);
  out.row_step = out.width * out.point_step;
  return out;
}

}  // namespace autoware::near_weak_return_filter
