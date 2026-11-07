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

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstring>
#include <string>
#include <array>
#include <unordered_map>
#include <utility>

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

// A dense grid over the occupied cell box, filled by a stable counting sort (no per-cell allocation): the same cells and
// the same order of points inside a cell as the hash-of-vectors it replaces, O(1) lookups.
struct DenseGrid
{
  int64_t lo[3]{0, 0, 0}, dim[3]{1, 1, 1};
  std::vector<uint32_t> start, items;
  bool ok{false};
  // keys: per element its integer cell (3 coordinates; 2D grids pass 0 as the third)
  void build(const std::vector<std::array<int64_t, 3>> & keys, size_t max_cells = 8000000)
  {
    ok = false;
    start.clear();
    items.clear();
    if (keys.empty()) {
      ok = true;
      dim[0] = dim[1] = dim[2] = 0;
      return;
    }
    int64_t hi[3];
    for (int a = 0; a < 3; ++a) lo[a] = hi[a] = keys[0][a];
    for (const auto & k : keys)
      for (int a = 0; a < 3; ++a) {
        lo[a] = std::min(lo[a], k[a]);
        hi[a] = std::max(hi[a], k[a]);
      }
    size_t cells = 1;
    for (int a = 0; a < 3; ++a) {
      dim[a] = hi[a] - lo[a] + 1;
      cells *= static_cast<size_t>(dim[a]);
      if (cells > max_cells) return;  // caller falls back
    }
    start.assign(cells + 1, 0);
    std::vector<uint32_t> cell_of(keys.size());
    for (size_t e = 0; e < keys.size(); ++e) {
      cell_of[e] = static_cast<uint32_t>(index(keys[e][0], keys[e][1], keys[e][2]));
      ++start[cell_of[e] + 1];
    }
    for (size_t c = 0; c < cells; ++c) start[c + 1] += start[c];
    items.resize(keys.size());
    std::vector<uint32_t> fill(start.begin(), start.end() - 1);
    for (size_t e = 0; e < keys.size(); ++e) items[fill[cell_of[e]]++] = static_cast<uint32_t>(e);
    ok = true;
  }
  int64_t index(int64_t x, int64_t y, int64_t z) const
  {
    return ((x - lo[0]) * dim[1] + (y - lo[1])) * dim[2] + (z - lo[2]);
  }
  bool inside(int64_t x, int64_t y, int64_t z) const
  {
    return x >= lo[0] && x < lo[0] + dim[0] && y >= lo[1] && y < lo[1] + dim[1] && z >= lo[2] && z < lo[2] + dim[2];
  }
  // [begin, end) into items for a cell, empty outside the box
  std::pair<const uint32_t *, const uint32_t *> cell(int64_t x, int64_t y, int64_t z) const
  {
    if (!inside(x, y, z)) return {nullptr, nullptr};
    const auto c = static_cast<size_t>(index(x, y, z));
    return {items.data() + start[c], items.data() + start[c + 1]};
  }
};

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
  const double pool_r = r_max + cell;
  auto idx_of = [cell](float v) { return static_cast<int64_t>(std::floor(v / cell)); };
  std::vector<P> pool;
  std::vector<std::array<int64_t, 3>> keys;
  for (size_t i = 0; i < n; ++i) {
    const uint8_t * p = cloud.data.data() + i * cloud.point_step;
    const float d = read_float(p, *fd);
    if (!(d < pool_r)) continue;
    const float x = read_float(p, *fx), y = read_float(p, *fy), z = read_float(p, *fz);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
    pool.push_back({x, y, z, static_cast<uint32_t>(i)});
    keys.push_back({idx_of(x), idx_of(y), idx_of(z)});
  }
  DenseGrid dg;
  dg.build(keys);
  std::unordered_map<int64_t, std::vector<P>> grid;  // fallback only (a pool spread over > 8 M cells)
  if (!dg.ok) {
    for (size_t e = 0; e < pool.size(); ++e) grid[cell_key(keys[e][0], keys[e][1], keys[e][2])].push_back(pool[e]);
  }

  for (const auto i : cand) {
    const uint8_t * p = cloud.data.data() + static_cast<size_t>(i) * cloud.point_step;
    const float x = read_float(p, *fx), y = read_float(p, *fy), z = read_float(p, *fz);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
    const double rho = params.neighbor_radius_ratio * read_float(p, *fd);
    const double rho2 = rho * rho;
    const int64_t cx = idx_of(x), cy = idx_of(y), cz = idx_of(z);
    int count = 0;
    if (dg.ok) {
      // the candidate's own cell first (most of its neighbours), then the 26 around: the count reaches its exit sooner;
      // the decision (more than max_neighbors or not) does not depend on the order
      for (int c = 0; c < 27 && count <= params.max_neighbors; ++c) {
        const int o = (c + 13) % 27;  // 13 = the centre cell (0, 0, 0)
        const auto r = dg.cell(cx + o / 9 - 1, cy + (o / 3) % 3 - 1, cz + o % 3 - 1);
        for (const uint32_t * e = r.first; e != r.second; ++e) {
          const auto & q = pool[*e];
          if (q.idx == i) continue;
          const double dx = q.x - x, dy = q.y - y, dz = q.z - z;
          if (dx * dx + dy * dy + dz * dz <= rho2 && ++count > params.max_neighbors) break;
        }
      }
    }
    for (int64_t ix = cx - 1; !dg.ok && ix <= cx + 1 && count <= params.max_neighbors; ++ix) {
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

namespace
{
struct UnionFind
{
  std::vector<uint32_t> parent;
  explicit UnionFind(size_t n) : parent(n)
  {
    for (size_t i = 0; i < n; ++i) parent[i] = static_cast<uint32_t>(i);
  }
  uint32_t find(uint32_t a)
  {
    while (parent[a] != a) {
      parent[a] = parent[parent[a]];
      a = parent[a];
    }
    return a;
  }
  void unite(uint32_t a, uint32_t b)
  {
    a = find(a);
    b = find(b);
    if (a != b) parent[std::max(a, b)] = std::min(a, b);
  }
};

// eigenvalues of a symmetric 3x3 matrix (closed form): largest, smallest
std::pair<double, double> eigen_extremes(const double a[6])  // xx, yy, zz, xy, xz, yz
{
  const double p1 = a[3] * a[3] + a[4] * a[4] + a[5] * a[5];
  const double q = (a[0] + a[1] + a[2]) / 3.0;
  if (p1 <= 1e-18) return {std::max({a[0], a[1], a[2]}), std::min({a[0], a[1], a[2]})};
  const double p2 = (a[0] - q) * (a[0] - q) + (a[1] - q) * (a[1] - q) + (a[2] - q) * (a[2] - q) + 2.0 * p1;
  const double p = std::sqrt(p2 / 6.0);
  const double b0 = (a[0] - q) / p, b1 = (a[1] - q) / p, b2 = (a[2] - q) / p;
  const double b3 = a[3] / p, b4 = a[4] / p, b5 = a[5] / p;
  const double det = b0 * (b1 * b2 - b5 * b5) - b3 * (b3 * b2 - b5 * b4) + b4 * (b3 * b5 - b1 * b4);
  const double r = std::clamp(det / 2.0, -1.0, 1.0);
  const double phi = std::acos(r) / 3.0;
  return {q + 2.0 * p * std::cos(phi), q + 2.0 * p * std::cos(phi + 2.0 * M_PI / 3.0)};
}

double largest_eigenvalue(const double a[6]) { return eigen_extremes(a).first; }

int64_t key2(int64_t a, int64_t b)
{
  constexpr int64_t off = 1 << 30;
  return ((a + off) << 32) | (b + off);
}
}  // namespace

std::optional<std::vector<bool>> find_dust(
  const PointCloud2 & cloud, const DustParams & params, DustStats & stats, std::vector<uint8_t> * reasons)
{
  stats = DustStats{};
  const auto fx = field(cloud, "x");
  const auto fy = field(cloud, "y");
  const auto fz = field(cloud, "z");
  const auto fi = field(cloud, "intensity");
  const auto fd = field(cloud, "distance");
  const auto fa = field(cloud, "azimuth");
  const auto fe = field(cloud, "elevation");
  if (!fx || !fy || !fz || !fi || !fd || !fa || !fe) return std::nullopt;
  for (const auto & f : {*fx, *fy, *fz, *fd, *fa, *fe}) {
    if (f.datatype != PointField::FLOAT32) return std::nullopt;
  }
  const size_t n = static_cast<size_t>(cloud.width) * cloud.height;
  if (cloud.point_step == 0 || cloud.data.size() < n * cloud.point_step) return std::nullopt;
  std::vector<bool> dust(n, false);
  if (reasons) reasons->assign(n, kOutside);
  if (n == 0 || params.angle_radius <= 0.0 || params.max_distance <= 0.0) return dust;
  // the frame gate's volume (aebdust_1106: the drop box may be wider than the volume the gate counts in)
  const double gate_y = params.gate_max_abs_y < 0.0 ? params.max_abs_y : std::min(params.gate_max_abs_y, params.max_abs_y);

  // the returns used for the segments (and as the dominance pool)
  struct S
  {
    float x, y, z, az, el, d;
    double inten;
    uint32_t idx;
  };
  std::vector<S> pts;
  const double seg_r = std::max(params.segment_range, params.max_distance);
  for (size_t i = 0; i < n; ++i) {
    const uint8_t * p = cloud.data.data() + i * cloud.point_step;
    const float d = read_float(p, *fd);
    if (!(d < seg_r)) continue;
    const float x = read_float(p, *fx), y = read_float(p, *fy), z = read_float(p, *fz);
    const float az = read_float(p, *fa), el = read_float(p, *fe);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(az) || !std::isfinite(el)) continue;
    if (
      params.segment_margin >= 0.0 &&
      (x <= params.min_x - params.segment_margin || std::abs(y) > params.max_abs_y + params.segment_margin ||
       z <= params.min_z - params.segment_margin || z > params.max_z + params.segment_margin)) {
      continue;
    }
    const auto inten = read_number(p, *fi);
    if (!inten) return std::nullopt;
    pts.push_back({x, y, z, az, el, d, *inten, static_cast<uint32_t>(i)});
  }
  const size_t m = pts.size();
  if (m == 0) return dust;
  auto in_volume = [&params](const S & q) {
    return q.d < params.max_distance && q.z > params.min_z && q.z <= params.max_z && q.x > params.min_x &&
           std::abs(q.y) <= params.max_abs_y;
  };
  // exact early-out: a dust frame needs >= min(frame_min, hold_min) weak candidates in the gate's volume; with fewer weak
  // returns there no gate can be open this frame, whatever the segments say (the result is the same, only cheaper).
  // A follower has no gate of its own: it is only called while the external gate is open.
  if (!params.follow) {
    size_t weak = 0;
    for (const auto & q : pts) {
      if (in_volume(q) && std::abs(q.y) <= gate_y && q.inten <= params.gate_max_intensity && q.inten <= params.max_intensity) {
        ++weak;
      }
    }
    if (weak < static_cast<size_t>(std::max(1, std::min(params.frame_min, params.hold_min)))) {
      if (reasons) {
        for (const auto & q : pts) {
          if (in_volume(q)) (*reasons)[q.idx] = kEarlyOut;
        }
      }
      return dust;
    }
  }

  // range-consistent angular neighbour pairs
  const double R = params.angle_radius;
  auto acell = [R](float v) { return static_cast<int64_t>(std::floor(v / R)); };
  std::vector<std::array<int64_t, 3>> akeys(m);
  for (uint32_t k = 0; k < m; ++k) akeys[k] = {acell(pts[k].az), acell(pts[k].el), 0};
  DenseGrid adg;
  adg.build(akeys);
  std::unordered_map<int64_t, std::vector<uint32_t>> ag;  // fallback only
  if (!adg.ok) {
    for (uint32_t k = 0; k < m; ++k) ag[key2(akeys[k][0], akeys[k][1])].push_back(k);
  }
  std::vector<std::pair<uint32_t, uint32_t>> pairs;
  pairs.reserve(m * 6);
  std::vector<int> cons(m, 0);
  const double R2 = R * R;
  auto try_pair = [&](uint32_t k, uint32_t j) {
    const double da = pts[j].az - pts[k].az, de = pts[j].el - pts[k].el;
    if (da * da + de * de > R2) return;
    const double dmin = std::min(pts[j].d, pts[k].d);
    if (std::abs(pts[j].d - pts[k].d) >= params.range_tol_abs + params.range_tol_rel * dmin) return;
    pairs.emplace_back(std::min(k, j), std::max(k, j));
    ++cons[k];
    ++cons[j];
  };
  if (adg.ok) {
    // each unordered pair once: the own cell (j > k) and the 4 cells after it in raster order (all of their points);
    // the same pair set as visiting all 9 cells with j > k (its order does not matter downstream)
    static const int fwd[4][2] = {{0, 1}, {1, -1}, {1, 0}, {1, 1}};
    for (uint32_t k = 0; k < m; ++k) {
      const int64_t ca = akeys[k][0], ce = akeys[k][1];
      const auto own = adg.cell(ca, ce, 0);
      for (const uint32_t * e = own.first; e != own.second; ++e) {
        if (*e > k) try_pair(k, *e);
      }
      for (const auto & f : fwd) {
        const auto r = adg.cell(ca + f[0], ce + f[1], 0);
        for (const uint32_t * e = r.first; e != r.second; ++e) try_pair(k, *e);
      }
    }
  }
  for (uint32_t k = 0; !adg.ok && k < m; ++k) {
    const int64_t ca = akeys[k][0], ce = akeys[k][1];
    for (int64_t ia = ca - 1; ia <= ca + 1; ++ia) {
      for (int64_t ie = ce - 1; ie <= ce + 1; ++ie) {
        const auto it = ag.find(key2(ia, ie));
        if (it == ag.end()) continue;
        for (const auto j : it->second) {
          if (j > k) try_pair(k, j);
        }
      }
    }
  }

  // segments: connected cores (>= core consistent neighbours); a non-core return joins a core segment it touches
  auto segments = [&](int core_min, std::vector<uint32_t> & root, std::vector<uint32_t> & size) {
    UnionFind uf(m);
    for (const auto & pr : pairs) {
      if (cons[pr.first] >= core_min && cons[pr.second] >= core_min) uf.unite(pr.first, pr.second);
    }
    root.assign(m, UINT32_MAX);
    size.assign(m, 0);
    for (uint32_t k = 0; k < m; ++k) {
      if (cons[k] >= core_min) {
        root[k] = uf.find(k);
        ++size[root[k]];
      }
    }
  };
  std::vector<uint32_t> root3, size3, root2, size2;
  segments(params.surface_core, root3, size3);
  segments(params.line_core, root2, size2);
  // line segments' spread: std along the main axis of their cores
  std::vector<std::array<double, 9>> mom(m);  // n, sx, sy, sz, sxx, syy, szz, sxy, sxz (syz below)
  std::vector<double> syz(m, 0.0);
  for (uint32_t k = 0; k < m; ++k) {
    if (root2[k] == UINT32_MAX) continue;
    auto & a = mom[root2[k]];
    const double x = pts[k].x, y = pts[k].y, z = pts[k].z;
    a[0] += 1; a[1] += x; a[2] += y; a[3] += z; a[4] += x * x; a[5] += y * y; a[6] += z * z; a[7] += x * y; a[8] += x * z;
    syz[root2[k]] += y * z;
  }
  std::vector<bool> line_ok(m, false);
  for (uint32_t r = 0; r < m; ++r) {
    if (size2[r] == 0) continue;
    if (static_cast<int>(size2[r]) >= params.line_min) {
      line_ok[r] = true;
      continue;
    }
    const auto & a = mom[r];
    const double c = a[0];
    const double mx = a[1] / c, my = a[2] / c, mz = a[3] / c;
    const double cov[6] = {a[4] / c - mx * mx, a[5] / c - my * my, a[6] / c - mz * mz,
                           a[7] / c - mx * my, a[8] / c - mx * mz, syz[r] / c - my * mz};
    line_ok[r] = std::sqrt(std::max(0.0, largest_eigenvalue(cov))) >= params.line_extent;
  }
  // structure membership (cores, and non-cores touching a qualifying segment): 1 = surface, 2 = line (diagnostics only;
  // either keeps the return)
  std::vector<uint8_t> structure(m, 0);
  for (uint32_t k = 0; k < m; ++k) {
    if (root3[k] != UINT32_MAX && static_cast<int>(size3[root3[k]]) >= params.surface_min) structure[k] = 1;
    else if (root2[k] != UINT32_MAX && line_ok[root2[k]]) structure[k] = 2;
  }
  for (const auto & pr : pairs) {
    for (int s = 0; s < 2; ++s) {
      const uint32_t a = s ? pr.second : pr.first, b = s ? pr.first : pr.second;
      if (structure[a]) continue;
      if (root3[b] != UINT32_MAX && cons[a] < params.surface_core && static_cast<int>(size3[root3[b]]) >= params.surface_min) structure[a] = 1;
      else if (root2[b] != UINT32_MAX && cons[a] < params.line_core && line_ok[root2[b]]) structure[a] = 2;
    }
  }
  // candidates: near, weak enough, not on a structure
  std::vector<bool> cand(m, false);
  size_t ncand = 0;
  for (uint32_t k = 0; k < m; ++k) {
    if (!in_volume(pts[k])) continue;
    if (pts[k].inten > params.max_intensity) {
      if (reasons) (*reasons)[pts[k].idx] = kBright;
    } else if (structure[k]) {
      if (reasons) (*reasons)[pts[k].idx] = structure[k] == 1 ? kSurface : kLine;
    } else {
      cand[k] = true;
      ++ncand;
    }
  }
  stats.candidates = ncand;
  if (ncand == 0) return dust;
  // dominance: within local_radius, candidates are >= local_share of the pool
  const double rho = params.local_radius;
  if (rho <= 0.0 || params.local_share <= 0.0) {
    for (uint32_t k = 0; k < m; ++k) {
      if (!cand[k]) continue;
      dust[pts[k].idx] = true;
      if (reasons) (*reasons)[pts[k].idx] = kDust;
      if (std::abs(pts[k].y) <= gate_y) {
        ++stats.gate_dominant;
        if (pts[k].inten <= params.gate_max_intensity) ++stats.gate_count;
      }
    }
    stats.dominant = ncand;
    return dust;
  }
  // per voxel: all returns and candidates; a candidate's neighbourhood = its voxel and the 26 around it
  struct V
  {
    uint32_t all{0}, cand{0};
    double s[9]{0, 0, 0, 0, 0, 0, 0, 0, 0};  // sx sy sz sxx syy szz sxy sxz syz (all returns)
    double c[9]{0, 0, 0, 0, 0, 0, 0, 0, 0};  // the same over the candidates only
    double w[9]{0, 0, 0, 0, 0, 0, 0, 0, 0};  // ... and over the weak candidates (intensity <= gate_max_intensity)
    uint32_t weak{0};
  };
  auto cidx = [rho](float v) { return static_cast<int64_t>(std::floor(v / rho)); };
  // only voxels next to a candidate matter: a dense box over the candidates' voxels +- 1
  int64_t blo[3] = {INT64_MAX, INT64_MAX, INT64_MAX}, bhi[3] = {INT64_MIN, INT64_MIN, INT64_MIN};
  for (uint32_t k = 0; k < m; ++k) {
    if (!cand[k]) continue;
    const int64_t c3[3] = {cidx(pts[k].x), cidx(pts[k].y), cidx(pts[k].z)};
    for (int a = 0; a < 3; ++a) {
      blo[a] = std::min(blo[a], c3[a] - 1);
      bhi[a] = std::max(bhi[a], c3[a] + 1);
    }
  }
  const int64_t bd[3] = {bhi[0] - blo[0] + 1, bhi[1] - blo[1] + 1, bhi[2] - blo[2] + 1};
  const size_t bcells = static_cast<size_t>(bd[0]) * static_cast<size_t>(bd[1]) * static_cast<size_t>(bd[2]);
  const bool dense = bcells <= 2000000;
  auto bin = [&](int64_t x, int64_t y, int64_t z) {
    return x >= blo[0] && x <= bhi[0] && y >= blo[1] && y <= bhi[1] && z >= blo[2] && z <= bhi[2];
  };
  auto bix = [&](int64_t x, int64_t y, int64_t z) {
    return static_cast<size_t>(((x - blo[0]) * bd[1] + (y - blo[1])) * bd[2] + (z - blo[2]));
  };
  std::vector<V> dv(dense ? bcells : 0);
  std::unordered_map<int64_t, V> g3;  // fallback only
  for (uint32_t k = 0; k < m; ++k) {
    const int64_t vx = cidx(pts[k].x), vy = cidx(pts[k].y), vz = cidx(pts[k].z);
    if (dense && !bin(vx, vy, vz)) continue;  // next to no candidate: never part of a neighbourhood
    auto & v = dense ? dv[bix(vx, vy, vz)] : g3[cell_key(vx, vy, vz)];
    ++v.all;
    if (cand[k]) ++v.cand;
    const double x = pts[k].x, y = pts[k].y, z = pts[k].z;
    v.s[0] += x; v.s[1] += y; v.s[2] += z; v.s[3] += x * x; v.s[4] += y * y; v.s[5] += z * z;
    v.s[6] += x * y; v.s[7] += x * z; v.s[8] += y * z;
    if (cand[k]) {
      const double mo[9] = {x, y, z, x * x, y * y, z * z, x * y, x * z, y * z};
      for (int q = 0; q < 9; ++q) v.c[q] += mo[q];
      if (pts[k].inten <= params.gate_max_intensity) {
        for (int q = 0; q < 9; ++q) v.w[q] += mo[q];
        ++v.weak;
      }
    }
  }
  auto voxel = [&](int64_t x, int64_t y, int64_t z) -> const V * {
    if (dense) {
      if (!bin(x, y, z)) return nullptr;
      const V & v = dv[bix(x, y, z)];
      return v.all ? &v : nullptr;
    }
    const auto g = g3.find(cell_key(x, y, z));
    return g == g3.end() ? nullptr : &g->second;
  };
  // decided once per voxel holding candidates: -1 not yet, 0 not dominant, 1 dust, 2 a plane
  std::vector<int8_t> vmemo(dense ? bcells : 0, -1);
  std::unordered_map<int64_t, int8_t> vdust;  // fallback memo
  size_t ndom = 0;
  for (uint32_t k = 0; k < m; ++k) {
    if (!cand[k]) continue;
    const int64_t cx = cidx(pts[k].x), cy = cidx(pts[k].y), cz = cidx(pts[k].z);
    int8_t decided = dense ? vmemo[bix(cx, cy, cz)] : -1;
    if (!dense) {
      const auto it = vdust.find(cell_key(cx, cy, cz));
      if (it != vdust.end()) decided = it->second;
    }
    if (decided < 0) {
      size_t all = 0, c = 0;
      double sm[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
      double cm[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
      double wm[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
      size_t wk = 0;
      for (int64_t ix = cx - 1; ix <= cx + 1; ++ix) {
        for (int64_t iy = cy - 1; iy <= cy + 1; ++iy) {
          for (int64_t iz = cz - 1; iz <= cz + 1; ++iz) {
            const V * g = voxel(ix, iy, iz);
            if (!g) continue;
            all += g->all;
            c += g->cand;
            for (int q = 0; q < 9; ++q) {
              sm[q] += g->s[q];
              cm[q] += g->c[q];
              wm[q] += g->w[q];
            }
            wk += g->weak;
          }
        }
      }
      bool is_dust = static_cast<double>(c) >= params.local_share * static_cast<double>(all);
      // a plane is not dust: all returns there (the road at grazing incidence), the dust-like ones, or the weak
      // dust-like ones alone (a dark surface with dropouts: its remaining returns still lie on it, also when the
      // bright returns of a sloped road around its base are dust-like too)
      auto planar = [&](const double * mo, size_t cnt) {
        if (static_cast<int>(cnt) < std::max(3, params.planar_min_points)) return false;
        const double nn = static_cast<double>(cnt);
        const double mx = mo[0] / nn, my = mo[1] / nn, mz = mo[2] / nn;
        const double cov[6] = {mo[3] / nn - mx * mx, mo[4] / nn - my * my, mo[5] / nn - mz * mz,
                               mo[6] / nn - mx * my, mo[7] / nn - mx * mz, mo[8] / nn - my * mz};
        const auto ev = eigen_extremes(cov);
        return ev.first > 0.0 && std::max(0.0, ev.second) < params.planar_max * ev.first;
      };
      decided = is_dust ? 1 : 0;
      if (is_dust && params.planar_max > 0.0 && (planar(sm, all) || (params.planar_subsets && (planar(cm, c) || planar(wm, wk))))) decided = 2;
      if (dense) {
        vmemo[bix(cx, cy, cz)] = decided;
      } else {
        vdust.emplace(cell_key(cx, cy, cz), decided);
      }
    }
    if (reasons) (*reasons)[pts[k].idx] = decided == 1 ? kDust : (decided == 2 ? kPlane : kNotDominant);
    if (decided == 1) {
      dust[pts[k].idx] = true;
      ++ndom;
      if (std::abs(pts[k].y) <= gate_y) {
        ++stats.gate_dominant;
        if (pts[k].inten <= params.gate_max_intensity) ++stats.gate_count;
      }
    }
  }
  stats.dominant = ndom;
  return dust;
}

void DustStage::external_gate_open(double stamp_sec) { last_external_open_ = stamp_sec; }

bool DustStage::external_open_at(double stamp_sec) const
{
  return std::abs(stamp_sec - last_external_open_) <= params_.follow_sec;
}

std::optional<std::vector<bool>> DustStage::process(
  const PointCloud2 & cloud, double stamp_sec, DustStats & stats, std::vector<uint8_t> * reasons)
{
  if (params_.follow) {
    // [nero 2026-11-06 aebdust_1106] a side lidar: the front E1R's dust gate decides; closed = nothing looked at
    if (!external_open_at(stamp_sec)) {
      stats = DustStats{};
      const size_t n = static_cast<size_t>(cloud.width) * cloud.height;
      if (reasons) reasons->assign(n, kOutside);
      return std::vector<bool>(n, false);
    }
    auto dust = find_dust(cloud, params_, stats, reasons);
    if (dust) stats.gate_open = true;
    return dust;
  }
  // [nero 2026-11-06 aebdust_1106] exact early-out by the gate's state: outside a hold only a burst (>= frame_min weak
  // returns) can open the gate, inside it >= hold_min keep it open; find_dust returns early below min(frame_min, hold_min)
  const double since_burst = stamp_sec - last_dust_frame_;
  const bool in_hold = since_burst >= 0.0 && since_burst <= params_.hold_sec;
  DustParams p = params_;
  if (!in_hold) p.hold_min = p.frame_min;
  auto dust = find_dust(cloud, p, stats, reasons);
  if (!dust) return dust;
  const bool burst = static_cast<int>(stats.gate_count) >= params_.frame_min && stats.gate_count > 0 &&
                     static_cast<double>(stats.gate_count) >= params_.gate_weak_share * static_cast<double>(stats.gate_dominant);
  if (burst) last_dust_frame_ = stamp_sec;
  const double since = stamp_sec - last_dust_frame_;
  stats.gate_open = burst || (static_cast<int>(stats.gate_count) >= params_.hold_min && stats.gate_count > 0 &&
                              since >= 0.0 && since <= params_.hold_sec);
  if (!stats.gate_open) std::fill(dust->begin(), dust->end(), false);
  return dust;
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
