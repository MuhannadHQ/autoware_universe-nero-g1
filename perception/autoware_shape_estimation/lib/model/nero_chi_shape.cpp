// Copyright 2026 Nero (wallbox_1107). Licensed under the Apache License, Version 2.0.

#include "autoware/shape_estimation/model/nero_chi_shape.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace autoware::shape_estimation
{
namespace model
{

namespace
{
// NERO_SHAPE_UNKNOWN_CHI=<chi_m>,<min_hull_area_m2>,<grid_m>,<max_points>,<tol_out_m> (wallbox_1107, QUEUE WALLBOX): the
// footprint of a cluster (UNKNOWN) is its chi-shape instead of its convex hull: Duckham, Kulik, Worboys, Galton 2008, "Efficient
// generation of simple polygons for characterizing the shape of a set of points in the plane" (Pattern Recognition 41). From the
// Delaunay triangulation (= the convex hull), the boundary triangle behind the longest boundary edge longer than chi_m is removed
// while the outline stays a simple polygon (its third vertex is not on the outline yet). The vertices are cluster points: a wall that
// bends round the robot, an L-shaped corner or a recess keeps the outline of its points instead of a hull over the open road
// between them. The pose (position, orientation) stays exactly as with the hull; only the footprint vertices change. A cluster whose
// hull is smaller than min_hull_area_m2 keeps its hull, and so does every case the rule cannot do safely (degenerate triangulation,
// a pinched outline, an invalid polygon, a point left outside). Unset / 0 = the car (convex hull).
struct NeroChiCfg
{
  bool on{false};
  double chi_mm{800.0};
  double min_area_mm2{0.5e6};
  double grid_mm{20.0};
  std::size_t max_points{1500};
  double tol_out_mm{50.0};
  double hull_reach_mm{0.0};  // field 6 (wallbox-3): keep the hull when it reaches <= this beyond the chi outline (0 = off)
  double corridor_m{0.0};     // field 7 (wallbox-4): only clusters whose hull comes within this of the planned trajectory (0 = all)
};

const NeroChiCfg & neroChiCfg()
{
  static const NeroChiCfg cfg = [] {
    NeroChiCfg c;
    const char * e = std::getenv("NERO_SHAPE_UNKNOWN_CHI");
    if (e == nullptr || *e == '\0') return c;
    std::vector<double> v;
    std::string s(e);
    std::size_t pos = 0;
    while (pos <= s.size()) {
      const std::size_t q = s.find(',', pos);
      const std::string tok = s.substr(pos, q == std::string::npos ? std::string::npos : q - pos);
      char * end = nullptr;
      const double x = std::strtod(tok.c_str(), &end);
      v.push_back((end != tok.c_str()) ? x : 0.0);
      if (q == std::string::npos) break;
      pos = q + 1;
    }
    if (v.empty() || !(v[0] > 0.0)) return c;
    c.on = true;
    c.chi_mm = v[0] * 1000.0;
    if (v.size() > 1 && v[1] >= 0.0) c.min_area_mm2 = v[1] * 1e6;
    if (v.size() > 2 && v[2] > 0.0) c.grid_mm = v[2] * 1000.0;
    if (v.size() > 3 && v[3] >= 16.0) c.max_points = static_cast<std::size_t>(v[3]);
    if (v.size() > 4 && v[4] >= 0.0) c.tol_out_mm = v[4] * 1000.0;
    if (v.size() > 5 && v[5] > 0.0) c.hull_reach_mm = v[5] * 1000.0;
    if (v.size() > 6 && v[6] > 0.0) c.corridor_m = v[6];
    return c;
  }();
  return cfg;
}

inline double cross3(const cv::Point2d & a, const cv::Point2d & b, const cv::Point2d & c)
{
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool segmentsCross(
  const cv::Point2d & a, const cv::Point2d & b, const cv::Point2d & c, const cv::Point2d & d)
{
  const double d1 = cross3(c, d, a);
  const double d2 = cross3(c, d, b);
  const double d3 = cross3(a, b, c);
  const double d4 = cross3(a, b, d);
  if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
    return true;
  }
  auto on = [](const cv::Point2d & p, const cv::Point2d & q, const cv::Point2d & r, double cr) {
    return std::abs(cr) < 1e-9 && std::min(p.x, q.x) <= r.x && r.x <= std::max(p.x, q.x) &&
           std::min(p.y, q.y) <= r.y && r.y <= std::max(p.y, q.y);
  };
  return on(c, d, a, d1) || on(c, d, b, d2) || on(a, b, c, d3) || on(a, b, d, d4);
}

// a simple counter-clockwise ring: positive area, no two non-adjacent edges touching
bool simpleCcw(const std::vector<cv::Point2d> & r)
{
  const std::size_t n = r.size();
  if (n < 3) return false;
  double area2 = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto & p = r[i];
    const auto & q = r[(i + 1) % n];
    area2 += p.x * q.y - q.x * p.y;
  }
  if (!(area2 > 1e-6)) return false;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      if (j == i + 1 || (i == 0 && j == n - 1)) continue;
      if (segmentsCross(r[i], r[(i + 1) % n], r[j], r[(j + 1) % n])) return false;
    }
  }
  return true;
}

// drop vertices without losing coverage: a vertex goes when every ORIGINAL ring vertex between its two neighbours lies at most
// 10 mm outside their new chord (what the ring loses is within 10 mm of it) and at most tol_out inside it (the outline grows into
// a dent of at most tol_out); the smallest deviation first; never a chord that touches the rest of the ring
void simplifyOut(std::vector<cv::Point2d> & r, double tol_out)
{
  constexpr double tol_in = 10.0;
  const std::vector<cv::Point2d> orig = r;
  std::vector<std::size_t> id(r.size());
  for (std::size_t i = 0; i < id.size(); ++i) id[i] = i;  // the current ring's indices into orig
  const std::size_t m = orig.size();
  while (r.size() > 3) {
    const std::size_t n = r.size();
    double best_d = 1e300;
    std::size_t best_i = n;
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t ia = (i + n - 1) % n;
      const std::size_t ib = (i + 1) % n;
      const auto & a = r[ia];
      const auto & b = r[ib];
      const double L = std::hypot(b.x - a.x, b.y - a.y);
      if (L < 1e-9) continue;
      double out = 0.0;
      double in = 0.0;
      for (std::size_t k = (id[ia] + 1) % m; k != id[ib]; k = (k + 1) % m) {
        const auto & q = orig[k];
        const double tp = ((q.x - a.x) * (b.x - a.x) + (q.y - a.y) * (b.y - a.y)) / (L * L);
        if (tp < 0.0 || tp > 1.0) {  // beyond the chord's ends (a spike along its line): never dropped
          out = 1e300;
          break;
        }
        const double sd = cross3(a, b, q) / L;  // > 0: left of a->b = inside (counter-clockwise ring)
        out = std::max(out, -sd);
        in = std::max(in, sd);
      }
      if (out <= tol_in && in <= tol_out) {
        const double d = std::max(out, in);
        if (d < best_d) {
          best_d = d;
          best_i = i;
        }
      }
    }
    if (best_i == n) break;
    // the new edge a-b must not touch any edge that is not adjacent to it (the rest of the ring is unchanged)
    const std::size_t ia = (best_i + n - 1) % n;
    const std::size_t ib = (best_i + 1) % n;
    bool crosses = false;
    for (std::size_t j = 0; j < n && !crosses; ++j) {
      const std::size_t j2 = (j + 1) % n;
      if (j == ia || j == best_i || j2 == ia || j == ib || j2 == best_i) continue;  // edges sharing a or b, or removed
      crosses = segmentsCross(r[ia], r[ib], r[j], r[j2]);
    }
    if (crosses) break;
    r.erase(r.begin() + static_cast<std::ptrdiff_t>(best_i));
    id.erase(id.begin() + static_cast<std::ptrdiff_t>(best_i));
  }
}

// the chi-shape ring (mm, counter-clockwise) of the points, or false
bool neroChiShape(
  const std::vector<cv::Point> & pts_mm, const NeroChiCfg & cfg, std::vector<cv::Point2d> & ring)
{
  // 1. one point per grid cell (the grid doubles until max_points)
  std::vector<cv::Point> P;
  double g = std::max(1.0, cfg.grid_mm);
  for (int pass = 0; pass < 6; ++pass) {
    std::unordered_set<std::int64_t> seen;
    P.clear();
    for (const auto & p : pts_mm) {
      const auto gx = static_cast<std::int64_t>(std::floor(p.x / g));
      const auto gy = static_cast<std::int64_t>(std::floor(p.y / g));
      if (seen.insert((gx << 32) ^ (gy & 0xffffffffLL)).second) P.push_back(p);
    }
    if (P.size() <= cfg.max_points) break;
    g *= 2.0;
  }
  if (P.size() < 4 || P.size() > cfg.max_points) return false;
  // 2. Delaunay (OpenCV Subdiv2D; triangles with a virtual outer vertex are not listed)
  int minx = P[0].x, maxx = P[0].x, miny = P[0].y, maxy = P[0].y;
  std::unordered_map<std::int64_t, int> index;
  for (std::size_t i = 0; i < P.size(); ++i) {
    minx = std::min(minx, P[i].x);
    maxx = std::max(maxx, P[i].x);
    miny = std::min(miny, P[i].y);
    maxy = std::max(maxy, P[i].y);
    index[(static_cast<std::int64_t>(P[i].x) << 32) ^ (static_cast<std::int64_t>(P[i].y) & 0xffffffffLL)] =
      static_cast<int>(i);
  }
  std::vector<cv::Vec6f> tl;
  try {
    // a rect much larger than the points puts Subdiv2D's virtual outer vertices far away: hull points then keep their real
    // triangles (with the outer vertices at 3x a tight rect some near-collinear hull points had only virtual ones)
    const int ext = std::max(maxx - minx, maxy - miny) * 20 + 1000;
    cv::Subdiv2D sub(cv::Rect(minx - ext, miny - ext, maxx - minx + 2 * ext, maxy - miny + 2 * ext));
    for (const auto & p : P) sub.insert(cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y)));
    sub.getTriangleList(tl);
  } catch (const cv::Exception &) {
    return false;
  }
  if (tl.empty()) return false;
  std::vector<std::array<int, 3>> T;
  T.reserve(tl.size());
  for (const auto & v : tl) {
    std::array<int, 3> t{};
    bool ok = true;
    for (int k = 0; k < 3 && ok; ++k) {
      const auto it = index.find(
        (static_cast<std::int64_t>(std::lround(v[2 * k])) << 32) ^
        (static_cast<std::int64_t>(std::lround(v[2 * k + 1])) & 0xffffffffLL));
      if (it == index.end()) {
        ok = false;
      } else {
        t[k] = it->second;
      }
    }
    if (!ok) continue;
    const cv::Point2d a(P[t[0]]), b(P[t[1]]), c(P[t[2]]);
    const double cr = cross3(a, b, c);
    if (std::abs(cr) < 1e-6) continue;
    if (cr < 0) std::swap(t[1], t[2]);  // counter-clockwise
    T.push_back(t);
  }
  if (T.empty()) return false;
  // 3. chi: remove boundary triangles behind the longest boundary edges > chi while the outline stays simple
  auto key = [](int u, int v) {
    return (static_cast<std::int64_t>(std::min(u, v)) << 32) | static_cast<std::int64_t>(std::max(u, v));
  };
  std::unordered_map<std::int64_t, std::array<int, 2>> edge_tris;
  for (std::size_t i = 0; i < T.size(); ++i) {
    for (int k = 0; k < 3; ++k) {
      auto & e = edge_tris.try_emplace(key(T[i][k], T[i][(k + 1) % 3]), std::array<int, 2>{-1, -1}).first->second;
      (e[0] < 0 ? e[0] : e[1]) = static_cast<int>(i);
    }
  }
  std::vector<char> alive(T.size(), 1);
  std::vector<char> on_b(P.size(), 0);
  std::unordered_set<std::int64_t> boundary;
  using QE = std::pair<double, std::int64_t>;
  std::priority_queue<QE> q;
  auto len = [&](std::int64_t k) {
    const int u = static_cast<int>(k >> 32);
    const int v = static_cast<int>(k & 0xffffffffLL);
    return std::hypot(double(P[u].x - P[v].x), double(P[u].y - P[v].y));
  };
  for (const auto & [k, ts] : edge_tris) {
    if (ts[1] < 0) {
      boundary.insert(k);
      on_b[static_cast<std::size_t>(k >> 32)] = 1;
      on_b[static_cast<std::size_t>(k & 0xffffffffLL)] = 1;
      q.emplace(len(k), k);
    }
  }
  while (!q.empty()) {
    const auto [l, k] = q.top();
    q.pop();
    if (l <= cfg.chi_mm) break;
    if (boundary.count(k) == 0) continue;
    const auto & ts = edge_tris[k];
    int t = -1;
    int n_alive = 0;
    for (int ti : ts) {
      if (ti >= 0 && alive[static_cast<std::size_t>(ti)]) {
        t = ti;
        ++n_alive;
      }
    }
    if (n_alive != 1) continue;
    const int u = static_cast<int>(k >> 32);
    const int v = static_cast<int>(k & 0xffffffffLL);
    int c = -1;
    for (int x : T[static_cast<std::size_t>(t)]) {
      if (x != u && x != v) c = x;
    }
    if (c < 0 || on_b[static_cast<std::size_t>(c)]) continue;
    alive[static_cast<std::size_t>(t)] = 0;
    boundary.erase(k);
    on_b[static_cast<std::size_t>(c)] = 1;
    for (int w : {u, v}) {
      const auto ne = key(w, c);
      boundary.insert(ne);
      q.emplace(len(ne), ne);
    }
  }
  // 4. walk the outline (counter-clockwise: the interior on the left of every boundary edge)
  std::unordered_map<int, int> nxt;
  for (std::size_t i = 0; i < T.size(); ++i) {
    if (!alive[i]) continue;
    for (int k = 0; k < 3; ++k) {
      const int u = T[i][k];
      const int v = T[i][(k + 1) % 3];
      if (boundary.count(key(u, v)) == 0) continue;
      if (!nxt.emplace(u, v).second) return false;  // a pinched outline
    }
  }
  if (nxt.size() < 3) return false;
  const int start = nxt.begin()->first;
  std::vector<cv::Point2d> r;
  int cur = start;
  do {
    r.emplace_back(P[static_cast<std::size_t>(cur)]);
    const auto it = nxt.find(cur);
    if (it == nxt.end() || r.size() > nxt.size()) return false;
    cur = it->second;
  } while (cur != start);
  if (r.size() != nxt.size()) return false;
  // 5. fewer vertices without losing coverage; 6. a simple ring that holds every point (within grid + 10 mm)
  simplifyOut(r, cfg.tol_out_mm);
  if (!simpleCcw(r)) return false;
  std::vector<cv::Point2f> rf;
  rf.reserve(r.size());
  for (const auto & p : r) rf.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
  const double tol = g * 1.5 + 10.0;
  for (const auto & p : pts_mm) {
    if (cv::pointPolygonTest(rf, cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y)), true) < -tol) {
      return false;
    }
  }
  // 7. field 6: an outline its convex hull covers within hull_reach keeps the hull (the car's footprint): only a wall bending round
  // the robot, an L corner or a misfit plinth wall needs the outline, and every planner module pays for concave polygons (obstacle_stop
  // +7 ms p95 at 11-02 D with ~6.5 concave UNKNOWNs per message)
  if (cfg.hull_reach_mm > 0.0) {
    std::vector<cv::Point2f> hull;
    cv::convexHull(rf, hull);
    double reach = 0.0;
    const std::size_t n = r.size();
    for (std::size_t i = 0; i < hull.size() && reach <= cfg.hull_reach_mm; ++i) {
      const cv::Point2d a(hull[i]);
      const cv::Point2d b(hull[(i + 1) % hull.size()]);
      const int steps = std::max(1, static_cast<int>(std::ceil(std::hypot(b.x - a.x, b.y - a.y) / 100.0)));
      for (int k = 0; k <= steps; ++k) {
        const double t = static_cast<double>(k) / steps;
        const cv::Point2d p(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y));
        double d = 1e300;
        for (std::size_t j = 0; j < n; ++j) {
          const cv::Point2d & u = r[j];
          const cv::Point2d & w = r[(j + 1) % n];
          const double vx = w.x - u.x, vy = w.y - u.y, l2 = vx * vx + vy * vy;
          const double tt = l2 > 0.0 ? std::clamp(((p.x - u.x) * vx + (p.y - u.y) * vy) / l2, 0.0, 1.0) : 0.0;
          d = std::min(d, std::hypot(p.x - u.x - tt * vx, p.y - u.y - tt * vy));
        }
        reach = std::max(reach, d);
      }
    }
    if (reach <= cfg.hull_reach_mm) return false;  // the hull (as the car)
  }
  ring.swap(r);
  return true;
}
}  // namespace

bool neroChiEnabled()
{
  return neroChiCfg().on;
}

// field 7 (wallbox-4): the planned trajectory in the clusters' frame for the message being processed (set by the node per message;
// nullptr = no fresh plan: every cluster as field 7 unset). A cluster whose convex hull stays >= corridor_m from every trajectory
// segment keeps its hull (the car's footprint) and costs no triangulation: the chi-shape is for what can reach the robot's way.
namespace
{
const std::vector<std::array<double, 2>> * g_nero_chi_path = nullptr;
double neroChiPtSeg(double px, double py, double ax, double ay, double bx, double by)
{
  const double vx = bx - ax, vy = by - ay;
  const double l2 = vx * vx + vy * vy;
  double t = l2 > 0.0 ? ((px - ax) * vx + (py - ay) * vy) / l2 : 0.0;
  t = std::clamp(t, 0.0, 1.0);
  return std::hypot(px - ax - t * vx, py - ay - t * vy);
}
// the hull (absolute, the clusters' frame) lies >= x from every segment of the path
bool neroChiHullOutside(const std::vector<cv::Point2d> & h, const std::vector<std::array<double, 2>> & path, const double x)
{
  double x0 = 1e300, x1 = -1e300, y0 = 1e300, y1 = -1e300;
  for (const auto & p : h) {
    x0 = std::min(x0, p.x); x1 = std::max(x1, p.x); y0 = std::min(y0, p.y); y1 = std::max(y1, p.y);
  }
  x0 -= x; x1 += x; y0 -= x; y1 += x;
  const std::size_t n = h.size();
  for (std::size_t i = 1; i < path.size(); ++i) {
    const auto & a = path[i - 1];
    const auto & b = path[i];
    if (std::max(a[0], b[0]) < x0 || std::min(a[0], b[0]) > x1 || std::max(a[1], b[1]) < y0 || std::min(a[1], b[1]) > y1) continue;
    const cv::Point2d A(a[0], a[1]);
    const cv::Point2d B(b[0], b[1]);
    bool in = false;  // a path point inside the hull
    for (std::size_t k = 0; k < n; ++k) {
      const auto & p = h[k];
      const auto & q = h[(k + 1) % n];
      if (segmentsCross(p, q, A, B)) return false;
      if (neroChiPtSeg(p.x, p.y, a[0], a[1], b[0], b[1]) < x) return false;
      if (neroChiPtSeg(a[0], a[1], p.x, p.y, q.x, q.y) < x || neroChiPtSeg(b[0], b[1], p.x, p.y, q.x, q.y) < x) return false;
      if (((p.y > a[1]) != (q.y > a[1])) && a[0] < (q.x - p.x) * (a[1] - p.y) / (q.y - p.y) + p.x) in = !in;
    }
    if (in) return false;
  }
  return true;
}
}  // namespace

bool neroChiCorridorWanted()
{
  return neroChiCfg().on && neroChiCfg().corridor_m > 0.0;
}

void neroChiSetCorridor(const std::vector<std::array<double, 2>> * path_in_cluster_frame)
{
  g_nero_chi_path = path_in_cluster_frame;
}

bool neroChiFootprint(
  const pcl::PointCloud<pcl::PointXYZ> & cluster, const geometry_msgs::msg::Pose & pose,
  autoware_perception_msgs::msg::Shape & shape)
{
  // the convex-hull model's footprint (vertices relative to the pose, identity orientation) becomes the chi-shape of the same
  // points (mm, relative to the pose); the pose stays as estimated
  const auto & cfg = neroChiCfg();
  if (!cfg.on || shape.type != autoware_perception_msgs::msg::Shape::POLYGON || shape.footprint.points.size() < 3) {
    return false;
  }
  double a2 = 0.0;
  const auto & f = shape.footprint.points;
  for (std::size_t i = 0; i < f.size(); ++i) {
    const auto & p = f[i];
    const auto & q = f[(i + 1) % f.size()];
    a2 += static_cast<double>(p.x) * q.y - static_cast<double>(q.x) * p.y;
  }
  if (std::abs(a2) * 0.5e6 < cfg.min_area_mm2) return false;  // m2 -> mm2
  if (cfg.corridor_m > 0.0 && g_nero_chi_path != nullptr && g_nero_chi_path->size() >= 2) {  // field 7: far from the way: the hull
    std::vector<cv::Point2d> h;
    h.reserve(f.size());
    for (const auto & p : f) h.emplace_back(pose.position.x + p.x, pose.position.y + p.y);
    if (neroChiHullOutside(h, *g_nero_chi_path, cfg.corridor_m)) return false;
  }
  std::vector<cv::Point> pts;
  pts.reserve(cluster.size());
  for (const auto & p : cluster) {
    pts.emplace_back(
      static_cast<int>(std::lround((p.x - pose.position.x) * 1000.0)),
      static_cast<int>(std::lround((p.y - pose.position.y) * 1000.0)));
  }
  std::vector<cv::Point2d> ring;
  if (!neroChiShape(pts, cfg, ring)) return false;
  shape.footprint.points.clear();
  for (const auto & r : ring) {
    geometry_msgs::msg::Point32 point;
    point.x = static_cast<float>(r.x / 1000.0);
    point.y = static_cast<float>(r.y / 1000.0);
    point.z = 0.0F;
    shape.footprint.points.push_back(point);
  }
  return true;
}

}  // namespace model
}  // namespace autoware::shape_estimation
