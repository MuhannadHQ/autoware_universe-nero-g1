// NERO_SOA_SIDE_FAST (10amimprovement 2026-11-16, sideperf_1117): NERO_SOA_SIDE_LANE's "does the planned body leave the lane"
// (neroPlanLeavesLane) with the SAME answer, faster.
// Profile (gperftools, 11-08 19:15 logsim window, set1117): static avoidance 183 ms at 19:15:51.40 vs 47 ms without the switch;
// the extra was neroPlanLeavesLane -> utils::isEgoWithinOriginalLane(lane polygon, pose) for every shifted path point: four
// boost::geometry::distance(corner, polygon) calls per point, each a point-in-polygon walk over the whole combined
// current-lanes polygon (hundreds of points per bound on the dense lanelets) - an in-lane nudge tests every point.
// Fast path: the polygon's edges indexed in 2 m bands of y (built once per lane set, kept with it); per corner the winding
// number over the edges of its band (the edges that can cross its horizontal line), and the distance to those edges.
// A corner closer than 0.1 mm to any edge takes boost's own call (the original computation) - only there can floating
// point tell inside / on / outside differently. Anywhere else the winding number decides exactly what boost's distance
// (0 inside or on, > 0 outside) decides. 0 / unset = the car; 1 = fast; 2 = check: both computed, the ORIGINAL result
// used, mismatches counted ([NERO_SOA_SIDE_FAST] lines every 20 s and at exit).
#ifndef NERO_SIDE_FAST_HPP_
#define NERO_SIDE_FAST_HPP_

#include "autoware/behavior_path_planner_common/parameters.hpp"
#include "autoware/behavior_path_planner_common/utils/utils.hpp"

#include <autoware_utils/geometry/boost_polygon_utils.hpp>
#include <rclcpp/rclcpp.hpp>

#include <boost/geometry/algorithms/distance.hpp>

#include <lanelet2_core/primitives/Lanelet.h>
#include <lanelet2_core/primitives/Polygon.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

namespace nero_side_fast
{
inline int mode()
{
  static const int m = [] {
    const char * v = std::getenv("NERO_SOA_SIDE_FAST");
    const int r = v ? std::atoi(v) : 0;
    if (r > 0) {
      RCLCPP_INFO(
        rclcpp::get_logger("static_obstacle_avoidance"),
        "[NERO_SOA_SIDE_FAST] %d (1: lane polygon edges indexed by band, winding number per corner, boost's own call "
        "within 0.1 mm of an edge; 2: check = both computed, the original used, mismatches counted)", r);
    }
    return r;
  }();
  return m;
}

struct Stats
{
  std::atomic<uint64_t> calls{0}, corners{0}, near{0}, mismatch{0}, builds{0};
  std::atomic<int64_t> last_print_ms{0};
};
inline Stats & stats()
{
  static Stats s;
  return s;
}
inline void print(const char * why)
{
  auto & s = stats();
  RCLCPP_INFO(
    rclcpp::get_logger("static_obstacle_avoidance"),
    "[NERO_SOA_SIDE_FAST] %s mode %d: footprints %lu (%lu corners, %lu near an edge -> boost), %lu mismatches, %lu indexes built",
    why, mode(), static_cast<unsigned long>(s.calls.load()), static_cast<unsigned long>(s.corners.load()),
    static_cast<unsigned long>(s.near.load()), static_cast<unsigned long>(s.mismatch.load()),
    static_cast<unsigned long>(s.builds.load()));
}
inline void maybe_print()
{
  static struct AtExit
  {
    ~AtExit()
    {
      if (mode() > 0) print("exit");
    }
  } at_exit;
  const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
  auto & s = stats();
  int64_t last = s.last_print_ms.load();
  if (now - last >= 20000 && s.last_print_ms.compare_exchange_strong(last, now)) print("totals");
}

constexpr double kEps = 1.0e-4;  // m: closer to an edge than this = boost's own call
constexpr double kBand = 2.0;    // m

struct Index
{
  std::vector<lanelet::Id> ids;
  std::size_t n{0};
  double fx{0.0}, fy{0.0};  // the polygon's first point (with ids and size: the key)
  std::vector<std::array<double, 4>> e;  // edges i -> i+1 (closing edge included)
  double xmin{0.0}, xmax{0.0}, ymin{0.0}, ymax{0.0};
  std::vector<std::vector<uint32_t>> bands;
  int band(const double y) const
  {
    const int k = static_cast<int>(std::floor((y - ymin) / kBand));
    return std::clamp(k, 0, static_cast<int>(bands.size()) - 1);
  }
};

inline std::shared_ptr<const Index> index(const lanelet::BasicPolygon2d & poly, const std::vector<lanelet::Id> & ids)
{
  static std::mutex mtx;
  static std::vector<std::shared_ptr<const Index>> cache;
  if (poly.empty()) return nullptr;
  {
    std::lock_guard<std::mutex> lk(mtx);
    for (const auto & ix : cache) {
      if (ix->ids == ids && ix->n == poly.size() && ix->fx == poly.front().x() && ix->fy == poly.front().y()) return ix;
    }
  }
  auto ix = std::make_shared<Index>();
  ix->ids = ids;
  ix->n = poly.size();
  ix->fx = poly.front().x();
  ix->fy = poly.front().y();
  ix->xmin = ix->ymin = std::numeric_limits<double>::max();
  ix->xmax = ix->ymax = std::numeric_limits<double>::lowest();
  for (const auto & p : poly) {
    ix->xmin = std::min(ix->xmin, p.x());
    ix->xmax = std::max(ix->xmax, p.x());
    ix->ymin = std::min(ix->ymin, p.y());
    ix->ymax = std::max(ix->ymax, p.y());
  }
  const std::size_t nb = static_cast<std::size_t>(std::floor((ix->ymax - ix->ymin) / kBand)) + 1;
  ix->bands.resize(nb);
  ix->e.reserve(poly.size());
  for (std::size_t i = 0; i < poly.size(); ++i) {
    const auto & a = poly[i];
    const auto & b = poly[(i + 1) % poly.size()];
    ix->e.push_back({a.x(), a.y(), b.x(), b.y()});
    const int k0 = ix->band(std::min(a.y(), b.y()) - kEps);
    const int k1 = ix->band(std::max(a.y(), b.y()) + kEps);
    for (int k = k0; k <= k1; ++k) ix->bands[k].push_back(static_cast<uint32_t>(i));
  }
  stats().builds++;
  std::lock_guard<std::mutex> lk(mtx);
  cache.insert(cache.begin(), ix);
  if (cache.size() > 4) cache.pop_back();
  return ix;
}

// 1 = inside (boost distance 0), 0 = outside (distance > 0), -1 = within kEps of an edge (ask boost)
inline int classify(const Index & ix, const double px, const double py)
{
  if (px < ix.xmin - kEps || px > ix.xmax + kEps || py < ix.ymin - kEps || py > ix.ymax + kEps) return 0;
  int w = 0;
  for (const auto i : ix.bands[ix.band(py)]) {
    const auto & s = ix.e[i];
    const double ax = s[0], ay = s[1], bx = s[2], by = s[3];
    // distance to the segment
    const double vx = bx - ax, vy = by - ay, wx = px - ax, wy = py - ay;
    const double c2 = vx * vx + vy * vy;
    double t = c2 > 0.0 ? (wx * vx + wy * vy) / c2 : 0.0;
    t = std::clamp(t, 0.0, 1.0);
    const double dx = wx - t * vx, dy = wy - t * vy;
    if (dx * dx + dy * dy < kEps * kEps) return -1;
    // winding number (crossings of the ray to +x, half-open in y)
    const double side = vx * wy - wx * vy;  // > 0: p left of a -> b
    if (ay <= py) {
      if (by > py && side > 0.0) ++w;
    } else {
      if (by <= py && side < 0.0) --w;
    }
  }
  return w != 0 ? 1 : 0;
}

// = utils::isEgoWithinOriginalLane(lane_polygon, pose, p, outer_margin 0.0)
inline bool within_lane(
  const Index & ix, const lanelet::BasicPolygon2d & lane_polygon, const geometry_msgs::msg::Pose & pose,
  const ::BehaviorPathPlannerParameters & p)
{
  stats().calls++;
  const auto vehicle_poly =
    autoware_utils::to_footprint(pose, p.base_link2front, p.base_link2rear, p.vehicle_width);
  for (const auto & c : vehicle_poly.outer()) {
    stats().corners++;
    const int k = classify(ix, c.x(), c.y());
    if (k == 1) continue;
    if (k == 0) return false;
    stats().near++;
    const auto dist = boost::geometry::distance(c, lane_polygon);
    if (dist > 0.0) return false;
  }
  return true;
}
}  // namespace nero_side_fast

#endif  // NERO_SIDE_FAST_HPP_
