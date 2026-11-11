// Copyright 2026 Nero (trjrate, 2026-11-09, QUEUE TRJ3)
//
// NERO_IX_GEOM_CACHE: the intersection module's per-object lanelet test (checkAngleForTargetLanelets ->
// lanelet::utils::isInLanelet) without rebuilding the lanelet's polygon on every call.
//   unset / 0 = the car: lanelet::utils::isInLanelet on every call (it builds polygon2d().basicPolygon() each time:
//       joinSubStrings + an inversion check with an R-tree over every bound point; the module's attention lanelets are
//       merged lanelets made once per module instance, e.g. the left turn 146634's oncoming 35320 has 662 points per bound,
//       and the test runs per object x attention lanelet x instance every cycle).
//   1 = fast, the SAME result: the polygon is built once per lanelet object (its data pointer + inverted flag; the entry
//       holds the lanelet, so its data cannot be freed and the address reused while cached; a lanelet is never changed
//       after it is made; entries unused for 30 s are dropped) and a bounding-box reject 1 um wider than the radius skips
//       the distance when the point is clearly outside (the polygon lies inside its box: the result cannot change).
//   2 = check: both computed on every call and compared; upstream's result is used; totals every 10 s and at exit as
//       [NERO_IX_GEOM].
// trjrate-2 (2026-11-09), the same switch: util::generateInterpolatedPath's spline of the whole path at 0.1 m per module
//   instance per cycle (behavior_velocity_planner_common's splineInterpolate = motion_utils::resamplePath, whose segment
//   lookup scans from the path end for every one of ~3000 samples) by nero_resample.hpp's exact replica (binary-search
//   lookup, lane ids moved): the same path bit for bit (2 = check, every field of every point compared); and the module's
//   0.1 m path is no longer copied three times per instance per cycle on its way out of prepareIntersectionData.
// trjrate-2 v9, the same switch: lanelet2_extension getLaneletAngle (getClosestSegment over the centerline allocates three
//   objects per segment) by the same scan without the allocations, blocks of 16 segments whose bounding box is farther than
//   the best so far (rounding margin) skipped: the same angle bit for bit (2 = check).
// Header-only (inline functions with function-local statics: one instance per library), no new source file.
#pragma once

#include <autoware_lanelet2_extension/utility/utilities.hpp>

#include <geometry_msgs/msg/pose.hpp>

#include <boost/geometry/algorithms/distance.hpp>

#include <lanelet2_core/geometry/LineString.h>
#include <lanelet2_core/geometry/Point.h>
#include <lanelet2_core/geometry/Polygon.h>  // boost::geometry adapters for lanelet's Eigen points / polygons
#include <lanelet2_core/primitives/Lanelet.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <array>
#include <utility>

namespace autoware::behavior_velocity_planner::nero_geom
{
inline int mode()
{
  static const int m = []() {
    const char * e = std::getenv("NERO_IX_GEOM_CACHE");
    if (!e || !*e) return 0;
    const int v = std::atoi(e);
    return (v == 1 || v == 2) ? v : 0;
  }();
  return m;
}

struct Stats
{
  std::atomic<uint64_t> calls{0}, built{0}, reject{0}, mismatch{0}, dropped{0};
  std::atomic<uint64_t> sp_calls{0}, sp_mismatch{0}, sp_reused{0};
  std::atomic<uint64_t> la_calls{0}, la_blocks{0}, la_skipped{0}, la_mismatch{0};
  std::atomic<int64_t> last_print_ns{0};
};
inline Stats & stats()
{
  static Stats s;
  return s;
}
inline void print_totals(const char * when)
{
  const auto & s = stats();
  std::fprintf(
    stderr,
    "[NERO_IX_GEOM] %s mode %d: isInLanelet calls %llu (polygons built %llu, dropped unused %llu, box rejects %llu, "
    "MISMATCH %llu) | spline calls %llu (geometry reused %llu, MISMATCH %llu) | lanelet angle calls %llu (blocks %llu, box-skipped %llu, "
    "MISMATCH %llu)\n",
    when, mode(), (unsigned long long)s.calls.load(), (unsigned long long)s.built.load(),
    (unsigned long long)s.dropped.load(), (unsigned long long)s.reject.load(),
    (unsigned long long)s.mismatch.load(), (unsigned long long)s.sp_calls.load(),
    (unsigned long long)s.sp_reused.load(), (unsigned long long)s.sp_mismatch.load(), (unsigned long long)s.la_calls.load(),
    (unsigned long long)s.la_blocks.load(), (unsigned long long)s.la_skipped.load(),
    (unsigned long long)s.la_mismatch.load());
  std::fflush(stderr);
}
struct ExitPrinter
{
  ~ExitPrinter()
  {
    if (mode() != 0) print_totals("exit");
  }
};

struct PolyEntry
{
  lanelet::ConstLanelet keep;  // holds the lanelet's data: its address cannot be reused while cached
  lanelet::BasicPolygon2d poly;
  double minx{0.0}, miny{0.0}, maxx{0.0}, maxy{0.0};
  int64_t last_use_ns{0};
};

// lanelet::utils::isInLanelet with the same result (mode 1), checked against it (mode 2), or it (mode 0)
inline bool isInLanelet(
  const geometry_msgs::msg::Pose & pose, const lanelet::ConstLanelet & ll, const double radius)
{
  const int m = mode();
  if (m == 0 || !ll.constData()) return lanelet::utils::isInLanelet(pose, ll, radius);
  static ExitPrinter at_exit;
  (void)at_exit;
  static std::mutex mtx;
  static std::unordered_map<uintptr_t, PolyEntry> cache;
  static int64_t last_sweep_ns = 0;
  const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
  auto & s = stats();
  s.calls++;
  bool fast = false;
  {
    std::lock_guard<std::mutex> lock(mtx);  // held for the whole test: an entry is never dropped while in use
    if (now - last_sweep_ns > 10'000'000'000LL) {
      if (last_sweep_ns != 0) {
        for (auto it = cache.begin(); it != cache.end();) {
          if (now - it->second.last_use_ns > 30'000'000'000LL) {
            it = cache.erase(it);
            s.dropped++;
          } else {
            ++it;
          }
        }
        print_totals("10s");
      }
      last_sweep_ns = now;
    }
    const uintptr_t key =
      reinterpret_cast<uintptr_t>(ll.constData().get()) ^ static_cast<uintptr_t>(ll.inverted() ? 1 : 0);
    auto it = cache.find(key);
    if (it == cache.end()) {
      PolyEntry e;
      e.keep = ll;
      e.poly = ll.polygon2d().basicPolygon();  // exactly what upstream's isInLanelet measures against
      e.minx = e.miny = std::numeric_limits<double>::max();
      e.maxx = e.maxy = std::numeric_limits<double>::lowest();
      for (const auto & p : e.poly) {
        e.minx = std::min(e.minx, p.x());
        e.miny = std::min(e.miny, p.y());
        e.maxx = std::max(e.maxx, p.x());
        e.maxy = std::max(e.maxy, p.y());
      }
      s.built++;
      it = cache.emplace(key, std::move(e)).first;
    }
    auto & e = it->second;
    e.last_use_ns = now;
    constexpr double eps = 1.0e-9;  // upstream's
    const lanelet::BasicPoint2d p(pose.position.x, pose.position.y);
    const double r = radius + eps;
    const double rr = r + 1.0e-6;  // the reject is 1 um wider than upstream's bar: rounding can never flip it
    const double dx = std::max({e.minx - p.x(), 0.0, p.x() - e.maxx});
    const double dy = std::max({e.miny - p.y(), 0.0, p.y() - e.maxy});
    if (rr > 0.0 && dx * dx + dy * dy >= rr * rr) {
      s.reject++;
      fast = false;
    } else {
      fast = boost::geometry::distance(p, e.poly) < r;
    }
  }
  if (m == 2) {
    const bool ref = lanelet::utils::isInLanelet(pose, ll, radius);
    if (ref != fast) s.mismatch++;
    return ref;
  }
  return fast;
}

// trjrate-2 v9: lanelet::utils::getLaneletAngle(lanelet, point) with the same result
struct SegBlocks
{
  lanelet::ConstLineString3d keep;
  std::vector<std::array<double, 4>> blocks;  // minx, miny, maxx, maxy of segments [1 + 16 b, 16 + 16 b]
  std::atomic<int64_t> last_use_ns{0};
};
constexpr size_t kSegBlock = 16;
inline std::shared_ptr<const SegBlocks> kept_seg_blocks(const lanelet::ConstLineString3d & ls, const int64_t now)
{
  static std::mutex mtx;
  static std::unordered_map<uintptr_t, std::shared_ptr<SegBlocks>> store;
  static int64_t last_sweep_ns = 0;
  const uintptr_t key =
    reinterpret_cast<uintptr_t>(ls.constData().get()) ^ static_cast<uintptr_t>(ls.inverted() ? 1 : 0);
  {
    std::lock_guard<std::mutex> lock(mtx);
    if (now - last_sweep_ns > 10'000'000'000LL) {
      if (last_sweep_ns != 0) {
        for (auto it = store.begin(); it != store.end();) {
          if (now - it->second->last_use_ns.load() > 30'000'000'000LL) {
            it = store.erase(it);
          } else {
            ++it;
          }
        }
      }
      last_sweep_ns = now;
    }
    if (store.size() > 20000) store.clear();
    const auto it = store.find(key);
    if (it != store.end()) {
      it->second->last_use_ns = now;
      return it->second;
    }
  }
  auto e = std::make_shared<SegBlocks>();
  e->keep = ls;
  const size_t n = ls.size();
  for (size_t first = 1; first < n; first += kSegBlock) {
    const size_t last = std::min(n - 1, first + kSegBlock - 1);
    std::array<double, 4> b{std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
                            std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest()};
    for (size_t i = first - 1; i <= last; ++i) {
      b[0] = std::min(b[0], static_cast<double>(ls[i].x()));
      b[1] = std::min(b[1], static_cast<double>(ls[i].y()));
      b[2] = std::max(b[2], static_cast<double>(ls[i].x()));
      b[3] = std::max(b[3], static_cast<double>(ls[i].y()));
    }
    e->blocks.push_back(b);
  }
  e->last_use_ns = now;
  std::lock_guard<std::mutex> lock(mtx);
  return store.emplace(key, e).first->second;
}
inline double getLaneletAngle(const lanelet::ConstLanelet & lanelet, const geometry_msgs::msg::Point & search_point)
{
  const int m = mode();
  if (m == 0) return lanelet::utils::getLaneletAngle(lanelet, search_point);
  static ExitPrinter at_exit;
  (void)at_exit;
  auto & s = stats();
  s.la_calls++;
  const auto cl = lanelet.centerline();
  const size_t n = cl.size();
  if (n < 2) return lanelet::utils::getLaneletAngle(lanelet, search_point);  // upstream's own (empty segment) path
  const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
  const auto sb = kept_seg_blocks(cl, now);
  const lanelet::BasicPoint2d search_pt(search_point.x, search_point.y);
  lanelet::BasicLineString2d seg(2);
  double min_distance = std::numeric_limits<double>::max();
  size_t best = 0;
  for (size_t b = 0; b < sb->blocks.size(); ++b) {
    s.la_blocks++;
    if (min_distance != std::numeric_limits<double>::max()) {
      const auto & bb = sb->blocks[b];
      const double dx = std::max({bb[0] - search_pt.x(), 0.0, search_pt.x() - bb[2]});
      const double dy = std::max({bb[1] - search_pt.y(), 0.0, search_pt.y() - bb[3]});
      if (std::sqrt(dx * dx + dy * dy) * (1.0 - 1.0e-9) - 1.0e-9 > min_distance) {
        s.la_skipped++;
        continue;
      }
    }
    const size_t first = 1 + b * kSegBlock;
    const size_t last = std::min(n - 1, first + kSegBlock - 1);
    for (size_t i = first; i <= last; ++i) {
      seg[0] = lanelet::BasicPoint2d(cl[i - 1].x(), cl[i - 1].y());
      seg[1] = lanelet::BasicPoint2d(cl[i].x(), cl[i].y());
      const double distance = lanelet::geometry::distance2d(seg, search_pt);
      if (distance < min_distance) {
        best = i;
        min_distance = distance;
      }
    }
  }
  if (best == 0) return lanelet::utils::getLaneletAngle(lanelet, search_point);
  const double fast = std::atan2(cl[best].y() - cl[best - 1].y(), cl[best].x() - cl[best - 1].x());
  if (m == 2) {
    const double ref = lanelet::utils::getLaneletAngle(lanelet, search_point);
    uint64_t x = 0;
    uint64_t y = 0;
    std::memcpy(&x, &ref, sizeof(x));
    std::memcpy(&y, &fast, sizeof(y));
    if (x != y) s.la_mismatch++;
    return ref;
  }
  return fast;
}
}  // namespace autoware::behavior_velocity_planner::nero_geom
