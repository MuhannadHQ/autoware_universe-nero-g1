// Copyright 2026 Nero (trjrate, 2026-11-09, QUEUE TRJ3)
//
// NERO_BPP_GEOM_FAST: three per-cycle geometry steps of behavior_path_planner's drivable lanes with the SAME results, so
// that their cost no longer grows with the map's bound point density (the car's 35313 / 35320 have 662 points per bound):
//   (a) utils::expandLanelets: lanelet::utils::getExpandedLanelet (lanelet2 offsetNoThrow + checkForInversion: an R-tree
//       over every bound point per call) is kept per (input lanelet data, inverted, left offset, right offset) for a
//       lanelet whose two bounds are map line strings (id > 0: never changed at run time); the entry holds the input
//       lanelet so its data cannot be freed and its address reused while cached; anything else is computed;
//   (b) utils::getOverlappedLaneletId: each drivable lane's lanelet polygons built once per call (upstream rebuilds both
//       polygons for every lane pair) and a pair whose bounding boxes are apart cannot overlap: skipped;
//   (c) drivable_area_expansion::expand_bound's self-intersection removal: a segment pair whose bounding boxes are apart by
//       more than 1 nm cannot intersect: skipped (upstream tests every pair: O(n^2) over the bound's points).
//   trjrate-2 (2026-11-09, the rest of TRJ3's margin), the same switch, the same results:
//   (d) utils::isInLanelets / isInLanelet (goal planner's isPathValid: every path point x every path lanelet) on a lanelet
//       polygon kept per lanelet object (upstream rebuilds polygon2d().basicPolygon() per call) + a box reject 1 um wider
//       than the radius;
//   (e) lanelet::utils::query::getClosestLanelet at BPP call sites: the same scan on kept polygons; a lanelet whose
//       bounding-box distance already exceeds the running minimum (with a rounding margin) cannot become a candidate and
//       is skipped; ties (several candidates) go to upstream's call on the candidates (the same order, the same angle rule);
//   (f) utils::postProcess's lane-overlap test after the goal on kept polygons, a pair with boxes apart cannot intersect;
//   (g) utils::resamplePathWithSpline: its "an s within 0.2 m already" test by binary search (upstream scans the whole
//       list per sample: O(n^2) on 1000-point paths) and motion_utils::resamplePath with a binary-search segment lookup
//       (nero_resample.hpp; upstream's calc_closest_segment_indices scans from the path end per sample);
//   (h) expand_bound's self-intersection loop: the segment pairs not box-apart are found through a grid, then tested in the
//       same order with the same rule as (c).
//   trjrate-2 v9, the same switch, the same results:
//   (i) utils::isEgoWithinOriginalLane(lanes, pose, ...): the combined lane shape (lanelet2_extension combineLaneletsShape: a
//       new lanelet with every bound point copied) kept per lanelet list (static avoidance asks it for every candidate-path
//       point in its hold / yield stop searches);
//   (j) lanelet2_extension getClosestSegment / getLaneletAngle / getClosestCenterPose: the same scan over the centerline
//       (the same per-segment boost distance, the first strictly smallest wins) without the 3 heap objects per segment, and
//       blocks of 16 segments whose bounding box is farther than the best so far (rounding margin) skipped;
//   (k) point-in-lanelet tests (isCentroidWithinLanelet(s), static avoidance's NERO blockers) on kept polygons.
//   trjrate-2 v10, the same switch, the same results:
//   (l) route_handler.getClosestLaneletWithinRoute(pose) (= getClosestLanelet over EVERY route lanelet, each polygon rebuilt):
//       the route lanelets among the map's R-tree hits within 10 m (then 40 m) of the pose, on kept polygons; used only when
//       exactly one lanelet holds the minimum (any tie or a minimum not provably inside the searched box: the route
//       handler's own call, whose order decides ties).
//   unset / 0 = the car (every step as upstream). 1 = fast. 2 = check: both computed on every call and compared exactly
//   (every bound point x / y / z, ids and attributes; the overlap index; the expanded bound); upstream's result is used;
//   totals every 10 s and at exit as [NERO_BPP_GEOM].
// Note (log only): lanelet2_extension's throttled "Fail to expand lanelet" error is printed when an expansion is computed,
// so with the cache a lanelet whose expansion inverts logs it once instead of once a second.
#pragma once

#include <autoware_lanelet2_extension/utility/query.hpp>
#include <autoware_lanelet2_extension/utility/utilities.hpp>

#include <autoware/route_handler/route_handler.hpp>

#include <boost/geometry/algorithms/comparable_distance.hpp>
#include <boost/geometry/algorithms/distance.hpp>
#include <boost/geometry/algorithms/intersects.hpp>
#include <boost/geometry/algorithms/within.hpp>

#include <geometry_msgs/msg/pose.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <lanelet2_core/geometry/LineString.h>
#include <lanelet2_core/geometry/Point.h>
#include <lanelet2_core/geometry/Polygon.h>  // boost::geometry adapters for lanelet's Eigen points / polygons
#include <lanelet2_core/primitives/Lanelet.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace autoware::behavior_path_planner::nero_geom_fast
{
inline int mode()
{
  static const int m = []() {
    const char * e = std::getenv("NERO_BPP_GEOM_FAST");
    if (!e || !*e) return 0;
    const int v = std::atoi(e);
    return (v == 1 || v == 2) ? v : 0;
  }();
  return m;
}

struct Stats
{
  std::atomic<uint64_t> calls{0}, hits{0}, bypass{0}, mismatch{0}, cleared{0};
  std::atomic<uint64_t> ov_calls{0}, ov_pairs{0}, ov_skipped{0}, ov_mismatch{0};
  std::atomic<uint64_t> eb_calls{0}, eb_pairs{0}, eb_skipped{0}, eb_mismatch{0};
  std::atomic<uint64_t> poly_built{0}, poly_dropped{0};
  std::atomic<uint64_t> in_calls{0}, in_reject{0}, in_mismatch{0};
  std::atomic<uint64_t> cl_calls{0}, cl_lanelets{0}, cl_skipped{0}, cl_ties{0}, cl_mismatch{0};
  std::atomic<uint64_t> pp_calls{0}, pp_pairs{0}, pp_skipped{0}, pp_mismatch{0};
  std::atomic<uint64_t> rs_calls{0}, rs_mismatch{0}, rp_calls{0}, rp_mismatch{0};
  std::atomic<uint64_t> ew_calls{0}, ew_built{0}, ew_mismatch{0};
  std::atomic<uint64_t> cs_calls{0}, cs_blocks{0}, cs_skipped{0}, cs_mismatch{0};
  std::atomic<uint64_t> wi_calls{0}, wi_mismatch{0};
  std::atomic<uint64_t> rc_calls{0}, rc_fast{0}, rc_mismatch{0};
  std::atomic<int64_t> last_print_ns{0};
};
inline Stats & stats()
{
  static Stats s;
  return s;
}
inline void poly_census(size_t & n, size_t & n_map, size_t & vertices);
inline void print_totals(const char * when)
{
  const auto & s = stats();
  size_t pn = 0;
  size_t pn_map = 0;
  size_t pv = 0;
  if (mode() != 0) poly_census(pn, pn_map, pv);
  std::fprintf(
    stderr,
    "[NERO_BPP_GEOM] %s mode %d: (a) getExpandedLanelet calls %llu (cached %llu, not map bounds %llu, cache cleared %llu, "
    "MISMATCH %llu) | (b) overlap calls %llu (lanelet pairs %llu, box-skipped %llu, MISMATCH %llu) | (c) expand_bound "
    "calls %llu (segment pairs %llu, box-skipped %llu, MISMATCH %llu) | polygons kept %llu (dropped unused %llu) | (d) "
    "isInLanelet calls %llu (box rejects %llu, MISMATCH %llu) | (e) closest lanelet calls %llu (lanelets %llu, "
    "box-skipped %llu, ties %llu, MISMATCH %llu) | (f) postProcess overlap calls %llu (pairs %llu, box-skipped %llu, "
    "MISMATCH %llu) | (g) resamplePathWithSpline calls %llu (MISMATCH %llu), resamplePath calls %llu (MISMATCH %llu) | "
    "polygon store now %zu (map lanelets %zu, vertices %zu) | (i) ego-within-lane calls %llu (combined shapes built %llu, "
    "MISMATCH %llu) | (j) closest centre segment calls %llu (blocks %llu, box-skipped %llu, MISMATCH %llu) | (k) "
    "within-lanelet calls %llu (MISMATCH %llu) | (l) route closest lanelet calls %llu (decided fast %llu, MISMATCH %llu)\n",
    when, mode(), (unsigned long long)s.calls.load(), (unsigned long long)s.hits.load(),
    (unsigned long long)s.bypass.load(), (unsigned long long)s.cleared.load(),
    (unsigned long long)s.mismatch.load(), (unsigned long long)s.ov_calls.load(),
    (unsigned long long)s.ov_pairs.load(), (unsigned long long)s.ov_skipped.load(),
    (unsigned long long)s.ov_mismatch.load(), (unsigned long long)s.eb_calls.load(),
    (unsigned long long)s.eb_pairs.load(), (unsigned long long)s.eb_skipped.load(),
    (unsigned long long)s.eb_mismatch.load(), (unsigned long long)s.poly_built.load(),
    (unsigned long long)s.poly_dropped.load(), (unsigned long long)s.in_calls.load(),
    (unsigned long long)s.in_reject.load(), (unsigned long long)s.in_mismatch.load(),
    (unsigned long long)s.cl_calls.load(), (unsigned long long)s.cl_lanelets.load(),
    (unsigned long long)s.cl_skipped.load(), (unsigned long long)s.cl_ties.load(),
    (unsigned long long)s.cl_mismatch.load(), (unsigned long long)s.pp_calls.load(),
    (unsigned long long)s.pp_pairs.load(), (unsigned long long)s.pp_skipped.load(),
    (unsigned long long)s.pp_mismatch.load(), (unsigned long long)s.rs_calls.load(),
    (unsigned long long)s.rs_mismatch.load(), (unsigned long long)s.rp_calls.load(),
    (unsigned long long)s.rp_mismatch.load(), pn, pn_map, pv, (unsigned long long)s.ew_calls.load(),
    (unsigned long long)s.ew_built.load(), (unsigned long long)s.ew_mismatch.load(),
    (unsigned long long)s.cs_calls.load(), (unsigned long long)s.cs_blocks.load(),
    (unsigned long long)s.cs_skipped.load(), (unsigned long long)s.cs_mismatch.load(),
    (unsigned long long)s.wi_calls.load(), (unsigned long long)s.wi_mismatch.load(),
    (unsigned long long)s.rc_calls.load(), (unsigned long long)s.rc_fast.load(),
    (unsigned long long)s.rc_mismatch.load());
  std::fflush(stderr);
}
struct ExitPrinter
{
  ~ExitPrinter()
  {
    if (mode() != 0) print_totals("exit");
  }
};
inline void maybe_print()
{
  static ExitPrinter at_exit;
  (void)at_exit;
  const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
  auto & last = stats().last_print_ns;
  int64_t prev = last.load();
  if (now - prev > 10'000'000'000LL && last.compare_exchange_strong(prev, now)) {
    if (prev != 0) print_totals("10s");
  }
}

inline bool same_bound(const lanelet::ConstLineString3d & a, const lanelet::ConstLineString3d & b)
{
  if (a.size() != b.size() || a.id() != b.id()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    const auto & p = a[i];
    const auto & q = b[i];
    if (p.x() != q.x() || p.y() != q.y() || p.z() != q.z()) return false;
  }
  return true;
}
inline bool same_lanelet(const lanelet::ConstLanelet & a, const lanelet::ConstLanelet & b)
{
  return a.id() == b.id() && same_bound(a.leftBound(), b.leftBound()) &&
         same_bound(a.rightBound(), b.rightBound()) && a.attributes() == b.attributes();
}

// (a) lanelet::utils::getExpandedLanelet with the same result (mode 1), checked (mode 2), or it (mode 0)
inline lanelet::ConstLanelet getExpandedLanelet(
  const lanelet::ConstLanelet & ll, const double left_offset, const double right_offset)
{
  const int m = mode();
  if (m == 0) return lanelet::utils::getExpandedLanelet(ll, left_offset, right_offset);
  stats().calls++;
  maybe_print();
  if (!ll.constData() || ll.leftBound().id() <= 0 || ll.rightBound().id() <= 0) {
    stats().bypass++;
    return lanelet::utils::getExpandedLanelet(ll, left_offset, right_offset);
  }
  using Key = std::tuple<uintptr_t, bool, uint64_t, uint64_t>;
  uint64_t lo = 0;
  uint64_t ro = 0;
  std::memcpy(&lo, &left_offset, sizeof(lo));
  std::memcpy(&ro, &right_offset, sizeof(ro));
  const Key key{reinterpret_cast<uintptr_t>(ll.constData().get()), ll.inverted(), lo, ro};
  struct Entry
  {
    lanelet::ConstLanelet input;  // keeps the input's data alive: its address is not reused while cached
    lanelet::ConstLanelet output;
  };
  static std::mutex mtx;
  static std::map<Key, Entry> cache;
  {
    std::lock_guard<std::mutex> lock(mtx);
    const auto it = cache.find(key);
    if (it != cache.end()) {
      stats().hits++;
      if (m == 2) {
        const auto fresh = lanelet::utils::getExpandedLanelet(ll, left_offset, right_offset);
        if (!same_lanelet(fresh, it->second.output)) stats().mismatch++;
        return fresh;
      }
      return it->second.output;
    }
  }
  const auto out = lanelet::utils::getExpandedLanelet(ll, left_offset, right_offset);
  // the only lazily computed member of a lanelet (LaneletData::centerline_): computed here once, so threads sharing the
  // cached lanelet afterwards only read it
  (void)out.centerline();
  std::lock_guard<std::mutex> lock(mtx);
  if (cache.size() > 20000) {
    cache.clear();
    stats().cleared++;
  }
  cache.emplace(key, Entry{ll, out});
  return out;
}

// a 2D axis-aligned box; apart() = separated by more than `margin` in x or y (then nothing inside them can touch)
struct Box
{
  double minx{0.0}, miny{0.0}, maxx{0.0}, maxy{0.0};
  template <class Pts>
  static Box of(const Pts & pts)
  {
    Box b{std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
          std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest()};
    for (const auto & p : pts) {
      b.minx = std::min(b.minx, static_cast<double>(p.x()));
      b.miny = std::min(b.miny, static_cast<double>(p.y()));
      b.maxx = std::max(b.maxx, static_cast<double>(p.x()));
      b.maxy = std::max(b.maxy, static_cast<double>(p.y()));
    }
    return b;
  }
  bool apart(const Box & o, const double margin) const
  {
    return o.minx > maxx + margin || minx > o.maxx + margin || o.miny > maxy + margin ||
           miny > o.maxy + margin;
  }
};

// ---------------------------------------------------------------------------------------------------------------------
// trjrate-2: a lanelet polygon kept per lanelet object, exactly lanelet.polygon2d().basicPolygon() (what upstream's
// isInLanelet / getClosestLanelet measure against). Key = the data pointer + inverted flag; the entry holds the lanelet
// (its data cannot be freed and the address reused while kept); a lanelet's bounds never change after it is made; entries
// unused for 30 s are dropped (a sweep every 10 s); shared_ptr entries so a caller keeps its polygon while another thread
// sweeps.
struct PolyEntry
{
  lanelet::ConstLanelet keep;
  lanelet::BasicPolygon2d poly;
  Box box;
  std::atomic<int64_t> last_use_ns{0};
};
inline int64_t now_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}
struct PolyStore
{
  std::mutex mtx;
  std::unordered_map<uintptr_t, std::shared_ptr<PolyEntry>> map;
  int64_t last_sweep_ns{0};
};
inline PolyStore & poly_store()
{
  static PolyStore * s = new PolyStore;  // never destroyed: the exit totals read it during static destruction
  return *s;
}
// store census for the totals line: entries, of them map lanelets (id and both bound ids > 0), vertices kept
inline void poly_census(size_t & n, size_t & n_map, size_t & vertices)
{
  auto & st = poly_store();
  std::lock_guard<std::mutex> lock(st.mtx);
  n = st.map.size();
  n_map = 0;
  vertices = 0;
  for (const auto & kv : st.map) {
    const auto & ll = kv.second->keep;
    if (ll.id() > 0 && ll.leftBound().id() > 0 && ll.rightBound().id() > 0) ++n_map;
    vertices += kv.second->poly.size();
  }
}
inline std::shared_ptr<const PolyEntry> kept_polygon(const lanelet::ConstLanelet & ll)
{
  auto & st = poly_store();
  const int64_t now = now_ns();
  const uintptr_t key =
    reinterpret_cast<uintptr_t>(ll.constData().get()) ^ static_cast<uintptr_t>(ll.inverted() ? 1 : 0);
  {
    std::lock_guard<std::mutex> lock(st.mtx);
    if (now - st.last_sweep_ns > 10'000'000'000LL) {
      if (st.last_sweep_ns != 0) {
        for (auto it = st.map.begin(); it != st.map.end();) {
          if (now - it->second->last_use_ns.load() > 30'000'000'000LL) {
            it = st.map.erase(it);
            stats().poly_dropped++;
          } else {
            ++it;
          }
        }
      }
      st.last_sweep_ns = now;
    }
    if (st.map.size() > 20000) {  // a bound on memory whatever creates lanelets: start again (results unchanged)
      stats().poly_dropped += st.map.size();
      st.map.clear();
    }
    const auto it = st.map.find(key);
    if (it != st.map.end()) {
      it->second->last_use_ns = now;
      return it->second;
    }
  }
  auto e = std::make_shared<PolyEntry>();
  e->keep = ll;
  e->poly = ll.polygon2d().basicPolygon();
  e->box = Box::of(e->poly);
  e->last_use_ns = now;
  std::lock_guard<std::mutex> lock(st.mtx);
  const auto ins = st.map.emplace(key, e);
  if (ins.second) stats().poly_built++;
  return ins.first->second;
}
// squared distance from p to the box (0 inside): a lower bound of the squared distance to anything inside the box
inline double box_dist2(const Box & b, const double x, const double y)
{
  const double dx = std::max({b.minx - x, 0.0, x - b.maxx});
  const double dy = std::max({b.miny - y, 0.0, y - b.maxy});
  return dx * dx + dy * dy;
}

// (d) lanelet::utils::isInLanelet(pose, ll, radius) with the same result
inline bool isInLaneletFast(
  const geometry_msgs::msg::Pose & pose, const lanelet::ConstLanelet & ll, const double radius)
{
  const auto e = kept_polygon(ll);
  constexpr double eps = 1.0e-9;  // upstream's
  const lanelet::BasicPoint2d p(pose.position.x, pose.position.y);
  const double r = radius + eps;
  const double rr = r + 1.0e-6;  // the reject is 1 um wider than upstream's bar: rounding can never flip it
  if (rr > 0.0 && box_dist2(e->box, p.x(), p.y()) >= rr * rr) {
    stats().in_reject++;
    return false;
  }
  return boost::geometry::distance(p, e->poly) < r;
}
inline bool isInLanelet(
  const geometry_msgs::msg::Pose & pose, const lanelet::ConstLanelet & ll, const double radius = 0.0)
{
  const int m = mode();
  if (m == 0 || !ll.constData()) return lanelet::utils::isInLanelet(pose, ll, radius);
  stats().in_calls++;
  maybe_print();
  const bool fast = isInLaneletFast(pose, ll, radius);
  if (m == 2) {
    const bool ref = lanelet::utils::isInLanelet(pose, ll, radius);
    if (ref != fast) stats().in_mismatch++;
    return ref;
  }
  return fast;
}

// (e) lanelet::utils::query::getClosestLanelet(lanelets, pose, &out) with the same result and the same return value
inline bool getClosestLaneletFast(
  const lanelet::ConstLanelets & lanelets, const geometry_msgs::msg::Pose & search_pose,
  lanelet::ConstLanelet * closest_lanelet_ptr)
{
  if (closest_lanelet_ptr == nullptr || lanelets.empty()) {
    return lanelet::utils::query::getClosestLanelet(lanelets, search_pose, closest_lanelet_ptr);
  }
  bool found = false;
  const lanelet::BasicPoint2d search_point(search_pose.position.x, search_pose.position.y);
  lanelet::ConstLanelets candidate_lanelets;
  double min_distance = std::numeric_limits<double>::max();
  constexpr double eps = std::numeric_limits<double>::epsilon();
  for (const auto & llt : lanelets) {
    stats().cl_lanelets++;
    if (!llt.constData()) return lanelet::utils::query::getClosestLanelet(lanelets, search_pose, closest_lanelet_ptr);
    const auto e = kept_polygon(llt);
    // a lower bound of boost's comparable (squared) distance; shrunk by a relative 1e-9 and 1e-12 absolute so that no
    // rounding of either value can make a skipped lanelet a tie or a new minimum
    const double lb = box_dist2(e->box, search_point.x(), search_point.y()) * (1.0 - 1.0e-9) - 1.0e-12;
    if (min_distance != std::numeric_limits<double>::max() && lb > min_distance + 2.0 * eps) {
      stats().cl_skipped++;
      continue;
    }
    const double distance = boost::geometry::comparable_distance(e->poly, search_point);
    if (std::abs(distance - min_distance) <= eps) {
      candidate_lanelets.push_back(llt);
    } else if (distance < min_distance) {
      found = true;
      candidate_lanelets.clear();
      candidate_lanelets.push_back(llt);
      min_distance = distance;
    }
  }
  if (candidate_lanelets.size() == 1) {
    *closest_lanelet_ptr = candidate_lanelets.at(0);
    return found;
  }
  // several candidates within epsilon of the minimum (or none): upstream on the candidates gives the same list in the same
  // order (each is within epsilon of the first) and then its own angle rule
  stats().cl_ties++;
  if (candidate_lanelets.empty()) {
    return lanelet::utils::query::getClosestLanelet(lanelets, search_pose, closest_lanelet_ptr);
  }
  lanelet::utils::query::getClosestLanelet(candidate_lanelets, search_pose, closest_lanelet_ptr);
  return found;
}
inline bool same_id_and_data(const lanelet::ConstLanelet & a, const lanelet::ConstLanelet & b)
{
  return a.id() == b.id() && a.constData() == b.constData() && a.inverted() == b.inverted();
}
inline bool getClosestLanelet(
  const lanelet::ConstLanelets & lanelets, const geometry_msgs::msg::Pose & search_pose,
  lanelet::ConstLanelet * closest_lanelet_ptr)
{
  const int m = mode();
  if (m == 0) return lanelet::utils::query::getClosestLanelet(lanelets, search_pose, closest_lanelet_ptr);
  stats().cl_calls++;
  maybe_print();
  if (m == 2 && closest_lanelet_ptr != nullptr) {
    lanelet::ConstLanelet fast_ll = *closest_lanelet_ptr;  // the output is left as it was when nothing is found
    const bool fast = getClosestLaneletFast(lanelets, search_pose, &fast_ll);
    const bool ref = lanelet::utils::query::getClosestLanelet(lanelets, search_pose, closest_lanelet_ptr);
    if (fast != ref || !same_id_and_data(fast_ll, *closest_lanelet_ptr)) stats().cl_mismatch++;
    return ref;
  }
  return getClosestLaneletFast(lanelets, search_pose, closest_lanelet_ptr);
}

// (f) boost::geometry::intersects(a.polygon2d().basicPolygon(), b.polygon2d().basicPolygon()) with the same result
inline bool lanelets_intersect(const lanelet::ConstLanelet & a, const lanelet::ConstLanelet & b)
{
  const int m = mode();
  if (m == 0 || !a.constData() || !b.constData()) {
    return boost::geometry::intersects(a.polygon2d().basicPolygon(), b.polygon2d().basicPolygon());
  }
  stats().pp_pairs++;
  const auto ea = kept_polygon(a);
  const auto eb = kept_polygon(b);
  bool fast = false;
  if (ea->box.apart(eb->box, 1.0e-6)) {
    stats().pp_skipped++;
  } else {
    fast = boost::geometry::intersects(ea->poly, eb->poly);
  }
  if (m == 2) {
    const bool ref =
      boost::geometry::intersects(a.polygon2d().basicPolygon(), b.polygon2d().basicPolygon());
    if (ref != fast) stats().pp_mismatch++;
    return ref;
  }
  return fast;
}



// (i) the combined lane polygon (= combineLaneletsShape(lanes).polygon2d().basicPolygon()) kept per lanelet list (the data
// pointers + inverted flags in order; the entry holds the lanelets); 16 lists kept, the least recently used replaced
inline std::shared_ptr<const lanelet::BasicPolygon2d> combined_lane_polygon(const lanelet::ConstLanelets & lanes)
{
  struct Entry
  {
    std::vector<std::pair<uintptr_t, bool>> key;
    lanelet::ConstLanelets keep;
    std::shared_ptr<const lanelet::BasicPolygon2d> poly;
    int64_t last_use_ns{0};
  };
  static std::mutex mtx;
  static std::vector<Entry> cache;
  std::vector<std::pair<uintptr_t, bool>> key;
  key.reserve(lanes.size());
  for (const auto & ll : lanes) key.emplace_back(reinterpret_cast<uintptr_t>(ll.constData().get()), ll.inverted());
  const int64_t now = now_ns();
  {
    std::lock_guard<std::mutex> lock(mtx);
    for (auto & e : cache) {
      if (e.key == key) {
        e.last_use_ns = now;
        return e.poly;
      }
    }
  }
  auto poly = std::make_shared<const lanelet::BasicPolygon2d>(
    lanelet::utils::combineLaneletsShape(lanes).polygon2d().basicPolygon());
  stats().ew_built++;
  std::lock_guard<std::mutex> lock(mtx);
  if (cache.size() >= 16) {
    auto oldest = std::min_element(cache.begin(), cache.end(), [](const auto & a, const auto & b) {
      return a.last_use_ns < b.last_use_ns;
    });
    cache.erase(oldest);
  }
  cache.push_back(Entry{key, lanes, poly, now});
  return poly;
}

// (j) a linestring's segment boxes in blocks of 16 segments (segment i = points i-1, i), kept per linestring object
struct SegBlocks
{
  lanelet::ConstLineString3d keep;
  std::vector<Box> blocks;
  std::atomic<int64_t> last_use_ns{0};
};
constexpr size_t kSegBlock = 16;
inline std::shared_ptr<const SegBlocks> kept_seg_blocks(const lanelet::ConstLineString3d & ls)
{
  static std::mutex mtx;
  static std::unordered_map<uintptr_t, std::shared_ptr<SegBlocks>> store;
  static int64_t last_sweep_ns = 0;
  const int64_t now = now_ns();
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
    Box b{std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
          std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest()};
    for (size_t i = first - 1; i <= last; ++i) {
      b.minx = std::min(b.minx, static_cast<double>(ls[i].x()));
      b.miny = std::min(b.miny, static_cast<double>(ls[i].y()));
      b.maxx = std::max(b.maxx, static_cast<double>(ls[i].x()));
      b.maxy = std::max(b.maxy, static_cast<double>(ls[i].y()));
    }
    e->blocks.push_back(b);
  }
  e->last_use_ns = now;
  std::lock_guard<std::mutex> lock(mtx);
  return store.emplace(key, e).first->second;
}
// the segment lanelet2_extension's getClosestSegment(search_pt, ls) returns, as its end index i (points i-1, i); 0 = the
// empty segment (fewer than 2 points, or no distance below the initial maximum)
inline size_t closest_segment_index(const lanelet::BasicPoint2d & search_pt, const lanelet::ConstLineString3d & ls)
{
  const size_t n = ls.size();
  if (n < 2) return 0;
  stats().cs_calls++;
  const auto sb = kept_seg_blocks(ls);
  lanelet::BasicLineString2d seg(2);
  double min_distance = std::numeric_limits<double>::max();
  size_t best = 0;
  for (size_t b = 0; b < sb->blocks.size(); ++b) {
    stats().cs_blocks++;
    if (min_distance != std::numeric_limits<double>::max()) {
      // the box distance is a lower bound of every segment distance in the block; shrunk for rounding
      const double lb = std::sqrt(box_dist2(sb->blocks[b], search_pt.x(), search_pt.y())) * (1.0 - 1.0e-9) - 1.0e-9;
      if (lb > min_distance) {
        stats().cs_skipped++;
        continue;
      }
    }
    const size_t first = 1 + b * kSegBlock;
    const size_t last = std::min(n - 1, first + kSegBlock - 1);
    for (size_t i = first; i <= last; ++i) {
      seg[0] = lanelet::BasicPoint2d(ls[i - 1].x(), ls[i - 1].y());
      seg[1] = lanelet::BasicPoint2d(ls[i].x(), ls[i].y());
      const double distance = lanelet::geometry::distance2d(seg, search_pt);
      if (distance < min_distance) {
        best = i;
        min_distance = distance;
      }
    }
  }
  return best;
}
inline double lanelet_angle_from(const lanelet::ConstLineString3d & cl, const size_t i)
{
  return std::atan2(cl[i].y() - cl[i - 1].y(), cl[i].x() - cl[i - 1].x());
}
inline bool bits_eq(const double a, const double b)
{
  uint64_t x = 0;
  uint64_t y = 0;
  std::memcpy(&x, &a, sizeof(x));
  std::memcpy(&y, &b, sizeof(y));
  return x == y;
}
// = lanelet::utils::getLaneletAngle(lanelet, search_point)
inline double getLaneletAngle(const lanelet::ConstLanelet & lanelet, const geometry_msgs::msg::Point & search_point)
{
  const int m = mode();
  if (m == 0) return lanelet::utils::getLaneletAngle(lanelet, search_point);
  maybe_print();
  const auto cl = lanelet.centerline();
  const size_t i = closest_segment_index(lanelet::BasicPoint2d(search_point.x, search_point.y), cl);
  if (i == 0) return lanelet::utils::getLaneletAngle(lanelet, search_point);  // upstream's own (empty segment) path
  const double fast = lanelet_angle_from(cl, i);
  if (m == 2) {
    const double ref = lanelet::utils::getLaneletAngle(lanelet, search_point);
    if (!bits_eq(ref, fast)) stats().cs_mismatch++;
    return ref;
  }
  return fast;
}
// = lanelet::utils::getClosestCenterPose(lanelet, search_point)
inline geometry_msgs::msg::Pose getClosestCenterPose(
  const lanelet::ConstLanelet & lanelet, const geometry_msgs::msg::Point & search_point)
{
  const int m = mode();
  if (m == 0) return lanelet::utils::getClosestCenterPose(lanelet, search_point);
  maybe_print();
  const auto cl = lanelet.centerline();
  if (cl.size() < 2) return lanelet::utils::getClosestCenterPose(lanelet, search_point);
  const size_t i = closest_segment_index(lanelet::BasicPoint2d(search_point.x, search_point.y), cl);
  if (i == 0) return lanelet::utils::getClosestCenterPose(lanelet, search_point);
  // the same arithmetic as upstream on the same segment's points
  const lanelet::BasicPoint2d front(cl[i - 1].x(), cl[i - 1].y());
  const lanelet::BasicPoint2d back(cl[i].x(), cl[i].y());
  const Eigen::Vector2d direction((back - front).normalized());
  const Eigen::Vector2d xf(front);
  const Eigen::Vector2d x(search_point.x, search_point.y);
  const Eigen::Vector2d p = xf + (x - xf).dot(direction) * direction;
  geometry_msgs::msg::Pose fast;
  fast.position.x = p.x();
  fast.position.y = p.y();
  fast.position.z = search_point.z;
  tf2::Quaternion q;
  q.setRPY(0, 0, lanelet_angle_from(cl, i));
  fast.orientation = tf2::toMsg(q);
  if (m == 2) {
    const auto ref = lanelet::utils::getClosestCenterPose(lanelet, search_point);
    if (
      !bits_eq(ref.position.x, fast.position.x) || !bits_eq(ref.position.y, fast.position.y) ||
      !bits_eq(ref.position.z, fast.position.z) || !bits_eq(ref.orientation.x, fast.orientation.x) ||
      !bits_eq(ref.orientation.y, fast.orientation.y) || !bits_eq(ref.orientation.z, fast.orientation.z) ||
      !bits_eq(ref.orientation.w, fast.orientation.w)) {
      stats().cs_mismatch++;
    }
    return ref;
  }
  return fast;
}
// (k) boost::geometry::within(point, lanelet.polygon2d().basicPolygon()) on the kept polygon (+ an exact box reject)
inline bool withinLanelet(const lanelet::BasicPoint2d & p, const lanelet::ConstLanelet & ll)
{
  const int m = mode();
  if (m == 0 || !ll.constData()) return boost::geometry::within(p, ll.polygon2d().basicPolygon());
  stats().wi_calls++;
  maybe_print();
  const auto e = kept_polygon(ll);
  // a point outside the polygon's bounding box is not within it
  const bool fast = (p.x() < e->box.minx || p.x() > e->box.maxx || p.y() < e->box.miny || p.y() > e->box.maxy)
                      ? false
                      : boost::geometry::within(p, e->poly);
  if (m == 2) {
    const bool ref = boost::geometry::within(p, ll.polygon2d().basicPolygon());
    if (ref != fast) stats().wi_mismatch++;
    return ref;
  }
  return fast;
}


// (l) = route_handler.getClosestLaneletWithinRoute(pose, &out)
inline bool getClosestLaneletWithinRoute(
  const autoware::route_handler::RouteHandler & rh, const geometry_msgs::msg::Pose & pose,
  lanelet::ConstLanelet * closest_lanelet)
{
  const int m = mode();
  if (m == 0 || closest_lanelet == nullptr) return rh.getClosestLaneletWithinRoute(pose, closest_lanelet);
  stats().rc_calls++;
  maybe_print();
  bool decided = false;
  lanelet::ConstLanelet fast_ll;
  const auto map = rh.getLaneletMapPtr();
  if (map) {
    const lanelet::BasicPoint2d p(pose.position.x, pose.position.y);
    constexpr double eps = std::numeric_limits<double>::epsilon();
    for (const double R : {10.0, 40.0}) {
      const lanelet::BoundingBox2d box(
        lanelet::BasicPoint2d(p.x() - R, p.y() - R), lanelet::BasicPoint2d(p.x() + R, p.y() + R));
      const auto hits = map->laneletLayer.search(box);
      std::vector<std::pair<double, lanelet::ConstLanelet>> ds;
      double best = std::numeric_limits<double>::max();
      for (const auto & ll : hits) {
        const lanelet::ConstLanelet cll = ll;
        if (!rh.isRouteLanelet(cll)) continue;
        const auto e = kept_polygon(cll);
        const double d = boost::geometry::comparable_distance(e->poly, p);  // upstream's arguments, the same polygon
        ds.emplace_back(d, cll);
        best = std::min(best, d);
      }
      // every route lanelet outside the box is at least R away (its box misses the search box): comparable >= R^2
      if (ds.empty() || !(best < R * R * (1.0 - 1.0e-9))) continue;
      size_t close = 0;
      for (const auto & dl : ds) {
        if (dl.first <= best + 4.0 * eps) {
          ++close;
          fast_ll = dl.second;
        }
      }
      decided = (close == 1);  // a tie: upstream decides by its route order
      break;
    }
  }
  if (decided) stats().rc_fast++;
  if (!decided || m == 2) {
    const bool ref = rh.getClosestLaneletWithinRoute(pose, closest_lanelet);
    if (decided && (!ref || !same_id_and_data(fast_ll, *closest_lanelet))) stats().rc_mismatch++;
    return ref;
  }
  *closest_lanelet = fast_ll;
  return true;
}

}  // namespace autoware::behavior_path_planner::nero_geom_fast
