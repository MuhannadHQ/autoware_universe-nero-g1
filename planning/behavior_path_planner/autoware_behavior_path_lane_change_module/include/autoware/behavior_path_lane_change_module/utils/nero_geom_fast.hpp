// NERO_LC_GEOM_FAST (10amimprovement 2026-11-16, gapperf_1117): the lane change candidate loop's two hottest per-point
// geometry calls, computed with the SAME numbers and decisions, faster.
// Profile (gperftools, 11-08 17:28 logsim window, set1117): with gap-ahead's extra prepare durations the candidate loop
// (get_path_using_path_shifter) took 2455 of 18190 BPP samples vs 803 without; per candidate:
//  (a) path_footprint_exceeds_target_lane_bound: for EVERY path point (~300, the candidate runs 300 m along the target lane)
//      lanelet2's toArcCoordinates on the combined target-lane bound. For a line of >= 50 points lanelet2
//      (LineStringGeometry.cpp projectedPointL2PWithTree, 1.2.1) BUILDS AN R-TREE OF ALL SEGMENTS ON EVERY CALL, then walks
//      it in nearest order. Here the combined bound and that same R-tree (same values, same order, same bgi::linear<8>
//      packing = the same tree) are built once per (lanes, side) and lanelet2's own query loop / update rule / signed
//      distance (signedDistanceImpl: project + distance + isLeftOf) run on it. toArcCoordinates' arc length (unused: only
//      .distance is read) is not computed.
//  (b) construct_candidate_path: motion_utils::findNearestIndex(points, POSE) without max_dist / max_yaw computes a yaw
//      deviation (tf2 -> atan2) for every new minimum, which can never reject (|yaw| > DBL_MAX is false): the POINT
//      overload returns the same index (strict '<' first minimum in both).
// 0 / unset = the car; 1 = fast; 2 = check: both computed, the ORIGINAL result used, mismatches counted
// ([NERO_LC_GEOM_FAST] lines every 20 s and at exit). Never for decisions of its own: the numbers are lanelet2's.
#ifndef AUTOWARE__BEHAVIOR_PATH_LANE_CHANGE_MODULE__UTILS__NERO_GEOM_FAST_HPP_
#define AUTOWARE__BEHAVIOR_PATH_LANE_CHANGE_MODULE__UTILS__NERO_GEOM_FAST_HPP_

#include <autoware_lanelet2_extension/utility/utilities.hpp>
#include <rclcpp/rclcpp.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>

#include <lanelet2_core/geometry/LineString.h>
#include <lanelet2_core/primitives/Lanelet.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace lanelet::geometry::internal
{
// lanelet2 1.2.1 signedDistanceImpl's body with the closest segment given (same namespace, same unqualified calls)
template <typename LineStringT, typename PointT, typename SegmentT>
std::pair<double, ProjectedPointInfo<traits::BasicPointT<PointT>>> neroSignedDistanceWithSegment(
  const LineStringT & lineString, const PointT & p, const SegmentT & nextSegment)
{
  const auto basicP = utils::toBasicPoint(p);
  const auto projPoint = lanelet::geometry::project(utils::toBasicSegment(nextSegment), basicP);
  const auto d = distance(projPoint, p);
  ProjectedPointInfo<traits::BasicPointT<PointT>> ppInfo{nextSegment, projPoint};
  const auto isLeft = isLeftOf(lineString, p, ppInfo);
  return {isLeft ? d : -d, ppInfo};
}
}  // namespace lanelet::geometry::internal

namespace autoware::behavior_path_planner::utils::lane_change::nero_geom_fast
{
inline int mode()
{
  static const int m = [] {
    const char * v = std::getenv("NERO_LC_GEOM_FAST");
    const int r = v ? std::atoi(v) : 0;
    if (r > 0) {
      RCLCPP_INFO(
        rclcpp::get_logger("lane_change"),
        "[NERO_LC_GEOM_FAST] %d (1: target-bound R-tree kept per lanes + nearest index by position, 2: check = both "
        "computed, the original used, mismatches counted)", r);
    }
    return r;
  }();
  return m;
}

struct Stats
{
  std::atomic<uint64_t> bound_calls{0}, bound_pts{0}, bound_mismatch{0}, trees{0}, small{0};
  std::atomic<uint64_t> nearest_calls{0}, nearest_mismatch{0};
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
    rclcpp::get_logger("lane_change"),
    "[NERO_LC_GEOM_FAST] %s mode %d: bound checks %lu (%lu points, %lu mismatches, %lu trees built, %lu short bounds) | "
    "nearest %lu (%lu mismatches)", why, mode(), static_cast<unsigned long>(s.bound_calls.load()),
    static_cast<unsigned long>(s.bound_pts.load()), static_cast<unsigned long>(s.bound_mismatch.load()),
    static_cast<unsigned long>(s.trees.load()), static_cast<unsigned long>(s.small.load()),
    static_cast<unsigned long>(s.nearest_calls.load()), static_cast<unsigned long>(s.nearest_mismatch.load()));
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

// the combined target lanes' bound on one side + lanelet2's per-query R-tree, built once
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
using TreePointT = lanelet::ConstPoint2d;
using TreeSegmT = lanelet::Segment<TreePointT>;
using Box = bg::model::box<bg::model::point<double, 2, bg::cs::cartesian>>;
using Node = std::pair<Box, TreeSegmT>;
using RTree = bgi::rtree<Node, bgi::linear<8>>;
constexpr std::size_t kRtreeThres = 50;  // lanelet2 1.2.1 LineStringGeometry.cpp RtreeThres

struct Bound
{
  std::vector<lanelet::Id> ids;
  bool left{false};
  lanelet::ConstLanelet lane;    // the combined lanelet (keeps the bound's points alive)
  lanelet::ConstLineString2d ls;  // its left / right bound in 2d
  std::unique_ptr<RTree> tree;    // only for >= kRtreeThres points (lanelet2 brute-forces shorter lines)
};

inline std::shared_ptr<const Bound> bound(const lanelet::ConstLanelets & lanes, const bool left)
{
  std::vector<lanelet::Id> ids;
  ids.reserve(lanes.size());
  for (const auto & l : lanes) ids.push_back(l.id());
  // a few recent (lanes, side) pairs; the lanelets' geometry is fixed for an id within one map (the first and last point
  // of the bound are compared too, so a reloaded map with other geometry builds anew)
  static std::mutex mtx;
  static std::vector<std::shared_ptr<const Bound>> cache;
  {
    std::lock_guard<std::mutex> lk(mtx);
    for (const auto & b : cache) {
      if (b->left != left || b->ids != ids) continue;
      const auto & ref = left ? lanes.front().leftBound2d() : lanes.front().rightBound2d();
      const auto & ref2 = left ? lanes.back().leftBound2d() : lanes.back().rightBound2d();
      if (
        !ref.empty() && !ref2.empty() && !b->ls.empty() && b->ls.front().x() == ref.front().x() &&
        b->ls.front().y() == ref.front().y() && b->ls.back().x() == ref2.back().x() &&
        b->ls.back().y() == ref2.back().y()) {
        return b;
      }
    }
  }
  auto b = std::make_shared<Bound>();
  b->ids = ids;
  b->left = left;
  b->lane = lanelet::utils::combineLaneletsShape(lanes);
  b->ls = left ? b->lane.leftBound2d() : b->lane.rightBound2d();
  if (b->ls.size() >= kRtreeThres) {
    std::vector<Node> values;
    values.reserve(b->ls.size() - 1);
    for (std::size_t i = 0; i + 1 < b->ls.size(); ++i) {
      const auto & p1 = b->ls[i];
      const auto & p2 = b->ls[i + 1];
      Box box;
      bg::envelope(std::make_pair(lanelet::utils::toBasicPoint(p1), lanelet::utils::toBasicPoint(p2)), box);
      values.emplace_back(box, TreeSegmT(p1, p2));
    }
    b->tree = std::make_unique<RTree>(values.begin(), values.end());
    stats().trees++;
  }
  std::lock_guard<std::mutex> lk(mtx);
  cache.insert(cache.begin(), b);
  if (cache.size() > 8) cache.pop_back();
  return b;
}

// lanelet2 1.2.1 ProjectedPointL2PResult::update's projection (LineStringGeometry.cpp projectedPoint(l1, l2, p))
inline lanelet::BasicPoint2d projected_point(
  lanelet::BasicPoint2d l1, lanelet::BasicPoint2d l2, lanelet::BasicPoint2d p)
{
  lanelet::BasicPoint2d v = l2 - l1;
  lanelet::BasicPoint2d w = p - l1;
  const auto c1 = w.dot(v);
  if (c1 <= 0) {
    return l1;
  }
  const auto c2 = v.dot(v);
  if (c2 <= c1) {
    return l2;
  }
  const auto b = c1 / c2;
  return v * b + l1;
}

// = lanelet::geometry::signedDistance(b.ls, p) (lanelet2 1.2.1: closestSegment = projectedPointL2PWithTree on the same tree)
inline double signed_distance(const Bound & b, const lanelet::BasicPoint2d & p)
{
  if (!b.tree) {
    stats().small++;
    return lanelet::geometry::signedDistance(b.ls, p);
  }
  std::optional<double> distance;
  TreeSegmT segm{};
  for (auto qIt = b.tree->qbegin(bgi::nearest(p, unsigned(b.tree->size()))); qIt != b.tree->qend(); ++qIt) {
    const auto & nearest = *qIt;
    auto dBox = boost::geometry::distance(nearest.first, p);
    if (!!distance && *distance < dBox) {
      break;
    }
    // update()
    const auto projP = projected_point(
      lanelet::utils::toBasicPoint(nearest.second.first), lanelet::utils::toBasicPoint(nearest.second.second), p);
    const auto dNew = (projP - p).norm();
    if (!distance || *distance > dNew) {
      distance = dNew;
      segm = nearest.second;
    }
    if (*distance == 0.) {
      break;
    }
  }
  return lanelet::geometry::internal::neroSignedDistanceWithSegment(b.ls, p, segm).first;
}
}  // namespace autoware::behavior_path_planner::utils::lane_change::nero_geom_fast

#endif  // AUTOWARE__BEHAVIOR_PATH_LANE_CHANGE_MODULE__UTILS__NERO_GEOM_FAST_HPP_
