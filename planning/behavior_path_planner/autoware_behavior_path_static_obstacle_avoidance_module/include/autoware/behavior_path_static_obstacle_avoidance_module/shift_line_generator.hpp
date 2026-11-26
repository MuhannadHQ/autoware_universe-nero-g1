// Copyright 2023 TIER IV, Inc.
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

#ifndef AUTOWARE__BEHAVIOR_PATH_STATIC_OBSTACLE_AVOIDANCE_MODULE__SHIFT_LINE_GENERATOR_HPP_
#define AUTOWARE__BEHAVIOR_PATH_STATIC_OBSTACLE_AVOIDANCE_MODULE__SHIFT_LINE_GENERATOR_HPP_

#include "autoware/behavior_path_planner_common/utils/path_shifter/path_shifter.hpp"
#include "autoware/behavior_path_static_obstacle_avoidance_module/data_structs.hpp"
#include "autoware/behavior_path_static_obstacle_avoidance_module/helper.hpp"
#include "autoware/behavior_path_static_obstacle_avoidance_module/type_alias.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace autoware::behavior_path_planner::utils::static_obstacle_avoidance
{

// NERO_SOA_REPLAN_ANCHOR="hyst_m[,absent_s]" (ownlane-3 / Fable 2026-11-25, QUEUE OWNLANE-1122; Ahmed: "shouldnt it replan instead of
// hold one plan at the begenning of the monouver"): the whole avoidance manoeuvre is re-planned every cycle from the targets' current
// needs and, when that plan differs from the registered one, it REPLACES the registered lines, anchored to the robot's state (the
// plan's shift and lateral rate at ego). See shift_line_generator.cpp (neroAnchor*) and scene.cpp (neroAnchorResolve / neroAnchorApply).
// hyst_m: a target's registered level is dropped once its current need is this much lower (0.2). absent_s: the lines of a target that is
// still ahead and has not been a target for this long are dropped (1.0; 0 = never, the stock memory). Unset / 0 = the car.
// With it NERO_SOA_REPLAN_HORIZON is ignored and NERO_SOA_FLOOR_SPAN's reading is superseded.
// hold_m (third field, default 30; 0 = off, fb16): WEAVE (b) "1 hold the shift" (Ahmed, CAP1122-WEAVE): where this cycle's merged
// shift dips between two places that both need more within hold_m of each other (a V: down to a row's lower need and back up for the
// next close target), the dip is held at the lower of the two levels; the held level is never above a current need ahead.
namespace nero_ra
{
struct Cfg
{
  bool on{false};
  double hyst{0.2};
  double absent_s{1.0};
  double hold_m{30.0};
};
inline const Cfg & cfg()
{
  static const Cfg c = [] {
    Cfg r;
    const char * e = std::getenv("NERO_SOA_REPLAN_ANCHOR");
    if (e != nullptr) {
      double a = 0.0, b = 1.0, h = 30.0;
      const int n = std::sscanf(e, "%lf,%lf,%lf", &a, &b, &h);
      if (n >= 1 && a > 0.0) {
        r.on = true;
        r.hyst = a;
        if (n >= 2 && b >= 0.0) r.absent_s = b;
        if (n >= 3 && h >= 0.0) r.hold_m = h;
      }
    }
    std::fprintf(
      stderr, "[NERO_SOA] replan_anchor=%d hyst=%.2f absent_s=%.2f hold_m=%.1f\n", r.on, r.hyst, r.absent_s, r.hold_m);
    return r;
  }();
  return c;
}
}  // namespace nero_ra

using autoware::behavior_path_planner::PathShifter;
using autoware::behavior_path_planner::helper::static_obstacle_avoidance::AvoidanceHelper;

class ShiftLineGenerator
{
public:
  explicit ShiftLineGenerator(const std::shared_ptr<AvoidanceParameters> & parameters)
  : parameters_{parameters}
  {
  }

  void update(AvoidancePlanningData & data, DebugData & debug);

  void setRawRegisteredShiftLine(
    const AvoidLineArray & shift_lines, const AvoidancePlanningData & data);

  void setData(const std::shared_ptr<const PlannerData> & data) { data_ = data; }

  void setHelper(const std::shared_ptr<AvoidanceHelper> & helper) { helper_ = helper; }

  void setPathShifter(const PathShifter & path_shifter)
  {
    base_offset_ = path_shifter.getBaseOffset();
    if (path_shifter.getShiftLines().empty()) {
      last_ = std::nullopt;
    } else {
      last_ = path_shifter.getLastShiftLine().value();
    }
    // NERO_SOA_FLOOR_SPAN (ownlane-2): the registered line that ends furthest along the path. PathShifter::getLastShiftLine()
    // runs max_element with a greater-than comparator, so last_ is the line that ends NEAREST (kept as it is: upstream uses it)
    nero_furthest_ = std::nullopt;
    for (const auto & sl : path_shifter.getShiftLines()) {
      if (!nero_furthest_ || sl.end_idx > nero_furthest_.value().end_idx) {
        nero_furthest_ = sl;
      }
    }
    nero_ps_ = path_shifter;  // NERO_SOA_REPLAN_HORIZON: the registered lines, split into committed / re-plannable in update()
  }

  void reset()
  {
    last_ = std::nullopt;
    nero_furthest_ = std::nullopt;
    base_offset_ = 0.0;
    raw_.clear();
    raw_registered_.clear();
  }

  AvoidLineArray generate(const AvoidancePlanningData & data, DebugData & debug) const;

  AvoidLineArray getRawRegisteredShiftLine() const { return raw_registered_; }

  // NERO_SOA_REPLAN_ANCHOR: the generator's memory after a re-plan was registered. keep_until_lon >= -1e8: the registered line in
  // progress was kept (the memory before that place stays); n_chord: the first n registered lines were made by the anchor (carrier,
  // moved start, relaxed return): their memory is their own chord
  void neroAnchorSetMemory(
    const AvoidLineArray & lines, const AvoidancePlanningData & data, const double keep_until_lon, const size_t n_chord);
  // why this cycle's fresh plan differs from the registered one ("" = it does not) and where (m from ego)
  const std::string & neroAnchorWhy() const { return nero_ra_why_; }
  double neroAnchorDiffLon() const { return nero_ra_diff_lon_; }
  // a line end of the fresh plan that some raw line of this cycle (a target's avoid / plateau / return line, a memory line) ends at;
  // false for the end of the fill from ego (the linear plan's "hold ego's level for the prepare distance": not a place a level is needed)
  bool neroAnchorIsNeed(const AvoidLine & line) const;
  // this cycle's merged shift (the max of the raw lines, before the trim simplifies it to few lines) at a path point
  double neroAnchorEnvelope(const size_t idx, const double fallback) const
  {
    return idx < nero_ra_total_.size() ? nero_ra_total_.at(idx) : fallback;
  }

private:
  /**
   * @brief Calculate the shift points (start/end point, shift length) from the object lateral
   * and longitudinal positions in the Frenet coordinate. The jerk limit is also considered here.
   * @param avoidance data.
   * @param debug data.
   * @return processed shift lines.
   */
  AvoidOutlines generateAvoidOutline(AvoidancePlanningData & data, DebugData & debug) const;

  /*
   * @brief merge avoid outlines.
   * @param original shift lines.
   * @param debug data.
   * @return processed shift lines.
   */
  AvoidOutlines applyMergeProcess(
    const AvoidOutlines & outlines, const AvoidancePlanningData & data, DebugData & debug) const;

  /*
   * @brief fill gap between two shift lines.
   * @param original shift lines.
   * @param debug data.
   * @return processed shift lines.
   */
  AvoidOutlines applyFillGapProcess(
    const AvoidOutlines & outlines, const AvoidancePlanningData & data, DebugData & debug) const;

  /*
   * @brief generate candidate shift lines.
   * @param one-shot shift lines.
   * @param debug data.
   */
  AvoidLineArray generateCandidateShiftLine(
    const AvoidLineArray & shift_lines, const AvoidancePlanningData & data,
    DebugData & debug) const;

  /**
   * @brief clean up raw shift lines.
   * @param target shift lines.
   * @param debug data.
   * @return processed shift lines.
   * @details process flow:
   * 1. combine raw shirt lines and previous registered shift lines.
   * 2. add return shift line.
   * 3. merge raw shirt lines.
   * 4. trim unnecessary shift lines.
   */
  AvoidLineArray applyPreProcess(
    const AvoidOutlines & outlines, const AvoidancePlanningData & data, DebugData & debug) const;

  /*
   * @brief fill gap among shift lines.
   * @param original shift lines.
   * @param debug data.
   * @return processed shift lines.
   */
  AvoidLineArray applyFillGapProcess(
    const AvoidLineArray & shift_lines, const AvoidancePlanningData & data,
    DebugData & debug) const;

  /*
   * @brief merge negative & positive shift lines.
   * @param original shift lines.
   * @param debug data.
   * @return processed shift lines.
   */
  AvoidLineArray applyMergeProcess(
    const AvoidLineArray & shift_lines, const AvoidancePlanningData & data,
    DebugData & debug) const;

  /*
   * @brief add return shift line from ego position.
   * @param current raw shift line.
   * @param current registered shift line.
   * @param debug data.
   */
  AvoidLineArray applyCombineProcess(
    const AvoidLineArray & shift_lines, const AvoidLineArray & registered_lines,
    [[maybe_unused]] DebugData & debug) const;

  /*
   * @brief add return shift line from ego position.
   * @param shift lines which the return shift is added.
   * Pick up the last shift point, which is the most farthest from ego, from the current candidate
   * avoidance points and registered points in the shifter. If the last shift length of the point is
   * non-zero, add a return-shift to center line from the point. If there is no shift point in
   * candidate avoidance points nor registered points, and base_shift > 0, add a return-shift to
   * center line from ego.
   */
  AvoidLineArray addReturnShiftLine(
    const AvoidLineArray & shift_lines, const AvoidancePlanningData & data,
    DebugData & debug) const;

  /*
   * @brief extract shift lines from total shift lines based on their gradient.
   * @param shift length data.
   * @return extracted shift lines.
   */
  AvoidLineArray extractShiftLinesFromLine(
    const AvoidancePlanningData & data, ShiftLineData & shift_line_data) const;

  /*
   * @brief remove unnecessary avoid points
   * @param original shift lines.
   * @param debug data.
   * @return processed shift lines.
   * @details
   * - Combine avoid points that have almost same gradient
   * - Quantize the shift length to reduce the shift point noise
   * - Change the shift length to the previous one if the deviation is small.
   * - Remove unnecessary return shift (back to the center line).
   */
  AvoidLineArray applyTrimProcess(const AvoidLineArray & shift_lines, DebugData & debug) const;

  /*
   * @brief extract new shift lines based on current shifted path. the module makes a RTC request
   * for those new shift lines.
   * @param candidate shift lines.
   * @return new shift lines.
   */
  AvoidLineArray findNewShiftLine(const AvoidLineArray & shift_lines, DebugData & debug) const;

  /*
   * @brief generate total shift line. total shift line has shift length and gradient array.
   * @param raw shift lines.
   * @param total shift line.
   */
  void generateTotalShiftLine(
    const AvoidLineArray & avoid_points, const AvoidancePlanningData & data,
    ShiftLineData & shift_line_data) const;

  /*
   * @brief quantize shift line length.
   * @param target shift lines.
   * @param threshold. shift length is quantized by this value. (if it is 0.3[m], the output shift
   * length is 0.0, 0.3, 0.6...)
   */
  void applyQuantizeProcess(AvoidLineArray & shift_lines, const double threshold) const;

  /*
   * @brief trim shift line whose relative longitudinal distance is less than threshold.
   * @param target shift lines.
   * @param threshold.
   */
  void applySmallShiftFilter(AvoidLineArray & shift_lines, const double threshold) const;

  /*
   * @brief merge multiple shift lines whose relative gradient is less than threshold.
   * @param target shift lines.
   * @param threshold.
   */
  void applySimilarGradFilter(AvoidLineArray & shift_lines, const double threshold) const;

  /**
   * @brief update path index of the registered objects. remove old objects whose end point is
   * behind ego pose.
   */
  void updateRegisteredRawShiftLines(const AvoidancePlanningData & data);

  std::shared_ptr<const PlannerData> data_;

  std::shared_ptr<AvoidanceParameters> parameters_;

  std::shared_ptr<AvoidanceHelper> helper_;

  std::optional<ShiftLine> last_{std::nullopt};

  AvoidLineArray raw_;

  AvoidLineArray raw_registered_;

  std::optional<ShiftLine> nero_furthest_{std::nullopt};  // NERO_SOA_FLOOR_SPAN

  // NERO_SOA_REPLAN_HORIZON (ownlane-3): registered lines that start beyond ego + the nominal prepare distance are re-plannable.
  // The generator reads the registered path made of the COMMITTED lines only (those that start within the horizon); raw
  // registered lines of a target whose manoeuvre has not started are replaced by its current outline when its need fell.
  void neroReplanUpdate(const AvoidancePlanningData & data);
  void neroReplanRaw(const AvoidOutlines & outlines, const AvoidancePlanningData & data);
  double neroRegShift(const geometry_msgs::msg::Point & p) const;
  PathShifter nero_ps_;
  ShiftedPath nero_committed_;
  bool nero_rh_active_{false};  // some registered line starts beyond the horizon: the generator reads nero_committed_
  double nero_rh_horizon_{0.0};  // [m] from ego
  size_t nero_rh_n_committed_{0};
  size_t nero_rh_n_replannable_{0};
  std::optional<ShiftLine> nero_rh_furthest_{std::nullopt};  // the committed line that ends furthest (NERO_SOA_FLOOR_SPAN)
  double nero_rh_committed_end_{-1.0e9};  // [m] from ego: where the furthest committed line ends (-1e9: none)
  std::vector<std::string> nero_rh_started_;  // targets with a registered raw line that starts within the horizon (their avoidance is under way)

  // NERO_SOA_REPLAN_ANCHOR
  void neroAnchorMemory(const AvoidLineArray & current, const AvoidancePlanningData & data);
  AvoidLineArray neroAnchorCandidates(const AvoidLineArray & merged, const AvoidancePlanningData & data, DebugData & debug) const;
  mutable bool nero_ra_keep_near_{false};  // the trim keeps the lines that start within the prepare distance
  mutable std::string nero_ra_why_;
  mutable double nero_ra_diff_lon_{0.0};
  mutable std::vector<double> nero_ra_total_;
  // fb16 WEAVE (b): the merged shift's dips between two close higher places are held (generateTotalShiftLine); 1 = a held point
  void neroAnchorHold(const AvoidancePlanningData & data, ShiftLineData & sl) const;
  mutable std::vector<char> nero_ra_held_;
  // fb18: the trimmed plan is never under this cycle's merged needs (hold included) by more than quantize_size / 2 for two path
  // points in a row: where the trim (similar-gradient combine, short-line drop, the extraction's missed level step) cut under
  // them, the line is split on the needs there. The path indices of the knots it put in (they are needs for the carrier's pin).
  AvoidLineArray neroAnchorCover(const AvoidLineArray & trimmed, const AvoidancePlanningData & data) const;
  mutable std::vector<size_t> nero_ra_cover_;
  std::unordered_map<std::string, double> nero_ra_seen_;  // target uuid -> the last time [s] it was a target
  std::unordered_map<std::string, double> nero_ra_low_;  // target uuid -> since when [s] its need is smaller than its memory

  double base_offset_{0.0};
};

}  // namespace autoware::behavior_path_planner::utils::static_obstacle_avoidance

#endif  // AUTOWARE__BEHAVIOR_PATH_STATIC_OBSTACLE_AVOIDANCE_MODULE__SHIFT_LINE_GENERATOR_HPP_
