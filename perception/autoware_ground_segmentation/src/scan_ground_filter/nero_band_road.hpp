// NERO (rampground, QUEUE PITCHG, 2026-10-31): the obstacle band's ceiling measured from the ROAD AHEAD, not from the
// tilted base_link plane. The crop box keeps points up to max_z (vehicle_height + margin_max_z) above base_link. With the
// body pitched / rolled against the road ahead (ramp or bump crest, turn exit) that ceiling sits far above the road
// 20-30 m out, and things hanging over the road (a branch 0.85-1.7 m up, 10-27 20:48:03.8) enter the band, cluster into an
// UNKNOWN and stop the robot. A level body never sees them.
//
// After the ground filter, per frame: a road profile per 1 deg azimuth sector from the filter's own GROUND points (median z
// per 0.5 m range bin, walked outward from the robot with a slope trend; a bin off the trend is not road and the walk never
// re-anchors past a jump), the road height under each obstacle point (interpolated between the road bins around it, within
// 8 m; near side only with a rising trend; neighbour sectors +-2 deg as a fallback; otherwise unknown = kept). An obstacle
// point is taken out of the obstacle cloud only if ALL hold:
//   - it is ahead of the robot within cone_deg of straight ahead (|y| <= x tan(cone_deg): where the robot drives; a
//     lower lot or embankment beside the road is not the road ahead) and within rmax;
//   - the road under it is known and lies at least max(gate_m, range x tan(gate_deg)) below the base plane (the tilt really
//     lifted the ceiling there);
//   - it is more than band + margin above that road (band = the crop's own height, so the band is [road, road + band] as on
//     level ground);
//   - nothing stands under it: no obstacle point of its 3 x 3 cells of 0.5 m is within support_h of the road (or has an
//     unknown road), and no GROUND point there lies ground_low..support_h above the road (a wall's foot the filter calls
//     ground). Anything standing on the road keeps all its points.
// Nothing is reclassified as ground and nothing is added. Switch:
//   NERO_GROUND_BAND_ROAD=<band>,<margin>,<support_h>,<ground_low>,<gate_m>,<gate_deg>,<rmax>,<cone_deg>
// (unset = the car: this code never runs). Plain C++ on float arrays: the same file runs in the offline harness.
#ifndef SCAN_GROUND_FILTER__NERO_BAND_ROAD_HPP_
#define SCAN_GROUND_FILTER__NERO_BAND_ROAD_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace autoware::ground_segmentation::nero
{
class BandRoad
{
public:
  float band = 0.52f, margin = 0.15f, support_h = 0.3f, ground_low = 0.15f, gate = 0.3f, gate_deg = 2.0f, rmax = 30.0f,
        cone_deg = 40.0f;

  // parse "<band>,<margin>,<support_h>,<ground_low>,<gate_m>,<gate_deg>,<rmax>,<cone_deg>"; false = invalid (the caller
  // leaves it off)
  bool parse(const char * s)
  {
    if (!s || !*s) return false;
    float v[8];
    char tail = 0;
    if (std::sscanf(s, "%f,%f,%f,%f,%f,%f,%f,%f%c", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &tail) != 8)
      return false;
    if (!(v[0] > 0.0f && v[1] >= 0.0f && v[2] >= 0.0f && v[3] >= 0.0f && v[3] <= v[2] && v[4] >= 0.0f && v[5] >= 0.0f &&
          v[5] < 45.0f && v[6] > 2.0f && v[6] <= kRmax && v[7] > 0.0f && v[7] < 80.0f))
      return false;
    band = v[0]; margin = v[1]; support_h = v[2]; ground_low = v[3]; gate = v[4]; gate_deg = v[5]; rmax = v[6];
    cone_deg = v[7];
    tan_gate_ = std::tan(static_cast<double>(gate_deg) * M_PI / 180.0);
    tan_cone_ = static_cast<float>(std::tan(static_cast<double>(cone_deg) * M_PI / 180.0));
    return true;
  }

  struct Stats
  {
    size_t removed = 0;
    // (set when removed > 0) drop20: the road 20 m straight ahead (base z)
    float r_min = 0.0f, r_max = 0.0f, h_min = 0.0f, drop20 = 0.0f;
  };

  // xyz: n points (x, y, z consecutive at `stride` floats); ng: 1 = obstacle (in/out: removed points set to 0)
  Stats apply(const float * xyz, size_t stride, size_t n, std::vector<uint8_t> & ng)
  {
    Stats st;
    build_profile(xyz, stride, n, ng);
    // candidates (before the support test): ahead within the cone and rmax, the road under them known and gated, above
    // band + margin
    const double hs = std::min(static_cast<double>(band), static_cast<double>(support_h));
    cand_.clear();
    cest_.clear();
    float bx0 = 0.0f, bx1 = 0.0f, by0 = 0.0f, by1 = 0.0f;  // the candidates' bounding box
    for (size_t i = 0; i < n; ++i) {
      if (!ng[i]) continue;
      const float * p = xyz + i * stride;
      if (!(p[0] > 0.0f) || !(std::abs(p[1]) <= p[0] * tan_cone_)) continue;
      const float r = std::hypot(p[0], p[1]);
      if (!(r < rmax) || !act_[sector(p[0], p[1])]) continue;  // (no hot sector around: its road passes no gate)
      const double e = road_at_nb(p[0], p[1]), h = static_cast<double>(p[2]) - e;
      if (!std::isfinite(h) || !(h > static_cast<double>(band) + margin)) continue;
      if (!(e < -std::max(static_cast<double>(gate), r * tan_gate_))) continue;
      if (cand_.empty()) { bx0 = bx1 = p[0]; by0 = by1 = p[1]; }
      bx0 = std::min(bx0, p[0]); bx1 = std::max(bx1, p[0]); by0 = std::min(by0, p[1]); by1 = std::max(by1, p[1]);
      cand_.push_back(static_cast<uint32_t>(i));
      cest_.push_back(e);
    }
    if (cand_.empty()) return st;
    // only the 3 x 3 cells around a candidate can support it: everything below looks at those cells only (a point in
    // them lies < 2 cells = 1 m from a candidate in x and in y: the box test is a cheap superset)
    near_.assign(kNX * kNY, 0);
    for (const uint32_t i : cand_) dilate_into(near_, cell(xyz[i * stride], xyz[i * stride + 1]));
    bx0 -= kBox; bx1 += kBox; by0 -= kBox; by1 += kBox;
    // support cells: an obstacle point near the road (or with an unknown road) marks its cell; so does a GROUND point
    // ground_low..support_h above the road
    sup_.assign(kNX * kNY, 0);
    for (size_t i = 0; i < n; ++i) {
      if (!ng[i]) continue;
      const float * p = xyz + i * stride;
      if (!(p[0] >= bx0 && p[0] <= bx1 && p[1] >= by0 && p[1] <= by1)) continue;
      const int c = cell(p[0], p[1]);
      if (!near_[c] || sup_[c]) continue;
      const double h = static_cast<double>(p[2]) - road_at_nb(p[0], p[1]);
      if (!std::isfinite(h) || h <= hs) sup_[c] = 1;
    }
    for (size_t i = 0; i < n; ++i) {
      if (ng[i]) continue;
      const float * p = xyz + i * stride;
      if (!(p[0] >= bx0 && p[0] <= bx1 && p[1] >= by0 && p[1] <= by1)) continue;
      const int c = cell(p[0], p[1]);
      if (!near_[c] || sup_[c]) continue;
      const double h = static_cast<double>(p[2]) - road_at_nb(p[0], p[1]);
      if (std::isfinite(h) && h > ground_low && h <= hs) sup_[c] = 1;
    }
    for (size_t k = 0; k < cand_.size(); ++k) {
      const uint32_t i = cand_[k];
      const float * p = xyz + i * stride;
      if (supported(cell(p[0], p[1]))) continue;
      const float r = std::hypot(p[0], p[1]);
      const double h = static_cast<double>(p[2]) - cest_[k];
      ng[i] = 0;
      if (!st.removed || r < st.r_min) st.r_min = r;
      if (!st.removed || r > st.r_max) st.r_max = r;
      if (!st.removed || h < st.h_min) st.h_min = static_cast<float>(h);
      ++st.removed;
    }
    if (st.removed) st.drop20 = static_cast<float>(road_at(20.0f, 0.0f));
    return st;
  }

private:
  static constexpr int kNS = 360;                      // 1 deg sectors
  static constexpr float kRB = 0.5f;                   // range bin (m)
  static constexpr float kRmax = 60.0f;                // profile range
  static constexpr int kNB = 120;                      // kRmax / kRB
  static constexpr double kT0 = 0.15, kG = 0.03;       // road bin iff |z - pred| <= kT0 + kG * dr
  static constexpr double kGap = 8.0, kNear = 3.0;     // interpolation gap; near-side-only reach
  static constexpr int kNBS = 2;                       // neighbour sectors for the fallback
  static constexpr float kCS = 0.5f, kX0 = -60.0f, kY0 = -60.0f;
  static constexpr int kNX = 320, kNY = 240;           // support grid x -60..100, y -60..60
  static constexpr float kBox = 1.01f;                 // > 2 cells: the candidates' box grown by it holds their 3 x 3 cells
  static constexpr float kA0 = 0.99999663461921584f, kA1 = -0.3331830260567602f, kA2 = 0.19813210704817072f,
                         kA3 = -0.13247511695101466f, kA4 = 0.079810996780818336f, kA5 = -0.033725753083124796f,
                         kA6 = 0.006842562354758594f;  // atan(t) ~ t (kA0 + kA1 t^2 + ... + kA6 t^12), t in [0, 1]
  static constexpr float kPif = 3.14159265358979f, kPi2f = 1.57079632679490f, kDegf = 57.29577951308232f;
  static constexpr float kGuardS = 2e-3f, kGuardB = 2e-4f;  // sector (deg) / bin guards of fast_cells
  static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

  // What the profile needs (the rest cannot change any result; the per-frame cost is what decides the car's load):
  //  - every road query is for a point in a candidate's 3 x 3 cells (candidates: x > 0, r < rmax), i.e. x > -0.5 and
  //    r < rmax + 1.42; a query with r < 2 returns 0 before it looks at a sector. With r >= 2 and x > -0.5 the point's
  //    angle is within +-104.5 deg, so its sector +-kNBS lies in 73..286 = angles within +-107 deg: x >= -0.2924 r.
  //    The test keeps x >= -0.3 r.
  //  - a query in bin b reads road bins up to b + kGap / kRB (interpolation partner), and a road bin depends only on the
  //    bins before it: bins from nbp() on are never read.
  //  - a sector's road bins (its walk: one median per bin) are computed on its first query only, and the candidate
  //    search queries only points whose sector +-kNBS holds a HOT sector (build_profile).
  int nbp() const { return std::min(kNB, static_cast<int>((rmax + 1.5f) / kRB) + 1 + static_cast<int>(kGap / kRB) + 2); }

  std::vector<double> rd_, tr_, cest_;
  std::vector<int> prv_, nxt_;
  std::vector<uint32_t> cnt_, start_;
  std::vector<float> zs_, px_, py_, pz_, pr_;
  std::vector<int32_t> pc_;
  std::vector<uint8_t> pok_, hot_, act_, walked_;
  int nb_ = 0;
  std::vector<uint8_t> sup_, near_;
  std::vector<uint32_t> cand_;
  double tan_gate_ = std::tan(2.0 * M_PI / 180.0);
  float tan_cone_ = 0.8390996f;  // tan(40 deg)

  static int sector(float x, float y)
  {
    const float deg = std::atan2(y, x) * 57.29577951308232f + 180.0f;
    return static_cast<int>(deg) % kNS;
  }
  // (sector * kNB + bin, or -1 for bin >= nb) of m points; ok = 0: within a guard of an edge (the caller takes the
  // exact calls). Written so that GCC vectorises it on x86 and aarch64: selects of constants only, and no sqrt (r =
  // sqrt(x^2 + y^2) comes in).
  static void fast_cells(
    const float * __restrict X, const float * __restrict Y, const float * __restrict R, size_t m, int32_t nb,
    int32_t * __restrict C, uint8_t * __restrict OK)
  {
    for (size_t k = 0; k < m; ++k) {
      const float x = X[k], y = Y[k];
      const float ax = std::fabs(x), ay = std::fabs(y);
      const float mx = ax > ay ? ax : ay, mn = ax > ay ? ay : ax;
      const float t = mn / (mx + 1e-30f), t2 = t * t;
      float a = t * (kA0 + t2 * (kA1 + t2 * (kA2 + t2 * (kA3 + t2 * (kA4 + t2 * (kA5 + t2 * kA6))))));
      a = (ay > ax ? kPi2f : 0.0f) + (ay > ax ? -1.0f : 1.0f) * a;  // the octant
      a = (x < 0.0f ? kPif : 0.0f) + (x < 0.0f ? -1.0f : 1.0f) * a;
      a = (y < 0.0f ? -1.0f : 1.0f) * a;
      const float d = a * kDegf + 180.0f;  // 0..360
      const int32_t s = static_cast<int32_t>(d);
      const float fs = d - static_cast<float>(s);
      const float q = R[k] * (1.0f / kRB);
      const int32_t bq = static_cast<int32_t>(q);
      const float fb = q - static_cast<float>(bq);
      OK[k] = (mx > 0.0f) & (fs >= kGuardS) & (fs <= 1.0f - kGuardS) & (fb >= kGuardB) & (fb <= 1.0f - kGuardB);
      C[k] = bq < nb ? s * kNB + bq : -1;
    }
  }
  static void dilate_into(std::vector<uint8_t> & g, int c)
  {
    const int ix = c / kNY, iy = c % kNY;
    for (int a = std::max(ix - 1, 0); a <= std::min(ix + 1, kNX - 1); ++a)
      for (int b = std::max(iy - 1, 0); b <= std::min(iy + 1, kNY - 1); ++b) g[a * kNY + b] = 1;
  }
  bool supported(int c) const  // any support cell in the 3 x 3 cells around c
  {
    const int ix = c / kNY, iy = c % kNY;
    for (int a = std::max(ix - 1, 0); a <= std::min(ix + 1, kNX - 1); ++a)
      for (int b = std::max(iy - 1, 0); b <= std::min(iy + 1, kNY - 1); ++b)
        if (sup_[a * kNY + b]) return true;
    return false;
  }
  static int cell(float x, float y)
  {
    const int ix = std::clamp(static_cast<int>((x - kX0) / kCS), 0, kNX - 1);
    const int iy = std::clamp(static_cast<int>((y - kY0) / kCS), 0, kNY - 1);
    return ix * kNY + iy;
  }

  void build_profile(const float * xyz, size_t stride, size_t n, const std::vector<uint8_t> & ng)
  {
    const int nb = nbp();
    const size_t nc = static_cast<size_t>(kNS) * kNB;
    const float r2max = (nb * kRB + 1.0f) * (nb * kRB + 1.0f);
    // 1. the ground points (the filter's classification) the profile needs, compacted (branch-free)
    if (px_.size() < n + 1) {
      px_.resize(n + 1);
      py_.resize(n + 1);
      pz_.resize(n + 1);
      pr_.resize(n + 1);
    }
    size_t m = 0;
    for (size_t i = 0; i < n; ++i) {
      const float * p = xyz + i * stride;
      const float r2 = p[0] * p[0] + p[1] * p[1];
      px_[m] = p[0];
      py_[m] = p[1];
      pz_[m] = p[2];
      pr_[m] = r2;
      // (NaN fails r2 < r2max)
      m += static_cast<size_t>(!ng[i] & (r2 < r2max) & !((p[0] < 0.0f) & (p[0] * p[0] > 0.09f * r2)));
    }
    for (size_t k = 0; k < m; ++k) pr_[k] = std::sqrt(pr_[k]);
    // 2. their (sector, bin) = the same integers as sector(x, y) and int(hypot(x, y) / kRB), in one straight loop the
    //    compiler vectorises: a polynomial atan (fitted on [0, 1], max error 1.6e-5 deg) and a sqrt in float. Values
    //    within a guard of a sector / bin edge (2e-3 deg / 2e-4 bin, > 10 x the error of either side) are flagged and
    //    take the exact calls (3.), so every integer is the exact one (checked on every point of the w31 / w4 / w2 /
    //    10-12 replays and a dense synthetic sweep).
    if (pc_.size() < m) {
      pc_.resize(m);
      pok_.resize(m);
    }
    fast_cells(px_.data(), py_.data(), pr_.data(), m, nb, pc_.data(), pok_.data());
    for (size_t k = 0; k < m; ++k) {  // 3. the flagged few: the exact calls
      if (pok_[k]) continue;
      const int bq = static_cast<int>(std::hypot(px_[k], py_[k]) / kRB);
      pc_[k] = bq >= 0 && bq < nb ? sector(px_[k], py_[k]) * kNB + bq : -1;
    }
    // counts per (sector, bin) and the HOT sectors: a sector none of whose ground points lies below
    // -max(gate, (range - 9 m) tan(gate_deg)) + 1 mm has every road bin above that line (a bin's median is one of its
    // points). An estimate at range r reads bins < r + 8.5 m (its own, the previous road bin, the next within kGap) and
    // stays at or above the lowest bin it reads (interpolation, + the near side's rising trend, medians of neighbours),
    // so it lies >= 1 mm above the gate -max(gate, r tan(gate_deg)): a point whose sector +-kNBS holds no hot sector is
    // never a candidate.
    cnt_.assign(nc + 1, 0);
    hot_.assign(kNS, 0);
    const double g0 = static_cast<double>(gate) - 1e-3, reach = kGap + 2.0 * kRB;
    for (size_t k = 0; k < m; ++k) {
      if (pc_[k] < 0) continue;
      ++cnt_[pc_[k] + 1];
      const double z = pz_[k];
      if (z < -g0 && z < -std::max(static_cast<double>(gate), std::max(0.0, pr_[k] - reach) * tan_gate_) + 1e-3)
        hot_[pc_[k] / kNB] = 1;
    }
    act_.assign(kNS, 0);
    for (int sct = 0; sct < kNS; ++sct)
      if (hot_[sct])
        for (int d = -kNBS; d <= kNBS; ++d) act_[((sct + d) % kNS + kNS) % kNS] = 1;
    start_.resize(nc + 1);
    start_[0] = 0;
    for (size_t c = 0; c < nc; ++c) start_[c + 1] = start_[c] + cnt_[c + 1];
    zs_.resize(start_[nc]);
    std::vector<uint32_t> & fill = cnt_;  // reuse as write cursor
    std::copy(start_.begin(), start_.end() - 1, fill.begin());
    for (size_t k = 0; k < m; ++k)
      if (pc_[k] >= 0) zs_[fill[pc_[k]]++] = pz_[k];
    walked_.assign(kNS, 0);  // each sector is walked on its first road query (walk())
    nb_ = nb;
  }

  // walk sector s outward: road bins (rd_, NaN = not road), the trend at each bin (tr_), previous / next road bin
  void walk(int s)
  {
    walked_[s] = 1;
    const size_t nc = static_cast<size_t>(kNS) * kNB;
    const int nb = nb_;
    if (rd_.size() != nc) {  // bins from nbp() on stay "no road" (never read)
      rd_.assign(nc, kNaN);
      tr_.assign(nc, 0.0);
      prv_.assign(nc, -1);
      nxt_.assign(nc, kNB);
    }
    double lr = 0.0, lz = 0.0, tr = 0.0;
    int last = -1;
    for (int j = 0; j < nb; ++j) {
      const size_t c = static_cast<size_t>(s) * kNB + j;
      const uint32_t m = start_[c + 1] - start_[c];
      rd_[c] = kNaN;
      if (m) {
        float * b0 = zs_.data() + start_[c];
        if (m > 2) std::nth_element(b0, b0 + m / 2, b0 + m);  // the upper median
        const double zj = m == 1 ? b0[0] : m == 2 ? std::max(b0[0], b0[1]) : b0[m / 2];
        const double rj = (j + 0.5) * kRB, dr = rj - lr;
        if (std::abs(zj - (lz + tr * dr)) <= kT0 + kG * dr) {
          const double slope = dr > 0.25 ? (zj - lz) / std::max(dr, 1e-3) : tr;
          tr = std::clamp(0.5 * tr + 0.5 * slope, -0.15, 0.15);
          lz = zj; lr = rj; rd_[c] = zj;
          last = j;
        }
      }
      tr_[c] = tr;
      prv_[c] = last;
    }
    int next = kNB;
    for (int j = nb - 1; j >= 0; --j) {
      const size_t c = static_cast<size_t>(s) * kNB + j;
      if (std::isfinite(rd_[c])) next = j;
      nxt_[c] = next;
    }
  }

  double road_in(int s, double r)
  {
    if (r < 2.0) return 0.0;
    if (!walked_[s]) walk(s);
    const int b = std::min(static_cast<int>(r / kRB), kNB - 1);
    const size_t c = static_cast<size_t>(s) * kNB + b;
    const int p = prv_[c], q = nxt_[c];
    const double pr = (p + 0.5) * kRB, qr = (q + 0.5) * kRB;
    if (p >= 0 && q < kNB && qr - pr <= kGap) {
      const double zp = rd_[static_cast<size_t>(s) * kNB + p], zq = rd_[static_cast<size_t>(s) * kNB + q];
      const double w = std::clamp((r - pr) / std::max(qr - pr, 1e-3), 0.0, 1.0);
      return zp + w * (zq - zp);
    }
    if (p >= 0 && r - pr <= kNear) {
      const size_t cp = static_cast<size_t>(s) * kNB + p;
      return rd_[cp] + std::max(tr_[cp], 0.0) * (r - pr);
    }
    return kNaN;
  }
  double road_at(float x, float y) { return road_in(sector(x, y), std::hypot(x, y)); }
  double road_at_nb(float x, float y)
  {
    const int s = sector(x, y);
    const double r = std::hypot(x, y);
    const double e = road_in(s, r);
    if (std::isfinite(e)) return e;
    double v[2 * kNBS];
    int k = 0;
    for (int d = -kNBS; d <= kNBS; ++d) {
      if (!d) continue;
      const double ed = road_in(((s + d) % kNS + kNS) % kNS, r);
      if (std::isfinite(ed)) v[k++] = ed;
    }
    if (!k) return kNaN;
    std::sort(v, v + k);
    return k % 2 ? v[k / 2] : 0.5 * (v[k / 2 - 1] + v[k / 2]);
  }
};
}  // namespace autoware::ground_segmentation::nero

#endif  // SCAN_GROUND_FILTER__NERO_BAND_ROAD_HPP_
