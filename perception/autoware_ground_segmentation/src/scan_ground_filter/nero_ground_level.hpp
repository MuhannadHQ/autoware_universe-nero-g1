// [nero 2026-11-05 jolt] NERO_GROUND_LEVEL (QUEUE PTH, Ahmed 11-05 12:4x "change the floor filter in addition to the height
// filter to see everything always no matter what"): classify ground and cut the obstacle band's ceiling in the ROAD frame of each
// cloud instead of the body frame. The road plane (pitch, roll relative to base_link) is fitted from the cloud's own low points
// near the robot, so a body pitched / rolled by a pothole, a dip or braking no longer lifts the road into the band or squeezes a
// real object out of it; a uniform slope is unchanged (the plane follows it). Output points stay in base_link (only the selection
// changes). Switch: "1" (ceiling = the node's vehicle height + the crop's margin, 0.52 m on the car) or "<ceiling m>[,<max tilt
// deg>]". The crop box before the node must pass points above the ceiling (launch: margin_max_z raised with this switch).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace autoware::ground_segmentation::nero
{
struct GroundLevel
{
  float ceiling = -1.0f;      // band ceiling over base_link's plane after levelling (m); < 0: the node's vehicle height
  float max_tilt_deg = 8.0f;  // a fit tilted more than this is not trusted: no levelling for that cloud
  float x0 = 2.0f, x1 = 12.0f, yh = 2.5f, q = 0.2f;
  size_t min_bin = 20, min_bins = 4;
  double inlier_tol = 0.08;  // m: a bin further than this from the robust line is not road
  // last fit (for the log line)
  float pitch = 0.0f, roll = 0.0f;
  bool ok = false;
  float cp = 1.0f, sp = 0.0f, cr = 1.0f, sr = 0.0f;
  float icpt_x = 0.0f;  // the fitted road line's height at x = 0 (base_link), from the pitch bins (jolt-3)
  std::vector<float> bx[10], by[10];
  std::vector<float> xs, zs, ys, zy, in_x, in_z;  // per-cloud work buffers (reused)
  std::vector<double> pair_buf;

  // "1" or "<ceiling>[,<max tilt deg>]", numbers as plain decimals (digits and one '.', no sign / exponent / spaces): the launch
  // file's nero_ground_level_valid() accepts exactly the same strings (test/data/nero_ground_level_switch.txt holds both sides).
  static bool plain_decimal(const std::string & t)
  {
    size_t dots = 0, digits = 0;
    for (const char c : t) {
      if (c == '.') ++dots;
      else if (c >= '0' && c <= '9') ++digits;
      else return false;
    }
    return digits > 0 && dots <= 1;
  }

  bool parse(const char * s)
  {
    if (!s) return false;
    const std::string v(s);
    if (v == "1") return true;
    const size_t comma = v.find(',');
    const std::string a = v.substr(0, comma), b = comma == std::string::npos ? std::string() : v.substr(comma + 1);
    if (!plain_decimal(a) || (comma != std::string::npos && !plain_decimal(b))) return false;
    const float c = std::strtof(a.c_str(), nullptr);
    if (!(c > 0.1f && c < 5.0f)) return false;
    float t = max_tilt_deg;
    if (comma != std::string::npos) {
      t = std::strtof(b.c_str(), nullptr);
      if (!(t > 0.5f && t < 20.0f)) return false;
    }
    ceiling = c;
    max_tilt_deg = t;
    return true;
  }

  static float quantile(std::vector<float> & v, float q)
  {
    const size_t k = static_cast<size_t>(q * static_cast<float>(v.size() - 1));
    std::nth_element(v.begin(), v.begin() + static_cast<long>(k), v.end());
    return v[k];
  }

  static bool line(const std::vector<float> & px, const std::vector<float> & pz, float & slope)
  {
    float icpt = 0.0f;
    return line(px, pz, slope, icpt);
  }

  static bool line(const std::vector<float> & px, const std::vector<float> & pz, float & slope, float & icpt)
  {
    const size_t n = px.size();
    if (n < 2) return false;
    double sx = 0, sz = 0, sxx = 0, sxz = 0;
    for (size_t i = 0; i < n; ++i) {
      sx += px[i]; sz += pz[i]; sxx += static_cast<double>(px[i]) * px[i]; sxz += static_cast<double>(px[i]) * pz[i];
    }
    const double den = static_cast<double>(n) * sxx - sx * sx;
    if (std::abs(den) < 1e-9) return false;
    slope = static_cast<float>((static_cast<double>(n) * sxz - sx * sz) / den);
    icpt = static_cast<float>((sz - static_cast<double>(slope) * sx) / static_cast<double>(n));
    return true;
  }

  static double median(std::vector<double> & v)
  {
    std::sort(v.begin(), v.end());
    const size_t k = v.size() / 2;
    return v.size() % 2 ? v[k] : 0.5 * (v[k - 1] + v[k]);
  }

  // jolt-3: a line through the bins that ignores a minority of bins off the road (a kerb / pavement or a parked car filling a
  // side bin, a wall across the far bins): Theil-Sen slope (median of the pairwise slopes) + median intercept, then least
  // squares on the bins within inlier_tol of that line (>= min_bins of them, else no fit). With every bin on one line this
  // is the plain least-squares fit.
  bool robust_line(const std::vector<float> & px, const std::vector<float> & pz, float & slope)
  {
    float icpt = 0.0f;
    return robust_line(px, pz, slope, icpt);
  }

  bool robust_line(const std::vector<float> & px, const std::vector<float> & pz, float & slope, float & icpt)
  {
    const size_t n = px.size();
    if (n < min_bins) return false;
    pair_buf.clear();
    for (size_t i = 0; i < n; ++i)
      for (size_t j = i + 1; j < n; ++j)
        pair_buf.push_back((static_cast<double>(pz[j]) - pz[i]) / (static_cast<double>(px[j]) - px[i]));
    const double b = median(pair_buf);
    pair_buf.clear();
    for (size_t i = 0; i < n; ++i) pair_buf.push_back(pz[i] - b * px[i]);
    const double a = median(pair_buf);
    in_x.clear(); in_z.clear();
    for (size_t i = 0; i < n; ++i) {
      if (std::abs(pz[i] - (a + b * px[i])) < inlier_tol) { in_x.push_back(px[i]); in_z.push_back(pz[i]); }
    }
    return in_x.size() >= min_bins && line(in_x, in_z, slope, icpt);
  }

  // fit the road plane from xyz (stride floats per point, x y z at 0 1 2); sets pitch / roll / ok
  void fit(const float * p, size_t n, size_t stride)
  {
    for (auto & b : bx) b.clear();
    for (auto & b : by) b.clear();
    const float zlim_a = 0.3f, zlim_b = std::tan(max_tilt_deg * static_cast<float>(M_PI) / 180.0f);
    for (size_t i = 0; i < n; ++i, p += stride) {
      const float x = p[0], y = p[1], z = p[2];
      if (!(x >= x0 && x < x1) || !(std::abs(y) < yh) || !(z < zlim_a + zlim_b * x) || !(z > -zlim_a - zlim_b * x)) continue;
      bx[std::min<size_t>(9, static_cast<size_t>((x - x0) / (x1 - x0) * 10.0f))].push_back(z);
      if (x >= 3.0f && x < 8.0f) by[std::min<size_t>(9, static_cast<size_t>((y + yh) / (2.0f * yh) * 10.0f))].push_back(z);
    }
    xs.clear(); zs.clear(); ys.clear(); zy.clear();
    for (size_t k = 0; k < 10; ++k) {
      if (bx[k].size() >= min_bin) { xs.push_back(x0 + (x1 - x0) * (static_cast<float>(k) + 0.5f) / 10.0f); zs.push_back(quantile(bx[k], q)); }
      if (by[k].size() >= min_bin) { ys.push_back(-yh + 2.0f * yh * (static_cast<float>(k) + 0.5f) / 10.0f); zy.push_back(quantile(by[k], q)); }
    }
    float bxs = 0.0f, bys = 0.0f;
    float axs = 0.0f;
    ok = xs.size() >= min_bins && ys.size() >= min_bins && robust_line(xs, zs, bxs, axs) && robust_line(ys, zy, bys);
    icpt_x = ok ? axs : 0.0f;
    pitch = ok ? std::atan(bxs) : 0.0f;
    roll = ok ? std::atan(bys) : 0.0f;
    const float lim = max_tilt_deg * static_cast<float>(M_PI) / 180.0f;
    if (ok && (std::abs(pitch) > lim || std::abs(roll) > lim)) ok = false;
    // the rotation's sines / cosines once per cloud (jolt-3: per point they were ~0.7 ms of the ~1 ms levelling cost)
    cp = std::cos(pitch); sp = std::sin(pitch); cr = std::cos(roll); sr = std::sin(roll);
  }

  // one point into the road frame (pitch first, then roll; about base_link's origin): z first, so a point over the ceiling
  // costs only that; identity when the last fit is not trusted
  float road_z(const float * p, float & z1) const
  {
    z1 = -sp * p[0] + cp * p[2];
    return -sr * p[1] + cr * z1;
  }
  void road_xy(const float * p, float z1, float & x, float & y) const
  {
    x = cp * p[0] + sp * p[2];
    y = cr * p[1] + sr * z1;
  }

  // jolt-3: the height a point is measured from in the road frame, so the band's ceiling follows the ROAD and not an extrapolation
  // of it: ahead where the fit has data (x >= x0) the fitted road line itself (its intercept included: a road that only starts to
  // rise / fall a few metres ahead is not a tilted body), at the robot (x <= x_front, the vehicle's front) the robot's own plane
  // (base_link: its wheels stand on the road there), linearly between. Points behind the robot are not levelled at all (the node).
  float ref_z(const float * p, float x_front) const
  {
    const float base = -sr * p[1] - cr * sp * p[0];  // base_link's plane z = 0 at (x, y), in the road frame
    const float road = icpt_x * cp * cr;              // the fitted road plane, in the road frame
    if (p[0] >= x0) return road;
    if (p[0] <= x_front || x0 <= x_front) return base;
    const float w = (p[0] - x_front) / (x0 - x_front);
    return w * road + (1.0f - w) * base;
  }

  // jolt-8 (jg9): the coordinates the grid filter classifies a point ahead in. A rotation about base_link's origin moves the ground
  // right in front of the bumper by x sin(tilt): fits of 0.3-0.4 deg = 1.5-3 cm at 2.3 m, enough to flip low dust returns the car
  // calls ground into the obstacle cloud (10-29 19:00:42.80: the car's AEB on residual dust in 4 of 7 placements, the car 1 of 21).
  // Near the robot its wheels stand on the road, so base_link IS the road there: classify in base_link up to xn0 ahead, in the road
  // frame from xn1 (where a jolt lifts the road 0.3-1 m), linearly between (as ref_z blends the ceiling's reference).
  float xn0 = 3.0f, xn1 = 6.0f;
  float near_w(float x) const
  {
    if (!(x > xn0)) return 0.0f;
    if (!(x < xn1)) return 1.0f;
    return (x - xn0) / (xn1 - xn0);
  }
  // one point's classification coordinates: p in base_link, z / z1 from road_z(p, z1); out may not alias p
  void classify_xyz(const float * p, float z, float z1, float * out) const
  {
    const float w = near_w(p[0]);
    if (w == 0.0f) {
      out[0] = p[0]; out[1] = p[1]; out[2] = p[2];
      return;
    }
    float x = 0.0f, y = 0.0f;
    road_xy(p, z1, x, y);
    if (w == 1.0f) {
      out[0] = x; out[1] = y; out[2] = z;
      return;
    }
    out[0] = p[0] + w * (x - p[0]); out[1] = p[1] + w * (y - p[1]); out[2] = p[2] + w * (z - p[2]);
  }

  // rotate xyz in place into the road frame (tests / tools; the node uses road_z / road_xy)
  void rotate(float * p, size_t n, size_t stride) const
  {
    if (!ok) return;
    for (size_t i = 0; i < n; ++i, p += stride) {
      float z1 = 0.0f, x = 0.0f, y = 0.0f;
      const float z = road_z(p, z1);
      road_xy(p, z1, x, y);
      p[0] = x; p[1] = y; p[2] = z;
    }
  }
};
}  // namespace autoware::ground_segmentation::nero
