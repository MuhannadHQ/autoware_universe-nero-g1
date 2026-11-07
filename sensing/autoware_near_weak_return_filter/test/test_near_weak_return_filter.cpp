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

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

using autoware::near_weak_return_filter::find_dropped;
using autoware::near_weak_return_filter::Params;
using autoware::near_weak_return_filter::remove_points;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

namespace
{
struct Pt
{
  float x, y, z;
  uint8_t i;
};

// x y z intensity return_type channel(2) azimuth elevation distance time_stamp: the E1R's PointXYZIRCAEDT layout
PointCloud2 make_cloud(const std::vector<Pt> & pts)
{
  PointCloud2 c;
  auto add = [&](const char * n, uint32_t off, uint8_t t) {
    PointField f;
    f.name = n;
    f.offset = off;
    f.datatype = t;
    f.count = 1;
    c.fields.push_back(f);
  };
  add("x", 0, PointField::FLOAT32);
  add("y", 4, PointField::FLOAT32);
  add("z", 8, PointField::FLOAT32);
  add("intensity", 12, PointField::UINT8);
  add("return_type", 13, PointField::UINT8);
  add("channel", 14, PointField::UINT16);
  add("azimuth", 16, PointField::FLOAT32);
  add("elevation", 20, PointField::FLOAT32);
  add("distance", 24, PointField::FLOAT32);
  add("time_stamp", 28, PointField::UINT32);
  c.point_step = 32;
  c.height = 1;
  c.width = pts.size();
  c.row_step = c.width * c.point_step;
  c.data.assign(c.row_step, 0);
  for (size_t k = 0; k < pts.size(); ++k) {
    uint8_t * p = c.data.data() + k * 32;
    const float d = std::sqrt(pts[k].x * pts[k].x + pts[k].y * pts[k].y + pts[k].z * pts[k].z);
    std::memcpy(p, &pts[k].x, 4);
    std::memcpy(p + 4, &pts[k].y, 4);
    std::memcpy(p + 8, &pts[k].z, 4);
    p[12] = pts[k].i;
    std::memcpy(p + 24, &d, 4);
  }
  return c;
}

// a surface patch at range r: rows / cols on a 0.625 deg grid, all with intensity i
std::vector<Pt> patch(double r, int rows, int cols, uint8_t i)
{
  std::vector<Pt> v;
  const double step = 0.625 * M_PI / 180.0;
  for (int a = 0; a < cols; ++a)
    for (int e = 0; e < rows; ++e) {
      const double az = a * step, el = e * step;
      v.push_back(
        {static_cast<float>(r * std::cos(el) * std::cos(az)),
         static_cast<float>(r * std::cos(el) * std::sin(az)), static_cast<float>(r * std::sin(el)), i});
    }
  return v;
}
}  // namespace

TEST(NearWeakReturnFilter, DropsIsolatedWeakNearReturns)
{
  // scattered weak returns (dust): ~5 pixels apart
  std::vector<Pt> pts;
  for (int k = 0; k < 20; ++k) pts.push_back({0.8f + 0.01f * (k % 3), -0.4f + 0.04f * k, 0.1f, 2});
  const auto drop = find_dropped(make_cloud(pts), Params{});
  ASSERT_TRUE(drop.has_value());
  for (bool b : *drop) EXPECT_TRUE(b);
}

TEST(NearWeakReturnFilter, KeepsBlackSurface)
{
  // a black surface (intensity 1) at 1.0 m returns in every pixel: never dropped, edges and corners included
  const auto pts = patch(1.0, 6, 6, 1);
  const auto drop = find_dropped(make_cloud(pts), Params{});
  ASSERT_TRUE(drop.has_value());
  for (bool b : *drop) EXPECT_FALSE(b);
}

TEST(NearWeakReturnFilter, KeepsFarAndBright)
{
  std::vector<Pt> pts = {{2.0f, 0.0f, 0.1f, 1}, {0.8f, 0.0f, 0.1f, 4}, {1.49f, 0.0f, 0.0f, 3}};
  const auto drop = find_dropped(make_cloud(pts), Params{});
  ASSERT_TRUE(drop.has_value());
  EXPECT_FALSE((*drop)[0]);  // beyond max_distance
  EXPECT_FALSE((*drop)[1]);  // brighter than max_intensity
  EXPECT_TRUE((*drop)[2]);   // near, weak, alone
}

TEST(NearWeakReturnFilter, RemovePointsKeepsBytesAndOrder)
{
  const auto c = make_cloud({{0.5f, 0, 0, 1}, {3.0f, 0, 0, 9}, {0.6f, 0, 0, 2}, {4.0f, 0, 0, 7}});
  const auto out = remove_points(c, {true, false, true, false});
  ASSERT_EQ(out.width, 2u);
  EXPECT_EQ(0, std::memcmp(out.data.data(), c.data.data() + 32, 32));
  EXPECT_EQ(0, std::memcmp(out.data.data() + 32, c.data.data() + 96, 32));
}

TEST(NearWeakReturnFilter, MissingDistanceFieldIsNotFiltered)
{
  auto c = make_cloud({{0.5f, 0, 0, 1}});
  c.fields.pop_back();
  c.fields.pop_back();  // drop time_stamp and distance
  EXPECT_FALSE(find_dropped(c, Params{}).has_value());
}

TEST(NearWeakReturnFilter, EmptyCloud)
{
  const auto c = make_cloud({});
  const auto drop = find_dropped(c, Params{});
  ASSERT_TRUE(drop.has_value());
  EXPECT_TRUE(drop->empty());
  EXPECT_EQ(remove_points(c, *drop).width, 0u);
}

// [nero 2026-11-05 AEBD2] the dust stage
namespace
{
using autoware::near_weak_return_filter::DustParams;
using autoware::near_weak_return_filter::DustStage;
using autoware::near_weak_return_filter::DustStats;

struct Ret
{
  double az, el, d;
  uint8_t i;
};

// returns in base_link from an E1R at (1.19, 0, 0.16) looking along +x: azimuth / elevation / distance as the driver
// writes them (the sensor frame's azimuth = atan2(-y, x))
PointCloud2 e1r_cloud(const std::vector<Ret> & rs)
{
  std::vector<Pt> pts;
  for (const auto & r : rs) {
    const double ux = std::cos(r.el) * std::cos(r.az), uy = -std::cos(r.el) * std::sin(r.az), uz = std::sin(r.el);
    pts.push_back({static_cast<float>(1.19 + r.d * ux), static_cast<float>(r.d * uy), static_cast<float>(0.16 + r.d * uz), r.i});
  }
  auto c = make_cloud(pts);
  for (size_t k = 0; k < rs.size(); ++k) {
    uint8_t * p = c.data.data() + k * 32;
    const float az = rs[k].az, el = rs[k].el, d = rs[k].d;
    std::memcpy(p + 16, &az, 4);
    std::memcpy(p + 20, &el, 4);
    std::memcpy(p + 24, &d, 4);
  }
  return c;
}

uint32_t lcg_state = 12345;
double rnd()
{
  lcg_state = lcg_state * 1664525u + 1013904223u;
  return (lcg_state >> 8) / 16777216.0;
}

// a dust volume: n returns on random pixels of a 0.01 rad grid, at random ranges 0.7-1.6 m, intensity 2
std::vector<Ret> dust(int n)
{
  std::vector<Ret> v;
  for (int k = 0; k < n; ++k) {
    const double az = std::round((rnd() - 0.5) * 60.0) * 0.01, el = std::round(rnd() * 20.0) * 0.01;
    v.push_back({az, el, 0.7 + 0.9 * rnd(), 2});
  }
  return v;
}

// a black wall facing the sensor at range r: every pixel of a 0.01 rad grid
std::vector<Ret> wall(double r, double az0, double az1, double el0, double el1, uint8_t i)
{
  std::vector<Ret> v;
  for (double az = az0; az <= az1 + 1e-9; az += 0.01)
    for (double el = el0; el <= el1 + 1e-9; el += 0.01) v.push_back({az, el, r / (std::cos(az) * std::cos(el)), i});
  return v;
}

size_t count_drops(const std::optional<std::vector<bool>> & d, size_t from, size_t to)
{
  size_t n = 0;
  for (size_t k = from; k < to; ++k) n += (*d)[k] ? 1 : 0;
  return n;
}
}  // namespace

TEST(NearDustStage, EmptyCloud)
{
  DustParams p;
  p.enabled = true;
  DustStage s(p);
  DustStats st;
  const auto d = s.process(e1r_cloud({}), 0.0, st);
  ASSERT_TRUE(d.has_value());
  EXPECT_TRUE(d->empty());
}

TEST(NearDustStage, MissingAzimuthIsNotFiltered)
{
  auto c = e1r_cloud(dust(200));
  c.fields.erase(c.fields.begin() + 6);  // azimuth
  DustParams p;
  p.enabled = true;
  DustStage s(p);
  DustStats st;
  EXPECT_FALSE(s.process(c, 0.0, st).has_value());
}

TEST(NearDustStage, DustVolumeDroppedOnlyInABurst)
{
  DustParams p;
  p.enabled = true;
  {
    DustStage s(p);
    DustStats st;
    const auto v = dust(600);
    const auto d = s.process(e1r_cloud(v), 0.0, st);
    ASSERT_TRUE(d.has_value());
    EXPECT_TRUE(st.gate_open);
    EXPECT_GT(count_drops(d, 0, v.size()), v.size() * 8 / 10);
  }
  {
    DustStage s(p);  // a handful of dust-like returns and no burst before: nothing is touched
    DustStats st;
    const auto v = dust(30);
    const auto d = s.process(e1r_cloud(v), 0.0, st);
    ASSERT_TRUE(d.has_value());
    EXPECT_FALSE(st.gate_open);
    EXPECT_EQ(count_drops(d, 0, v.size()), 0u);
  }
}

TEST(NearDustStage, KeepsABlackWallInsideTheDust)
{
  DustParams p;
  p.enabled = true;
  DustStage s(p);
  DustStats st;
  auto v = wall(1.4, -0.15, 0.15, 0.0, 0.12, 1);  // black, 0.4 x 0.17 m
  const size_t nw = v.size();
  const auto dv = dust(600);
  v.insert(v.end(), dv.begin(), dv.end());
  const auto d = s.process(e1r_cloud(v), 0.0, st);
  ASSERT_TRUE(d.has_value());
  EXPECT_TRUE(st.gate_open);
  EXPECT_EQ(count_drops(d, 0, nw), 0u);              // the wall: every return kept
  EXPECT_GT(count_drops(d, nw, v.size()), 300u);      // the dust around it: removed
}

TEST(NearDustStage, BrightCarSideAloneNeverOpensTheGate)
{
  DustParams p;
  p.enabled = true;
  DustStage s(p);
  DustStats st;
  auto v = dust(200);
  for (auto & r : v) r.i = 120;  // fragmented but bright (reflectors, paint): the weak share is the gate's test
  const auto d = s.process(e1r_cloud(v), 0.0, st);
  ASSERT_TRUE(d.has_value());
  EXPECT_FALSE(st.gate_open);
  EXPECT_EQ(count_drops(d, 0, v.size()), 0u);
}

TEST(NearDustStage, HoldsOneSecondWhileWeakDustRemains)
{
  DustParams p;
  p.enabled = true;
  DustStage s(p);
  DustStats st;
  s.process(e1r_cloud(dust(600)), 10.0, st);
  EXPECT_TRUE(st.gate_open);
  const auto v = dust(40);  // the tail of the cloud: below the burst count
  auto d = s.process(e1r_cloud(v), 10.5, st);
  EXPECT_TRUE(st.gate_open);
  EXPECT_GT(count_drops(d, 0, v.size()), 0u);
  d = s.process(e1r_cloud(v), 11.6, st);  // more than hold_sec after the burst
  EXPECT_FALSE(st.gate_open);
  EXPECT_EQ(count_drops(d, 0, v.size()), 0u);
}

// [nero 2026-11-06 aebdust_1106] the gate's own volume, the follower, planes need returns
namespace
{
// dust beside the robot, |y| 1.4-1.9 (inside the drop box, outside the gate's volume)
std::vector<Ret> side_dust(int n)
{
  std::vector<Ret> v;
  while (static_cast<int>(v.size()) < n) {
    const double az = -0.7 - 0.6 * rnd(), el = std::round(rnd() * 20.0) * 0.01, d = 1.6 + 0.8 * rnd();
    const double y = -d * std::cos(el) * std::sin(az), x = 1.19 + d * std::cos(el) * std::cos(az);
    if (y > 1.4 && y < 1.9 && x > 1.05) v.push_back({az, el, d, 2});
  }
  return v;
}
}  // namespace

TEST(NearDustStage, GateCountsOnlyItsOwnVolume)
{
  DustParams p;
  p.enabled = true;
  {
    DustStage s(p);  // a burst beside the robot alone (outside |y| <= gate_max_abs_y) never opens the gate
    DustStats st;
    const auto v = side_dust(600);
    const auto d = s.process(e1r_cloud(v), 0.0, st);
    ASSERT_TRUE(d.has_value());
    EXPECT_FALSE(st.gate_open);
    EXPECT_EQ(count_drops(d, 0, v.size()), 0u);
  }
  {
    DustStage s(p);  // with a burst in front, the same side dust is dropped too (the wider box)
    DustStats st;
    auto v = side_dust(300);
    const size_t ns = v.size();
    const auto dv = dust(600);
    v.insert(v.end(), dv.begin(), dv.end());
    const auto d = s.process(e1r_cloud(v), 0.0, st);
    ASSERT_TRUE(d.has_value());
    EXPECT_TRUE(st.gate_open);
    EXPECT_GT(count_drops(d, 0, ns), ns * 7 / 10);
  }
}

TEST(NearDustStage, FollowerActsOnlyWithinTheLeadersGate)
{
  DustParams p;
  p.enabled = true;
  p.follow = true;
  DustStage s(p);
  DustStats st;
  const auto v = dust(600);
  auto d = s.process(e1r_cloud(v), 20.0, st);  // no open frame of the leader yet: nothing looked at
  ASSERT_TRUE(d.has_value());
  EXPECT_FALSE(st.gate_open);
  EXPECT_EQ(count_drops(d, 0, v.size()), 0u);
  s.external_gate_open(20.0);
  d = s.process(e1r_cloud(v), 20.1, st);
  EXPECT_TRUE(st.gate_open);
  EXPECT_GT(count_drops(d, 0, v.size()), v.size() * 8 / 10);
  d = s.process(e1r_cloud(v), 20.0 + p.follow_sec + 0.05, st);  // the leader's gate is stale
  EXPECT_FALSE(st.gate_open);
  EXPECT_EQ(count_drops(d, 0, v.size()), 0u);
  const auto few = dust(5);  // no own gate: a follower acts on a handful of dust-like returns while the leader's is open
  s.external_gate_open(30.0);
  d = s.process(e1r_cloud(few), 30.05, st);
  EXPECT_TRUE(st.gate_open);
}

TEST(NearDustStage, FollowerKeepsABlackWallInsideTheDust)
{
  DustParams p;
  p.enabled = true;
  p.follow = true;
  DustStage s(p);
  DustStats st;
  auto v = wall(1.4, -0.15, 0.15, 0.0, 0.12, 1);
  const size_t nw = v.size();
  const auto dv = dust(600);
  v.insert(v.end(), dv.begin(), dv.end());
  s.external_gate_open(5.0);
  const auto d = s.process(e1r_cloud(v), 5.0, st);
  ASSERT_TRUE(d.has_value());
  EXPECT_EQ(count_drops(d, 0, nw), 0u);
  EXPECT_GT(count_drops(d, nw, v.size()), 300u);
}

TEST(NearDustStage, KeepsADarkFaceWithDropoutsInsideTheDust)
{
  DustParams p;
  p.enabled = true;
  DustStage s(p);
  DustStats st;
  std::vector<Ret> v;
  for (const auto & r : wall(1.6, -0.2, 0.2, 0.0, 0.2, 1)) {
    if (rnd() < 0.5) v.push_back(r);  // half of its pixels return nothing (a dark, partly absorbing face)
  }
  const size_t nf = v.size();
  const auto dv = dust(600);
  v.insert(v.end(), dv.begin(), dv.end());
  const auto d = s.process(e1r_cloud(v), 0.0, st);
  ASSERT_TRUE(d.has_value());
  EXPECT_TRUE(st.gate_open);
  EXPECT_LE(count_drops(d, 0, nf), nf / 20);  // at most a few edge returns
}

TEST(NearDustStage, HoldStaysOpenOnAShortTail)
{
  DustParams p;
  p.enabled = true;
  DustStage s(p);
  DustStats st;
  s.process(e1r_cloud(dust(600)), 10.0, st);
  EXPECT_TRUE(st.gate_open);
  const auto v = dust(12);  // a tail of a few weak returns (>= hold_min of them dust-like)
  const auto d = s.process(e1r_cloud(v), 10.3, st);
  EXPECT_TRUE(st.gate_open);
  EXPECT_GT(count_drops(d, 0, v.size()), 0u);
}
