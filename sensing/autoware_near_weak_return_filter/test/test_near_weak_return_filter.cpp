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
