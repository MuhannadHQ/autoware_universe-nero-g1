// [nero 2026-11-05 jolt] NERO_GROUND_LEVEL (QUEUE PTH): the road-frame ground + band ceiling of the scan ground filter.
// - the switch parse against the table the launch side is checked on too (test/data/nero_ground_level_switch.txt);
// - the road plane fit and the rotation on synthetic roads;
// - the node: on a level body the switch gives exactly the car's output (car = crop at the vehicle height, no switch);
//   with the body pitched nose-down 4 deg an object 10 m ahead that the car's base_link crop cuts away is kept, the road is ground,
//   and the output stays in base_link.

#include "../src/scan_ground_filter/nero_ground_level.hpp"
#include "../src/scan_ground_filter/node.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using autoware::ground_segmentation::nero::GroundLevel;

namespace
{
constexpr float kDeg = static_cast<float>(M_PI) / 180.0f;

struct P
{
  float x, y, z;
};

std::vector<P> road(float pitch, float roll, float x_max = 30.0f, float step = 0.1f)
{
  // a road plane rising ahead by pitch and to the left by roll (base_link), the robot's own footprint left out
  std::vector<P> v;
  for (float x = -10.0f; x < x_max; x += step) {
    for (float y = -6.0f; y <= 6.0f; y += step) {
      if (std::abs(x) < 1.5f && std::abs(y) < 0.6f) continue;
      v.push_back({x, y, x * std::tan(pitch) + y * std::tan(roll)});
    }
  }
  return v;
}

void add_box(std::vector<P> & v, float cx, float cy, float w, float h, float pitch, float step = 0.05f)
{
  // a box standing on the pitched road (its faces sampled), height h over the road
  for (float x = cx - w / 2; x <= cx + w / 2; x += step) {
    for (float y = cy - w / 2; y <= cy + w / 2; y += step) {
      for (float z = 0.05f; z <= h; z += step) {
        const bool face = std::abs(x - (cx - w / 2)) < step / 2 || std::abs(y - (cy - w / 2)) < step / 2 ||
                          std::abs(y - (cy + w / 2)) < step / 2;
        if (face) v.push_back({x, y, x * std::tan(pitch) + z});
      }
    }
  }
}

sensor_msgs::msg::PointCloud2 to_msg(const std::vector<P> & pts, float max_z)
{
  // what the common crop box hands the ground filter: x y z intensity, base_link, z < max_z
  sensor_msgs::msg::PointCloud2 c;
  c.header.frame_id = "base_link";
  c.height = 1;
  c.is_dense = true;
  c.is_bigendian = false;
  c.point_step = 16;
  c.fields.resize(4);
  const char * names[4] = {"x", "y", "z", "intensity"};
  for (int i = 0; i < 4; ++i) {
    c.fields[i].name = names[i];
    c.fields[i].offset = 4 * i;
    c.fields[i].datatype = sensor_msgs::msg::PointField::FLOAT32;
    c.fields[i].count = 1;
  }
  size_t n = 0;
  for (const auto & p : pts) n += p.z < max_z ? 1 : 0;
  c.width = static_cast<uint32_t>(n);
  c.row_step = c.point_step * c.width;
  c.data.resize(c.row_step);
  size_t k = 0;
  for (const auto & p : pts) {
    if (!(p.z < max_z)) continue;
    const float q[4] = {p.x, p.y, p.z, 10.0f};
    std::memcpy(&c.data[k * 16], q, 16);
    ++k;
  }
  return c;
}

std::vector<P> from_msg(const sensor_msgs::msg::PointCloud2 & c)
{
  std::vector<P> v;
  for (sensor_msgs::PointCloud2ConstIterator<float> ix(c, "x"), iy(c, "y"), iz(c, "z"); ix != ix.end(); ++ix, ++iy, ++iz)
    v.push_back({*ix, *iy, *iz});
  return v;
}
}  // namespace

// ---------------------------------------------------------------- parse
TEST(NeroGroundLevel, ParseTable)
{
  const auto path =
    ament_index_cpp::get_package_share_directory("autoware_ground_segmentation") + "/data/nero_ground_level_switch.txt";
  std::ifstream f(path);
  ASSERT_TRUE(f.good()) << path;
  std::string line;
  int rows = 0;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    const size_t a = line.find('\''), b = line.find('\'', a + 1);
    ASSERT_NE(b, std::string::npos) << line;
    const std::string v = line.substr(a + 1, b - a - 1);
    std::vector<std::string> cols;
    size_t s = b + 1;
    while (s < line.size()) {
      const size_t t = line.find('\t', s + 1);
      cols.push_back(line.substr(s + 1, (t == std::string::npos ? line.size() : t) - s - 1));
      s = t == std::string::npos ? line.size() : t;
    }
    ASSERT_GE(cols.size(), 1u) << line;
    GroundLevel g;
    const bool on = g.parse(v.c_str());
    EXPECT_EQ(on, cols[0] == "1") << "'" << v << "'";
    if (on && cols.size() >= 3) {
      if (cols[1] == "vh") {
        EXPECT_LT(g.ceiling, 0.0f) << "'" << v << "'";  // the node's vehicle height
      } else {
        EXPECT_FLOAT_EQ(g.ceiling, std::stof(cols[1])) << "'" << v << "'";
      }
      EXPECT_FLOAT_EQ(g.max_tilt_deg, std::stof(cols[2])) << "'" << v << "'";
    }
    if (!on) {  // a refused value leaves the defaults alone
      EXPECT_LT(g.ceiling, 0.0f) << "'" << v << "'";
      EXPECT_FLOAT_EQ(g.max_tilt_deg, 8.0f) << "'" << v << "'";
    }
    ++rows;
  }
  EXPECT_GE(rows, 40);
  GroundLevel g;
  EXPECT_FALSE(g.parse(nullptr));
}

// ---------------------------------------------------------------- fit + rotate
TEST(NeroGroundLevel, FitRecoversThePlane)
{
  for (const float pd : {-6.0f, -2.0f, 0.0f, 1.0f, 4.0f}) {
    for (const float rd : {-3.0f, 0.0f, 2.5f}) {
      auto pts = road(pd * kDeg, rd * kDeg);
      add_box(pts, 7.0f, 0.5f, 0.8f, 1.2f, pd * kDeg);  // an object on the road must not move the plane
      GroundLevel g;
      g.fit(&pts[0].x, pts.size(), 3);
      ASSERT_TRUE(g.ok) << pd << " " << rd;
      EXPECT_NEAR(g.pitch / kDeg, pd, 0.05) << pd << " " << rd;
      // the y-bins span x 3-8 m, so on a pitched road a tall object in one of them moves its 20th percentile a little:
      // 0.16 deg at 6 deg pitch with this 1.2 m box (6 mm at 2 m to the side)
      EXPECT_NEAR(g.roll / kDeg, rd, 0.25) << pd << " " << rd;
    }
  }
}

TEST(NeroGroundLevel, KerbAndWallDoNotTiltThePlane)
{
  // 10-29 20:40:22.65 (jolt-3): a least-squares roll followed two side bins on the pavement (0.15-0.4 m over the road) and lifted
  // the near road on the other side into the band; the robust line leaves such bins out
  for (const float pd : {-4.0f, 0.0f, 3.0f}) {
    const float pitch = pd * kDeg;
    auto pts = road(pitch, 0.0f);
    for (auto & p : pts)
      if (p.y < -1.5f) p.z += 0.25f;  // pavement on the right
    for (float x = 10.6f; x < 12.0f; x += 0.1f)  // a wall across the far bins
      for (float y = -2.5f; y < 2.5f; y += 0.05f)
        for (float z = 0.0f; z < 1.5f; z += 0.05f) pts.push_back({x, y, x * std::tan(pitch) + z});
    GroundLevel g;
    g.fit(&pts[0].x, pts.size(), 3);
    ASSERT_TRUE(g.ok) << pd;
    EXPECT_NEAR(g.pitch / kDeg, pd, 0.1) << pd;
    EXPECT_NEAR(g.roll / kDeg, 0.0, 0.1) << pd;
  }
}

TEST(NeroGroundLevel, RotateLevelsTheRoad)
{
  const float pitch = 4.0f * kDeg, roll = -2.0f * kDeg;
  auto pts = road(pitch, roll);
  GroundLevel g;
  g.fit(&pts[0].x, pts.size(), 3);
  ASSERT_TRUE(g.ok);
  auto lev = pts;
  g.rotate(&lev[0].x, lev.size(), 3);
  float worst = 0.0f;
  for (const auto & p : lev)
    if (p.x > 2.0f && p.x < 25.0f) worst = std::max(worst, std::abs(p.z));
  EXPECT_LT(worst, 0.06f);  // 4 deg would put the road 25 m ahead 1.75 m up in base_link
  // road_z / road_xy == rotate, bit for bit
  for (size_t i = 0; i < pts.size(); i += 97) {
    float z1 = 0.0f, x = 0.0f, y = 0.0f;
    const float z = g.road_z(&pts[i].x, z1);
    g.road_xy(&pts[i].x, z1, x, y);
    EXPECT_EQ(x, lev[i].x);
    EXPECT_EQ(y, lev[i].y);
    EXPECT_EQ(z, lev[i].z);
  }
}

TEST(NeroGroundLevel, ClassifyCoordinatesBlendFromTheRobot)
{
  // jolt-8 (jg9): base_link up to xn0 (3 m) ahead and behind, the road frame (road_z / road_xy) from xn1 (6 m), linear between
  const float pitch = 2.0f * kDeg, roll = 1.0f * kDeg;
  auto pts = road(pitch, roll);
  GroundLevel g;
  g.fit(&pts[0].x, pts.size(), 3);
  ASSERT_TRUE(g.ok);
  for (const float x : {-5.0f, 0.0f, 1.3f, 2.3f, 2.99f, 3.0f, 4.5f, 6.0f, 9.0f}) {
    const float p[3] = {x, 0.4f, x * std::tan(pitch) + 0.4f * std::tan(roll) + 0.09f};
    float z1 = 0.0f, rx = 0.0f, ry = 0.0f, c[3];
    const float rz = g.road_z(p, z1);
    g.road_xy(p, z1, rx, ry);
    g.classify_xyz(p, rz, z1, c);
    if (x <= 3.0f) {  // the car's coordinates, bit for bit
      EXPECT_EQ(c[0], p[0]) << x;
      EXPECT_EQ(c[1], p[1]) << x;
      EXPECT_EQ(c[2], p[2]) << x;
    } else if (x >= 6.0f) {  // jg8's road frame, bit for bit
      EXPECT_EQ(c[0], rx) << x;
      EXPECT_EQ(c[1], ry) << x;
      EXPECT_EQ(c[2], rz) << x;
    } else {  // halfway
      EXPECT_NEAR(c[2], 0.5f * (p[2] + rz), 1e-6f);
      EXPECT_NEAR(c[0], 0.5f * (p[0] + rx), 1e-6f);
    }
  }
  // what jg8 did at the bumper: the rotation about base_link moved a point 2.3 m ahead by x sin(tilt) (here ~8 cm)
  const float q[3] = {2.3f, 0.4f, 2.3f * std::tan(pitch) + 0.09f};
  float z1 = 0.0f;
  EXPECT_GT(q[2] - g.road_z(q, z1), 0.06f);
}

TEST(NeroGroundLevel, UntrustedFits)
{
  {  // tilted beyond max_tilt_deg: no levelling, rotate is the identity
    auto pts = road(10.0f * kDeg, 0.0f);
    GroundLevel g;
    g.fit(&pts[0].x, pts.size(), 3);
    EXPECT_FALSE(g.ok);
    auto lev = pts;
    g.rotate(&lev[0].x, lev.size(), 3);
    for (size_t i = 0; i < pts.size(); i += 31) EXPECT_EQ(lev[i].z, pts[i].z);
  }
  {  // too few points ahead (a wall / a blocked view): no levelling
    std::vector<P> pts;
    for (float x = 2.0f; x < 3.0f; x += 0.1f)
      for (float y = -2.0f; y < 2.0f; y += 0.1f) pts.push_back({x, y, 0.0f});
    GroundLevel g;
    g.fit(&pts[0].x, pts.size(), 3);
    EXPECT_FALSE(g.ok);
  }
  {  // a max tilt from the switch
    auto pts = road(5.0f * kDeg, 0.0f);
    GroundLevel g;
    ASSERT_TRUE(g.parse("0.52,4"));
    g.fit(&pts[0].x, pts.size(), 3);
    EXPECT_FALSE(g.ok);
  }
}

// ---------------------------------------------------------------- the node (friend fixture name of node.hpp)
class ScanGroundFilterTest : public ::testing::Test
{
protected:
  void SetUp() override { rclcpp::init(0, nullptr); }
  void TearDown() override { rclcpp::shutdown(); }

  std::shared_ptr<autoware::ground_segmentation::ScanGroundFilterComponent> make(const char * sw, const char * band = nullptr)
  {
    const auto share_dir = ament_index_cpp::get_package_share_directory("autoware_ground_segmentation");
    auto params = YAML::LoadFile(share_dir + "/config/scan_ground_filter.param.yaml")["/**"]["ros__parameters"];
    std::vector<rclcpp::Parameter> p;
    for (const auto & kv : std::vector<std::pair<std::string, double>>{
           {"wheel_radius", 0.2}, {"wheel_width", 0.1}, {"wheel_base", 1.0}, {"wheel_tread", 0.7}, {"front_overhang", 0.4},
           {"rear_overhang", 0.4}, {"left_overhang", 0.05}, {"right_overhang", 0.05}, {"vehicle_height", 0.52},
           {"max_steer_angle", 0.6}})
      p.emplace_back(kv.first, kv.second);
    for (const char * k : {"global_slope_max_angle_deg", "local_slope_max_angle_deg", "split_points_distance_tolerance",
                           "split_height_distance", "non_ground_height_threshold", "grid_size_m", "grid_mode_switch_radius",
                           "detection_range_z_max", "low_priority_region_x", "center_pcl_shift", "radial_divider_angle_deg"})
      p.emplace_back(k, params[k].as<float>());
    for (const char * k : {"use_virtual_ground_point", "use_recheck_ground_cluster", "use_lowest_point",
                           "publish_processing_time_detail"})
      p.emplace_back(k, params[k].as<bool>());
    p.emplace_back("gnd_grid_buffer_size", params["gnd_grid_buffer_size"].as<int>());
    p.emplace_back("elevation_grid_mode", true);  // the car's mode
    rclcpp::NodeOptions o;
    o.parameter_overrides(p);
    if (sw) setenv("NERO_GROUND_LEVEL", sw, 1);
    else unsetenv("NERO_GROUND_LEVEL");
    if (band) setenv("NERO_GROUND_BAND_ROAD", band, 1);  // rampground_1031 (on the car)
    else unsetenv("NERO_GROUND_BAND_ROAD");
    auto n = std::make_shared<autoware::ground_segmentation::ScanGroundFilterComponent>(o);
    unsetenv("NERO_GROUND_LEVEL");
    unsetenv("NERO_GROUND_BAND_ROAD");
    return n;
  }

  std::vector<P> run(autoware::ground_segmentation::ScanGroundFilterComponent & n, const sensor_msgs::msg::PointCloud2 & in)
  {
    auto ptr = std::make_shared<sensor_msgs::msg::PointCloud2>(in);
    sensor_msgs::msg::PointCloud2 out;
    autoware::pointcloud_preprocessor::TransformInfo ti;
    n.faster_filter(ptr, nullptr, out, ti);
    return from_msg(out);
  }
};

TEST_F(ScanGroundFilterTest, LevelBodyEqualsTheCar)
{
  // level road, a 0.4 m box 10 m ahead, a branch 1.0-1.4 m over the lane 14 m ahead (over the ceiling)
  auto pts = road(0.0f, 0.0f);
  add_box(pts, 10.0f, 0.0f, 0.6f, 0.4f, 0.0f);
  for (float x = 13.5f; x < 14.5f; x += 0.05f)
    for (float z = 1.0f; z < 1.4f; z += 0.05f) pts.push_back({x, 0.2f, z});
  auto car = make(nullptr);
  const auto out_car = run(*car, to_msg(pts, 0.52f));  // the car: crop max_z = vehicle height
  car.reset();
  auto lev = make("1");
  const auto out_lev = run(*lev, to_msg(pts, 0.52f + 1.5f));  // the switch: crop + 1.5, the node cuts
  ASSERT_EQ(out_car.size(), out_lev.size());
  for (size_t i = 0; i < out_car.size(); ++i) {
    EXPECT_EQ(out_car[i].x, out_lev[i].x);
    EXPECT_EQ(out_car[i].y, out_lev[i].y);
    EXPECT_EQ(out_car[i].z, out_lev[i].z);
  }
  size_t box = 0;
  for (const auto & p : out_lev) box += (std::abs(p.x - 10.0f) < 0.4f && std::abs(p.y) < 0.4f) ? 1 : 0;
  EXPECT_GT(box, 20u);
}

TEST_F(ScanGroundFilterTest, PitchedBodyKeepsTheObject)
{
  // the body pitched nose-down 4 deg: the road ahead rises in base_link (0.70 m at 10 m); a 0.4 m box on it 10 m ahead
  const float pitch = 4.0f * kDeg;
  auto pts = road(pitch, 0.0f);
  add_box(pts, 10.0f, 0.0f, 0.6f, 0.4f, pitch);
  const auto box_at = [](const std::vector<P> & v) {
    size_t n = 0;
    for (const auto & p : v) n += (std::abs(p.x - 10.0f) < 0.4f && std::abs(p.y) < 0.4f) ? 1 : 0;
    return n;
  };
  const auto road_kept = [pitch](const std::vector<P> & v) {  // road points (within 3 cm of the road) left as obstacles
    size_t n = 0;
    for (const auto & p : v) n += (std::abs(p.z - p.x * std::tan(pitch)) < 0.03f && std::abs(p.x - 10.0f) > 0.5f) ? 1 : 0;
    return n;
  };
  auto car = make(nullptr);
  const auto out_car = run(*car, to_msg(pts, 0.52f));
  car.reset();
  auto lev = make("1");
  const auto out_lev = run(*lev, to_msg(pts, 0.52f + 1.5f));
  EXPECT_EQ(box_at(out_car), 0u);   // the car: the box is above the base_link crop
  EXPECT_GT(box_at(out_lev), 20u);  // levelled: kept
  EXPECT_EQ(road_kept(out_lev), 0u);
  for (const auto & p : out_lev)  // output in base_link: the box's points keep their base_link height (road there 0.70 m)
    if (std::abs(p.x - 10.0f) < 0.4f && std::abs(p.y) < 0.4f) {
      EXPECT_GT(p.z, 0.70f);
    }
}

TEST_F(ScanGroundFilterTest, RoadRisingAheadKeepsTheCarBehindAndAtTheBumper)
{
  // jolt-3 (10-29 19:01:00, W6a): a level body before a road that starts to rise 3 m ahead at 6 deg. The road line fitted 2-12 m
  // ahead tilts ~5 deg but does not pass through the robot (intercept ~ -0.2 m): rotating about base_link alone lifted the flat
  // road behind into the band's ceiling (objects behind cut) and raised the ceiling at the bumper (a dust patch at 0.55-0.7 m kept).
  const float rise = 6.0f * kDeg;
  std::vector<P> pts;
  for (float x = -10.0f; x < 30.0f; x += 0.1f)
    for (float y = -6.0f; y <= 6.0f; y += 0.1f) {
      if (std::abs(x) < 1.5f && std::abs(y) < 0.6f) continue;
      pts.push_back({x, y, x > 3.0f ? (x - 3.0f) * std::tan(rise) : 0.0f});
    }
  add_box(pts, -4.0f, 0.0f, 0.6f, 0.4f, 0.0f);  // a 0.4 m box 4 m behind
  for (float x = 1.6f; x < 2.0f; x += 0.03f)      // a patch 0.55-0.7 m over the flat road just past the bumper (front 1.25 m)
    for (float y = -0.1f; y < 0.4f; y += 0.03f)
      for (float z = 0.55f; z < 0.7f; z += 0.03f) pts.push_back({x, y, z});
  GroundLevel g;
  g.fit(&pts[0].x, pts.size(), 3);
  ASSERT_TRUE(g.ok);
  EXPECT_GT(g.pitch / kDeg, 3.0f);    // the line through the rising road ...
  EXPECT_LT(g.icpt_x, -0.1f);         // ... does not pass through the robot
  auto car = make(nullptr);
  const auto out_car = run(*car, to_msg(pts, 0.52f));
  car.reset();
  auto lev = make("1");
  const auto out_lev = run(*lev, to_msg(pts, 0.52f + 1.5f));
  const auto count = [](const std::vector<P> & v, float x0, float x1, float y0, float y1, float z0) {
    size_t n = 0;
    for (const auto & p : v) n += (p.x > x0 && p.x < x1 && p.y > y0 && p.y < y1 && p.z > z0) ? 1 : 0;
    return n;
  };
  const size_t behind_car = count(out_car, -4.4f, -3.6f, -0.4f, 0.4f, 0.1f), behind_lev = count(out_lev, -4.4f, -3.6f, -0.4f, 0.4f, 0.1f);
  EXPECT_GT(behind_car, 20u);
  EXPECT_EQ(behind_lev, behind_car);                              // behind: the car's band
  EXPECT_EQ(count(out_lev, 1.55f, 2.05f, -0.15f, 0.45f, 0.52f), 0u);  // at the bumper: nothing over the car's ceiling
  for (const auto & p : out_lev) {                                 // behind the robot: exactly the car's points
    if (p.x <= 0.0f) {
      bool found = false;
      for (const auto & c : out_car) if (c.x == p.x && c.y == p.y && c.z == p.z) { found = true; break; }
      if (!found) { ADD_FAILURE() << "behind point not in the car's output " << p.x << " " << p.y << " " << p.z; break; }
    }
  }
}

TEST_F(ScanGroundFilterTest, RoadFallingSidewaysKeepsTheCarsBand)
{
  // jolt-4 (lidarinsert s1_087 recess): the road falls 3 deg toward a wall on the right; 0.52 m over the LOCAL road sits lower in
  // base_link there and cut the wall's band the car keeps (a low object beside the wall lost its track). Levelled = the car's band
  // plus what the road frame adds: every point the car keeps in the wall is kept.
  const float roll = 3.0f * kDeg;
  auto pts = road(0.0f, roll);
  for (float x = 3.0f; x < 10.0f; x += 0.05f)  // a wall 5 m to the right from the road up to 1.5 m
    for (float z = -5.0f * std::tan(roll); z < 1.5f; z += 0.04f) pts.push_back({x, -5.0f, z});
  auto car = make(nullptr);
  const auto out_car = run(*car, to_msg(pts, 0.52f));
  car.reset();
  auto lev = make("1");
  const auto out_lev = run(*lev, to_msg(pts, 0.52f + 1.5f));
  const auto wall = [](const std::vector<P> & v, float z0, float z1) {
    size_t n = 0;
    for (const auto & p : v) n += (p.x > 3.0f && p.x < 10.0f && std::abs(p.y + 5.0f) < 0.05f && p.z > z0 && p.z < z1) ? 1 : 0;
    return n;
  };
  EXPECT_GT(wall(out_car, 0.3f, 0.52f), 100u);                // the car keeps the wall up to its base_link ceiling
  EXPECT_EQ(wall(out_lev, 0.3f, 0.52f), wall(out_car, 0.3f, 0.52f));  // so does the levelled band
  EXPECT_EQ(wall(out_lev, 0.52f, 2.0f), 0u);                  // and nothing over the car's ceiling (the road's is 0.26 m lower here)
}

TEST_F(ScanGroundFilterTest, NoseUpBranchLeftOutAsByTheCarsBandRoad)
{
  // jolt-5 (w31 20:48:03.9, PITCHG): the body nose-up 3 deg, the road ahead falls in base_link (-1.05 m at 20 m). A branch
  // 1.0-1.3 m over the lane 20 m ahead lies under base_link's 0.52 m: the car's crop keeps it and the car's NERO_GROUND_BAND_ROAD
  // (rampground_1031) leaves it out (floating, > 0.67 m over the road, the road well below the body's plane). With
  // NERO_GROUND_LEVEL the car's ceiling is kept too (jolt-4), so the band road must run in base_link as on the car: in the road
  // frame its gate never opens and the branch came back. A wall standing on the road keeps its band (supported) as on the car.
  const char * band = "0.52,0.15,0.3,0.15,0.3,2.0,30,40";  // the car's value
  const float pitch = -3.0f * kDeg;
  const float t = std::tan(pitch);
  auto pts = road(pitch, 0.0f);
  for (float x = 19.5f; x < 20.5f; x += 0.05f)
    for (float h = 1.0f; h < 1.3f; h += 0.05f) pts.push_back({x, 0.2f, x * t + h});
  for (float x = 14.0f; x < 17.0f; x += 0.05f)  // a wall 4 m to the left, standing on the road, up to 1.5 m over it
    for (float h = 0.0f; h < 1.5f; h += 0.04f) pts.push_back({x, 4.0f, x * t + h});
  const auto branch = [](const std::vector<P> & v) {
    size_t n = 0;
    for (const auto & p : v) n += (p.x > 19.4f && p.x < 20.6f && std::abs(p.y - 0.2f) < 0.05f) ? 1 : 0;
    return n;
  };
  const auto wall = [t](const std::vector<P> & v, float h0, float h1) {
    size_t n = 0;
    for (const auto & p : v) {
      const float h = p.z - p.x * t;
      n += (p.x > 13.9f && p.x < 17.1f && std::abs(p.y - 4.0f) < 0.05f && h > h0 && h < h1) ? 1 : 0;
    }
    return n;
  };
  auto plain = make(nullptr);  // the car's crop without the band road: the branch is in the band
  EXPECT_GT(branch(run(*plain, to_msg(pts, 0.52f))), 20u);
  plain.reset();
  auto car = make(nullptr, band);
  const auto out_car = run(*car, to_msg(pts, 0.52f));
  car.reset();
  auto lev = make("1", band);
  const auto out_lev = run(*lev, to_msg(pts, 0.52f + 1.5f));
  EXPECT_EQ(branch(out_car), 0u);  // the car's band road leaves it out
  EXPECT_EQ(branch(out_lev), 0u);  // and so does the levelled filter
  EXPECT_GT(wall(out_car, 0.52f, 1.3f), 100u);                     // the wall's upper band kept by the car (supported)
  EXPECT_EQ(wall(out_lev, 0.52f, 1.3f), wall(out_car, 0.52f, 1.3f));  // the same with the levelled filter
}

TEST_F(ScanGroundFilterTest, InvalidSwitchIsTheCar)
{
  auto pts = road(4.0f * kDeg, 0.0f);
  add_box(pts, 10.0f, 0.0f, 0.6f, 0.4f, 4.0f * kDeg);
  const auto in = to_msg(pts, 0.52f);
  auto car = make(nullptr);
  const auto out_car = run(*car, in);
  car.reset();
  for (const char * bad : {"0", "true", "1 ", "0x1"}) {
    auto n = make(bad);  // refused at start (an ERROR line): the node runs the car's code path
    const auto out = run(*n, in);
    ASSERT_EQ(out.size(), out_car.size()) << bad;
    for (size_t i = 0; i < out.size(); ++i) EXPECT_EQ(out[i].z, out_car[i].z) << bad;
  }
}

TEST_F(ScanGroundFilterTest, NearBumperClassifiedAsTheCar)
{
  // jolt-8 (jg9; 10-29 19:00:42.80, AEB on residual dust): with a tilted road fit, jg8 classified the points right in front of the
  // bumper rotated about base_link (1.5-3 cm at 2.3 m on the field's 0.3-0.4 deg fits) and low dust returns the car calls ground
  // became obstacle points. Near the robot jg9 classifies in base_link: every point within ~2.2 m of the grid origin (its cells and
  // every cell before them on the ray lie under xn0) is classified exactly as by the car.
  const float pitch = 2.0f * kDeg, roll = 1.0f * kDeg;
  auto pts = road(pitch, roll);
  const auto road_z = [&](float x, float y) { return x * std::tan(pitch) + y * std::tan(roll); };
  for (float x = 1.6f; x < 2.9f; x += 0.07f)  // dust-like single returns 3-30 cm over the road just past the bumper
    for (float y = -0.8f; y <= 0.8f; y += 0.13f)
      for (float dz = 0.03f; dz < 0.31f; dz += 0.03f) pts.push_back({x, y, road_z(x, y) + dz});
  add_box(pts, 10.0f, 0.0f, 0.6f, 0.4f, pitch);  // and a real 0.4 m box 10 m ahead
  auto car = make(nullptr);
  const auto out_car = run(*car, to_msg(pts, 0.52f));
  car.reset();
  auto lev = make("1");
  const auto out_lev = run(*lev, to_msg(pts, 0.52f + 1.5f));
  const auto near = [](const P & p) { return p.x > 1.5f && p.x < 2.5f && std::abs(p.y) < 1.0f && p.z < 0.52f; };
  std::vector<P> nc, nl;
  for (const auto & p : out_car) if (near(p)) nc.push_back(p);
  for (const auto & p : out_lev) if (near(p)) nl.push_back(p);
  EXPECT_GT(nc.size(), 50u);  // the case has near obstacle points to compare
  ASSERT_EQ(nl.size(), nc.size());
  for (size_t i = 0; i < nc.size(); ++i) {
    EXPECT_EQ(nl[i].x, nc[i].x);
    EXPECT_EQ(nl[i].y, nc[i].y);
    EXPECT_EQ(nl[i].z, nc[i].z);
  }
  size_t box = 0;  // the box ahead still kept
  for (const auto & p : out_lev) box += (std::abs(p.x - 10.0f) < 0.4f && std::abs(p.y) < 0.4f) ? 1 : 0;
  EXPECT_GT(box, 20u);
}
