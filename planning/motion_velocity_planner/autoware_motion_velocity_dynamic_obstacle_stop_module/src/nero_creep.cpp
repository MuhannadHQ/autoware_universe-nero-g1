#include "nero_creep.hpp"

#include <rclcpp/rclcpp.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <utility>

namespace autoware::motion_velocity_planner::dynamic_obstacle_stop
{
namespace
{
double now_s()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
std::string key_of(const autoware_perception_msgs::msg::PredictedObject & o)
{
  return std::string(reinterpret_cast<const char *>(o.object_id.uuid.data()), o.object_id.uuid.size());
}
struct Hist
{
  std::deque<std::pair<double, double>> s;  // (t, v)
  double last{0.0};
};
std::unordered_map<std::string, Hist> & hists()
{
  static std::unordered_map<std::string, Hist> h;
  return h;
}
}  // namespace

const NeroCreepParam & nero_creep_param()
{
  static const NeroCreepParam p = [] {
    NeroCreepParam r;
    if (const char * e = std::getenv("NERO_DOS_CREEP")) {
      std::sscanf(e, "%lf,%lf,%lf,%lf,%lf", &r.v_creep, &r.t_react, &r.a_stop, &r.acc_th, &r.hold_s);
    }
    return r;
  }();
  return p;
}

void nero_creep_observe(const autoware_perception_msgs::msg::PredictedObject & object)
{
  if (nero_creep_param().v_creep <= 0.0) return;
  const double t = now_s();
  auto & h = hists()[key_of(object)];
  const double v = std::abs(object.kinematics.initial_twist_with_covariance.twist.linear.x);
  if (!h.s.empty() && t - h.s.back().first < 0.02) return;  // once per cycle
  h.s.emplace_back(t, v);
  h.last = t;
  while (!h.s.empty() && t - h.s.front().first > nero_creep_param().hold_s + 1.5) h.s.pop_front();
  static double last_prune = 0.0;
  if (t - last_prune > 5.0) {
    last_prune = t;
    for (auto it = hists().begin(); it != hists().end();) {
      if (t - it->second.last > 10.0) it = hists().erase(it); else ++it;
    }
  }
}

double nero_creep_reach(const autoware_perception_msgs::msg::PredictedObject & object)
{
  const auto & p = nero_creep_param();
  if (p.v_creep <= 0.0) return -1.0;
  const auto it = hists().find(key_of(object));
  if (it == hists().end() || it->second.s.size() < 3) return -1.0;
  const auto & s = it->second.s;
  const double t = s.back().first;
  const double v = s.back().second;
  if (v > p.v_creep) return -1.0;
  if (t - s.front().first < p.hold_s - 0.2) return -1.0;  // not seen long enough to call it a creeper
  double v_min_1s = v;
  for (const auto & [ti, vi] : s) {
    if (t - ti > p.hold_s) continue;
    if (vi > p.v_creep) return -1.0;  // was faster within hold_s
    if (t - ti <= 1.0) v_min_1s = std::min(v_min_1s, vi);
  }
  const double acc = v - v_min_1s;  // speed gained within the last second
  if (acc > p.acc_th) return -1.0;  // pulling away: not creeping
  static double last_log = 0.0;
  const double reach = v * p.t_react + v * v / (2.0 * p.a_stop);
  if (t - last_log > 1.0) {
    last_log = t;
    RCLCPP_INFO(rclcpp::get_logger("nero_dos_creep"), "[NERO_DOS] creeper v %.2f acc %.2f -> reach %.2f m", v, acc, reach);
  }
  return reach;
}
bool nero_creep_launching(const autoware_perception_msgs::msg::PredictedObject & object)
{
  const auto & p = nero_creep_param();
  if (p.v_creep <= 0.0) return false;
  const auto it = hists().find(key_of(object));
  if (it == hists().end() || it->second.s.size() < 2) return false;
  const auto & s = it->second.s;
  const double t = s.back().first, v = s.back().second;
  if (v < 0.5) return false;
  double v_min_1s = v;
  for (const auto & [ti, vi] : s) {
    if (t - ti <= 1.0) v_min_1s = std::min(v_min_1s, vi);
  }
  return v - v_min_1s > p.acc_th;  // pulling away from (near) standstill
}
}  // namespace autoware::motion_velocity_planner::dynamic_obstacle_stop
