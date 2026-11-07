// NERO (kerbout 2026-11-06, QUEUE KERBOUT): a vehicle that stood beside the robot's way and starts to move (nero_kerb.hpp).
// The Python replica /work/ahmed-kerbout/tools/kerbsim.py is the specification of this code (same steps, same numbers).

#include "nero_kerb.hpp"

#include "type_alias.hpp"

#include <autoware_utils/geometry/boost_polygon_utils.hpp>
#include <autoware_utils/geometry/geometry.hpp>
#include <autoware_utils/ros/uuid_helper.hpp>

#include <boost/geometry.hpp>

#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_core/geometry/LaneletMap.h>
#include <tf2/utils.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace autoware::motion_velocity_planner
{
namespace
{
using autoware_utils::Polygon2d;
constexpr double kDt = 0.1;
constexpr double kTStand = 2.0, kRStand = 0.3, kVStand = 0.6, kRefLag = 0.5, kTKeep = 30.0;
constexpr double kAlongMax = 40.0, kLatDeg = 40.0;
constexpr double kDStart = 0.25, kKNoise = 3.0;
constexpr int kNStart = 3;
constexpr double kDStep = 0.1, kDRigid = 0.15, kDPrev = 0.03, kAgeMin = 1.0, kDlMax = 0.4;
constexpr double kVConf = 0.4, kVlatMin = 0.15, kKVlat = 0.4, kKSteady = 2.5, kBack = 0.15;
constexpr double kVFwdMax = 4.0, kAMeasMax = 1.0, kKBeside = 0.5, kHoldS = 1.0, kReleaseS = 1.5, kAHold = 1.0;

double fold(const double y, const double ref)
{
  const double d = std::remainder(y - ref, 2.0 * M_PI);
  return d > M_PI_2 ? y - M_PI : (d < -M_PI_2 ? y + M_PI : y);
}

Polygon2d make_box(const double x, const double y, const double yaw, const double front, const double rear, const double w)
{
  const double c = std::cos(yaw), s = std::sin(yaw);
  Polygon2d p;
  const std::array<std::pair<double, double>, 5> pts{
    {{front, w / 2}, {front, -w / 2}, {-rear, -w / 2}, {-rear, w / 2}, {front, w / 2}}};
  for (const auto & [a, b] : pts) {
    p.outer().emplace_back(x + c * a - s * b, y + s * a + c * b);
  }
  boost::geometry::correct(p);
  return p;
}

// the trajectory as a polyline: arc length, offsets, poses (once per cycle; no motion_utils path copies per query)
struct Path
{
  static constexpr size_t kBlk = 16;
  std::vector<double> x, y, s, yaw, v;
  std::vector<double> bcx, bcy, br;  // block bounding circles
  explicit Path(const std::vector<autoware_planning_msgs::msg::TrajectoryPoint> & tp)
  {
    for (const auto & q : tp) {
      const double px = q.pose.position.x, py = q.pose.position.y;
      if (!x.empty() && std::hypot(px - x.back(), py - y.back()) < 1e-3) {
        continue;
      }
      s.push_back(x.empty() ? 0.0 : s.back() + std::hypot(px - x.back(), py - y.back()));
      x.push_back(px);
      y.push_back(py);
      v.push_back(q.longitudinal_velocity_mps);
    }
    for (size_t i = 0; i + 1 < x.size(); ++i) {
      yaw.push_back(std::atan2(y[i + 1] - y[i], x[i + 1] - x[i]));
    }
    // blocks of kBlk segments: a bounding circle each (the nearest-segment search skips blocks that cannot hold it)
    for (size_t b = 0; b * kBlk + 1 < x.size(); ++b) {
      const size_t i0 = b * kBlk, i1 = std::min(i0 + kBlk, x.size() - 1);  // points i0 .. i1
      double cx = 0.0, cy = 0.0;
      for (size_t i = i0; i <= i1; ++i) {
        cx += x[i];
        cy += y[i];
      }
      cx /= static_cast<double>(i1 - i0 + 1);
      cy /= static_cast<double>(i1 - i0 + 1);
      double r = 0.0;
      for (size_t i = i0; i <= i1; ++i) {
        r = std::max(r, std::hypot(x[i] - cx, y[i] - cy));
      }
      bcx.push_back(cx);
      bcy.push_back(cy);
      br.push_back(r);
    }
  }
  bool ok() const { return x.size() >= 2; }
  // arc length of the projection and the signed offset (left +): the nearest segment, ties to the first (exactly the full
  // scan's answer; blocks whose bounding circle is farther than the best so far are skipped)
  std::pair<double, double> frenet(const double px, const double py) const
  {
    double best = std::numeric_limits<double>::max(), bs = 0.0, bd = 0.0;
    size_t bi = 0;
    const auto scan = [&](const size_t b) {
      const size_t i0 = b * kBlk, i1 = std::min(i0 + kBlk, x.size() - 1);
      for (size_t i = i0; i < i1; ++i) {
        const double dx = x[i + 1] - x[i], dy = y[i + 1] - y[i], l2 = dx * dx + dy * dy;
        const double u = std::clamp(((px - x[i]) * dx + (py - y[i]) * dy) / l2, 0.0, 1.0);
        const double qx = x[i] + u * dx, qy = y[i] + u * dy, d = std::hypot(px - qx, py - qy);
        if (d < best || (d == best && i < bi)) {
          best = d;
          bi = i;
          bs = s[i] + u * std::sqrt(l2);
          bd = std::copysign(d, dx * (py - y[i]) - dy * (px - x[i]));
        }
      }
    };
    size_t b0 = 0;
    double lb0 = std::numeric_limits<double>::max();
    for (size_t b = 0; b < br.size(); ++b) {
      const double lb = std::hypot(px - bcx[b], py - bcy[b]) - br[b];
      if (lb < lb0) {
        lb0 = lb;
        b0 = b;
      }
    }
    scan(b0);
    for (size_t b = 0; b < br.size(); ++b) {
      if (b != b0 && std::hypot(px - bcx[b], py - bcy[b]) - br[b] <= best + 1e-6) {
        scan(b);
      }
    }
    return {bs, bd};
  }
  size_t seg(const double q) const
  {
    const auto it = std::upper_bound(s.begin(), s.end(), q);
    const size_t i = it == s.begin() ? 0 : static_cast<size_t>(it - s.begin()) - 1;
    return std::min(i, x.size() - 2);
  }
  std::array<double, 3> pose(double q) const
  {
    q = std::clamp(q, 0.0, s.back());
    const size_t i = seg(q);
    const double u = (q - s[i]) / std::max(s[i + 1] - s[i], 1e-9);
    return {x[i] + u * (x[i + 1] - x[i]), y[i] + u * (y[i + 1] - y[i]), yaw[i]};
  }
  double vat(double q) const
  {
    q = std::clamp(q, 0.0, s.back());
    const size_t i = seg(q);
    const double u = (q - s[i]) / std::max(s[i + 1] - s[i], 1e-9);
    return v[i] + u * (v[i + 1] - v[i]);
  }
};

struct Sample
{
  double t, x, y, yaw, l, w, v;
};
struct Anchor
{
  double x, y, yaw, l, sd, t_conf;
  int kerb{-1};  // -1 not judged yet
};
struct Start
{
  double t_start{0.0};
  int dir{1};
  bool conf{false};
  double t_ask{-1.0};
  double ask{-1.0};  // < 0 none; 0 hold; > 0 deceleration
  double a_hold{kAHold};  // a hold's deceleration
  double still{0.0};
};
struct Track
{
  std::deque<Sample> h;
  std::optional<Anchor> a;
  int cnt{0};
  std::deque<std::array<double, 3>> dh;  // (t, front end, rear end) displacements from the anchor
  std::deque<bool> coast;                // the last 5 samples: coasting (no measurement)
  std::optional<Start> st;
  double last_seen{0.0};
};
struct State
{
  std::unordered_map<std::string, Track> tracks;
  std::unordered_map<std::string, std::pair<double, Anchor>> vanished;
  double last_stamp{-1.0};
  std::optional<VelocityLimit> last_limit;
};
State & state()
{
  static State s;
  return s;
}

// the sample nearest to `ago` seconds before t, within tol (the planner does not see every objects message)
const Sample * sample_ago(const std::deque<Sample> & h, const double t, const double ago, const double tol)
{
  const Sample * best = nullptr;
  for (auto it = h.rbegin(); it != h.rend(); ++it) {
    const double e = std::abs((t - it->t) - ago);
    if (e <= tol && (!best || e < std::abs((t - best->t) - ago))) {
      best = &*it;
    }
    if (t - it->t > ago + tol) {
      break;
    }
  }
  return best;
}

bool is_vehicle(const uint8_t label)
{
  using autoware_perception_msgs::msg::ObjectClassification;
  return label == ObjectClassification::CAR || label == ObjectClassification::TRUCK ||
         label == ObjectClassification::BUS || label == ObjectClassification::TRAILER ||
         label == ObjectClassification::MOTORCYCLE;
}

// "at the kerb": the anchor lies in no non-route road lanelet whose local direction is within kLatDeg of its heading
bool at_kerb(const route_handler::RouteHandler & rh, const Anchor & a)
{
  const auto map = rh.getLaneletMapPtr();
  if (!map) {
    return true;
  }
  const lanelet::BasicPoint2d p(a.x, a.y);
  for (const auto & [dist, ll] : lanelet::geometry::findWithin2d(map->laneletLayer, p, 0.0)) {
    if (rh.isRouteLanelet(ll)) {
      continue;
    }
    const std::string sub = ll.attributeOr(lanelet::AttributeName::Subtype, "");
    if (sub != "road" && sub != "road_shoulder") {
      continue;
    }
    const auto c = ll.centerline2d();
    if (c.size() < 2) {
      continue;
    }
    size_t bi = 0;
    double bd = std::numeric_limits<double>::max();
    for (size_t i = 0; i + 1 < c.size(); ++i) {
      const double d = std::hypot(c[i].x() - a.x, c[i].y() - a.y);
      if (d < bd) {
        bd = d;
        bi = i;
      }
    }
    const double dir = std::atan2(c[bi + 1].y() - c[bi].y(), c[bi + 1].x() - c[bi].x());
    if (std::abs(std::remainder(a.yaw - dir, 2.0 * M_PI)) < kLatDeg * M_PI / 180.0) {
      return false;
    }
  }
  return true;
}
}  // namespace

const NeroKerbParam & nero_kerb_param()
{
  static const NeroKerbParam p = [] {
    NeroKerbParam r;
    if (const char * e = std::getenv("NERO_CRUISE_KERB")) {
      r.on = std::sscanf(
               e, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &r.gap_m, &r.a_max, &r.v_slow, &r.d_side, &r.d_conf, &r.t_claim,
               &r.k_pull, &r.a_poss) >= 7;
    }
    r.raw = std::getenv("NERO_CRUISE_KERB") ? std::getenv("NERO_CRUISE_KERB") : "";
    if (const char * e = std::getenv("NERO_CRUISE_KERB_DEBUG")) {
      r.debug_level = std::atoi(e);
      r.debug = r.debug_level >= 1;
    }
    return r;
  }();
  return p;
}

void nero_kerb(
  const std::vector<autoware_planning_msgs::msg::TrajectoryPoint> & traj_points,
  const std::shared_ptr<const PlannerData> & planner_data, const double limit_min_jerk, const double max_jerk,
  const rclcpp::Logger & logger, VelocityPlanningResult & result)
{
  const auto & P = nero_kerb_param();
  static bool announced = false;
  if (!announced) {  // one line at the first cycle: read / not understood / unset (= the car, silent)
    announced = true;
    if (P.on) {
      RCLCPP_INFO(
        logger, "[NERO_CRUISE_KERB] on: gap %.2f m, a_max %.2f, slow robot < %.2f m/s, beside <= %.2f m, confirmed %.2f m, claim %.1f s, "
        "pull-out %.2f, possible-start ease-off %.2f (0 = off)", P.gap_m, P.a_max, P.v_slow, P.d_side, P.d_conf, P.t_claim, P.k_pull,
        P.a_poss);
    } else if (!P.raw.empty()) {
      RCLCPP_WARN(logger, "[NERO_CRUISE_KERB] '%s' not understood: off (the car)", P.raw.c_str());
    }
  }
  if (!P.on) {
    return;
  }
  auto & S = state();
  // never weaker than what the module already asks: the lower speed AND the stronger deceleration / jerk of the two
  const auto merge = [&](const VelocityLimit & msg) {
    if (!result.velocity_limit) {
      result.velocity_limit = msg;
      return;
    }
    auto & r = *result.velocity_limit;
    const bool had = r.use_constraints;
    r.max_velocity = std::min(r.max_velocity, msg.max_velocity);
    r.use_constraints = true;
    r.constraints.min_acceleration =
      had ? std::min(r.constraints.min_acceleration, msg.constraints.min_acceleration) : msg.constraints.min_acceleration;
    r.constraints.min_jerk = had ? std::min(r.constraints.min_jerk, msg.constraints.min_jerk) : msg.constraints.min_jerk;
    r.constraints.max_jerk = had ? std::max(r.constraints.max_jerk, msg.constraints.max_jerk) : msg.constraints.max_jerk;
  };
  const double t = rclcpp::Time(planner_data->predicted_objects_header.stamp).seconds();
  if (t == S.last_stamp) {  // no new objects message: this cycle asks what the last one asked
    if (S.last_limit) {
      merge(*S.last_limit);
    }
    return;
  }
  S.last_stamp = t;
  S.last_limit.reset();
  const Path T(traj_points);
  if (!T.ok()) {
    return;
  }
  const auto & vi = planner_data->vehicle_info_;
  const double front = vi.max_longitudinal_offset_m, rear = vi.rear_overhang_m, ew = vi.vehicle_width_m;
  const auto & ep = planner_data->current_odometry.pose.pose;
  const double ex = ep.position.x, ey = ep.position.y, eyaw = tf2::getYaw(ep.orientation);
  const double ev_signed = planner_data->current_odometry.twist.twist.linear.x;
  const double ev = std::max(0.0, ev_signed);
  // a robot reversing / on a backward trajectory is not this rule's (pull-out from a spot, parking): judge nothing, ask nothing
  const bool active = planner_data->is_driving_forward && ev_signed > -0.1;
  const auto [s_e, d_e] = T.frenet(ex, ey);
  const Polygon2d ego_now = make_box(ex, ey, eyaw, front, rear, ew);
  const int n = static_cast<int>(std::round(P.t_claim / kDt));
  const double jerk = std::abs(limit_min_jerk);
  // ego's footprints along the trajectory: decelerating at a (jerk-limited from 0), or going on ("go": the trajectory's speeds
  // reached at +0.5 m/s2)
  std::map<int, std::vector<Polygon2d>> ego_cache;
  const auto ego_motion = [&](const int key, const double a, const bool go) -> const std::vector<Polygon2d> & {
    const auto it = ego_cache.find(key);
    if (it != ego_cache.end()) {
      return it->second;
    }
    std::vector<Polygon2d> out;
    double v = ev, q = s_e, acc = 0.0;
    for (int k = 0; k < n; ++k) {
      if (go) {
        const double vt = std::max(T.vat(q), 0.0);
        v = v < vt ? std::min(vt, v + 0.5 * kDt) : std::max(vt, v - 1.0 * kDt);
      } else {
        acc = std::min(a, acc + jerk * kDt);
        v = std::max(0.0, v - acc * kDt);
      }
      q += v * kDt;
      const auto p = T.pose(q);
      out.push_back(make_box(p[0], p[1], p[2], front, rear, ew));
    }
    return ego_cache.emplace(key, std::move(out)).first->second;
  };
  const auto log_event = [&](const std::string & id, const char * kind, const std::string & info) {
    RCLCPP_INFO(logger, "[NERO_KERB] %s %s %s", id.substr(0, 8).c_str(), kind, info.c_str());
  };
  char buf[320];

  double a_req = 0.0, v_lim = std::numeric_limits<double>::max();
  bool any = false;
  for (const auto & object : planner_data->objects) {
    const auto & po = object->predicted_object;
    if (po.classification.empty() || !is_vehicle(po.classification.at(0).label)) {
      continue;
    }
    const std::string id = autoware_utils::to_hex_string(po.object_id);
    const auto & pose = po.kinematics.initial_pose_with_covariance.pose;
    const double x = pose.position.x, y = pose.position.y, yaw = tf2::getYaw(pose.orientation);
    const double l = po.shape.dimensions.x, w = po.shape.dimensions.y;
    const double vx = po.kinematics.initial_twist_with_covariance.twist.linear.x;
    auto & tr = S.tracks[id];
    tr.last_seen = t;
    tr.h.push_back({t, x, y, yaw, l, w, vx});
    while (!tr.h.empty() && t - tr.h.front().t > 12.0) {
      tr.h.pop_front();
    }
    const auto & H = tr.h;
    // 1. standing anchor
    {
      static thread_local std::vector<const Sample *> win;  // reused buffers (no allocation per object per cycle)
      win.clear();
      for (const auto & q : H) {
        if (kRefLag <= t - q.t && t - q.t <= kTStand + kRefLag) {
          win.push_back(&q);
        }
      }
      bool standing = !win.empty() && t - win.front()->t >= kTStand + kRefLag - 0.15;
      for (const auto * q : win) {
        standing = standing && std::abs(q->v) < kVStand;
      }
      if (standing) {
        static thread_local std::vector<double> xs, ys;
        xs.clear();
        ys.clear();
        for (const auto * q : win) {
          xs.push_back(q->x);
          ys.push_back(q->y);
        }
        const auto median = [](std::vector<double> & v) {
          const size_t m = v.size() / 2;
          std::nth_element(v.begin(), v.begin() + m, v.end());
          if (v.size() % 2) {
            return v[m];
          }
          const double hi = v[m];
          std::nth_element(v.begin(), v.begin() + m - 1, v.end());
          return 0.5 * (hi + v[m - 1]);
        };
        const double mx = median(xs), my = median(ys);
        for (const auto * q : win) {
          standing = standing && std::hypot(q->x - mx, q->y - my) <= kRStand;
        }
        if (standing && !tr.st) {
          const double yr = win.back()->yaw;
          static thread_local std::vector<double> yv, lv;
          yv.clear();
          lv.clear();
          for (const auto * q : win) {
            yv.push_back(fold(q->yaw, yr));
            lv.push_back(q->l);
          }
          const double ym = median(yv), ux = std::cos(ym), uy = std::sin(ym);
          double m1 = 0.0, m2 = 0.0;
          for (const auto * q : win) {
            const double a = (q->x - mx) * ux + (q->y - my) * uy;
            m1 += a;
            m2 += a * a;
          }
          m1 /= win.size();
          const double sd = std::sqrt(std::max(0.0, m2 / win.size() - m1 * m1));
          if (tr.a && std::hypot(tr.a->x - mx, tr.a->y - my) <= 0.15) {
            tr.a->t_conf = t;
            tr.a->sd = std::min(tr.a->sd, sd);
          } else if (!tr.a || t - tr.a->t_conf > 3.0) {
            tr.a = Anchor{mx, my, ym, median(lv), sd, t, -1};
            tr.cnt = 0;
            tr.dh.clear();
          }
        }
      }
    }
    if (!tr.a) {  // a re-born id takes over a vanished track's anchor by place
      for (auto it = S.vanished.begin(); it != S.vanished.end(); ++it) {
        const auto & [tl, va] = it->second;
        if (
          t - tl <= 3.0 && std::hypot(va.x - x, va.y - y) < 1.2 &&
          std::abs(std::remainder(yaw - va.yaw, M_PI)) < 30.0 * M_PI / 180.0) {
          tr.a = va;
          S.vanished.erase(it);
          break;
        }
      }
    }
    if (!tr.a || t - tr.a->t_conf > kTKeep) {
      continue;
    }
    auto & A = *tr.a;
    // 2. beside the robot's way
    if (!tr.st) {
      // too far to be beside the band within kAlongMax (exact: the path distance is >= the straight one): skip the projections
      const double reach = kAlongMax + rear + l / 2.0 + P.d_side + ew / 2.0 + std::hypot(l, w) / 2.0 + std::abs(d_e) + 1.0;
      if (std::hypot(x - ex, y - ey) > reach) {
        continue;
      }
    }
    const auto [s_o, d_o] = T.frenet(x, y);
    if (!tr.st) {
      // the same tests as below, decided from the centre before the corners (exact: every corner lies within half the box diagonal
      // of the centre, and a box across the path has its centre within that of it)
      if (s_o - s_e >= kAlongMax || s_o - s_e + l / 2.0 <= -rear || std::abs(d_o) - std::hypot(l, w) / 2.0 - ew / 2.0 > P.d_side) {
        continue;
      }
    }
    const Polygon2d ob = make_box(x, y, yaw, l / 2, l / 2, w);
    double dband = std::numeric_limits<double>::max();
    {
      int side = 0;
      bool both = false;
      for (size_t i = 0; i + 1 < ob.outer().size(); ++i) {
        const auto & c = ob.outer()[i];
        const double dc = T.frenet(c.x(), c.y()).second;
        const int sg = dc > 0.0 ? 1 : -1;
        both = both || (side != 0 && sg != side);
        side = sg;
        dband = std::min(dband, std::abs(dc) - ew / 2.0);
      }
      dband = (both || dband < 0.0) ? 0.0 : dband;
    }
    const double front_along = s_o - s_e + l / 2.0;
    const double path_yaw = T.pose(s_o)[2];
    if (tr.st) {
      const bool lead = dband == 0.0 && std::abs(std::remainder(yaw - path_yaw, M_PI)) < 30.0 * M_PI / 180.0 &&
                        s_o - l / 2.0 > s_e + front;
      if (dband > P.d_side + 1.0 || front_along < -rear || s_o - s_e >= kAlongMax || lead) {
        std::snprintf(buf, sizeof(buf), "%s (band %.2f front %.2f)", lead ? "lead" : "out", dband, front_along);
        log_event(id, "release", buf);
        tr.st.reset();
        continue;
      }
    } else if (!(0.0 < dband && dband <= P.d_side && front_along > -rear && s_o - s_e < kAlongMax)) {
      continue;
    }
    if (A.kerb < 0) {
      A.kerb = (!planner_data->route_handler || at_kerb(*planner_data->route_handler, A)) ? 1 : 0;
    }
    if (A.kerb == 0) {
      continue;
    }
    // 3. start: both ends away from the anchor, rigid sustained motion
    const double ux = std::cos(A.yaw), uy = std::sin(A.yaw), yf = fold(yaw, A.yaw);
    const double cx = std::cos(yf), cy = std::sin(yf);
    const double dfr = ((x + cx * l / 2) - (A.x + ux * A.l / 2)) * ux + ((y + cy * l / 2) - (A.y + uy * A.l / 2)) * uy;
    const double dbk = ((x - cx * l / 2) - (A.x - ux * A.l / 2)) * ux + ((y - cy * l / 2) - (A.y - uy * A.l / 2)) * uy;
    const double thr = std::max(kDStart, kKNoise * A.sd);
    const Sample * p5 = sample_ago(H, t, 0.5, 0.1);
    const double vpos = p5 ? std::hypot(x - p5->x, y - p5->y) / std::max(t - p5->t, 1e-3) : 0.0;
    const double age = t - H.front().t;
    // coasting = no measurement: the tracker keeps the shape and moves the box exactly by its own speed (both last steps)
    bool frozen = false;
    if (H.size() >= 3) {
      const auto & q0 = H[H.size() - 3];
      const auto & q1 = H[H.size() - 2];
      const auto & q2 = H.back();
      const bool same_shape = std::abs(q1.l - q0.l) < 1e-3 && std::abs(q2.l - q0.l) < 1e-3 &&
                              std::abs(q1.w - q0.w) < 1e-3 && std::abs(q2.w - q0.w) < 1e-3;
      const auto own_speed_step = [](const Sample & p, const Sample & q) {
        const double dt = q.t - p.t;
        const double ex = p.x + std::cos(p.yaw) * p.v * dt, ey = p.y + std::sin(p.yaw) * p.v * dt;
        return std::hypot(q.x - ex, q.y - ey) < 0.01;
      };
      frozen = same_shape && own_speed_step(q0, q1) && own_speed_step(q1, q2);
    }
    tr.coast.push_back(frozen);
    while (tr.coast.size() > 5) {
      tr.coast.pop_front();
    }
    const int n_coast = static_cast<int>(std::count(tr.coast.begin(), tr.coast.end(), true));
    tr.dh.push_back({t, dfr, dbk});
    while (tr.dh.size() > 12) {
      tr.dh.pop_front();
    }
    const auto dh_at = [&](const double ago) -> const std::array<double, 3> * {
      const std::array<double, 3> * best = nullptr;
      for (const auto & d : tr.dh) {
        const double e = std::abs((t - d[0]) - ago);
        if (e <= 0.1 && (!best || e < std::abs((t - (*best)[0]) - ago))) {
          best = &d;
        }
      }
      return best;
    };
    const auto * h5 = dh_at(0.5);
    const auto * h10 = dh_at(1.0);
    const double sgn = dfr > 0.0 ? 1.0 : -1.0;
    const bool rigid = h5 && h10 && sgn * (dfr - (*h5)[1]) >= kDStep && sgn * (dbk - (*h5)[2]) >= kDStep &&
                       std::abs((dfr - (*h5)[1]) - (dbk - (*h5)[2])) <= kDRigid &&
                       sgn * (((*h5)[1] + (*h5)[2]) - ((*h10)[1] + (*h10)[2])) / 2.0 >= kDPrev;
    const bool ok = dfr * dbk > 0.0 && std::min(std::abs(dfr), std::abs(dbk)) >= thr && rigid && age >= kAgeMin &&
                    std::abs(l - A.l) <= kDlMax && !frozen;
    if (P.debug_level >= 2 && std::hypot(x - ex, y - ey) < 15.0) {
      RCLCPP_INFO(
        logger, "[NERO_KERB2] %s band %.2f fa %.1f kerb %d dfr %.2f dbk %.2f thr %.2f rigid %d h5 %d h10 %d age %.1f dl %.2f fr %d vpos %.2f cnt %d",
        id.substr(0, 8).c_str(), dband, front_along, A.kerb, dfr, dbk, thr, rigid, h5 != nullptr, h10 != nullptr, age,
        std::abs(l - A.l), frozen, vpos, tr.cnt);
    }
    if (!tr.st) {
      tr.cnt = ok ? tr.cnt + 1 : 0;
      if (tr.cnt < kNStart) {
        continue;
      }
      tr.st = Start{};
      tr.st->t_start = t;
      tr.st->dir = dfr > 0.0 ? 1 : -1;
      std::snprintf(
        buf, sizeof(buf), "start d %.2f/%.2f thr %.2f vpos %.2f band %.2f along %.1f", dfr, dbk, thr, vpos, dband,
        s_o - s_e);
      log_event(id, "start", buf);
    }
    auto & st = *tr.st;
    // stood again -> release (and a new anchor)
    const double dts = H.size() >= 2 ? H.back().t - H[H.size() - 2].t : kDt;
    if (vpos < 0.1 && t - st.t_start > 1.0) {
      st.still += dts;
      if (st.still >= kReleaseS) {
        log_event(id, "release", "stood");
        tr.st.reset();
        // re-anchored where it stands now (a second move off this place counts at once)
        tr.a = Anchor{x, y, yf, l, std::max(A.sd, 0.04), t, A.kerb};
        tr.cnt = 0;
        tr.dh.clear();
        continue;
      }
    } else {
      st.still = 0.0;
    }
    // 4. evidence level
    const double net = st.dir * (dfr + dbk) / 2.0;
    // confirmed only on measured boxes: a coasting track (the tracker moving it by its own, possibly false, speed) confirms nothing.
    // steady: the centre advanced >= kVConf x 0.5 s along the start direction in EACH half of the last second, evenly (one re-fit
    // jump of a drifting track is not a move); one way: over the last 2 s it never fell back along that direction by more than
    // kBack (a box rocking back and forth on a parked car is not a move; a car pulling / reversing out does not jump back)
    bool steady = false, one_way = false;
    if (!st.conf && net >= P.d_conf) {
      const double ax = st.dir * ux, ay = st.dir * uy;
      const Sample * q5 = sample_ago(H, t, 0.5, 0.1);
      const Sample * q10 = sample_ago(H, t, 1.0, 0.1);
      if (q5 && q10) {
        const double h1 = (q5->x - q10->x) * ax + (q5->y - q10->y) * ay;
        const double h2 = (x - q5->x) * ax + (y - q5->y) * ay;
        steady = std::min(h1, h2) >= kVConf * 0.5 && std::max(h1, h2) <= kKSteady * std::min(h1, h2);
      }
      double runmax = -1e9, back = 0.0;
      for (const auto & q : H) {
        if (t - q.t > 2.0) {
          continue;
        }
        const double pr = (q.x - A.x) * ax + (q.y - A.y) * ay;
        runmax = std::max(runmax, pr);
        back = std::max(back, runmax - pr);
      }
      one_way = back <= kBack;
    }
    if (!st.conf && net >= P.d_conf && vpos >= kVConf && dfr * dbk > 0.0 && !frozen && n_coast <= 1 && steady && one_way) {
      st.conf = true;
      std::snprintf(buf, sizeof(buf), "net %.2f vpos %.2f along %.1f band %.2f", net, vpos, s_o - s_e, dband);
      log_event(id, "confirm", buf);
    }
    // a hold already asked stays a hold while the robot's speed rises past v_slow (it was still accelerating: latency)
    const bool slow = ev < P.v_slow || st.ask == 0.0;
    if (!active) {
      st.ask = -1.0;
      continue;
    }
    // optional 8th field a_poss (> 0): a POSSIBLE start ahead of a robot at speed asks at most a gentle a_poss (DECISIONS KERBOUT-SLOW)
    const bool poss_fast = !st.conf && !slow;
    if (poss_fast && P.a_poss <= 0.0) {
      if (st.ask >= 0.0) {
        log_event(id, "unask", "possible start, robot not slow");
        st.ask = -1.0;
      }
      continue;
    }
    // 5. claim
    const double v0o = std::max(vpos, 0.3);
    const Sample * p10 = sample_ago(H, t, 1.0, 0.15);
    double a_m = 0.0;
    if (st.conf && p5 && sample_ago(H, t, 1.0, 0.1)) {
      const Sample * q10 = sample_ago(H, t, 1.0, 0.1);
      const double v_old = std::hypot(p5->x - q10->x, p5->y - q10->y) / std::max(p5->t - q10->t, 1e-3);
      a_m = std::clamp((vpos - v_old) / 0.5, 0.0, kAMeasMax);
    }
    double mvx, mvy;
    if (p10) {
      mvx = (x - p10->x) / (t - p10->t);
      mvy = (y - p10->y) / (t - p10->t);
    } else {
      mvx = cx * st.dir * v0o;
      mvy = cy * st.dir * v0o;
    }
    const double mn = std::max(std::hypot(mvx, mvy), 1e-6), dvx = mvx / mn, dvy = mvy / mn;
    double vlat_m = 0.0;
    if (p10) {
      const double d_p = T.frenet(p10->x, p10->y).second;
      vlat_m = (std::abs(d_p) - std::abs(d_o)) / (t - p10->t);
    }
    const bool par = std::abs(std::remainder(A.yaw - path_yaw, M_PI)) < M_PI_4;
    // a car parked along the road pulls out into the lane of its own direction: facing ego's direction = ego's lane (it drifts
    // to ego's path); facing the other way (an oncoming-direction car leaving the kerb) = the other lane: only its measured
    // sideways motion counts
    const bool same_dir = std::abs(std::remainder(yf - path_yaw, 2.0 * M_PI)) < M_PI_2;
    if (!st.conf && !poss_fast && st.ask < 0.0 && !(vlat_m >= kVlatMin && vlat_m >= kKVlat * vpos)) {
      continue;  // a possible start acts only while the vehicle is measurably closing on ego's path
    }
    std::vector<Polygon2d> claim;
    claim.reserve(n);
    double v = v0o;
    if ((st.conf || poss_fast) && par) {
      const double yrel = std::remainder(std::atan2(dvy, dvx) - path_yaw, 2.0 * M_PI);
      const double box_rel = std::remainder(yf - path_yaw, M_PI);
      double s_c = s_o, d_c = d_o;
      for (int k = 0; k < n; ++k) {
        v = std::min(kVFwdMax, v + a_m * kDt);
        const double vl = std::max(same_dir ? P.k_pull * v : 0.0, vlat_m);
        s_c += v * std::cos(yrel) * kDt;
        d_c = std::copysign(std::max(std::abs(d_c) - vl * kDt, 0.0), d_c);
        const auto b = T.pose(s_c);
        claim.push_back(make_box(
          b[0] - d_c * std::sin(b[2]), b[1] + d_c * std::cos(b[2]), b[2] + box_rel, l / 2, l / 2, w));
      }
    } else {
      double xx = x, yy = y;
      for (int k = 0; k < n; ++k) {
        v = std::min(st.conf ? kVFwdMax : v0o, v + a_m * kDt);
        xx += dvx * v * kDt;
        yy += dvy * v * kDt;
        claim.push_back(make_box(xx, yy, yaw, l / 2, l / 2, w));
      }
    }
    // 6. robot options
    const double d_now = boost::geometry::distance(ob, ego_now);
    const auto dmin = [&](const std::vector<Polygon2d> & opt) {
      double d = std::numeric_limits<double>::max();
      for (size_t k = 0; k < std::min(opt.size(), claim.size()); ++k) {
        d = std::min(d, boost::geometry::distance(opt[k], claim[k]));
      }
      return d;
    };
    const bool ahead_now = (s_o - l / 2.0) - (s_e + front) > -kKBeside;
    double ask = -1.0;
    bool soft = false;
    if (slow) {
      const double d_go = dmin(ego_motion(-1, 0.0, true));
      const bool conflict = d_go < P.gap_m && d_go < d_now - 0.1;
      soft = d_go < P.gap_m + 1.5;  // a hold stays while going would pass the claim within gap + 1.5 m (a reverse pauses)
      // the hold's stop is real (jerk-limited from the robot's speed): the gentlest of 1.0 / 1.5 / 2.0 / a_max that keeps gap_m,
      // else a_max if stopping helps at all or the vehicle is ahead of the nose (no stop keeps the gap: stop firmly, as at speed)
      double d_hold = d_go, a_h = -1.0;
      if (conflict) {
        for (const double a : {kAHold, 1.5, 2.0, P.a_max}) {
          if (a > P.a_max) {
            continue;
          }
          const double dk = dmin(ego_motion(100000 + static_cast<int>(std::round(a * 100.0)), a, false));  // own cache keys
          d_hold = std::max(d_hold, dk);
          if (dk >= P.gap_m) {
            a_h = a;
            break;
          }
        }
        if (a_h < 0.0 && (d_hold > d_go + 0.05 || ahead_now)) {
          a_h = P.a_max;
        }
      }
      if (a_h >= 0.0) {
        ask = 0.0;
        st.a_hold = a_h;
      }
      std::snprintf(buf, sizeof(buf), "%s go %.2f hold %.2f a %.2f", st.conf ? "conf" : "poss", d_go, d_hold, st.a_hold);
    } else {
      const double d0 = dmin(ego_motion(0, 0.0, false));
      const bool conflict = d0 < P.gap_m && d0 < d_now - 0.1;
      soft = d0 < P.gap_m + 0.3;
      int used = std::snprintf(buf, sizeof(buf), "conf d0 %.2f", d0);
      if (conflict) {
        double best = d0;
        const int steps = static_cast<int>(std::round(P.a_max / 0.25));
        for (int k = 1; k <= steps; ++k) {
          const double dk = dmin(ego_motion(k, 0.25 * k, false));
          best = std::max(best, dk);
          if (dk >= P.gap_m) {
            ask = 0.25 * k;
            break;
          }
        }
        if (ask < 0.0 && (best >= d0 + 0.1 || ahead_now)) {
          ask = P.a_max;
        }
        if (poss_fast) {  // a possible start: at most a_poss, and only ahead of the nose
          ask = (ask > 0.0 && ahead_now) ? std::min(ask, P.a_poss) : -1.0;
        }
        std::snprintf(buf + used, sizeof(buf) - used, " ask %.2f best %.2f", ask, best);
      }
    }
    const double prev = st.ask;
    if (ask < 0.0 && prev >= 0.0 && (t - st.t_ask < kHoldS || soft)) {
      ask = prev;  // held
    }
    if (ask >= 0.0 && prev < 0.0) {
      st.t_ask = t;
    }
    if ((ask >= 0.0) != (prev >= 0.0)) {
      char ev_buf[400];
      std::snprintf(ev_buf, sizeof(ev_buf), "%s now %.2f v_ego %.2f", buf, d_now, ev);
      log_event(id, ask >= 0.0 ? (ask == 0.0 ? "hold" : "ask") : "unask", ev_buf);
    }
    st.ask = ask;
    if (P.debug_level >= 1) {
      RCLCPP_INFO(
        logger, "[NERO_KERB] %s %s %s v %.2f vpos %.2f vlat %.2f band %.2f along %.1f now %.2f | %s -> %.2f",
        id.substr(0, 8).c_str(), st.conf ? "conf" : "poss", par ? "par" : "perp", vx, vpos, vlat_m, dband, s_o - s_e,
        d_now, buf, ask);
    }
    if (ask >= 0.0) {
      any = true;
      if (ask == 0.0) {
        a_req = std::max(a_req, st.a_hold);
        v_lim = 0.0;
      } else {
        a_req = std::max(a_req, ask);
        v_lim = std::min(v_lim, std::max(0.0, ev - 2.0 * ask));
      }
    }
  }
  // tracks no longer published: their anchors stay findable by place for 3 s
  for (auto it = S.tracks.begin(); it != S.tracks.end();) {
    if (t - it->second.last_seen > 0.25) {  // (a track seen this cycle has last_seen == t)
      if (it->second.a) {
        S.vanished[it->first] = {it->second.last_seen, *it->second.a};
      }
      it = S.tracks.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = S.vanished.begin(); it != S.vanished.end();) {
    it = t - it->second.first > 3.0 ? S.vanished.erase(it) : std::next(it);
  }
  if (any) {
    VelocityLimit msg;
    msg.stamp = planner_data->predicted_objects_header.stamp;
    msg.sender = "obstacle_cruise";
    msg.use_constraints = true;
    msg.max_velocity = v_lim;
    msg.constraints.min_acceleration = -a_req;
    msg.constraints.max_jerk = max_jerk;
    msg.constraints.min_jerk = limit_min_jerk;
    S.last_limit = msg;
    merge(msg);
  }
}
}  // namespace autoware::motion_velocity_planner
