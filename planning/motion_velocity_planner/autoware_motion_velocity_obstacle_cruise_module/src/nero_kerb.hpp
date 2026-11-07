// NERO (kerbout 2026-11-06, QUEUE KERBOUT): a vehicle that stood beside the robot's way and starts to move. Unset = stock.
//   NERO_CRUISE_KERB="gap_m,a_max,v_slow,d_side,d_conf,t_claim,k_pull[,a_poss]" (e.g. 1.0,2.5,1.5,2.0,1.2,4.0,0.35)
//   a_poss (optional, default 0 = off): a POSSIBLE start ahead of a robot at speed asks at most this gentle deceleration (with the
//   pull-out claim); off because the car's tracker slides parked cars' boxes like slow starts (~6 gentle ease-offs / h on the 10-29
//   trips at 0.75): Ahmed's trade, DECISIONS KERBOUT-SLOW.
//   Field: 11-03 #21 (a car parked at the kerb pulls out just ahead of the passing robot: operator STOP, closest 0.31 m) and #38
//   (a car reverses out of a nose-in spot across the path while the robot starts toward it: handheld takeover at 2.9 m). No stock
//   module reacts to such a vehicle before its box is in the robot's band: dynamic_obstacle_stop counts only objects heading
//   > 135 deg against ego, beside ego "too close", >= 1.0 m/s; obstacle_cruise's cut-in needs >= 1.5 m/s and a moving robot;
//   obstacle_stop's outside check is off; obstacle_slow_down caps 10 km/h only once the car is "moving" and within 1.1 m.
//   Per vehicle (CAR TRUCK BUS TRAILER MOTORCYCLE), from its own published positions (predicted objects, one sample per objects
//   message):
//   1. standing anchor: its centre stayed within 0.3 m of the window median (|v| < 0.6) for 2 s, the window ending 0.5 s ago (a
//      start must not drag the anchor); the anchor's place is fixed once set (a window within 0.15 m only refreshes it), a re-born
//      id within 1.2 m / 30 deg of a vanished track's anchor (<= 3 s) takes it over. Kept 30 s after its last confirmation.
//   2. beside the robot's way: its box 0 < distance to ego's trajectory band <= d_side, its front ahead of ego's rear, < 40 m
//      along; "at the kerb": the anchor lies in no non-route road lanelet whose local direction is within 40 deg of its heading
//      (a car keeping a lane next to ego, or an oncoming car in its lane, is not parked at the kerb).
//   3. start ("possible"): both box ends (along the anchor heading, the current yaw folded onto it: mirrored boxes) are >=
//      max(0.25, 3 x the anchor's along-noise) away from the anchor the same way for 3 cycles, the track is >= 1 s old, its length
//      within 0.4 m of the anchor's, measured (not coasting: shape kept and moved exactly by its own speed), and the motion is
//      rigid and sustained: both ends advanced >= 0.1 m together (within 0.15 m) over the last 0.5 s and the box was already
//      advancing (>= 0.03 m) the 0.5 s before.
//      "confirmed": its net displacement along the start direction >= d_conf while moving >= 0.4 m/s, >= 4 of the last 5 samples
//      measured, steady (>= 0.2 m along the start direction in each half of the last second, max <= 2.5 x min) and one way (never
//      fell back > 0.15 m along it in the last 2 s: a parked car's box rocking back and forth is not a move).
//   4. claim over t_claim: its measured motion (positions over 1 s), straight; a confirmed starter parked along the road (within
//      45 deg of the path) also pulls out toward ego's path at max(k_pull x its speed, its measured closing speed) and keeps its
//      measured acceleration (<= 1.0 m/s2, <= 4 m/s).
//   5. robot: a possible start acts only while the robot is slower than v_slow and the vehicle is measurably closing on ego's path
//      (its offset over 1 s shrinks at >= 0.15 m/s and >= 0.4 x its speed); a slow robot is held (velocity limit 0) if going on
//      (the trajectory's speeds reached at +0.5 m/s2) brings it within gap_m of the claim (and closer than now): the hold's
//      deceleration is the gentlest of 1.0 / 1.5 / 2.0 / a_max (jerk-limited from the robot's speed) that keeps gap_m, else a_max
//      if stopping helps or the vehicle is ahead of ego's nose. A hold already asked stays a hold while the robot's speed rises past
//      v_slow (latency). A confirmed start at speed asks the smallest deceleration (0.25 steps, jerk-limited) that keeps gap_m,
//      else a_max when its rear is ahead of ego's nose (or braking buys >= 0.1 m): a velocity limit (the speed after 2 s at it,
//      with that deceleration as its constraint). An ask is held >= 1 s and while the claim stays within gap_m + 0.3 m (a hold:
//      while going would pass it within gap_m + 1.5 m). Nothing is judged while the robot reverses / drives a backward trajectory.
//   6. release: the vehicle stood again for 1.5 s, fell behind ego, moved beyond d_side + 1 m, or is a lead in the band driving
//      away ahead along the path (obstacle_cruise / obstacle_stop's).
//   NERO_CRUISE_KERB_DEBUG=1: one line per judged starter per cycle ([NERO_KERB] ...); events are logged whenever the switch is set.
#ifndef NERO_KERB_HPP_
#define NERO_KERB_HPP_

#include <autoware/motion_velocity_planner_common/planner_data.hpp>
#include <autoware/motion_velocity_planner_common/velocity_planning_result.hpp>
#include <rclcpp/rclcpp.hpp>

#include <autoware_planning_msgs/msg/trajectory_point.hpp>

#include <memory>
#include <string>
#include <vector>

namespace autoware::motion_velocity_planner
{
struct NeroKerbParam
{
  bool on{false};
  double gap_m{1.0};
  double a_max{2.5};
  double v_slow{1.5};
  double d_side{2.0};
  double d_conf{1.2};
  double t_claim{4.0};
  double k_pull{0.35};
  double a_poss{0.0};  // optional 8th field: 0 / absent = a possible start never acts at speed (the delivered setting)
  bool debug{false};
  int debug_level{0};
  std::string raw;  // the switch as read  // NERO_CRUISE_KERB_DEBUG: 1 = a line per judged starter per cycle, 2 = + the gate values of vehicles within 15 m
};

const NeroKerbParam & nero_kerb_param();

// judges every vehicle this cycle; lowers result.velocity_limit when a kerb starter asks the robot to brake / hold
void nero_kerb(
  const std::vector<autoware_planning_msgs::msg::TrajectoryPoint> & traj_points,
  const std::shared_ptr<const PlannerData> & planner_data, const double limit_min_jerk, const double max_jerk,
  const rclcpp::Logger & logger, VelocityPlanningResult & result);
}  // namespace autoware::motion_velocity_planner

#endif  // NERO_KERB_HPP_
