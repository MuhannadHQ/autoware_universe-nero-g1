// Copyright 2026 Nero. Licensed under the Apache License, Version 2.0.
//
// NERO_MOT_TURN_ONSET (yawlag, 2026-11-06): a moving vehicle's turn rate may start from zero.
// 0.44.2's bicycle model gives the slip angle (= its turn-rate state, w = v sin(slip) / l_r) a
// process noise proportional to sin(slip) ("d(w)/dt is assumed to be proportional to w"): at slip 0
// only the 0.3 deg/s floor is left, so a car driving straight cannot start turning in the filter;
// its yaw follows the boxes only through the (20 deg)^2 yaw measurement, ~0.6 s behind, and the
// delay-compensated publish extrapolates no turn. Upstream's current vehicle model (#11230, kept by
// #12628 / #12818) gives its lateral-motion state a constant process noise (q_cov_vel_lat =
// acc_lat^2 dt^2). Same idea in this model's state: for a vehicle moving forward at >= v_on, the
// slip-rate sd is at least yaw_acc x l_r / v (d(slip)/dt = l_r / v x d(w)/dt, w's rate = a yaw
// acceleration sd), capped at slip_rate_max. Unset = the car (the stock noise, untouched).
//   NERO_MOT_TURN_ONSET=<yaw_acc rad/s2>,<v_on m/s>,<slip_rate_max deg/s>
#ifndef AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__MOTION_MODEL__NERO_TURN_ONSET_HPP_
#define AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__MOTION_MODEL__NERO_TURN_ONSET_HPP_

#include <rclcpp/rclcpp.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace autoware::multi_object_tracker
{

struct NeroTurnOnsetParams
{
  bool enabled{false};
  double yaw_acc{0.0};        // [rad/s2] yaw acceleration sd of a moving vehicle
  double v_on{1.0};           // [m/s] forward speed from which it applies
  double slip_rate_max{0.0};  // [rad/s] cap of the slip-rate sd under the switch
  NeroTurnOnsetParams()
  {
    const char * env = std::getenv("NERO_MOT_TURN_ONSET");
    if (env == nullptr || env[0] == '\0' || std::string(env) == "0") return;
    double max_deg = 0.0;
    const int n = std::sscanf(env, "%lf,%lf,%lf", &yaw_acc, &v_on, &max_deg);
    enabled = n == 3 && yaw_acc > 0.0 && v_on > 0.01 && max_deg > 0.0;
    slip_rate_max = max_deg * M_PI / 180.0;
    if (enabled) {
      RCLCPP_INFO(
        rclcpp::get_logger("BicycleMotionModel"),
        "[NERO_MOT_TURN_ONSET] on: a vehicle moving forward at >= %.2f m/s may start turning: "
        "slip-rate sd >= %.2f rad/s2 x "
        "l_r / v, capped at %.1f deg/s",
        v_on, yaw_acc, max_deg);
    } else {
      RCLCPP_WARN(
        rclcpp::get_logger("BicycleMotionModel"),
        "[NERO_MOT_TURN_ONSET] '%s' not understood: off (the car)", env);
    }
  }
};

inline const NeroTurnOnsetParams & neroTurnOnsetParams()
{
  static const NeroTurnOnsetParams params;
  return params;
}

}  // namespace autoware::multi_object_tracker

#endif  // AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__MOTION_MODEL__NERO_TURN_ONSET_HPP_
