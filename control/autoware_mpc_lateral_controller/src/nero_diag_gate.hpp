// Copyright 2026 Nero
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

// [nero 2026-11-11 DIAGHZ] Rate limit for a /diagnostics status, switch NERO_DIAG_HZ.
//
// Why: with the full stack on nero-jetson (live replay 2026-11-11) /diagnostics carried ~600 arrays per second:
// gyro_odometer 221 (one per IMU and per twist message), the EKF 173, the controller 75 (two force_update per 30 Hz
// cycle), NDT 67 (one per 50 Hz EKF pose). Every reader pays for every array, the Python readers most.
//
// What: a status goes out at most NERO_DIAG_HZ times per second, and AT ONCE whenever its level or its message
// changes. A WARN or ERROR is never held back, and neither is its clearing. Unset, empty or not a number = as before
// (every call publishes). Below 2 Hz is refused (as before), so the diagnostic graph's ~1 s timeouts stay covered.
//
// The same file is in autoware_gyro_odometer, autoware_ndt_scan_matcher, autoware_mpc_lateral_controller and
// autoware_pid_longitudinal_controller (one copy per package, so no shared header changes and no wide rebuild).

#ifndef NERO_DIAG_GATE_HPP_
#define NERO_DIAG_GATE_HPP_

#include "autoware_utils/ros/diagnostics_interface.hpp"

#include <rclcpp/rclcpp.hpp>

#include <diagnostic_msgs/msg/diagnostic_status.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace nero_diag
{
// NERO_DIAG_HZ, read once per library. 0 = no limit (as before).
inline double rate_hz()
{
  static const double hz = []() {
    const char * e = std::getenv("NERO_DIAG_HZ");
    if (!e || !*e) {
      return 0.0;
    }
    char * end = nullptr;
    const double v = std::strtod(e, &end);
    if (end == e || !(v >= 2.0)) {
      fprintf(stderr, "[NERO_DIAG_HZ] '%s' refused (needs >= 2): every status is sent, as before\n", e);
      return 0.0;
    }
    fprintf(
      stderr, "[NERO_DIAG_HZ] on: each status at most %.1f per second, at once when its level or message changes\n",
      v);
    return v;
  }();
  return hz;
}

// One per status. pass() = send this one.
class Gate
{
public:
  bool pass(const int key, const std::string & message)
  {
    const double hz = rate_hz();
    if (hz <= 0.0) {
      return true;
    }
    const auto now = std::chrono::steady_clock::now();
    if (
      sent_ && key == key_ && message == message_ &&
      now - last_ < std::chrono::duration<double>(1.0 / hz)) {
      return false;
    }
    sent_ = true;
    key_ = key;
    message_ = message;
    last_ = now;
    return true;
  }

private:
  bool sent_{false};
  int key_{0};
  std::string message_;
  std::chrono::steady_clock::time_point last_{};
};

// autoware_utils::DiagnosticsInterface with the gate in publish(). Its methods are not virtual, so the member that
// holds it must have THIS type (calls through a base pointer would skip the level tracking).
class RateLimitedDiagnostics : public autoware_utils::DiagnosticsInterface
{
public:
  RateLimitedDiagnostics(rclcpp::Node * node, const std::string & diagnostic_name)
  : autoware_utils::DiagnosticsInterface(node, diagnostic_name)
  {
  }

  void clear()
  {
    autoware_utils::DiagnosticsInterface::clear();
    level_ = diagnostic_msgs::msg::DiagnosticStatus::OK;
    message_.clear();
  }

  // Same rule as the base: the level is the worst one given, the messages of non-OK calls are joined with "; ".
  void update_level_and_message(const int8_t level, const std::string & message)
  {
    autoware_utils::DiagnosticsInterface::update_level_and_message(level, message);
    if (level > diagnostic_msgs::msg::DiagnosticStatus::OK) {
      if (!message_.empty()) {
        message_ += "; ";
      }
      message_ += message;
    }
    level_ = std::max(level_, level);
  }

  void publish(const rclcpp::Time & publish_time_stamp)
  {
    if (gate_.pass(level_, message_)) {
      autoware_utils::DiagnosticsInterface::publish(publish_time_stamp);
    }
  }

private:
  int8_t level_{diagnostic_msgs::msg::DiagnosticStatus::OK};
  std::string message_;
  Gate gate_;
};
}  // namespace nero_diag

#endif  // NERO_DIAG_GATE_HPP_
