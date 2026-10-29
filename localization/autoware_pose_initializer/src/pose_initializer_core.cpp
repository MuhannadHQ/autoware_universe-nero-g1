// Copyright 2022 The Autoware Contributors
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

#include "pose_initializer_core.hpp"

#include "copy_vector_to_array.hpp"
#include "ekf_localization_trigger_module.hpp"
#include "gnss_module.hpp"
#include "localization_module.hpp"
#include "nero_init_gate.hpp"
#include "ndt_localization_trigger_module.hpp"
#include "pose_error_check_module.hpp"
#include "stop_check_module.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <thread>
#include <vector>

namespace autoware::pose_initializer
{
PoseInitializer::PoseInitializer(const rclcpp::NodeOptions & options)
: rclcpp::Node("pose_initializer", options)
{
  const auto node = autoware::component_interface_utils::NodeAdaptor(this);
  group_srv_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  node.init_pub(pub_state_);
  node.init_srv(srv_initialize_, this, &PoseInitializer::on_initialize, group_srv_);
  pub_reset_ = create_publisher<PoseWithCovarianceStamped>("pose_reset", 1);

  output_pose_covariance_ = get_covariance_parameter(this, "output_pose_covariance");
  gnss_particle_covariance_ = get_covariance_parameter(this, "gnss_particle_covariance");

  diagnostics_pose_reliable_ =
    std::make_unique<autoware_utils::DiagnosticsInterface>(this, "pose_initializer_status");

  if (declare_parameter<bool>("ekf_enabled")) {
    ekf_localization_trigger_ = std::make_unique<EkfLocalizationTriggerModule>(this);
  }
  if (declare_parameter<bool>("gnss_enabled")) {
    gnss_ = std::make_unique<GnssModule>(this);
  }
  if (declare_parameter<bool>("yabloc_enabled")) {
    yabloc_ = std::make_unique<LocalizationModule>(this, "yabloc_align");
  }
  if (declare_parameter<bool>("ndt_enabled")) {
    ndt_ = std::make_unique<LocalizationModule>(this, "ndt_align");
    ndt_localization_trigger_ = std::make_unique<NdtLocalizationTriggerModule>(this);
  }
  if (declare_parameter<bool>("stop_check_enabled")) {
    // Add 1.0 sec margin for twist buffer.
    stop_check_duration_ = declare_parameter<double>("stop_check_duration");
    stop_check_ = std::make_unique<StopCheckModule>(this, stop_check_duration_ + 1.0);
  }
  if (declare_parameter<bool>("pose_error_check_enabled")) {
    pose_error_check_ = std::make_unique<PoseErrorCheckModule>(this);
  }
  logger_configure_ = std::make_unique<autoware_utils::LoggerLevelConfigure>(this);

  if (const char * env = std::getenv("NERO_POSE_INIT_GATE"); env && *env) {
    double a = 0.0, b = 0.0, c = 0.0;
    const int n = std::sscanf(env, "%lf,%lf,%lf", &a, &b, &c);
    if (n >= 2 && a > 0.0 && b >= 0.0) {
      nero_gate_.enabled = true;
      nero_gate_.healthy_nvtl = a;
      nero_gate_.margin = b;
      if (n >= 3 && c > 0.0) nero_gate_.window_s = c;
    }
    RCLCPP_INFO(
      get_logger(), "[NERO_POSE_INIT_GATE] %s (healthy NVTL %.2f, margin %.2f, score age <= %.1f s)",
      nero_gate_.enabled ? "on" : "malformed, off", nero_gate_.healthy_nvtl, nero_gate_.margin,
      nero_gate_.window_s);
  }
  if (nero_gate_.enabled) {
    nero_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions opt;
    opt.callback_group = nero_group_;
    // the EKF's own output (what the robot uses as its pose) and NDT's public diagnostics
    nero_sub_pose_ = create_subscription<PoseWithCovarianceStamped>(
      "/localization/pose_with_covariance", rclcpp::QoS(1),
      [this](const PoseWithCovarianceStamped::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(nero_mutex_);
        nero_last_pose_ = *msg;
      },
      opt);
    nero_sub_diag_ = create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
      "/diagnostics", rclcpp::QoS(100),
      [this](const diagnostic_msgs::msg::DiagnosticArray::ConstSharedPtr msg) { nero_on_diag(msg); },
      opt);
  }

  change_state(State::Message::UNINITIALIZED);

  if (declare_parameter<bool>("user_defined_initial_pose.enable")) {
    const auto initial_pose_array =
      declare_parameter<std::vector<double>>("user_defined_initial_pose.pose");
    if (initial_pose_array.size() != 7) {
      throw std::invalid_argument(
        "Could not set user defined initial pose. The size of initial_pose is " +
        std::to_string(initial_pose_array.size()) + ". It must be 7.");
    }
    if (
      std::abs(initial_pose_array[3]) < 1e-6 && std::abs(initial_pose_array[4]) < 1e-6 &&
      std::abs(initial_pose_array[5]) < 1e-6 && std::abs(initial_pose_array[6]) < 1e-6) {
      throw std::invalid_argument("Input quaternion is invalid. All elements are close to zero.");
    }

    geometry_msgs::msg::Pose initial_pose;
    initial_pose.position.x = initial_pose_array[0];
    initial_pose.position.y = initial_pose_array[1];
    initial_pose.position.z = initial_pose_array[2];
    initial_pose.orientation.x = initial_pose_array[3];
    initial_pose.orientation.y = initial_pose_array[4];
    initial_pose.orientation.z = initial_pose_array[5];
    initial_pose.orientation.w = initial_pose_array[6];

    set_user_defined_initial_pose(initial_pose, true);
  }
}

void PoseInitializer::change_state(State::Message::_state_type state)
{
  state_.stamp = now();
  state_.state = state;
  pub_state_->publish(state_);
}

// To execute in the constructor, you need to call ros spin.
// Conversely, ros spin should not be called elsewhere
void PoseInitializer::change_node_trigger(bool flag, bool need_spin)
{
  try {
    if (ekf_localization_trigger_) {
      ekf_localization_trigger_->wait_for_service();
      ekf_localization_trigger_->send_request(flag, need_spin);
    }
    if (ndt_localization_trigger_) {
      ndt_localization_trigger_->wait_for_service();
      ndt_localization_trigger_->send_request(flag, need_spin);
    }
  } catch (const ServiceException & error) {
    throw;
  }
}

void PoseInitializer::set_user_defined_initial_pose(
  const geometry_msgs::msg::Pose initial_pose, bool need_spin)
{
  try {
    change_state(State::Message::INITIALIZING);
    change_node_trigger(false, need_spin);

    PoseWithCovarianceStamped pose;
    pose.header.frame_id = "map";
    pose.header.stamp = now();
    pose.pose.pose = initial_pose;
    pose.pose.covariance = output_pose_covariance_;
    pub_reset_->publish(pose);

    change_node_trigger(true, need_spin);
    change_state(State::Message::INITIALIZED);

    RCLCPP_INFO(get_logger(), "Set user defined initial pose");
  } catch (const ServiceException & error) {
    change_state(State::Message::UNINITIALIZED);
    RCLCPP_WARN(get_logger(), "Could not set user defined initial pose");
  }
}

void PoseInitializer::on_initialize(
  const Initialize::Service::Request::SharedPtr req,
  const Initialize::Service::Response::SharedPtr res)
{
  // NOTE: This function is not executed during initialization because mutually exclusive.
  if (stop_check_ && !stop_check_->isVehicleStopped(stop_check_duration_)) {
    throw ServiceException(
      Initialize::Service::Response::ERROR_UNSAFE, "The vehicle is not stopped.");
  }
  try {
    if (req->method == Initialize::Service::Request::AUTO) {
      // NERO_POSE_INIT_GATE: the running localization before it is switched off for the align
      const NeroSnapshot nero_before = nero_gate_.enabled ? nero_snapshot() : NeroSnapshot{};
      change_state(State::Message::INITIALIZING);
      change_node_trigger(false, false);

      auto pose =
        req->pose_with_covariance.empty() ? get_gnss_pose() : req->pose_with_covariance.front();
      std::optional<double> nero_gnss_sigma;
      if (nero_gate_.enabled && req->pose_with_covariance.empty() && gnss_) {
        const auto raw = gnss_->get_pose();  // gnss_poser's own covariance (get_gnss_pose overwrites it)
        nero_gnss_sigma = std::sqrt(std::max(raw.pose.covariance[0], raw.pose.covariance[7]));
      }
      const int64_t nero_align_called_ns = now().nanoseconds();
      bool reliable = true;
      if (ndt_) {
        std::tie(pose, reliable) = ndt_->align_pose(pose);
      } else if (yabloc_) {
        // If both the NDT and YabLoc initializer are enabled, prioritize NDT as it offers more
        // accuracy pose.
        std::tie(pose, reliable) = yabloc_->align_pose(pose);
      }

      diagnostics_pose_reliable_->clear();

      // check pose error between gnss pose and initial pose result
      if (pose_error_check_ && gnss_) {
        const auto latest_gnss_pose = get_gnss_pose();

        double gnss_error_2d;
        const bool is_gnss_pose_error_small = pose_error_check_->check_pose_error(
          latest_gnss_pose.pose.pose, pose.pose.pose, gnss_error_2d);

        diagnostics_pose_reliable_->add_key_value("gnss_pose_error_2d", gnss_error_2d);
        diagnostics_pose_reliable_->add_key_value(
          "is_gnss_pose_error_small", is_gnss_pose_error_small);
        if (!is_gnss_pose_error_small) {
          std::stringstream message;
          message << " Large error between Initial Pose and GNSS Pose.";
          diagnostics_pose_reliable_->update_level_and_message(
            diagnostic_msgs::msg::DiagnosticStatus::WARN, message.str());
        }
      }
      // check initial pose result and publish diagnostics
      diagnostics_pose_reliable_->add_key_value("is_initial_pose_reliable", reliable);
      if (!reliable) {
        std::stringstream message;
        message << "Initial Pose Estimation is Unstable.";
        diagnostics_pose_reliable_->update_level_and_message(
          diagnostic_msgs::msg::DiagnosticStatus::ERROR, message.str());
      }

      // NERO_POSE_INIT_GATE: do not replace the localization with a pose NDT rates unreliable, nor a healthy
      // localization with a pose that fits the lidar worse than it does
      std::string nero_refusal;
      if (nero_gate_.enabled && ndt_) {
        NeroGateInput in;
        in.reliable = reliable;
        in.align_score = nero_wait_align_score(nero_align_called_ns);
        in.has_previous = nero_before.initialized && nero_before.pose.has_value();
        in.current_nvtl = nero_before.nvtl;
        in.gnss_sigma = nero_gnss_sigma;
        nero_refusal = nero_gate_refusal({nero_gate_.healthy_nvtl, nero_gate_.margin}, in);
        const std::string score = in.align_score ? nero_fmt("%.2f", *in.align_score) : "unknown";
        const std::string current = in.current_nvtl ? nero_fmt("%.2f", *in.current_nvtl) : "none";
        diagnostics_pose_reliable_->add_key_value("nero_align_score", score);
        diagnostics_pose_reliable_->add_key_value("nero_current_nvtl", current);
        diagnostics_pose_reliable_->add_key_value(
          "nero_init_gate", std::string(nero_refusal.empty() ? "accepted" : "refused"));
        if (!nero_refusal.empty()) {
          diagnostics_pose_reliable_->update_level_and_message(
            diagnostic_msgs::msg::DiagnosticStatus::WARN, nero_refusal);
        }
        const auto & p = pose.pose.pose.position;
        RCLCPP_WARN(
          get_logger(), "[NERO_POSE_INIT_GATE] %s: align NDT %s at (%.2f, %.2f), current NDT %s (n %zu)%s%s",
          nero_refusal.empty() ? "accepted" : "refused", score.c_str(), p.x, p.y, current.c_str(),
          nero_before.nvtl_n, nero_refusal.empty() ? "" : ": ", nero_refusal.c_str());
      }

      diagnostics_pose_reliable_->publish(this->now());

      if (!nero_refusal.empty()) {
        if (nero_before.initialized && nero_before.pose) {
          // hand the running localization back to the EKF (a deactivation cleared its initial pose)
          auto previous = *nero_before.pose;
          previous.header.stamp = now();
          pub_reset_->publish(previous);
          change_node_trigger(true, false);
          change_state(State::Message::INITIALIZED);
        } else {
          change_state(State::Message::UNINITIALIZED);  // nothing to keep: as a failed boot init
        }
        res->status.success = false;
        res->status.code = Initialize::Service::Response::ERROR_ESTIMATION;
        res->status.message = nero_refusal;
        return;
      }

      pose.pose.covariance = output_pose_covariance_;
      pub_reset_->publish(pose);

      change_node_trigger(true, false);
      res->status.success = true;
      change_state(State::Message::INITIALIZED);

    } else if (req->method == Initialize::Service::Request::DIRECT) {
      if (req->pose_with_covariance.empty()) {
        std::stringstream message;
        message << "No input pose_with_covariance. If you want to use DIRECT method, please input "
                   "pose_with_covariance.";
        RCLCPP_ERROR_STREAM(get_logger(), message.str());
        throw ServiceException(
          autoware_common_msgs::msg::ResponseStatus::PARAMETER_ERROR, message.str());
      }
      auto pose = req->pose_with_covariance.front().pose.pose;
      set_user_defined_initial_pose(pose, false);
      res->status.success = true;

    } else {
      std::stringstream message;
      message << "Unknown method type (=" << std::to_string(req->method) << ")";
      RCLCPP_ERROR_STREAM(get_logger(), message.str());
      throw ServiceException(
        autoware_common_msgs::msg::ResponseStatus::PARAMETER_ERROR, message.str());
    }
  } catch (const ServiceException & error) {
    autoware_adapi_v1_msgs::msg::ResponseStatus respose_status;
    respose_status = error.status();
    res->status.success = respose_status.success;
    res->status.code = respose_status.code;
    res->status.message = respose_status.message;
    change_state(State::Message::UNINITIALIZED);
  }
}

void PoseInitializer::nero_on_diag(const diagnostic_msgs::msg::DiagnosticArray::ConstSharedPtr msg)
{
  for (const auto & st : msg->status) {
    const bool scan = st.name == "ndt_scan_matcher: scan_matching_status";
    const bool align = st.name == "ndt_scan_matcher: ndt_align_service_status";
    if (!scan && !align) continue;
    std::optional<double> nvtl;
    std::optional<double> best;
    std::optional<int64_t> called;
    bool activated = true;
    for (const auto & kv : st.values) {
      try {
        if (kv.key == "nearest_voxel_transformation_likelihood") nvtl = std::stod(kv.value);
        if (kv.key == "best_particle_score") best = std::stod(kv.value);
        if (kv.key == "service_call_time_stamp") called = std::stoll(kv.value);
        if (kv.key == "is_activated") activated = (kv.value == "True");
      } catch (const std::exception &) {
      }
    }
    std::lock_guard<std::mutex> lock(nero_mutex_);
    if (scan && nvtl && activated) {
      nero_nvtl_.emplace_back(rclcpp::Time(msg->header.stamp, get_clock()->get_clock_type()), *nvtl);
      while (nero_nvtl_.size() > 200) nero_nvtl_.pop_front();
    }
    if (align && best && called) nero_align_score_ = std::make_pair(*called, *best);
  }
}

PoseInitializer::NeroSnapshot PoseInitializer::nero_snapshot()
{
  NeroSnapshot snap;
  snap.initialized = state_.state == State::Message::INITIALIZED;
  const auto t = now();
  std::lock_guard<std::mutex> lock(nero_mutex_);
  if (nero_last_pose_ && (t - rclcpp::Time(nero_last_pose_->header.stamp, t.get_clock_type())).seconds() < 1.0) {
    snap.pose = nero_last_pose_;
  }
  // the last 10 tracking scores (1 s of scans), none older than window_s: NDT is off during an init, so an init
  // right after another one still sees the scores from before that one
  std::vector<double> v;
  for (auto it = nero_nvtl_.rbegin(); it != nero_nvtl_.rend() && v.size() < 10; ++it) {
    if ((t - it->first).seconds() <= nero_gate_.window_s) v.push_back(it->second);
  }
  snap.nvtl_n = v.size();
  if (v.size() >= 3) {
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    snap.nvtl = v[v.size() / 2];
  }
  return snap;
}

std::optional<double> PoseInitializer::nero_wait_align_score(int64_t called_ns)
{
  // NDT publishes its align diagnostic before it answers; the two travel on different topics
  for (int i = 0; i < 50; ++i) {
    {
      std::lock_guard<std::mutex> lock(nero_mutex_);
      if (nero_align_score_ && nero_align_score_->first >= called_ns) return nero_align_score_->second;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return std::nullopt;
}

geometry_msgs::msg::PoseWithCovarianceStamped PoseInitializer::get_gnss_pose()
{
  if (gnss_) {
    PoseWithCovarianceStamped pose = gnss_->get_pose();
    pose.pose.covariance = gnss_particle_covariance_;
    return pose;
  }
  throw ServiceException(
    Initialize::Service::Response::ERROR_GNSS_SUPPORT, "GNSS is not supported.");
}
}  // namespace autoware::pose_initializer

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::pose_initializer::PoseInitializer)
