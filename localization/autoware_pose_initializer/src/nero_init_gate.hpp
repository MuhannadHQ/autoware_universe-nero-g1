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

#ifndef NERO_INIT_GATE_HPP_
#define NERO_INIT_GATE_HPP_

#include <cstdio>
#include <optional>
#include <string>

namespace autoware::pose_initializer
{
// NERO_POSE_INIT_GATE: whether an AUTO init may replace the running localization.
struct NeroGateParam
{
  double healthy_nvtl{2.3};  // a running localization at or above this NDT score is healthy
  double margin{0.1};        // a healthy one is kept unless the new pose scores at least this close to it
};

struct NeroGateInput
{
  bool reliable{true};                 // NDT's own rating of the align (score > its converged threshold)
  std::optional<double> align_score;   // the align's best particle NVTL
  bool has_previous{false};            // INITIALIZED with a recent EKF pose to hand back
  std::optional<double> current_nvtl;  // median NDT score of the running localization
  std::optional<double> gnss_sigma;    // gnss_poser's sigma of a GNSS seed [m]
};

inline std::string nero_fmt(const char * f, const double v)
{
  char buf[64];
  std::snprintf(buf, sizeof(buf), f, v);
  return buf;
}

// The refusal message for the operator, or "" to accept.
inline std::string nero_gate_refusal(const NeroGateParam & p, const NeroGateInput & in)
{
  const bool healthy =
    in.has_previous && in.current_nvtl && *in.current_nvtl >= p.healthy_nvtl;
  const std::string score = in.align_score ? nero_fmt("%.2f", *in.align_score) : "unknown";
  const std::string now = in.current_nvtl ? nero_fmt("%.2f", *in.current_nvtl) : "none";
  std::string why;
  if (!in.reliable) {
    why = "Init refused: NDT score " + score + " is below " + nero_fmt("%.2f", p.healthy_nvtl) +
          " (unreliable).";
  } else if (healthy && in.align_score && *in.align_score < *in.current_nvtl - p.margin) {
    why = "Init refused: the new pose scores NDT " + score + ", the current localization " + now + ".";
  }
  if (why.empty()) return why;
  if (in.has_previous) {
    why += " Kept the current localization (NDT " + now + ").";
  } else {
    why += " Not localized: retry, or set the pose with its heading.";
  }
  if (in.gnss_sigma && *in.gnss_sigma > 1.0) {
    why += " GNSS fix sigma " + nero_fmt("%.0f", *in.gnss_sigma) + " m (no RTK).";
  }
  return why;
}
}  // namespace autoware::pose_initializer

#endif  // NERO_INIT_GATE_HPP_
