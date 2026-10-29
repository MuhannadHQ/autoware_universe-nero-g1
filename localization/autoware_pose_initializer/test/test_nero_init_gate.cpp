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

#include "../src/nero_init_gate.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>

using autoware::pose_initializer::nero_gate_refusal;
using autoware::pose_initializer::NeroGateInput;
using autoware::pose_initializer::NeroGateParam;
using ::testing::HasSubstr;

namespace
{
NeroGateInput healthy(const double align, const bool reliable = true)
{
  NeroGateInput in;
  in.reliable = reliable;
  in.align_score = align;
  in.has_previous = true;
  in.current_nvtl = 2.78;
  return in;
}
}  // namespace

// 10-27 20:38:11 (140436): healthy 2.78, a GNSS align NDT rated unstable (1.98) -> refused, localization kept
TEST(NeroInitGate, UnreliableAlignIsRefusedAndTheLocalizationKept)
{
  auto in = healthy(1.98, false);
  in.gnss_sigma = 16.2;
  const auto why = nero_gate_refusal(NeroGateParam{}, in);
  EXPECT_THAT(why, HasSubstr("NDT score 1.98 is below 2.30"));
  EXPECT_THAT(why, HasSubstr("Kept the current localization (NDT 2.78)"));
  EXPECT_THAT(why, HasSubstr("GNSS fix sigma 16 m (no RTK)"));
}

// w31 stand: healthy 2.70, an align 12.6 m along the road rated reliable (2.40) -> refused
TEST(NeroInitGate, ReliableButWorseThanAHealthyLocalizationIsRefused)
{
  auto in = healthy(2.40);
  in.current_nvtl = 2.70;
  const auto why = nero_gate_refusal(NeroGateParam{}, in);
  EXPECT_THAT(why, HasSubstr("the new pose scores NDT 2.40, the current localization 2.70"));
}

TEST(NeroInitGate, AsGoodOrBetterIsAccepted)
{
  EXPECT_EQ(nero_gate_refusal(NeroGateParam{}, healthy(2.70)), "");  // within the margin of 2.78
  EXPECT_EQ(nero_gate_refusal(NeroGateParam{}, healthy(2.93)), "");
}

// a running localization below the healthy level (140436 C 2.36 / 10-28 #1 2.1-2.2): only NDT's own rating counts
TEST(NeroInitGate, NotHealthyOnlyNeedsAReliableAlign)
{
  auto in = healthy(2.38);
  in.current_nvtl = 2.20;
  EXPECT_EQ(nero_gate_refusal(NeroGateParam{}, in), "");
  in.reliable = false;
  in.align_score = 2.26;
  EXPECT_THAT(nero_gate_refusal(NeroGateParam{}, in), HasSubstr("Kept the current localization (NDT 2.20)"));
}

// boot: nothing to keep
TEST(NeroInitGate, BootUnreliableIsRefusedWithoutAPreviousPose)
{
  NeroGateInput in;
  in.reliable = false;
  in.align_score = 2.05;
  const auto why = nero_gate_refusal(NeroGateParam{}, in);
  EXPECT_THAT(why, HasSubstr("Not localized"));
  in.reliable = true;
  in.align_score = 2.80;
  EXPECT_EQ(nero_gate_refusal(NeroGateParam{}, in), "");
}

// the align diagnostic did not arrive: NDT's rating still decides, the comparison is skipped
TEST(NeroInitGate, UnknownScoreFallsBackToTheRating)
{
  auto in = healthy(0.0);
  in.align_score.reset();
  EXPECT_EQ(nero_gate_refusal(NeroGateParam{}, in), "");
  in.reliable = false;
  EXPECT_THAT(nero_gate_refusal(NeroGateParam{}, in), HasSubstr("NDT score unknown"));
}
