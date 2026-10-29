#!/usr/bin/env python3

# Copyright 2026 Nero
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [nero] NERO_VEL_LIMIT_CLEAR_DEFAULT (upstream #11463): the robot's own sequence of limits and clears after an
# Autoware start (the bridge's link watchdog clear at every engage, the dashboard and pull-out limits, obstacle_cruise).
# Only a clear on a table that never held a limit differs: max_vel with the switch, 0 without it (0.44.2). Every other
# step gives the same output in both runs.

import os
import unittest

from ament_index_python import get_package_share_directory
from autoware_internal_planning_msgs.msg import VelocityLimit
from autoware_internal_planning_msgs.msg import VelocityLimitClearCommand
import launch
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
import launch_testing
import pytest
from rcl_interfaces.msg import Parameter
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.msg import ParameterValue
from rcl_interfaces.srv import SetParameters
import rclpy
import rclpy.qos

MAX_VEL = 5.56  # the car's max_vel (autoware_launch common.param.yaml)


@pytest.mark.launch_test
def generate_test_description():
    launch_file = os.path.join(
        get_package_share_directory("autoware_external_velocity_limit_selector"),
        "launch",
        "external_velocity_limit_selector.launch.xml",
    )
    return launch.LaunchDescription(
        [
            IncludeLaunchDescription(AnyLaunchDescriptionSource(launch_file)),
            launch_testing.actions.ReadyToTest(),
        ]
    )


class TestNeroVelocityLimitClearDefault(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.test_node = rclpy.create_node("test_node_nero_vlz")
        qos = rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_api = self.test_node.create_publisher(
            VelocityLimit, "/planning/scenario_planning/max_velocity_default", qos
        )
        self.pub_internal = self.test_node.create_publisher(
            VelocityLimit, "/planning/scenario_planning/max_velocity_candidates", qos
        )
        self.pub_clear = self.test_node.create_publisher(
            VelocityLimitClearCommand, "/planning/scenario_planning/clear_velocity_limit", qos
        )
        self.msg = None
        self.test_node.create_subscription(
            VelocityLimit,
            "/planning/scenario_planning/max_velocity",
            lambda m: setattr(self, "msg", m),
            1,
        )

    def tearDown(self):
        self.test_node.destroy_node()

    def set_max_vel(self, max_vel):
        cli = self.test_node.create_client(
            SetParameters, "/external_velocity_limit_selector/set_parameters"
        )
        while not cli.wait_for_service(timeout_sec=1.0):
            continue
        req = SetParameters.Request()
        req.parameters = [
            Parameter(
                name="max_vel",
                value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=max_vel),
            )
        ]
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(self.test_node, future)
        if future.result() is None:
            raise self.failureException("setting max_vel failed")

    def output(self):
        for _ in range(100):
            if self.msg is not None:
                break
            rclpy.spin_once(self.test_node, timeout_sec=0.1)
        out, self.msg = self.msg, None
        self.assertIsNotNone(out, "no output from the selector")
        return out

    def check(self, step, out, max_velocity, sender):
        self.assertAlmostEqual(out.max_velocity, max_velocity, places=4, msg=step)
        self.assertEqual(out.sender, sender, msg=step)

    def clear(self, sender):
        self.pub_clear.publish(VelocityLimitClearCommand(command=True, sender=sender))
        return self.output()

    def internal(self, sender, kmh, constraints=None):
        m = VelocityLimit(max_velocity=kmh / 3.6, sender=sender)
        if constraints is not None:
            m.use_constraints = True
            (
                m.constraints.min_acceleration,
                m.constraints.max_jerk,
                m.constraints.min_jerk,
            ) = constraints
        self.pub_internal.publish(m)
        return self.output()

    def api(self, kmh, sender="bridge"):
        self.pub_api.publish(VelocityLimit(max_velocity=kmh / 3.6, sender=sender))
        return self.output()

    def test_robot_sequence(self):
        switch = os.environ.get("NERO_VEL_LIMIT_CLEAR_DEFAULT") == "1"
        self.set_max_vel(MAX_VEL)

        # 1 the engage's watchdog clear, before any limit since the start (10-27 #1 / 10-28 #1): the only difference
        first = MAX_VEL if switch else 0.0
        self.check("1 first clear", self.clear("link_watchdog"), first, "")
        self.check("2 second clear, still empty", self.clear("link_watchdog"), first, "")

        # obstacle_cruise's candidate (incl. its 0) and its clear: the table empties by an erase -> max_vel
        self.check("3 obstacle_cruise 0", self.internal("obstacle_cruise", 0.0), 0.0, "obstacle_cruise")
        self.check("4 obstacle_cruise clear", self.clear("obstacle_cruise"), MAX_VEL, "")
        self.check("5 clear on an emptied table", self.clear("link_watchdog"), MAX_VEL, "")

        # the dashboard's limit (API), the watchdog's own cap and stop, each withdrawn by its clear
        kmh15 = 15.0 / 3.6
        self.check("6 dashboard 15", self.api(15.0), kmh15, "api")
        out = self.internal("link_watchdog", 5.0, (-1.0, 1.0, -1.0))
        self.check("7 watchdog cap 5", out, 5.0 / 3.6, "link_watchdog")
        self.assertTrue(out.use_constraints, "7 watchdog cap 5: constraints")
        self.check("8 watchdog clear", self.clear("link_watchdog"), kmh15, "api")
        self.check("9 watchdog stop", self.internal("link_watchdog", 0.0), 0.0, "link_watchdog")
        self.check("10 watchdog clear", self.clear("link_watchdog"), kmh15, "api")

        # the dashboard's 1.5 km/h, the pull-out creep zone's 1.0 km/h and back (API), obstacle_cruise under it
        self.check("11 dashboard 1.5", self.api(1.5), 1.5 / 3.6, "api")
        self.check("12 creep zone on", self.api(1.0, "bridge_pullout_leg1"), 1.0 / 3.6, "api")
        self.check("13 creep zone off", self.api(1.5), 1.5 / 3.6, "api")
        self.check("14 obstacle_cruise 0", self.internal("obstacle_cruise", 0.0), 0.0, "obstacle_cruise")
        self.check("15 obstacle_cruise clear", self.clear("obstacle_cruise"), 1.5 / 3.6, "api")
        self.check("16 leg-2 watchdog clear", self.clear("link_watchdog"), 1.5 / 3.6, "api")

        # a dashboard value above max_vel: max_vel
        self.check("17 dashboard 25", self.api(25.0), MAX_VEL, "")
        self.check("18 watchdog clear", self.clear("link_watchdog"), MAX_VEL, "")


@launch_testing.post_shutdown_test()
class TestProcessOutput(unittest.TestCase):
    def test_exit_code(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
