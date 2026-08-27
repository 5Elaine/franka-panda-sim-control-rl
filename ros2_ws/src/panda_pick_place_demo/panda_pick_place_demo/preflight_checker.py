#!/usr/bin/env python3

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time

from control_msgs.action import GripperCommand
from controller_manager_msgs.srv import ListControllers
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.srv import (
    ApplyPlanningScene,
    GetCartesianPath,
    GetMotionPlan,
    GetPlanningScene,
)
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


class PreflightChecker(Node):
    REQUIRED_CONTROLLERS = {
        "joint_state_broadcaster",
        "panda_arm_controller",
        "panda_hand_controller",
    }

    REQUIRED_JOINTS = {
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
    }

    def __init__(self) -> None:
        super().__init__("panda_pick_place_preflight_checker")

        self.results: list[CheckResult] = []

        self.clock_received = False
        self.clock_value_ns = 0

        self.joint_state_received = False
        self.joint_names: set[str] = set()

        clock_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(
            Clock,
            "/clock",
            self.clock_callback,
            clock_qos,
        )

        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            qos_profile_sensor_data,
        )

        self.controller_client = self.create_client(
            ListControllers,
            "/controller_manager/list_controllers",
        )

        self.service_clients = {
            "/plan_kinematic_path": self.create_client(
                GetMotionPlan,
                "/plan_kinematic_path",
            ),
            "/compute_cartesian_path": self.create_client(
                GetCartesianPath,
                "/compute_cartesian_path",
            ),
            "/apply_planning_scene": self.create_client(
                ApplyPlanningScene,
                "/apply_planning_scene",
            ),
            "/get_planning_scene": self.create_client(
                GetPlanningScene,
                "/get_planning_scene",
            ),
        }

        self.execute_client = ActionClient(
            self,
            ExecuteTrajectory,
            "/execute_trajectory",
        )

        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            "/panda_hand_controller/gripper_cmd",
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

    def clock_callback(self, message: Clock) -> None:
        self.clock_received = True
        self.clock_value_ns = (
            message.clock.sec * 1_000_000_000
            + message.clock.nanosec
        )

    def joint_state_callback(
        self,
        message: JointState,
    ) -> None:
        self.joint_state_received = True
        self.joint_names = set(message.name)

    def add_result(
        self,
        name: str,
        passed: bool,
        detail: str,
    ) -> None:
        self.results.append(
            CheckResult(name, passed, detail)
        )

    def wait_for_runtime_messages(
        self,
        timeout_sec: float = 5.0,
    ) -> None:
        deadline = time.monotonic() + timeout_sec

        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

            if (
                self.clock_received
                and self.joint_state_received
            ):
                break

        clock_publishers = len(
            self.get_publishers_info_by_topic("/clock")
        )

        self.add_result(
            "/clock",
            (
                clock_publishers > 0
                and self.clock_received
            ),
            (
                f"publishers={clock_publishers}, "
                f"time={self.clock_value_ns / 1e9:.3f} s"
                if self.clock_received
                else f"publishers={clock_publishers}, no message"
            ),
        )

        missing_joints = (
            self.REQUIRED_JOINTS - self.joint_names
        )

        self.add_result(
            "/joint_states",
            (
                self.joint_state_received
                and not missing_joints
            ),
            (
                f"received {len(self.joint_names)} joints"
                if not missing_joints
                else "missing: "
                + ", ".join(sorted(missing_joints))
            ),
        )

    def check_move_group_node(
        self,
        timeout_sec: float = 8.0,
    ) -> None:
        """
        Wait for ROS graph discovery.

        If the node name is still missing but MoveIt's planning
        service is available, accept the functional evidence.
        """
        deadline = time.monotonic() + timeout_sec
        exists = False

        while time.monotonic() < deadline:
            rclpy.spin_once(
                self,
                timeout_sec=0.1,
            )

            full_names = set()

            for name, namespace in (
                self.get_node_names_and_namespaces()
            ):
                if namespace == "/":
                    full_name = f"/{name}"
                else:
                    full_name = (
                        f"{namespace.rstrip('/')}/{name}"
                    )

                full_names.add(full_name)

            if "/move_group" in full_names:
                exists = True
                break

        if exists:
            self.add_result(
                "/move_group node",
                True,
                "node discovered",
            )
            return

        planning_service_available = (
            self.service_clients[
                "/plan_kinematic_path"
            ].wait_for_service(timeout_sec=2.0)
        )

        self.add_result(
            "/move_group node",
            planning_service_available,
            (
                "node name not discovered, but MoveIt "
                "planning service is available"
                if planning_service_available
                else
                "node and planning service unavailable"
            ),
        )

    def check_controllers(self) -> None:
        available = self.controller_client.wait_for_service(
            timeout_sec=5.0
        )

        if not available:
            self.add_result(
                "controllers",
                False,
                "list_controllers service unavailable",
            )
            return

        future = self.controller_client.call_async(
            ListControllers.Request()
        )

        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=5.0,
        )

        if (
            not future.done()
            or future.result() is None
        ):
            self.add_result(
                "controllers",
                False,
                "list_controllers request timed out",
            )
            return

        states = {
            controller.name: controller.state
            for controller
            in future.result().controller
        }

        failed = {
            name: states.get(name, "missing")
            for name in self.REQUIRED_CONTROLLERS
            if states.get(name) != "active"
        }

        self.add_result(
            "controllers",
            not failed,
            (
                "all required controllers active"
                if not failed
                else ", ".join(
                    f"{name}={state}"
                    for name, state in sorted(
                        failed.items()
                    )
                )
            ),
        )

    def check_services(self) -> None:
        for name, client in self.service_clients.items():
            available = client.wait_for_service(
                timeout_sec=3.0
            )

            self.add_result(
                name,
                available,
                (
                    "service available"
                    if available
                    else "service unavailable"
                ),
            )

    def check_actions(self) -> None:
        execute_available = (
            self.execute_client.wait_for_server(
                timeout_sec=5.0
            )
        )

        self.add_result(
            "/execute_trajectory",
            execute_available,
            (
                "action server available"
                if execute_available
                else "action server unavailable"
            ),
        )

        gripper_available = (
            self.gripper_client.wait_for_server(
                timeout_sec=5.0
            )
        )

        self.add_result(
            "/panda_hand_controller/gripper_cmd",
            gripper_available,
            (
                "action server available"
                if gripper_available
                else "action server unavailable"
            ),
        )

    def check_tcp_tf(
        self,
        timeout_sec: float = 5.0,
    ) -> None:
        deadline = time.monotonic() + timeout_sec
        transform = None
        last_error = None

        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

            try:
                transform = self.tf_buffer.lookup_transform(
                    "world",
                    "panda_hand_tcp",
                    Time(),
                    timeout=Duration(seconds=0.2),
                )
                break
            except TransformException as error:
                last_error = error

        if transform is None:
            self.add_result(
                "TF world -> panda_hand_tcp",
                False,
                str(last_error),
            )
            return

        translation = transform.transform.translation

        self.add_result(
            "TF world -> panda_hand_tcp",
            True,
            (
                f"x={translation.x:.3f}, "
                f"y={translation.y:.3f}, "
                f"z={translation.z:.3f}"
            ),
        )

    def run_checks(self) -> bool:
        self.get_logger().info(
            "Starting Panda pick-and-place preflight checks..."
        )

        self.wait_for_runtime_messages()
        self.check_move_group_node()
        self.check_controllers()
        self.check_services()
        self.check_actions()
        self.check_tcp_tf()

        print()
        print("=== PANDA PICK-AND-PLACE PREFLIGHT ===")

        for result in self.results:
            status = "PASS" if result.passed else "FAIL"

            print(
                f"[{status}] {result.name}: "
                f"{result.detail}"
            )

        passed_count = sum(
            result.passed for result in self.results
        )
        total_count = len(self.results)

        print()
        print(
            f"Summary: {passed_count}/{total_count} "
            "checks passed."
        )

        all_passed = passed_count == total_count

        if all_passed:
            print(
                "PREFLIGHT PASSED: "
                "the automated task may start."
            )
        else:
            print(
                "PREFLIGHT FAILED: "
                "do not start the automated task."
            )

        return all_passed


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PreflightChecker()

    try:
        success = node.run_checks()
    except Exception as error:
        node.get_logger().error(
            f"Unexpected preflight error: {error}"
        )
        success = False
    finally:
        node.destroy_node()
        rclpy.shutdown()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
