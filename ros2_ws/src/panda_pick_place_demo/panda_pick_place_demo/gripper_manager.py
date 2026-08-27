#!/usr/bin/env python3

from __future__ import annotations

import time

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState


class GripperManager:
    """Send gripper goals and verify their final results."""

    ACTION_NAME = "/panda_hand_controller/gripper_cmd"

    def __init__(self, node: Node) -> None:
        self.node = node

        self.client = ActionClient(
            node,
            GripperCommand,
            self.ACTION_NAME,
        )

        self.finger_position: float | None = None
        self.finger_velocity: float | None = None

        self.joint_state_subscription = (
            node.create_subscription(
                JointState,
                "/joint_states",
                self.joint_state_callback,
                qos_profile_sensor_data,
            )
        )

    def joint_state_callback(
        self,
        message: JointState,
    ) -> None:
        try:
            index = message.name.index(
                "panda_finger_joint1"
            )
        except ValueError:
            return

        if index < len(message.position):
            self.finger_position = message.position[index]

        if index < len(message.velocity):
            self.finger_velocity = message.velocity[index]

    def wait_for_grasp_closure(
        self,
        maximum_position: float = 0.0252,
        timeout_sec: float = 4.0,
        stable_duration: float = 0.30,
    ) -> bool:
        """
        Wait for the measured finger joint to close far enough
        before permitting Attach and Lift.
        """
        deadline = time.monotonic() + timeout_sec
        stable_since = None

        self.node.get_logger().info(
            "Waiting for measured finger closure: "
            f"panda_finger_joint1 <= "
            f"{maximum_position:.4f} m"
        )

        while time.monotonic() < deadline:
            rclpy.spin_once(
                self.node,
                timeout_sec=0.05,
            )

            position = self.finger_position

            if (
                position is not None
                and position <= maximum_position
            ):
                if stable_since is None:
                    stable_since = time.monotonic()

                if (
                    time.monotonic() - stable_since
                    >= stable_duration
                ):
                    self.node.get_logger().info(
                        "Measured finger closure verified: "
                        f"finger position={position:.4f} m"
                    )
                    return True
            else:
                stable_since = None

        current_text = (
            "unavailable"
            if self.finger_position is None
            else f"{self.finger_position:.4f} m"
        )

        self.node.get_logger().error(
            "Measured grasp closure verification failed. "
            f"Final finger position: {current_text}"
        )
        return False

    def command(
        self,
        position: float,
        max_effort: float = 0.0,
        timeout_sec: float = 20.0,
        position_tolerance: float = 0.003,
    ) -> bool:
        if not 0.0 <= position <= 0.04:
            self.node.get_logger().error(
                "Gripper position is outside the valid "
                f"range [0.0, 0.04]: {position}"
            )
            return False

        if not self.client.wait_for_server(
            timeout_sec=5.0
        ):
            self.node.get_logger().error(
                f"{self.ACTION_NAME} is unavailable."
            )
            return False

        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = max_effort

        self.node.get_logger().info(
            "Sending gripper goal: "
            f"position={position:.3f} m, "
            f"max_effort={max_effort:.1f}"
        )

        send_future = self.client.send_goal_async(goal)

        rclpy.spin_until_future_complete(
            self.node,
            send_future,
            timeout_sec=5.0,
        )

        if (
            not send_future.done()
            or send_future.result() is None
        ):
            self.node.get_logger().error(
                "Gripper goal request timed out."
            )
            return False

        goal_handle = send_future.result()

        if not goal_handle.accepted:
            self.node.get_logger().error(
                "Gripper goal was rejected."
            )
            return False

        self.node.get_logger().info(
            "Gripper goal accepted."
        )

        result_future = goal_handle.get_result_async()

        rclpy.spin_until_future_complete(
            self.node,
            result_future,
            timeout_sec=timeout_sec,
        )

        if (
            not result_future.done()
            or result_future.result() is None
        ):
            self.node.get_logger().error(
                "Gripper execution timed out."
            )
            return False

        wrapped_result = result_future.result()
        result = wrapped_result.result
        status = wrapped_result.status

        position_error = abs(
            result.position - position
        )

        self.node.get_logger().info(
            "Gripper result: "
            f"position={result.position:.4f} m, "
            f"error={position_error:.4f} m, "
            f"effort={result.effort:.3f}, "
            f"stalled={result.stalled}, "
            f"reached_goal={result.reached_goal}, "
            f"status={status}"
        )

        succeeded = (
            status == GoalStatus.STATUS_SUCCEEDED
            and result.reached_goal
            and position_error <= position_tolerance
        )

        if not succeeded:
            self.node.get_logger().error(
                "Gripper result verification failed."
            )
            return False

        self.node.get_logger().info(
            "Gripper command verified successfully."
        )
        return True
