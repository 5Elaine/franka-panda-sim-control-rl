#!/usr/bin/env python3

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

TARGET = [
    0.5,
    -0.4,
    0.3,
    -1.8,
    0.2,
    1.8,
    -0.5,
]

HOME = [
    0.0,
    0.0,
    0.0,
    -1.57079,
    0.0,
    1.57079,
    -0.7853,
]

BASELINE_DURATION = 2.0
TARGET_DURATION = 5.0
HOME_DURATION = 5.0

# 每个新目标在开始阶段连续发布几次，避免单次消息丢失。
COMMAND_REPEAT_DURATION = 0.5
COMMAND_TIMER_PERIOD = 0.1


class JointBenchmarkSequence(Node):
    def __init__(self, output_path: Path) -> None:
        super().__init__("joint_benchmark_sequence")

        self.output_path = output_path
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.csv_file = self.output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )
        self.writer = csv.writer(self.csv_file)

        self.writer.writerow(
            [
                "elapsed_time_s",
                "ros_stamp_s",
                "phase",
            ]
            + [f"q{i}" for i in range(1, 8)]
            + [f"dq{i}" for i in range(1, 8)]
            + [f"target{i}" for i in range(1, 8)]
        )

        self.command_publisher = self.create_publisher(
            Float64MultiArray,
            "/joint_command",
            10,
        )

        self.joint_state_subscription = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            50,
        )

        self.timer = self.create_timer(
            COMMAND_TIMER_PERIOD,
            self.sequence_callback,
        )

        self.start_time = time.monotonic()
        self.phase_start_time = self.start_time

        self.phase = "BASELINE"
        self.current_target: List[float] | None = None
        self.repeat_until = 0.0
        self.row_count = 0
        self.finished = False
        self.missing_warning_printed = False

        self.get_logger().info(
            f"Output CSV: {self.output_path}"
        )
        self.get_logger().info(
            f"Baseline duration: {BASELINE_DURATION:.1f} s"
        )
        self.get_logger().info(
            f"Target duration: {TARGET_DURATION:.1f} s"
        )
        self.get_logger().info(
            f"Home duration: {HOME_DURATION:.1f} s"
        )

    def start_phase(
        self,
        phase: str,
        target: List[float],
    ) -> None:
        now = time.monotonic()

        self.phase = phase
        self.phase_start_time = now
        self.current_target = target.copy()
        self.repeat_until = now + COMMAND_REPEAT_DURATION

        self.publish_current_target()

        self.get_logger().info(
            f"Started phase {phase}: {target}"
        )

    def publish_current_target(self) -> None:
        if self.current_target is None:
            return

        message = Float64MultiArray()
        message.data = self.current_target
        self.command_publisher.publish(message)

    def sequence_callback(self) -> None:
        now = time.monotonic()
        total_elapsed = now - self.start_time
        phase_elapsed = now - self.phase_start_time

        if self.phase == "BASELINE":
            if total_elapsed < BASELINE_DURATION:
                return

            subscription_count = (
                self.command_publisher.get_subscription_count()
            )

            if subscription_count < 1:
                self.get_logger().warning(
                    "Waiting for a /joint_command subscriber..."
                )
                return

            self.start_phase("TARGET", TARGET)
            return

        if now <= self.repeat_until:
            self.publish_current_target()

        if (
            self.phase == "TARGET"
            and phase_elapsed >= TARGET_DURATION
        ):
            self.start_phase("HOME", HOME)
            return

        if (
            self.phase == "HOME"
            and phase_elapsed >= HOME_DURATION
        ):
            self.finished = True
            self.get_logger().info(
                "Benchmark sequence completed."
            )

    def joint_state_callback(
        self,
        message: JointState,
    ) -> None:
        name_to_index: Dict[str, int] = {
            name: index
            for index, name in enumerate(message.name)
        }

        missing = [
            name
            for name in JOINT_NAMES
            if name not in name_to_index
        ]

        if missing:
            if not self.missing_warning_printed:
                self.get_logger().warning(
                    "Missing JointState names: "
                    + ", ".join(missing)
                )
                self.missing_warning_printed = True
            return

        positions: List[float] = []
        velocities: List[float] = []

        for name in JOINT_NAMES:
            index = name_to_index[name]

            if index >= len(message.position):
                return

            positions.append(
                float(message.position[index])
            )

            if index < len(message.velocity):
                velocities.append(
                    float(message.velocity[index])
                )
            else:
                velocities.append(math.nan)

        elapsed_time = (
            time.monotonic() - self.start_time
        )

        ros_stamp = (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1e-9
        )

        target_values = (
            self.current_target
            if self.current_target is not None
            else [math.nan] * 7
        )

        self.writer.writerow(
            [
                f"{elapsed_time:.9f}",
                f"{ros_stamp:.9f}",
                self.phase,
            ]
            + positions
            + velocities
            + target_values
        )

        self.csv_file.flush()
        self.row_count += 1

    def close(self) -> None:
        if not self.csv_file.closed:
            self.csv_file.close()

        self.get_logger().info(
            f"Saved {self.row_count} JointState samples."
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output benchmark CSV.",
    )

    arguments = parser.parse_args()

    rclpy.init()

    node = JointBenchmarkSequence(
        arguments.output.expanduser().resolve()
    )

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(
                node,
                timeout_sec=0.05,
            )

    except KeyboardInterrupt:
        pass

    finally:
        node.close()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
