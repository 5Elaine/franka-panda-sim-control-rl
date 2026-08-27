#!/usr/bin/env python3

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
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

TARGET = [
    0.5,
    -0.4,
    0.3,
    -1.8,
    0.2,
    1.8,
    -0.5,
]

ACTION_NAME = (
    "/panda_arm_controller/follow_joint_trajectory"
)

# 先安全地把当前 Gazebo 姿态移动到正式 Home。
INITIAL_HOME_DURATION = 2.0

# 正式 Target/Home 轨迹时长设为 0.8 s，
# 与 MuJoCo 中约 0.78 s 的实际响应尺度接近。
COMMAND_DURATION = 0.8

BASELINE_DURATION = 2.0
TARGET_PHASE_DURATION = 5.0
HOME_PHASE_DURATION = 5.0


class GazeboJointBenchmark(Node):
    def __init__(self, output_path: Path) -> None:
        super().__init__("gazebo_joint_benchmark_sequence")

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

        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            ACTION_NAME,
        )

        self.joint_state_subscription = (
            self.create_subscription(
                JointState,
                "/joint_states",
                self.joint_state_callback,
                100,
            )
        )

        self.start_wall_time = time.monotonic()
        self.phase = "WAITING"
        self.current_target: Optional[List[float]] = None
        self.latest_positions: Optional[List[float]] = None
        self.latest_velocities: Optional[List[float]] = None

        self.row_count = 0
        self.missing_warning_printed = False

        self.get_logger().info(
            f"Output CSV: {self.output_path}"
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

        self.latest_positions = positions
        self.latest_velocities = velocities

        elapsed_time = (
            time.monotonic() - self.start_wall_time
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

    def spin_for(self, duration: float) -> None:
        end_time = time.monotonic() + duration

        while (
            rclpy.ok()
            and time.monotonic() < end_time
        ):
            rclpy.spin_once(
                self,
                timeout_sec=0.05,
            )

    def wait_for_joint_states(
        self,
        timeout: float = 10.0,
    ) -> None:
        end_time = time.monotonic() + timeout

        while (
            rclpy.ok()
            and self.latest_positions is None
            and time.monotonic() < end_time
        ):
            rclpy.spin_once(
                self,
                timeout_sec=0.1,
            )

        if self.latest_positions is None:
            raise RuntimeError(
                "No valid /joint_states message was received."
            )

    def send_trajectory(
        self,
        target: List[float],
        duration: float,
        label: str,
    ) -> None:
        if not self.action_client.wait_for_server(
            timeout_sec=10.0
        ):
            raise RuntimeError(
                f"Action server unavailable: {ACTION_NAME}"
            )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = target

        whole_seconds = int(duration)
        nanoseconds = int(
            round(
                (duration - whole_seconds)
                * 1_000_000_000
            )
        )

        point.time_from_start.sec = whole_seconds
        point.time_from_start.nanosec = nanoseconds

        goal.trajectory.points = [point]

        self.get_logger().info(
            f"Sending {label} trajectory: {target}"
        )

        send_future = self.action_client.send_goal_async(
            goal
        )

        rclpy.spin_until_future_complete(
            self,
            send_future,
            timeout_sec=10.0,
        )

        if not send_future.done():
            raise RuntimeError(
                f"Timed out while sending {label} goal."
            )

        goal_handle = send_future.result()

        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(
                f"{label} trajectory goal was rejected."
            )

        self.get_logger().info(
            f"{label} goal accepted."
        )

        result_future = goal_handle.get_result_async()

        rclpy.spin_until_future_complete(
            self,
            result_future,
            timeout_sec=max(10.0, duration + 5.0),
        )

        if not result_future.done():
            raise RuntimeError(
                f"Timed out waiting for {label} result."
            )

        wrapped_result = result_future.result()

        if wrapped_result is None:
            raise RuntimeError(
                f"No result was returned for {label}."
            )

        result = wrapped_result.result

        if result.error_code != 0:
            raise RuntimeError(
                f"{label} failed: "
                f"error_code={result.error_code}, "
                f"error_string={result.error_string!r}"
            )

        self.get_logger().info(
            f"{label} trajectory completed successfully."
        )

    def run_sequence(self) -> None:
        self.wait_for_joint_states()

        # 初始化动作不纳入正式 Target/Home 指标。
        self.phase = "INITIALIZE_HOME"
        self.current_target = None

        self.send_trajectory(
            HOME,
            INITIAL_HOME_DURATION,
            "INITIAL_HOME",
        )

        # 给控制器一点额外稳定时间。
        self.spin_for(1.0)

        self.phase = "BASELINE"
        self.current_target = None

        self.get_logger().info(
            f"Recording Home baseline for "
            f"{BASELINE_DURATION:.1f} s."
        )

        self.spin_for(BASELINE_DURATION)

        # 正式 Target 阶段。
        self.phase = "TARGET"
        self.current_target = TARGET.copy()
        target_start = time.monotonic()

        self.send_trajectory(
            TARGET,
            COMMAND_DURATION,
            "TARGET",
        )

        target_elapsed = (
            time.monotonic() - target_start
        )

        if target_elapsed < TARGET_PHASE_DURATION:
            self.spin_for(
                TARGET_PHASE_DURATION
                - target_elapsed
            )

        # 正式 Home 阶段。
        self.phase = "HOME"
        self.current_target = HOME.copy()
        home_start = time.monotonic()

        self.send_trajectory(
            HOME,
            COMMAND_DURATION,
            "HOME",
        )

        home_elapsed = (
            time.monotonic() - home_start
        )

        if home_elapsed < HOME_PHASE_DURATION:
            self.spin_for(
                HOME_PHASE_DURATION
                - home_elapsed
            )

        self.phase = "COMPLETE"
        self.current_target = None

        self.get_logger().info(
            "Gazebo benchmark sequence completed."
        )

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

    node = GazeboJointBenchmark(
        arguments.output.expanduser().resolve()
    )

    try:
        node.run_sequence()

    except KeyboardInterrupt:
        pass

    finally:
        node.close()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
