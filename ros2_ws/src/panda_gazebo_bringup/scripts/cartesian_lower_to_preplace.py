#!/usr/bin/env python3

import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetCartesianPath
from tf2_ros import Buffer, TransformException, TransformListener


class CartesianApproach(Node):
    def __init__(self) -> None:
        super().__init__("cartesian_lower_to_preplace")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        self.cartesian_client = self.create_client(
            GetCartesianPath,
            "/compute_cartesian_path",
        )

        self.execute_client = ActionClient(
            self,
            ExecuteTrajectory,
            "/execute_trajectory",
        )

    def lookup_tcp(self):
        deadline = time.monotonic() + 8.0
        last_error = None

        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

            try:
                return self.tf_buffer.lookup_transform(
                    "world",
                    "panda_hand_tcp",
                    Time(),
                    timeout=Duration(seconds=0.2),
                )
            except TransformException as error:
                last_error = error

        raise RuntimeError(
            f"Unable to read world -> panda_hand_tcp TF: {last_error}"
        )

    def compute_path(self):
        current = self.lookup_tcp()

        start = current.transform.translation
        rotation = current.transform.rotation

        self.get_logger().info(
            "Current TCP: "
            f"x={start.x:.3f}, y={start.y:.3f}, z={start.z:.3f}"
        )

        waypoint = Pose()

        # Align approximately with the cube centre while descending.
        waypoint.position.x = start.x
        waypoint.position.y = start.y
        waypoint.position.z = start.z - 0.040

        # Preserve the current gripper orientation.
        waypoint.orientation.x = rotation.x
        waypoint.orientation.y = rotation.y
        waypoint.orientation.z = rotation.z
        waypoint.orientation.w = rotation.w

        if not self.cartesian_client.wait_for_service(
            timeout_sec=10.0
        ):
            raise RuntimeError(
                "/compute_cartesian_path is unavailable."
            )

        request = GetCartesianPath.Request()
        request.header.frame_id = "world"
        request.start_state.is_diff = True
        request.group_name = "panda_arm"
        request.link_name = "panda_hand_tcp"
        request.waypoints = [waypoint]

        # Interpolate approximately every 5 mm.
        request.max_step = 0.005

        # We additionally inspect joint changes ourselves below.
        request.jump_threshold = 0.0

        # Table and cube must be considered during computation.
        request.avoid_collisions = True


        # Limit TCP motion to approximately 5 cm/s.

        self.get_logger().info(
            "Computing collision-aware 4 cm vertical descent: "
            f"z={start.z:.3f} -> {waypoint.position.z:.3f} m"
        )

        future = self.cartesian_client.call_async(request)

        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=15.0,
        )

        if not future.done() or future.result() is None:
            raise RuntimeError(
                "Cartesian path request timed out."
            )

        response = future.result()

        self.get_logger().info(
            f"Cartesian path fraction: {response.fraction:.3f}"
        )

        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(
                "Cartesian path computation failed. "
                f"MoveIt error code: {response.error_code.val}"
            )

        # Never execute an incomplete Cartesian approach.
        if response.fraction < 0.99:
            raise RuntimeError(
                "Cartesian path is incomplete; execution cancelled."
            )

        trajectory = response.solution.joint_trajectory

        if len(trajectory.points) < 2:
            raise RuntimeError(
                "The returned trajectory has too few points."
            )

        # Additional safety check for unusually large adjacent joint jumps.
        maximum_joint_step = 0.0

        for previous, current_point in zip(
            trajectory.points[:-1],
            trajectory.points[1:],
        ):
            for q_previous, q_current in zip(
                previous.positions,
                current_point.positions,
            ):
                maximum_joint_step = max(
                    maximum_joint_step,
                    abs(q_current - q_previous),
                )

        self.get_logger().info(
            f"Trajectory points: {len(trajectory.points)}"
        )
        self.get_logger().info(
            "Maximum adjacent joint step: "
            f"{maximum_joint_step:.4f} rad"
        )

        if maximum_joint_step > 0.35:
            raise RuntimeError(
                "Unexpectedly large joint jump; execution cancelled."
            )

        return self.slow_trajectory(
            response.solution,
            factor=5.0,
        )

    def slow_trajectory(self, trajectory, factor: float = 5.0):
        """Stretch trajectory time and consistently reduce speed."""
        if factor <= 1.0:
            return trajectory

        points = trajectory.joint_trajectory.points

        if not points:
            raise RuntimeError("Cannot slow an empty trajectory.")

        for point in points:
            total_nanoseconds = (
                point.time_from_start.sec * 1_000_000_000
                + point.time_from_start.nanosec
            )

            scaled_nanoseconds = int(
                total_nanoseconds * factor
            )

            point.time_from_start.sec = (
                scaled_nanoseconds // 1_000_000_000
            )
            point.time_from_start.nanosec = (
                scaled_nanoseconds % 1_000_000_000
            )

            if point.velocities:
                point.velocities = [
                    velocity / factor
                    for velocity in point.velocities
                ]

            if point.accelerations:
                point.accelerations = [
                    acceleration / (factor * factor)
                    for acceleration in point.accelerations
                ]

        final_time = points[-1].time_from_start
        self.get_logger().info(
            f"Trajectory slowed by factor {factor:.1f}; "
            f"new duration = "
            f"{final_time.sec + final_time.nanosec / 1e9:.2f} s"
        )

        return trajectory

    def execute(self, trajectory) -> bool:
        if not self.execute_client.wait_for_server(
            timeout_sec=10.0
        ):
            self.get_logger().error(
                "/execute_trajectory is unavailable."
            )
            return False

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory

        self.get_logger().info(
            "Sending Cartesian trajectory for execution..."
        )

        send_future = self.execute_client.send_goal_async(goal)

        rclpy.spin_until_future_complete(
            self,
            send_future,
            timeout_sec=10.0,
        )

        if not send_future.done() or send_future.result() is None:
            self.get_logger().error(
                "Execution goal request timed out."
            )
            return False

        goal_handle = send_future.result()

        if not goal_handle.accepted:
            self.get_logger().error(
                "Execution goal was rejected."
            )
            return False

        self.get_logger().info(
            "Execution goal accepted."
        )

        result_future = goal_handle.get_result_async()

        rclpy.spin_until_future_complete(
            self,
            result_future,
            timeout_sec=30.0,
        )

        if not result_future.done() or result_future.result() is None:
            self.get_logger().error(
                "Cartesian execution timed out."
            )
            return False

        result = result_future.result().result

        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                "Cartesian execution failed. "
                f"MoveIt error code: {result.error_code.val}"
            )
            return False

        final = self.lookup_tcp().transform.translation

        self.get_logger().info(
            "Cartesian execution succeeded."
        )
        self.get_logger().info(
            "Final TCP: "
            f"x={final.x:.3f}, y={final.y:.3f}, z={final.z:.3f}"
        )

        return True


def main() -> None:
    rclpy.init()
    node = CartesianApproach()

    try:
        trajectory = node.compute_path()
        success = node.execute(trajectory)
    except Exception as error:
        node.get_logger().error(str(error))
        success = False
    finally:
        node.destroy_node()
        rclpy.shutdown()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
