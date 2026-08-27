#!/usr/bin/env python3

from __future__ import annotations

import math
import time

import rclpy
from geometry_msgs.msg import Pose, Quaternion
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from moveit_msgs.srv import GetCartesianPath, GetMotionPlan
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf2_ros import (
    Buffer,
    TransformException,
    TransformListener,
)


class MotionManager:
    """Plan, execute, and verify Panda arm motions."""

    TCP_LINK = "panda_hand_tcp"
    REFERENCE_FRAME = "world"
    PLANNING_GROUP = "panda_arm"

    def __init__(self, node: Node) -> None:
        self.node = node

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            node,
        )

        self.plan_client = node.create_client(
            GetMotionPlan,
            "/plan_kinematic_path",
        )

        self.cartesian_client = node.create_client(
            GetCartesianPath,
            "/compute_cartesian_path",
        )

        self.execute_client = ActionClient(
            node,
            ExecuteTrajectory,
            "/execute_trajectory",
        )

        self.latest_joint_positions: dict[str, float] = {}

        self.joint_state_subscription = (
            node.create_subscription(
                JointState,
                "/joint_states",
                self.joint_state_callback,
                qos_profile_sensor_data,
            )
        )

    def lookup_tcp_transform(
        self,
        timeout_sec: float = 8.0,
    ):
        deadline = time.monotonic() + timeout_sec
        last_error = None

        while time.monotonic() < deadline:
            rclpy.spin_once(
                self.node,
                timeout_sec=0.1,
            )

            try:
                return self.tf_buffer.lookup_transform(
                    self.REFERENCE_FRAME,
                    self.TCP_LINK,
                    Time(),
                    timeout=Duration(seconds=0.2),
                )
            except TransformException as error:
                last_error = error

        raise RuntimeError(
            "Unable to read "
            f"{self.REFERENCE_FRAME} -> {self.TCP_LINK} TF: "
            f"{last_error}"
        )

    def build_pose_constraints(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        orientation,
        constraint_name: str,
        position_radius: float = 0.01,
        orientation_tolerance: float = 0.15,
    ) -> Constraints:
        target_region = SolidPrimitive()
        target_region.type = SolidPrimitive.SPHERE
        target_region.dimensions = [position_radius]

        target_pose = Pose()
        target_pose.position.x = target_x
        target_pose.position.y = target_y
        target_pose.position.z = target_z
        target_pose.orientation.w = 1.0

        bounding_volume = BoundingVolume()
        bounding_volume.primitives = [target_region]
        bounding_volume.primitive_poses = [target_pose]

        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = (
            self.REFERENCE_FRAME
        )
        position_constraint.link_name = self.TCP_LINK
        position_constraint.constraint_region = (
            bounding_volume
        )
        position_constraint.weight = 1.0

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = (
            self.REFERENCE_FRAME
        )
        orientation_constraint.link_name = self.TCP_LINK
        orientation_constraint.orientation = orientation
        orientation_constraint.absolute_x_axis_tolerance = (
            orientation_tolerance
        )
        orientation_constraint.absolute_y_axis_tolerance = (
            orientation_tolerance
        )
        orientation_constraint.absolute_z_axis_tolerance = (
            orientation_tolerance
        )
        orientation_constraint.weight = 1.0

        constraints = Constraints()
        constraints.name = constraint_name
        constraints.position_constraints = [
            position_constraint
        ]
        constraints.orientation_constraints = [
            orientation_constraint
        ]

        return constraints

    def joint_state_callback(
        self,
        message: JointState,
    ) -> None:
        for index, joint_name in enumerate(message.name):
            if index < len(message.position):
                self.latest_joint_positions[joint_name] = (
                    message.position[index]
                )

    def verify_joint_positions(
        self,
        target_positions: dict[str, float],
        tolerance: float = 0.03,
        timeout_sec: float = 5.0,
    ) -> bool:
        deadline = time.monotonic() + timeout_sec

        while time.monotonic() < deadline:
            rclpy.spin_once(
                self.node,
                timeout_sec=0.1,
            )

            if all(
                name in self.latest_joint_positions
                for name in target_positions
            ):
                break

        missing = [
            name
            for name in target_positions
            if name not in self.latest_joint_positions
        ]

        if missing:
            self.node.get_logger().error(
                "Joint-state verification is missing: "
                + ", ".join(sorted(missing))
            )
            return False

        errors = {
            name: abs(
                self.latest_joint_positions[name]
                - target
            )
            for name, target in target_positions.items()
        }

        maximum_error = max(errors.values())

        self.node.get_logger().info(
            "Final Home joint errors: "
            + ", ".join(
                f"{name}={error:.4f}"
                for name, error in errors.items()
            )
        )

        if maximum_error > tolerance:
            self.node.get_logger().error(
                "Home joint verification failed: "
                f"maximum error={maximum_error:.4f} rad, "
                f"tolerance={tolerance:.4f} rad."
            )
            return False

        self.node.get_logger().info(
            "Home joint state verified successfully."
        )
        return True

    def move_to_joint_state_ompl(
        self,
        target_positions: dict[str, float],
        constraint_name: str,
        velocity_scaling: float = 0.10,
        acceleration_scaling: float = 0.10,
        joint_tolerance: float = 0.01,
    ) -> bool:
        if not self.plan_client.wait_for_service(
            timeout_sec=10.0
        ):
            self.node.get_logger().error(
                "/plan_kinematic_path is unavailable."
            )
            return False

        constraints = Constraints()
        constraints.name = constraint_name

        for joint_name, target in target_positions.items():
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = joint_name
            joint_constraint.position = target
            joint_constraint.tolerance_above = (
                joint_tolerance
            )
            joint_constraint.tolerance_below = (
                joint_tolerance
            )
            joint_constraint.weight = 1.0

            constraints.joint_constraints.append(
                joint_constraint
            )

        request = GetMotionPlan.Request()
        motion_request = request.motion_plan_request

        motion_request.group_name = self.PLANNING_GROUP
        motion_request.pipeline_id = "ompl"
        motion_request.start_state.is_diff = True
        motion_request.goal_constraints = [constraints]

        motion_request.num_planning_attempts = 5
        motion_request.allowed_planning_time = 5.0
        motion_request.max_velocity_scaling_factor = (
            velocity_scaling
        )
        motion_request.max_acceleration_scaling_factor = (
            acceleration_scaling
        )

        self.node.get_logger().info(
            "Planning collision-aware return to Home "
            "with OMPL..."
        )

        future = self.plan_client.call_async(request)

        rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=15.0,
        )

        if (
            not future.done()
            or future.result() is None
        ):
            self.node.get_logger().error(
                "Home planning request timed out."
            )
            return False

        response = future.result().motion_plan_response

        if (
            response.error_code.val
            != MoveItErrorCodes.SUCCESS
        ):
            self.node.get_logger().error(
                "Home planning failed. MoveIt error code: "
                f"{response.error_code.val}"
            )
            return False

        trajectory = response.trajectory.joint_trajectory

        if not trajectory.points:
            self.node.get_logger().error(
                "MoveIt returned an empty Home trajectory."
            )
            return False

        self.node.get_logger().info(
            "Home planning succeeded."
        )
        self.node.get_logger().info(
            f"Planning time: "
            f"{response.planning_time:.3f} s"
        )
        self.node.get_logger().info(
            f"Trajectory points: "
            f"{len(trajectory.points)}"
        )

        if not self.execute_trajectory(
            response.trajectory
        ):
            return False

        return self.verify_joint_positions(
            target_positions=target_positions,
            tolerance=0.03,
        )

    def slow_trajectory(
        self,
        trajectory,
        factor: float = 5.0,
    ):
        """Stretch trajectory time and reduce derivatives."""
        if factor <= 1.0:
            return trajectory

        points = trajectory.joint_trajectory.points

        if not points:
            raise RuntimeError(
                "Cannot slow an empty trajectory."
            )

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

        self.node.get_logger().info(
            f"Trajectory slowed by factor {factor:.1f}; "
            "new duration="
            f"{final_time.sec + final_time.nanosec / 1e9:.2f} s"
        )

        return trajectory

    def move_cartesian_to(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        motion_label: str,
        max_step: float = 0.005,
        minimum_fraction: float = 0.99,
        slowdown_factor: float = 5.0,
        position_tolerance: float = 0.015,
        skip_tolerance: float = 0.002,
    ) -> bool:
        current = self.lookup_tcp_transform()

        start = current.transform.translation
        rotation = current.transform.rotation

        start_error = math.sqrt(
            (start.x - target_x) ** 2
            + (start.y - target_y) ** 2
            + (start.z - target_z) ** 2
        )

        self.node.get_logger().info(
            "Current TCP: "
            f"x={start.x:.3f}, "
            f"y={start.y:.3f}, "
            f"z={start.z:.3f}"
        )

        if start_error <= skip_tolerance:
            self.node.get_logger().info(
                f"{motion_label}: TCP is already within "
                f"{skip_tolerance:.3f} m of the target."
            )
            return True

        waypoint = Pose()
        waypoint.position.x = target_x
        waypoint.position.y = target_y
        waypoint.position.z = target_z

        # Preserve the measured TCP orientation.
        waypoint.orientation.x = rotation.x
        waypoint.orientation.y = rotation.y
        waypoint.orientation.z = rotation.z
        waypoint.orientation.w = rotation.w

        if not self.cartesian_client.wait_for_service(
            timeout_sec=10.0
        ):
            self.node.get_logger().error(
                "/compute_cartesian_path is unavailable."
            )
            return False

        request = GetCartesianPath.Request()
        request.header.frame_id = self.REFERENCE_FRAME
        request.start_state.is_diff = True
        request.group_name = self.PLANNING_GROUP
        request.link_name = self.TCP_LINK
        request.waypoints = [waypoint]
        request.max_step = max_step
        request.jump_threshold = 0.0
        request.avoid_collisions = True

        self.node.get_logger().info(
            f"Computing {motion_label}: "
            f"target=({target_x:.3f}, "
            f"{target_y:.3f}, {target_z:.3f}) m"
        )

        future = self.cartesian_client.call_async(
            request
        )

        rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=15.0,
        )

        if (
            not future.done()
            or future.result() is None
        ):
            self.node.get_logger().error(
                "Cartesian-path request timed out."
            )
            return False

        response = future.result()

        self.node.get_logger().info(
            f"Cartesian path fraction: "
            f"{response.fraction:.3f}"
        )

        if (
            response.error_code.val
            != MoveItErrorCodes.SUCCESS
        ):
            self.node.get_logger().error(
                "Cartesian-path computation failed. "
                "MoveIt error code: "
                f"{response.error_code.val}"
            )
            return False

        if response.fraction < minimum_fraction:
            self.node.get_logger().error(
                "Cartesian path is incomplete; "
                "execution cancelled."
            )
            return False

        trajectory = response.solution
        points = trajectory.joint_trajectory.points

        if len(points) < 2:
            self.node.get_logger().error(
                "Cartesian trajectory has too few points."
            )
            return False

        maximum_joint_step = 0.0

        for previous, current_point in zip(
            points[:-1],
            points[1:],
        ):
            for q_previous, q_current in zip(
                previous.positions,
                current_point.positions,
            ):
                maximum_joint_step = max(
                    maximum_joint_step,
                    abs(q_current - q_previous),
                )

        self.node.get_logger().info(
            f"Trajectory points: {len(points)}"
        )
        self.node.get_logger().info(
            "Maximum adjacent joint step: "
            f"{maximum_joint_step:.4f} rad"
        )

        if maximum_joint_step > 0.35:
            self.node.get_logger().error(
                "Unexpectedly large adjacent joint step; "
                "execution cancelled."
            )
            return False

        trajectory = self.slow_trajectory(
            trajectory,
            factor=slowdown_factor,
        )

        if not self.execute_trajectory(trajectory):
            return False

        return self.verify_tcp_position(
            target_x=target_x,
            target_y=target_y,
            target_z=target_z,
            tolerance=position_tolerance,
        )

    def execute_trajectory(
        self,
        trajectory,
        timeout_sec: float = 40.0,
    ) -> bool:
        if not self.execute_client.wait_for_server(
            timeout_sec=10.0
        ):
            self.node.get_logger().error(
                "/execute_trajectory is unavailable."
            )
            return False

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory

        self.node.get_logger().info(
            "Sending planned trajectory for execution..."
        )

        send_future = self.execute_client.send_goal_async(
            goal
        )

        rclpy.spin_until_future_complete(
            self.node,
            send_future,
            timeout_sec=10.0,
        )

        if (
            not send_future.done()
            or send_future.result() is None
        ):
            self.node.get_logger().error(
                "Trajectory goal request timed out."
            )
            return False

        goal_handle = send_future.result()

        if not goal_handle.accepted:
            self.node.get_logger().error(
                "Trajectory execution goal was rejected."
            )
            return False

        self.node.get_logger().info(
            "Trajectory execution goal accepted."
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
                "Trajectory execution timed out."
            )
            return False

        result = result_future.result().result
        error_code = result.error_code.val

        if error_code != MoveItErrorCodes.SUCCESS:
            self.node.get_logger().error(
                "Trajectory execution failed. "
                f"MoveIt error code: {error_code}"
            )
            return False

        self.node.get_logger().info(
            "Trajectory execution succeeded."
        )
        return True

    def verify_tcp_position(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        tolerance: float = 0.02,
    ) -> bool:
        transform = self.lookup_tcp_transform()
        actual = transform.transform.translation

        error = math.sqrt(
            (actual.x - target_x) ** 2
            + (actual.y - target_y) ** 2
            + (actual.z - target_z) ** 2
        )

        self.node.get_logger().info(
            "Final TCP: "
            f"x={actual.x:.3f}, "
            f"y={actual.y:.3f}, "
            f"z={actual.z:.3f}, "
            f"position_error={error:.4f} m"
        )

        if error > tolerance:
            self.node.get_logger().error(
                "Final TCP position verification failed: "
                f"error {error:.4f} m exceeds "
                f"tolerance {tolerance:.4f} m."
            )
            return False

        self.node.get_logger().info(
            "Final TCP position verified successfully."
        )
        return True

    def move_to_pose_ompl(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        constraint_name: str,
        velocity_scaling: float = 0.20,
        acceleration_scaling: float = 0.20,
        target_orientation: Quaternion | None = None,
        orientation_tolerance: float = 0.08,
    ) -> bool:
        self.node.get_logger().info(
            "Reading current TCP orientation..."
        )

        current_transform = self.lookup_tcp_transform()
        current_position = (
            current_transform.transform.translation
        )
        current_orientation = (
            current_transform.transform.rotation
        )

        if target_orientation is None:
            orientation = current_orientation
        else:
            orientation = target_orientation

        current_error = math.sqrt(
            (current_position.x - target_x) ** 2
            + (current_position.y - target_y) ** 2
            + (current_position.z - target_z) ** 2
        )

        if (
            target_orientation is None
            and current_error <= 0.02
        ):
            self.node.get_logger().info(
                "TCP position is already within 0.020 m "
                "of the requested OMPL target; "
                "planning skipped."
            )
            return True

        if not self.plan_client.wait_for_service(
            timeout_sec=10.0
        ):
            self.node.get_logger().error(
                "/plan_kinematic_path is unavailable."
            )
            return False

        request = GetMotionPlan.Request()
        motion_request = request.motion_plan_request

        motion_request.group_name = self.PLANNING_GROUP
        motion_request.pipeline_id = "ompl"
        motion_request.num_planning_attempts = 5
        motion_request.allowed_planning_time = 5.0
        motion_request.max_velocity_scaling_factor = (
            velocity_scaling
        )
        motion_request.max_acceleration_scaling_factor = (
            acceleration_scaling
        )

        # Use the robot's current measured state.
        motion_request.start_state.is_diff = True

        motion_request.goal_constraints = [
            self.build_pose_constraints(
                target_x=target_x,
                target_y=target_y,
                target_z=target_z,
                orientation=orientation,
                constraint_name=constraint_name,
                orientation_tolerance=orientation_tolerance,
            )
        ]

        self.node.get_logger().info(
            "Planning and executing with OMPL: "
            f"TCP target=({target_x:.3f}, "
            f"{target_y:.3f}, {target_z:.3f}) m"
        )

        future = self.plan_client.call_async(request)

        rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=15.0,
        )

        if (
            not future.done()
            or future.result() is None
        ):
            self.node.get_logger().error(
                "Motion planning request timed out."
            )
            return False

        response = future.result().motion_plan_response
        error_code = response.error_code.val

        if error_code != MoveItErrorCodes.SUCCESS:
            self.node.get_logger().error(
                "OMPL planning failed. "
                f"MoveIt error code: {error_code}"
            )
            return False

        trajectory = response.trajectory.joint_trajectory

        if not trajectory.points:
            self.node.get_logger().error(
                "OMPL returned an empty trajectory."
            )
            return False

        self.node.get_logger().info(
            "OMPL planning succeeded."
        )
        self.node.get_logger().info(
            f"Planning time: "
            f"{response.planning_time:.3f} s"
        )
        self.node.get_logger().info(
            f"Trajectory points: "
            f"{len(trajectory.points)}"
        )

        final_positions = [
            round(value, 4)
            for value
            in trajectory.points[-1].positions
        ]

        self.node.get_logger().info(
            "Final planned joint positions: "
            f"{final_positions}"
        )

        if not self.execute_trajectory(
            response.trajectory
        ):
            return False

        return self.verify_tcp_position(
            target_x=target_x,
            target_y=target_y,
            target_z=target_z,
        )
