#!/usr/bin/env python3

import sys
import time

import rclpy
from rclpy.duration import Duration
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from moveit_msgs.srv import GetMotionPlan
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformException, TransformListener


class PregraspPlanner(Node):
    def __init__(self) -> None:
        super().__init__("plan_execute_transport_attached")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        self.plan_client = self.create_client(
            GetMotionPlan,
            "/plan_kinematic_path",
        )

        self.execute_client = ActionClient(
            self,
            ExecuteTrajectory,
            "/execute_trajectory",
        )

    def get_current_tcp_orientation(self):
        deadline = time.monotonic() + 8.0
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
                return transform.transform.rotation
            except TransformException as error:
                last_error = error

        raise RuntimeError(
            f"Unable to read world -> panda_hand_tcp TF: {last_error}"
        )

    def build_goal_constraints(self, orientation) -> Constraints:
        target_x = 0.55
        target_y = 0.25
        target_z = 0.505
        # A small spherical region around the target position.
        target_region = SolidPrimitive()
        target_region.type = SolidPrimitive.SPHERE
        target_region.dimensions = [0.01]

        target_pose = Pose()
        target_pose.position.x = target_x
        target_pose.position.y = target_y
        target_pose.position.z = target_z
        target_pose.orientation.w = 1.0

        bounding_volume = BoundingVolume()
        bounding_volume.primitives = [target_region]
        bounding_volume.primitive_poses = [target_pose]

        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = "world"
        position_constraint.link_name = "panda_hand_tcp"
        position_constraint.constraint_region = bounding_volume
        position_constraint.weight = 1.0

        # Preserve approximately the current TCP orientation.
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = "world"
        orientation_constraint.link_name = "panda_hand_tcp"
        orientation_constraint.orientation = orientation
        orientation_constraint.absolute_x_axis_tolerance = 0.15
        orientation_constraint.absolute_y_axis_tolerance = 0.15
        orientation_constraint.absolute_z_axis_tolerance = 0.15
        orientation_constraint.weight = 1.0

        constraints = Constraints()
        constraints.name = "transport_attached_cube"
        constraints.position_constraints = [position_constraint]
        constraints.orientation_constraints = [orientation_constraint]

        return constraints

    def execute_plan(self, trajectory) -> bool:
        self.get_logger().info(
            "Waiting for /execute_trajectory..."
        )

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
            "Sending planned trajectory for execution..."
        )

        send_future = self.execute_client.send_goal_async(goal)

        rclpy.spin_until_future_complete(
            self,
            send_future,
            timeout_sec=10.0,
        )

        if not send_future.done() or send_future.result() is None:
            self.get_logger().error(
                "Trajectory goal request timed out."
            )
            return False

        goal_handle = send_future.result()

        if not goal_handle.accepted:
            self.get_logger().error(
                "Trajectory execution goal was rejected."
            )
            return False

        self.get_logger().info(
            "Trajectory execution goal accepted."
        )

        result_future = goal_handle.get_result_async()

        rclpy.spin_until_future_complete(
            self,
            result_future,
            timeout_sec=30.0,
        )

        if not result_future.done() or result_future.result() is None:
            self.get_logger().error(
                "Trajectory execution timed out."
            )
            return False

        result = result_future.result().result
        error_code = result.error_code.val

        if error_code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                "Trajectory execution failed. "
                f"MoveIt error code: {error_code}"
            )
            return False

        self.get_logger().info(
            "Trajectory execution succeeded."
        )
        return True

    def request_plan(self) -> bool:
        self.get_logger().info(
            "Reading current panda_hand_tcp orientation..."
        )

        orientation = self.get_current_tcp_orientation()

        if not self.plan_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                "/plan_kinematic_path is unavailable."
            )
            return False

        request = GetMotionPlan.Request()
        motion_request = request.motion_plan_request

        motion_request.group_name = "panda_arm"
        motion_request.pipeline_id = "ompl"
        motion_request.num_planning_attempts = 5
        motion_request.allowed_planning_time = 5.0
        motion_request.max_velocity_scaling_factor = 0.10
        motion_request.max_acceleration_scaling_factor = 0.10

        # Use the robot's current state as the planning start state.
        motion_request.start_state.is_diff = True
        motion_request.goal_constraints = [
            self.build_goal_constraints(orientation)
        ]

        self.get_logger().info(
            "Planning and executing with attached cube: TCP target = "
            "(0.55, 0.25, 0.505) m."
        )

        future = self.plan_client.call_async(request)

        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=15.0,
        )

        if not future.done() or future.result() is None:
            self.get_logger().error(
                "Motion planning request timed out."
            )
            return False

        response = future.result().motion_plan_response
        error_code = response.error_code.val

        if error_code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"Planning failed. MoveIt error code: {error_code}"
            )
            return False

        trajectory = response.trajectory.joint_trajectory

        self.get_logger().info("Planning succeeded.")
        self.get_logger().info(
            f"Planning time: {response.planning_time:.3f} s"
        )
        self.get_logger().info(
            f"Trajectory points: {len(trajectory.points)}"
        )

        if trajectory.points:
            final_positions = trajectory.points[-1].positions
            rounded = [round(value, 4) for value in final_positions]

            self.get_logger().info(
                f"Final planned joint positions: {rounded}"
            )

        self.get_logger().info(
            "Planning verification passed."
        )
        return self.execute_plan(response.trajectory)


def main() -> None:
    rclpy.init()
    node = PregraspPlanner()

    try:
        success = node.request_plan()
    except Exception as error:
        node.get_logger().error(str(error))
        success = False
    finally:
        node.destroy_node()
        rclpy.shutdown()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
