#!/usr/bin/env python3

import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
)
from moveit_msgs.srv import GetMotionPlan


HOME_JOINTS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.785,
    "panda_joint3": 0.0,
    "panda_joint4": -2.356,
    "panda_joint5": 0.0,
    "panda_joint6": 1.571,
    "panda_joint7": 0.785,
}


class ReturnHome(Node):
    def __init__(self) -> None:
        super().__init__("plan_execute_return_home")

        self.plan_client = self.create_client(
            GetMotionPlan,
            "/plan_kinematic_path",
        )

        self.execute_client = ActionClient(
            self,
            ExecuteTrajectory,
            "/execute_trajectory",
        )

    @staticmethod
    def build_home_constraints() -> Constraints:
        constraints = Constraints()
        constraints.name = "panda_home_joint_goal"

        for joint_name, position in HOME_JOINTS.items():
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = joint_name
            joint_constraint.position = position
            joint_constraint.tolerance_above = 0.01
            joint_constraint.tolerance_below = 0.01
            joint_constraint.weight = 1.0

            constraints.joint_constraints.append(
                joint_constraint
            )

        return constraints

    def plan(self):
        self.get_logger().info(
            "Waiting for /plan_kinematic_path..."
        )

        if not self.plan_client.wait_for_service(
            timeout_sec=10.0
        ):
            raise RuntimeError(
                "/plan_kinematic_path is unavailable."
            )

        request = GetMotionPlan.Request()
        motion_request = request.motion_plan_request

        motion_request.group_name = "panda_arm"
        motion_request.pipeline_id = "ompl"

        # Read the robot's current joint state as the start state.
        motion_request.start_state.is_diff = True

        motion_request.goal_constraints = [
            self.build_home_constraints()
        ]

        motion_request.num_planning_attempts = 5
        motion_request.allowed_planning_time = 5.0

        # Return slowly while near the table and placed cube.
        motion_request.max_velocity_scaling_factor = 0.10
        motion_request.max_acceleration_scaling_factor = 0.10

        self.get_logger().info(
            "Planning a collision-aware path back to Home..."
        )

        future = self.plan_client.call_async(request)

        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=15.0,
        )

        if not future.done() or future.result() is None:
            raise RuntimeError(
                "Motion planning request timed out."
            )

        response = future.result().motion_plan_response

        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(
                "Home planning failed. "
                f"MoveIt error code: {response.error_code.val}"
            )

        trajectory = response.trajectory.joint_trajectory

        if not trajectory.points:
            raise RuntimeError(
                "MoveIt returned an empty trajectory."
            )

        self.get_logger().info("Home planning succeeded.")
        self.get_logger().info(
            f"Planning time: {response.planning_time:.3f} s"
        )
        self.get_logger().info(
            f"Trajectory points: {len(trajectory.points)}"
        )

        final_positions = [
            round(value, 4)
            for value in trajectory.points[-1].positions
        ]

        self.get_logger().info(
            f"Final planned joint positions: {final_positions}"
        )

        return response.trajectory

    def execute(self, trajectory) -> bool:
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
            "Sending collision-aware Home trajectory..."
        )

        send_future = self.execute_client.send_goal_async(goal)

        rclpy.spin_until_future_complete(
            self,
            send_future,
            timeout_sec=10.0,
        )

        if not send_future.done() or send_future.result() is None:
            self.get_logger().error(
                "Execution request timed out."
            )
            return False

        goal_handle = send_future.result()

        if not goal_handle.accepted:
            self.get_logger().error(
                "Home trajectory was rejected."
            )
            return False

        self.get_logger().info(
            "Home trajectory accepted."
        )

        result_future = goal_handle.get_result_async()

        rclpy.spin_until_future_complete(
            self,
            result_future,
            timeout_sec=40.0,
        )

        if not result_future.done() or result_future.result() is None:
            self.get_logger().error(
                "Home trajectory execution timed out."
            )
            return False

        result = result_future.result().result

        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                "Home execution failed. "
                f"MoveIt error code: {result.error_code.val}"
            )
            return False

        self.get_logger().info(
            "Collision-aware return Home succeeded."
        )
        return True


def main() -> None:
    rclpy.init()
    node = ReturnHome()

    try:
        trajectory = node.plan()
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
