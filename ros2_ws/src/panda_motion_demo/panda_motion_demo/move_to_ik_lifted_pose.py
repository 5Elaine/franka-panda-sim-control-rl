import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class MoveToIKLiftedPose(Node):
    def __init__(self):
        super().__init__('move_to_ik_lifted_pose')

        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/panda_arm_controller/follow_joint_trajectory'
        )

        self.joint_names = [
            'panda_joint1',
            'panda_joint2',
            'panda_joint3',
            'panda_joint4',
            'panda_joint5',
            'panda_joint6',
            'panda_joint7',
        ]

        self.ik_positions = [
            -0.0000,
            -0.7432,
            -0.0000,
            -2.1996,
            -0.0000,
            1.4565,
            0.7850,
        ]

    def send_goal(self):
        self.get_logger().info('Waiting for arm action server...')
        self.action_client.wait_for_server()

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = self.ik_positions
        point.time_from_start.sec = 3

        goal_msg.trajectory.points.append(point)

        self.get_logger().info('Sending trajectory to IK lifted pose...')

        send_goal_future = self.action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected by controller.')
            rclpy.shutdown()
            return

        self.get_logger().info('Goal accepted by controller.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Motion finished with error code: {result.error_code}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    node = MoveToIKLiftedPose()
    node.send_goal()

    rclpy.spin(node)


if __name__ == '__main__':
    main()
