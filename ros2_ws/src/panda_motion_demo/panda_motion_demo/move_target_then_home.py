import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class PandaMoveTargetThenHome(Node):
    def __init__(self):
        super().__init__('panda_move_target_then_home')

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

        self.home_positions = [
            0.0,
            -0.785,
            0.0,
            -2.356,
            0.0,
            1.571,
            0.785,
        ]

        self.target_positions = [
            0.0,
            -0.5,
            0.0,
            -2.0,
            0.0,
            1.5,
            0.0,
        ]

        self.step = 'target'

    def send_trajectory(self, positions, duration_sec):
        self.get_logger().info('Waiting for action server...')
        self.action_client.wait_for_server()

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = duration_sec

        goal_msg.trajectory.points.append(point)

        self.get_logger().info(f'Sending trajectory to {self.step}...')
        send_goal_future = self.action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected by controller.')
            rclpy.shutdown()
            return

        self.get_logger().info('Goal accepted by controller.')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Trajectory finished with error code: {result.error_code}')

        if self.step == 'target':
            self.get_logger().info('Target reached. Now moving back to home...')
            self.step = 'home'
            self.send_trajectory(self.home_positions, 3)

        elif self.step == 'home':
            self.get_logger().info('Home reached. Task finished.')
            rclpy.shutdown()

    def start(self):
        self.send_trajectory(self.target_positions, 3)


def main(args=None):
    rclpy.init(args=args)

    node = PandaMoveTargetThenHome()
    node.start()

    rclpy.spin(node)


if __name__ == '__main__':
    main()
