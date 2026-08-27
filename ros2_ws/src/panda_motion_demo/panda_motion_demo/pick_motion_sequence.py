import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class PandaPickMotionSequence(Node):
    def __init__(self):
        super().__init__('panda_pick_motion_sequence')

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

        self.poses = {
            'home': [
                0.0,
                -0.785,
                0.0,
                -2.356,
                0.0,
                1.571,
                0.785,
            ],
            'pre_grasp': [
                0.2,
                -0.7,
                0.1,
                -2.1,
                0.0,
                1.4,
                0.6,
            ],
            'grasp': [
                0.25,
                -0.45,
                0.1,
                -2.0,
                0.0,
                1.35,
                0.55,
            ],
            'lift': [
                0.25,
                -0.85,
                0.1,
                -2.2,
                0.0,
                1.55,
                0.55,
            ],
        }

        self.sequence = [
            'home',
            'pre_grasp',
            'grasp',
            'lift',
            'home',
        ]

        self.current_index = 0

    def send_current_pose(self):
        pose_name = self.sequence[self.current_index]
        positions = self.poses[pose_name]

        self.get_logger().info(f'Moving to pose: {pose_name}')

        self.action_client.wait_for_server()

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = 3

        goal_msg.trajectory.points.append(point)

        send_goal_future = self.action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected by controller.')
            rclpy.shutdown()
            return

        self.get_logger().info('Goal accepted.')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Motion finished with error code: {result.error_code}')

        self.current_index += 1

        if self.current_index < len(self.sequence):
            self.send_current_pose()
        else:
            self.get_logger().info('Pick motion sequence finished.')
            rclpy.shutdown()

    def start(self):
        self.send_current_pose()


def main(args=None):
    rclpy.init(args=args)

    node = PandaPickMotionSequence()
    node.start()

    rclpy.spin(node)


if __name__ == '__main__':
    main()
