import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from control_msgs.action import GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint


class PandaPickDemo(Node):
    def __init__(self):
        super().__init__('panda_pick_demo')

        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/panda_arm_controller/follow_joint_trajectory'
        )

        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            '/panda_hand_controller/gripper_cmd'
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
            ('gripper', 'open'),
            ('arm', 'home'),
            ('arm', 'pre_grasp'),
            ('arm', 'grasp'),
            ('gripper', 'close'),
            ('arm', 'lift'),
            ('gripper', 'open'),
            ('arm', 'home'),
        ]

        self.current_index = 0

    def start(self):
        self.run_current_step()

    def run_current_step(self):
        if self.current_index >= len(self.sequence):
            self.get_logger().info('Pick demo finished.')
            rclpy.shutdown()
            return

        step_type, step_name = self.sequence[self.current_index]
        self.get_logger().info(f'Running step: {step_type} -> {step_name}')

        if step_type == 'arm':
            self.send_arm_goal(step_name)
        elif step_type == 'gripper':
            self.send_gripper_goal(step_name)

    def send_arm_goal(self, pose_name):
        self.arm_client.wait_for_server()

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = self.poses[pose_name]
        point.time_from_start.sec = 3

        goal_msg.trajectory.points.append(point)

        send_goal_future = self.arm_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.arm_goal_response_callback)

    def arm_goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Arm goal rejected.')
            rclpy.shutdown()
            return

        self.get_logger().info('Arm goal accepted.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.arm_result_callback)

    def arm_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Arm motion finished. error_code={result.error_code}')

        self.current_index += 1
        self.run_current_step()

    def send_gripper_goal(self, command_name):
        self.gripper_client.wait_for_server()

        goal_msg = GripperCommand.Goal()

        if command_name == 'open':
            goal_msg.command.position = 0.04
            goal_msg.command.max_effort = 10.0
        elif command_name == 'close':
            goal_msg.command.position = 0.0
            goal_msg.command.max_effort = 10.0

        send_goal_future = self.gripper_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.gripper_goal_response_callback)

    def gripper_goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Gripper goal rejected.')
            rclpy.shutdown()
            return

        self.get_logger().info('Gripper goal accepted.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.gripper_result_callback)

    def gripper_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(
            f'Gripper finished. reached_goal={result.reached_goal}, '
            f'position={result.position}'
        )

        self.current_index += 1
        self.run_current_step()


def main(args=None):
    rclpy.init(args=args)

    node = PandaPickDemo()
    node.start()

    rclpy.spin(node)


if __name__ == '__main__':
    main()
