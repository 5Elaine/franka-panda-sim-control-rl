import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import GripperCommand


class PandaGripperControl(Node):
    def __init__(self):
        super().__init__('panda_gripper_control')

        self.action_client = ActionClient(
            self,
            GripperCommand,
            '/panda_hand_controller/gripper_cmd'
        )

    def send_gripper_goal(self, position, max_effort):
        self.get_logger().info('Waiting for gripper action server...')
        self.action_client.wait_for_server()

        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = max_effort

        self.get_logger().info(
            f'Sending gripper goal: position={position}, max_effort={max_effort}'
        )

        send_goal_future = self.action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Gripper goal rejected.')
            rclpy.shutdown()
            return

        self.get_logger().info('Gripper goal accepted.')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(
            f'Gripper finished. reached_goal={result.reached_goal}, '
            f'position={result.position}, effort={result.effort}'
        )
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    node = PandaGripperControl()

    # 先测试打开夹爪
    node.send_gripper_goal(position=0.04, max_effort=10.0)

    rclpy.spin(node)


if __name__ == '__main__':
    main()
