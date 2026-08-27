import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from tf2_ros import Buffer, TransformListener
from moveit_msgs.srv import GetPositionIK
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class IKLiftAndExecute(Node):
    def __init__(self):
        super().__init__('ik_lift_and_execute')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')

        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/panda_arm_controller/follow_joint_trajectory'
        )

        self.base_frame = 'panda_link0'
        self.ee_frame = 'panda_link8'

        self.joint_names = [
            'panda_joint1',
            'panda_joint2',
            'panda_joint3',
            'panda_joint4',
            'panda_joint5',
            'panda_joint6',
            'panda_joint7',
        ]

        self.has_sent_ik_request = False
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        if self.has_sent_ik_request:
            return

        if not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /compute_ik service...')
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rclpy.time.Time()
            )

            t = transform.transform.translation
            q = transform.transform.rotation

            target_z = t.z + 0.05

            request = GetPositionIK.Request()

            request.ik_request.group_name = 'panda_arm'
            request.ik_request.ik_link_name = self.ee_frame
            request.ik_request.avoid_collisions = True
            request.ik_request.timeout.sec = 2

            request.ik_request.pose_stamped.header.frame_id = self.base_frame
            request.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()

            request.ik_request.pose_stamped.pose.position.x = t.x
            request.ik_request.pose_stamped.pose.position.y = t.y
            request.ik_request.pose_stamped.pose.position.z = target_z

            request.ik_request.pose_stamped.pose.orientation.x = q.x
            request.ik_request.pose_stamped.pose.orientation.y = q.y
            request.ik_request.pose_stamped.pose.orientation.z = q.z
            request.ik_request.pose_stamped.pose.orientation.w = q.w

            self.get_logger().info(
                f'Sending IK request for lifted target: '
                f'x={t.x:.3f}, y={t.y:.3f}, z={target_z:.3f}'
            )

            future = self.ik_client.call_async(request)
            future.add_done_callback(self.ik_response_callback)

            self.has_sent_ik_request = True

        except Exception:
            self.get_logger().info(
                f'Waiting for TF transform from {self.base_frame} to {self.ee_frame}...'
            )

    def ik_response_callback(self, future):
        response = future.result()

        self.get_logger().info(f'IK error code: {response.error_code.val}')

        if response.error_code.val != 1:
            self.get_logger().info('IK failed. No trajectory will be sent.')
            rclpy.shutdown()
            return

        joint_state = response.solution.joint_state

        ik_positions = []

        for joint_name in self.joint_names:
            if joint_name in joint_state.name:
                index = joint_state.name.index(joint_name)
                ik_positions.append(joint_state.position[index])
            else:
                self.get_logger().info(f'Missing joint in IK solution: {joint_name}')
                rclpy.shutdown()
                return

        formatted_positions = ', '.join([f'{p:.4f}' for p in ik_positions])
        self.get_logger().info(f'IK positions used for execution: [{formatted_positions}]')

        self.send_arm_goal(ik_positions)

    def send_arm_goal(self, positions):
        self.get_logger().info('Waiting for arm action server...')
        self.arm_client.wait_for_server()

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = 3

        goal_msg.trajectory.points.append(point)

        self.get_logger().info('Sending trajectory generated from IK...')

        future = self.arm_client.send_goal_async(goal_msg)
        future.add_done_callback(self.arm_goal_response_callback)

    def arm_goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Arm goal rejected by controller.')
            rclpy.shutdown()
            return

        self.get_logger().info('Arm goal accepted.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.arm_result_callback)

    def arm_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Arm motion finished with error code: {result.error_code}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    node = IKLiftAndExecute()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
