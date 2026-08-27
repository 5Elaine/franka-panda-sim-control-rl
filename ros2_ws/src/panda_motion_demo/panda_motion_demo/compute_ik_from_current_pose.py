import rclpy
from rclpy.node import Node

from tf2_ros import Buffer, TransformListener
from moveit_msgs.srv import GetPositionIK


class ComputeIKFromCurrentPose(Node):
    def __init__(self):
        super().__init__('compute_ik_from_current_pose')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')

        self.base_frame = 'panda_link0'
        self.ee_frame = 'panda_link8'

        self.timer = self.create_timer(1.0, self.timer_callback)
        self.has_sent_request = False

    def timer_callback(self):
        if self.has_sent_request:
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

            request = GetPositionIK.Request()

            request.ik_request.group_name = 'panda_arm'
            request.ik_request.ik_link_name = self.ee_frame
            request.ik_request.avoid_collisions = True
            request.ik_request.timeout.sec = 2

            request.ik_request.pose_stamped.header.frame_id = self.base_frame
            request.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()

            target_z = t.z + 0.05

            request.ik_request.pose_stamped.pose.position.x = t.x
            request.ik_request.pose_stamped.pose.position.y = t.y
            request.ik_request.pose_stamped.pose.position.z = target_z

            request.ik_request.pose_stamped.pose.orientation.x = q.x
            request.ik_request.pose_stamped.pose.orientation.y = q.y
            request.ik_request.pose_stamped.pose.orientation.z = q.z
            request.ik_request.pose_stamped.pose.orientation.w = q.w

            self.get_logger().info(
                f'Sending IK request for lifted pose: '
                f'x={t.x:.3f}, y={t.y:.3f}, z={target_z:.3f}'
            )

            future = self.ik_client.call_async(request)
            future.add_done_callback(self.ik_response_callback)

            self.has_sent_request = True

        except Exception:
            self.get_logger().info(
                f'Waiting for TF transform from {self.base_frame} to {self.ee_frame}...'
            )

    def ik_response_callback(self, future):
        response = future.result()

        self.get_logger().info(f'IK error code: {response.error_code.val}')

        joint_state = response.solution.joint_state

        self.get_logger().info('IK solution joint names and positions:')

        panda_joint_positions = []

        for name, position in zip(joint_state.name, joint_state.position):
            if name.startswith('panda_joint'):
                self.get_logger().info(f'{name}: {position:.4f}')
                panda_joint_positions.append(position)

        formatted_positions = ', '.join([f'{p:.4f}' for p in panda_joint_positions])
        self.get_logger().info(f'IK positions list: [{formatted_positions}]')

        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    node = ComputeIKFromCurrentPose()

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
