import math

import rclpy
from rclpy.node import Node

from tf2_ros import Buffer, TransformListener


class EndEffectorPoseListener(Node):
    def __init__(self):
        super().__init__('end_effector_pose_listener')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(1.0, self.timer_callback)

        self.base_frame = 'panda_link0'
        self.ee_frame = 'panda_link8'

    def timer_callback(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rclpy.time.Time()
            )

            t = transform.transform.translation
            q = transform.transform.rotation

            roll, pitch, yaw = self.quaternion_to_rpy(q.x, q.y, q.z, q.w)

            roll_deg = math.degrees(roll)
            pitch_deg = math.degrees(pitch)
            yaw_deg = math.degrees(yaw)

            if t.z > 0.635:
                height_status = 'LIFT_LEVEL'
            else:
                height_status = 'GRASP_LEVEL'

            self.get_logger().info(
                f'Position: x={t.x:.3f}, y={t.y:.3f}, z={t.z:.3f} m | '
                f'RPY: roll={roll_deg:.1f}°, pitch={pitch_deg:.1f}°, yaw={yaw_deg:.1f}° | '
                f'Height status: {height_status}'
            )

        except Exception as e:
            self.get_logger().warn(f'Could not get transform: {e}')

    def quaternion_to_rpy(self, x, y, z, w):
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)

    node = EndEffectorPoseListener()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
