import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimplePandaNode(Node):
    def __init__(self):
        super().__init__('simple_panda_node')

        self.publisher_ = self.create_publisher(
            String,
            'panda_demo_status',
            10
        )

        self.counter = 0

        self.timer = self.create_timer(
            1.0,
            self.timer_callback
        )

        self.get_logger().info('Simple Panda node with publisher is running!')

    def timer_callback(self):
        msg = String()
        msg.data = f'Panda demo node is alive: {self.counter}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = SimplePandaNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
