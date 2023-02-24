import rclpy
from rclpy.node import Node

from brains_custom_interfaces.msg import VelocityEstimation


class FSDSVelocityEstimation(Node):
    def __init__(self):
        super().__init__("FSDSVelocityEstimation")
        self.publisher = self.create_publisher(
            VelocityEstimation, "velocity_estimation", 10
        )
        self.timer = self.create_timer(0.01, self.callback)

    def callback(self):
        self.get_logger().info("Creating velocity estimation")
        self.publisher.publish(VelocityEstimation())


def main(args=None):
    rclpy.init(args=args)
    node = FSDSVelocityEstimation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
