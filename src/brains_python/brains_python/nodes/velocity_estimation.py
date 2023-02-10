#  Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import rclpy

from brains_custom_interfaces.msg import IMUData, GSSData, WSSData, VelocityEstimation
from .multi_subscription_node import MultiSubscriptionNode


class VelocityEstimationNode(MultiSubscriptionNode):
    def __init__(self):
        super().__init__(
            "VelocityEstimation",
            [
                {"topic": "imu_data", "msg_type": IMUData, "queue_size": 10},
                {"topic": "wss_data", "msg_type": WSSData, "queue_size": 10},
                {"topic": "gss_data", "msg_type": GSSData, "queue_size": 10},
            ],
            {
                "topic": "velocity_estimation",
                "msg_type": VelocityEstimation,
                "queue_size": 10,
            },
        )

    def processing(self, *args, **kwargs):
        self.get_logger().info("Processing")
        msg = VelocityEstimation()
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VelocityEstimationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
