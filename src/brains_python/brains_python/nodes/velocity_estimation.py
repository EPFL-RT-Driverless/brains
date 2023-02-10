#  Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import rclpy

from brains_custom_interfaces.msg import IMUData, GSSData, WSSData, VelocityEstimation
from brains_python.common import sleep
from .multi_subscription_node import MultiSubscriptionMixin


class VelocityEstimationNode(MultiSubscriptionMixin):
    def __init__(self):
        super().__init__(
            node_name="VelocityEstimation",
            subconfig=[
                {"topic": "imu_data", "msg_type": IMUData, "queue_size": 10},
                {"topic": "wss_data", "msg_type": WSSData, "queue_size": 10},
                {"topic": "gss_data", "msg_type": GSSData, "queue_size": 10},
            ],
        )
        self.vel_est_publisher = self.create_publisher(
            VelocityEstimation, "velocity_estimation", 10
        )

    def processing(self, imu_data: IMUData, wss_data: WSSData, gss_data: GSSData):
        self.get_logger().info("Creating Velocity estimation")
        msg = VelocityEstimation()
        sleep(0.002)
        self.vel_est_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VelocityEstimationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
