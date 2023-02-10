#  Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import rclpy

from brains_custom_interfaces.msg import (
    CarControls,
)
from .multi_subscription_node import MultiSubscriptionMixin


class ControlKnownTrackNode(MultiSubscriptionMixin):
    def __init__(self):
        super().__init__(
            node_name="ControlKnownTrack",
            subconfig=[
                {"topic": "car_controls", "msg_type": CarControls, "queue_size": 10}
            ],
        )

    def processing(
        self,
        car_controls: CarControls,
    ):
        self.get_logger().info("received car controls")


def main(args=None):
    rclpy.init(args=args)
    node = ControlKnownTrackNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
