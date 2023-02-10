#  Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import rclpy

from brains_custom_interfaces.msg import BoundingBoxes, RelConesPositions
from sensor_msgs.msg import PointCloud2
from brains_python.common import sleep
from .multi_subscription_node import MultiSubscriptionMixin


class VisionFusionNode(MultiSubscriptionMixin):
    def __init__(self):
        super().__init__(
            node_name="VelocityEstimation",
            subconfig=[
                {"topic": "point_cloud", "msg_type": PointCloud2, "queue_size": 10},
                {
                    "topic": "bounding_boxes",
                    "msg_type": BoundingBoxes,
                    "queue_size": 10,
                },
            ],
        )
        self.rel_cones_positions_publisher = self.create_publisher(
            RelConesPositions, "rel_cones_positions", 10
        )

    def processing(self, point_cloud: PointCloud2, bounding_boxes: BoundingBoxes):
        self.get_logger().info("running vision fusion")
        msg = RelConesPositions()
        sleep(0.05)
        self.rel_cones_positions_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VisionFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
