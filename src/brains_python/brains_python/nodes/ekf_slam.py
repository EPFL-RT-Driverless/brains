#  Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import rclpy

from brains_custom_interfaces.msg import (
    ConesObservations,
    VelocityEstimation,
    Pose2D,
    CenterLineWidths,
)
from brains_python.common import sleep
from .multi_subscription_node import MultiSubscriptionMixin


class EKFSLAMNode(MultiSubscriptionMixin):
    def __init__(self):
        super().__init__(
            node_name="EKFSLAM",
            subconfig=[
                {
                    "topic": "velocity_estimation",
                    "msg_type": VelocityEstimation,
                    "queue_size": 10,
                },
                {
                    "topic": "cones_observations",
                    "msg_type": ConesObservations,
                    "queue_size": 10,
                },
            ],
        )
        self.pose_publisher = self.create_publisher(Pose2D, "pose", 10)
        self.center_line_widths_publisher = self.create_publisher(
            CenterLineWidths, "center_line_widths", 10
        )

    def processing(
        self,
        velocity_estimation: VelocityEstimation,
        cones_observations: ConesObservations,
    ):
        self.get_logger().info("running EKF SLAM")
        sleep(0.01)
        self.pose_publisher.publish(Pose2D())
        self.center_line_widths_publisher.publish(CenterLineWidths())


def main(args=None):
    rclpy.init(args=args)
    node = EKFSLAMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
