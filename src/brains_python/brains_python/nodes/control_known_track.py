#  Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import rclpy

from brains_custom_interfaces.msg import (
    CarControls,
    ControlPrediction,
    VelocityEstimation,
    Pose2D,
    CenterLineWidths,
)
from brains_python.common import sleep
from .multi_subscription_node import MultiSubscriptionMixin


class ControlKnownTrackNode(MultiSubscriptionMixin):
    def __init__(self):
        super().__init__(
            node_name="ControlKnownTrack",
            subconfig=[
                {
                    "topic": "center_line_widths",
                    "msg_type": CenterLineWidths,
                    "queue_size": 10,
                },
                {"topic": "pose", "msg_type": Pose2D, "queue_size": 10},
                {
                    "topic": "velocity_estimation",
                    "msg_type": VelocityEstimation,
                    "queue_size": 10,
                },
            ],
        )
        self.car_controls_publisher = self.create_publisher(
            CarControls, "car_controls", 10
        )
        self.control_prediction_publisher = self.create_publisher(
            ControlPrediction, "control_prediction", 10
        )

    def processing(
        self,
        center_line_widths: CenterLineWidths,
        pose: Pose2D,
        velocity_estimation: VelocityEstimation,
    ):
        self.get_logger().info("Creating control on known track")
        sleep(0.05)
        self.car_controls_publisher.publish(CarControls())
        self.control_prediction_publisher.publish(ControlPrediction())


def main(args=None):
    rclpy.init(args=args)
    node = ControlKnownTrackNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
