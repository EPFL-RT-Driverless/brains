import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from brains_python.vision.camera_only import ConeObserverCameraOnly
from brains_custom_interfaces.msg import ConesObservations
from std_msgs.msg import Header


class CameraOnlyNode(Node):
    def __init__(self):
        super().__init__("cone_detection_node")
        camera_name = self.declare_parameter("camera_name", "camera").value
        camera_matrix_name = self.declare_parameter(
            "camera_matrix_name", "SimuMatrixReg"
        ).value
        self.subscription = self.create_subscription(
            Image, "/fsds/camera/" + camera_name, self.callback, 10
        )
        self.pub = self.create_publisher(
            ConesObservations, "/brains/cones_observations", 10
        )
        self.cone_observer = ConeObserverCameraOnly(self)

    def callback(self, msg: Image):
        # TODO: make sure that the image is in the right format
        start = self.get_clock().now()
        rho, theta, c = self.cone_observer.get_cones_observations(msg.data)
        stop = self.get_clock().now()
        self.get_logger().info(
            f"Processing time: {(stop.nanoseconds - start.nanoseconds) / 1e6} ms"
        )
        self.pub.publish(
            ConesObservations(
                header=Header(stamp=msg.header.stamp),
                rho=rho,
                theta=theta,
                colors=c,
            )
        )


def main(args=None):
    rclpy.init(args=args)
    cone_detection_node = CameraOnlyNode()
    rclpy.spin(cone_detection_node)
    cone_detection_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
