import rclpy
from rclpy.node import Node

from brains_custom_interfaces.msg import BoundingBoxes


class VideoFileAndYOLO(Node):
    def __init__(self):
        super().__init__("VideoFileAndYOLO")
        self.bounding_boxes_publisher = self.create_publisher(
            BoundingBoxes, "bounding_boxes", 10
        )
        self.bounding_boxes_timer = self.create_timer(0.05, self.callback)

    def callback(self):
        self.get_logger().info("Creating bounding boxes from video file")
        self.bounding_boxes_publisher.publish(BoundingBoxes())


def main(args=None):
    rclpy.init(args=args)
    node = VideoFileAndYOLO()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
