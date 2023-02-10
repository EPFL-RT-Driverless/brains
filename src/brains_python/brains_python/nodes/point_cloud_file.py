import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2


class PointCloudFile(Node):
    def __init__(self):
        super().__init__("PointCloudFile")
        self.point_cloud_publisher = self.create_publisher(
            PointCloud2, "point_cloud", 10
        )
        self.point_cloud_timer = self.create_timer(0.05, self.callback)

    def callback(self):
        self.get_logger().info("Creating point cloud")
        self.point_cloud_publisher.publish(PointCloud2())


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudFile()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
