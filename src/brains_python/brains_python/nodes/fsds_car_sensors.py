import rclpy
from rclpy.node import Node

from brains_custom_interfaces.msg import IMUData, GSSData, WSSData


class FSDSCarSensors(Node):
    def __init__(self):
        super().__init__("FSDSCarSensors")
        self.imu_publisher = self.create_publisher(IMUData, "imu_data", 10)
        self.gss_publisher = self.create_publisher(GSSData, "gss_data", 10)
        self.wss_publisher = self.create_publisher(WSSData, "wss_data", 10)
        self.imu_timer = self.create_timer(0.01, self.imu_callback)
        self.gss_timer = self.create_timer(0.01, self.gss_callback)
        self.wss_timer = self.create_timer(0.01, self.wss_callback)

    def imu_callback(self):
        self.get_logger().info("Creating IMU data")
        self.imu_publisher.publish(IMUData())

    def gss_callback(self):
        self.get_logger().info("Creating GSS data")
        self.gss_publisher.publish(GSSData())

    def wss_callback(self):
        self.get_logger().info("Creating WSS data")
        self.wss_publisher.publish(WSSData())


def main(args=None):
    rclpy.init(args=args)
    node = FSDSCarSensors()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
