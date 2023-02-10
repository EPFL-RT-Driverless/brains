import time

import rclpy
from rclpy.node import Node

from brains_custom_interfaces.msg import IMUData, GSSData, WSSData


def sleep(s):
    start = time.perf_counter()
    while time.perf_counter() - start < s:
        pass


class FSDSCarSensors(Node):
    def __init__(self):
        super().__init__("FSDSCarSensors")
        self.imu_publisher = self.create_publisher(IMUData, "imu_data", 10)
        self.gss_publisher = self.create_publisher(GSSData, "gss_data", 10)
        self.wss_publisher = self.create_publisher(WSSData, "wss_data", 10)
        self.timer = self.create_timer(0.01, self.timer_callback)

    def timer_callback(self):
        # self.get_logger().info("Publishing IMUData")
        self.imu_publisher.publish(IMUData())
        sleep(0.01)
        # self.get_logger().info("Publishing GSSData")
        self.gss_publisher.publish(GSSData())
        sleep(0.01)
        # self.get_logger().info("Publishing WSSData")
        self.wss_publisher.publish(WSSData())


def main(args=None):
    rclpy.init(args=args)
    node = FSDSCarSensors()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
