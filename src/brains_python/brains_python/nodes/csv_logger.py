import os
from typing import Optional
import numpy as np

import rclpy
from brains_custom_interfaces.msg import CarState, CarControls, CarControlsPrediction
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class CSVLogger(Node):
    def __init__(self):
        super().__init__("control_only_node")
        self.log_path = self.declare_parameter(
            "log_path",
            os.environ.get("BRAINS_SOURCE_DIR", "/tmp") + "/state_control_log.csv",
        )
        self.get_logger().info(f"Logging to {os.path.abspath(self.log_path.value)}")

        self.state_sub = self.create_subscription(
            CarState,
            "/fsds/car_state",
            self.state_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self.controls_sub = self.create_subscription(
            CarControls,
            "/fsds/car_controls",
            self.controls_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self.last_controls = CarControls()

        # write header to CSV file
        with open(os.path.abspath(self.log_path.value), "w") as f:
            f.write("timestamp,X,Y,phi,v_x,v_y,r,T,delta,dT,ddelta\n")

    @staticmethod
    def msgs_to_str(car_state: Optional[CarState], car_controls: Optional[CarControls]):
        assert car_state is not None or car_controls is not None
        if car_state is None:
            car_state_str = ",,,,,"
        else:
            car_state_str = f"{car_state.x:.5f}, {car_state.y:.5f}, {car_state.phi:.5f}, {car_state.v_x:.5f}, {car_state.v_y:.5f}, {car_state.r:.5f}"
        if car_controls is None:
            car_controls_str = ",,,"
        else:
            car_controls_str = f"{car_controls.throttle:.5f}, {car_controls.steering:.5f}, {car_controls.throttle_rate:.5f}, {car_controls.steering_rate:.5f}"
        return (
            (
                f"{car_state.header.stamp.sec + car_state.header.stamp.nanosec/1e9:.3f}"
                if car_state is not None
                else f"{car_controls.header.stamp.sec + car_controls.header.stamp.nanosec/1e9:.3f}"
            )
            + ", "
            + car_state_str
            + ", "
            + car_controls_str
            + "\n"
        )

    def state_callback(self, car_state: CarState):
        # start = self.get_clock().now()
        with open(self.log_path.value, "a") as f:
            f.write(CSVLogger.msgs_to_str(car_state, self.last_controls))
        # stop = self.get_clock().now()
        # self.get_logger().info(f"Writing to CSV took {(stop.nanoseconds - start.nanoseconds) / 1e6} ms")

    def controls_callback(self, car_controls: CarControls):
        self.last_controls = car_controls


def main(args=None):
    rclpy.init(args=args)
    control_only_node = CSVLogger()
    rclpy.spin(control_only_node)
    control_only_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
