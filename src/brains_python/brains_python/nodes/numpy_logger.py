import os

import rclpy
from brains_custom_interfaces.msg import CarState, CarControls, CarControlsPrediction
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class NumpyLogger(Node):
    def __init__(self):
        super().__init__("control_only_node")
        self.log_path = self.declare_parameter(
            "log_path",
            os.environ.get("BRAINS_SOURCE_DIR", "/tmp") + "/state_control_log.csv",
        )
        self.car_state_sub = Subscriber(
            self, CarState, "/fsds/car_state", qos_profile=10
        )
        self.car_controls_sub = Subscriber(
            self, CarControls, "/brains/car_controls", qos_profile=10
        )
        self.car_control_predictions_pub = self.create_publisher(
            CarControlsPrediction, "/car_control_predictions", 10
        )
        self.ats = ApproximateTimeSynchronizer(
            [self.car_state_sub, self.car_controls_sub], queue_size=30, slop=0.02
        )
        self.ats.registerCallback(self.callback)
        # write header to CSV file
        with open(os.path.abspath(self.log_path.value), "w") as f:
            f.write("timestamp, car_state, car_controls")

    def callback(self, car_state: CarState, car_controls: CarControls):
        # open CSV file, write to it in the format of:
        # timestamp, car_state, car_controls
        # and close the file
        start_time = self.get_clock().now().nanoseconds
        with open(self.log_path.value, "a") as f:
            f.write(
                f"{car_state.header.stamp}, {car_state.x}, {car_state.y}, {car_state.phi}, {car_state.v_x}, {car_state.v_y}, {car_state.r}, {car_controls.throttle}, {car_controls.steering}, {car_controls.throttle_rate}, {car_controls.steering_rate}"
            )
        end_time = self.get_clock().now().nanoseconds
        self.get_logger().info(
            f"Callback took {(end_time - start_time) * 1000000} ms to execute."
        )


def main(args=None):
    rclpy.init(args=args)
    control_only_node = NumpyLogger()
    rclpy.spin(control_only_node)
    control_only_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
