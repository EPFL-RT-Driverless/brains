import numpy as np
import rclpy
from rclpy.node import Node
from brains_custom_interfaces.msg import CarState, CarControls
from brains_python.control import (
    MotionPlannerParams,
    MotionPlannerController,
    CarParams,
    fsds_car_params,
)
from brains_python.control.ihm_acados import IHMAcadosParams, fsds_ihm_acados_params
from brains_python.control.stanley import StanleyParams, stanley_params_from_mission
from brains_python.common import Mission
import track_database as tdb
from brains_custom_interfaces.srv import RestartFSDS, EnableApiFSDS


class ControlOnly(Node):
    def __init__(self):
        super().__init__("control_only_node")
        track_name = self.declare_parameter("track_name", "fsds_competition_2")

        restart_client = self.create_client(RestartFSDS, "/fsds/restart")
        enable_api_client = self.create_client(EnableApiFSDS, "/fsds/enable_api")

        while not restart_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("restart service not available, waiting again...")
        while not enable_api_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("enable API service not available, waiting again...")
        restart_client.call(RestartFSDS.Request())
        self.get_logger().info("Restarted FSDS")
        enable_api_client.call(EnableApiFSDS.Request(enabled=True))
        self.get_logger().info("Enabled API")

        track = tdb.load_track(track_name.value)
        self.motion_planner_controller = MotionPlannerController(
            car_params=CarParams(**fsds_car_params),
            racing_controller_params=IHMAcadosParams(**fsds_ihm_acados_params),
            stopping_controller_params=StanleyParams(
                **stanley_params_from_mission(Mission.TRACKDRIVE)
            ),
            motion_planner_params=MotionPlannerParams(
                mission=Mission.TRACKDRIVE,
                center_points=track.center_line,
                widths=track.track_widths,
                psi_s=np.pi / 2,
                psi_e=np.pi / 2,
                closed=True,
                additional_attributes=[],
            ),
        )
        self.last_control = np.zeros(2)

        self.car_state_sub = self.create_subscription(
            CarState, "/fsds/car_state", self.callback, 10
        )
        self.car_controls_pub = self.create_publisher(
            CarControls, "/fsds/car_controls", 10
        )

    def callback(self, car_state: CarState):
        control_result = self.motion_planner_controller.compute_control(
            current_state=np.array(
                [
                    car_state.x,
                    car_state.y,
                    car_state.phi,
                    car_state.v_x,
                    car_state.v_y,
                    car_state.r,
                ]
            ),
            current_control=self.last_control,
        )
        self.last_control = control_result.control
        msg = CarControls()
        msg.throttle = control_result.control[0]
        msg.steering = control_result.control[1]
        msg.throttle_rate = control_result.control_derivative[0]
        msg.steering_rate = control_result.control_derivative[1]
        self.car_controls_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ControlOnly()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
