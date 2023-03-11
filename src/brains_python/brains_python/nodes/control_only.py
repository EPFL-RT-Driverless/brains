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
from std_msgs.msg import Empty


class ControlOnly(Node):
    def __init__(self):
        super().__init__("control_only_node")
        # node parameters
        track_name = self.declare_parameter("track_name", "fsds_competition_2")
        lap_count = self.declare_parameter("lap_count", 10)
        freq = self.declare_parameter("freq", 20.0)
        self.dt = 1 / freq.value

        # restart FSDS
        restart_client = self.create_client(RestartFSDS, "/fsds/restart")
        while not restart_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("restart service not available, waiting again...")
        future = restart_client.call_async(RestartFSDS.Request())
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info("Restarted FSDS")

        # load track
        track = tdb.load_track(track_name.value)
        car_params = CarParams(**fsds_car_params)
        car_params.W = 1.5
        self.motion_planner_controller = MotionPlannerController(
            car_params=car_params,
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
            max_lap_count=lap_count.value,
        )
        self.last_control = np.zeros(2)
        self.last_control_stamp = 0.0

        # declare publishers and subscribers
        self.car_state_sub = self.create_subscription(
            CarState, "/fsds/car_state", self.callback, 10
        )
        self.car_controls_pub = self.create_publisher(
            CarControls, "/fsds/car_controls", 10
        )

    def callback(self, car_state: CarState):
        if self.last_control_stamp == 0.0:
            self.last_control_stamp = (
                car_state.header.stamp.sec + car_state.header.stamp.nanosec * 1e-9
            )
        else:
            dt = (
                car_state.header.stamp.sec
                + car_state.header.stamp.nanosec * 1e-9
                - self.last_control_stamp
            )
            if dt >= self.dt * 0.9:
                self.last_control_stamp = (
                    car_state.header.stamp.sec + car_state.header.stamp.nanosec * 1e-9
                )
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
                msg.header.stamp = self.get_clock().now().to_msg()
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
