import numpy as np
import rclpy
import track_database as tdb
from rclpy.node import Node

from std_msgs.msg import Header
from brains_custom_interfaces.msg import CarControls, CarState, ControlPhase
from brains_custom_interfaces.srv import RestartFSDS
from brains_python.common import Mission
from brains_python.control import (
    CarParams,
    MotionPlannerController,
    MotionPlannerParams,
    fsds_car_params,
)
from brains_python.control.ihm_acados import IHMAcadosParams, fsds_ihm_acados_params
from brains_python.control.stanley import StanleyParams, stanley_params_from_mission


class ControlOnly(Node):
    def __init__(self):
        super().__init__("control_only_node")
        # node parameters
        track_name = self.declare_parameter("track_name", "fsds_competition_2")
        if track_name.value == "acceleration":
            mission = Mission.ACCELERATION
        elif track_name.value == "short_skidpad":
            mission = Mission.SHORT_SKIDPAD
        elif track_name.value == "skidpad":
            mission = Mission.SKIDPAD
        else:
            mission = Mission.TRACKDRIVE

        lap_count = self.declare_parameter("lap_count", 10)
        freq = self.declare_parameter("freq", 20.0)
        self.dt = 1 / freq.value
        v_x_max = self.declare_parameter("v_x_max", 10.0)
        a_y_max = self.declare_parameter("a_y_max", 7.0)
        W = self.declare_parameter("W", 1.5)

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
        car_params.W = W.value
        car_params.v_x_max = v_x_max.value
        car_params.a_y_max = a_y_max.value
        self.motion_planner_controller = MotionPlannerController(
            car_params=car_params,
            racing_controller_params=IHMAcadosParams(**fsds_ihm_acados_params),
            stopping_controller_params=StanleyParams(
                **stanley_params_from_mission(mission)
            ),
            motion_planner_params=MotionPlannerParams(
                mission=mission,
                center_points=track.center_line,
                widths=track.track_widths,
                psi_s=np.pi / 2,
                psi_e=np.pi / 2,
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
        self.control_phase_pub = self.create_publisher(
            ControlPhase, "/brains/control_phase", 10
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
                start = self.get_clock().now()
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
                if (
                    self.motion_planner_controller.stopping
                    and np.abs(car_state.v_x) < 0.1
                ):
                    self.control_phase_pub.publish(
                        ControlPhase(
                            header=Header(stamp=self.get_clock().now().to_msg()),
                            phase=ControlPhase.STOPPED,
                        )
                    )
                    # TODO: send special hand brake command ?
                    self.get_logger().info("Stopped the car")
                    raise RuntimeError("Stopped the car")

                self.control_phase_pub.publish(
                    ControlPhase(
                        header=Header(stamp=self.get_clock().now().to_msg()),
                        phase=ControlPhase.STOPPING
                        if self.motion_planner_controller.stopping
                        else ControlPhase.RACING,
                    )
                )

                self.last_control = control_result.control
                stop = self.get_clock().now()

                self.get_logger().info(
                    f"Control computation took {(stop.nanoseconds - start.nanoseconds) / 1e6} ms"
                )

                self.car_controls_pub.publish(
                    CarControls(
                        header=Header(stamp=self.get_clock().now().to_msg()),
                        throttle=control_result.control[0] + np.random.rand() * 0.1,
                        steering=control_result.control[1] + np.random.rand() * 0.1,
                        throttle_rate=control_result.control_derivative[0],
                        steering_rate=control_result.control_derivative[1],
                    )
                )


def main(args=None):
    rclpy.init(args=args)
    node = ControlOnly()
    try:
        rclpy.spin(node)
    # except KeyboardInterrupt:
    #     node.get_logger().info("KeyboardInterrupt")
    #     node.destroy_node()
    except RuntimeError:
        node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
