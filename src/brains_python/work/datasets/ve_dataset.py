import numpy as np

from brains_python import Mission
from brains_python.control import (
    MotionPlannerController,
    fsds_car_params,
    CarParams,
    MotionPlannerParams,
    StanleyParams,
    stanley_params_from_mission,
)
from brains_python.control.utils import SimulationMode, ClosedLoopRun
from common import missions_tracks, max_longitudinal_speeds
from track_database import load_track


def bruh(mission: Mission, track_name: str, v_x_max: float, csv: bool = False):
    track = load_track(track_name)
    car_params = CarParams(**fsds_car_params)
    car_params.v_x_max = v_x_max

    motion_planner_controller_instance = MotionPlannerController(
        car_params=car_params,
        racing_controller_params=StanleyParams(**stanley_params_from_mission(mission)),
        # racing_controller_params=IHMAcadosParams(**fsds_ihm_acados_params),
        stopping_controller_params=StanleyParams(
            **stanley_params_from_mission(mission)
        ),
        motion_planner_params=MotionPlannerParams(
            mission=mission,
            center_points=track.center_line,
            widths=track.track_widths,
            additional_attributes=[],
        ),
        max_lap_count=1,
    )

    instance = ClosedLoopRun(
        mission=mission,
        track=track,
        car_params=car_params,
        motion_planner_controller=motion_planner_controller_instance,
        sampling_time=0.01,
        simulation_mode=SimulationMode.SIMIL,
        max_time=50.0,
        delay=0.0,
        verbosity_level=127,
    )
    imu_linear_accelerations = []
    imu_angular_velocities = []
    wheel_speeds = []
    gss_velocities = []

    def callback(s: ClosedLoopRun):
        imu_data = s.fsds_client.low_level_client.getImuData()
        imu_linear_accelerations.append(imu_data.linear_acceleration.to_numpy_array())
        imu_angular_velocities.append(imu_data.angular_velocity.to_numpy_array())
        wss_data = s.fsds_client.low_level_client.simGetWheelStates()
        wheel_speeds.append(
            [wss_data.fl_rpm, wss_data.fr_rpm, wss_data.rl_rpm, wss_data.rr_rpm]
        )
        gss_data = s.fsds_client.low_level_client.getGroundSpeedSensorData()
        gss_velocities.append(gss_data.linear_velocity.to_numpy_array())

    instance.submit_callback(callback)
    instance.run()

    imu_linear_accelerations = np.array(imu_linear_accelerations)
    imu_angular_velocities = np.array(imu_angular_velocities)
    wheel_speeds = np.array(wheel_speeds)
    gss_velocities = np.array(gss_velocities)
    if csv:
        # noinspection PyTypeChecker
        np.savetxt(
            f"data/ve_dataset/{track_name}_{v_x_max}.csv",
            np.hstack(
                (
                    instance.states[:, 3:5]
                    + np.random.multivariate_normal(
                        np.zeros(2), np.diag([1.0, 0.1]) ** 2, instance.states.shape[0]
                    ),
                    imu_angular_velocities[:, 2],
                    imu_linear_accelerations[:, 2],
                    gss_velocities[:, 2],
                    wheel_speeds,
                    np.tile(
                        np.array([0.2, 0.2, 0.3, 0.3]), (instance.states.shape[0], 1)
                    )
                    * instance.controls[:, 0]
                    * 675.0,
                    instance.controls[:, 1],
                    instance.states[:, 3:6],
                )
            ),
            delimiter=",",
            header=",".join([f"y{i}" for i in range(16)] + [f"x{i}" for i in range(3)]),
        )
    else:
        np.savez_compressed(
            f"data/ve_dataset/{track_name}_{v_x_max}.npz",
            states=instance.states,
            controls=instance.controls,
            control_derivatives=instance.control_derivatives,
            imu_linear_accelerations=imu_linear_accelerations,
            imu_angular_velocities=imu_angular_velocities,
            wheel_speeds=wheel_speeds,
            gss_velocities=gss_velocities,
        )


def combine_csv_files(files: list[str], output_file: str):
    """
    combines the specified CSV files by adding a column called exp_id
    it uses numpy to load the files and then saves them again to increase performance
    """
    data = np.vstack(
        [
            np.hstack(
                (
                    np.ones(),
                    np.loadtxt(f, delimiter=",", skiprows=1, usecols=range(1, 16)),
                )
            )
            for f in files
        ]
    )
    np.savetxt(
        output_file,
        data,
        delimiter=",",
        header="exp_id,states,controls,control_derivatives,imu_linear_accelerations,imu_angular_velocities,wheel_speeds,gss_velocities",
    )


if __name__ == "__main__":
    mission, track_name = missions_tracks[3]
    for v_x_max in max_longitudinal_speeds[:]:
        bruh(mission, track_name, v_x_max)

    # combine_csv_files(
    #     [
    #         f"data/ve_dataset/{track_name}_{v_x_max}.csv"
    #         for v_x_max in max_longitudinal_speeds
    #     ],
    #     f"data/ve_dataset/{track_name}.csv",
    # )
