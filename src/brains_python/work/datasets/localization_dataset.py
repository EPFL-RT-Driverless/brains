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
from common import max_longitudinal_speeds
from track_database import load_track


def bruh(mission: Mission, track_name: str, v_x_max: float):
    track = load_track(track_name)
    car_params = CarParams(**fsds_car_params)
    car_params.v_x_max = v_x_max

    motion_planner_controller_instance = MotionPlannerController(
        car_params=car_params,
        racing_controller_params=StanleyParams(**stanley_params_from_mission(mission)),
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
    rel_cones_positions = []

    def callback(s: ClosedLoopRun):
        imu_data = s.fsds_client.low_level_client.getImuData()
        imu_linear_accelerations.append(imu_data.linear_acceleration.to_numpy_array())
        imu_angular_velocities.append(imu_data.angular_velocity.to_numpy_array())
        rel_cones_positions.append(instance.fsds_client.find_cones(s.states[-1]))

    instance.submit_callback(callback)
    instance.run()

    imu_linear_accelerations = np.array(imu_linear_accelerations)
    imu_angular_velocities = np.array(imu_angular_velocities)
    rel_cones_positions = np.array(rel_cones_positions)
    np.savez_compressed(
        f"data/localization_dataset/{track_name}_{v_x_max}.npz",
        global_cones_positions=np.vstack((track.right_cones, track.left_cones)),
        states=instance.states,
        controls=instance.controls,
        control_derivatives=instance.control_derivatives,
        imu_linear_accelerations=imu_linear_accelerations,
        imu_angular_velocities=imu_angular_velocities,
        rel_cones_positions=rel_cones_positions,
    )


if __name__ == "__main__":
    hey = [
        (Mission.SHORT_SKIDPAD, "short_skidpad"),
        (Mission.TRACKDRIVE, "fsds_competition_1"),
        (Mission.TRACKDRIVE, "fsds_competition_2"),
        (Mission.TRACKDRIVE, "fsds_competition_3"),
        (Mission.TRACKDRIVE, "fsds_default"),
    ]
    mission, track = hey[0]
    for v_x_max in max_longitudinal_speeds[:]:
        bruh(mission, track, v_x_max)
