from time import perf_counter

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
from fsds_client.utils import to_eulerian_angles_vectorized


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
        verbosity_level=0,
    )
    imu_orientations = []
    rel_cones_positions = []

    def callback(s: ClosedLoopRun):
        start = perf_counter()
        imu_data = s.fsds_client.low_level_client.getImuData()
        imu_orientations.append(imu_data.orientation.to_numpy_array())
        rel_cones_positions.append(instance.fsds_client.find_cones(s.states[-1]))
        print("imu fetch: ", 1000 * (perf_counter() - start), " ms")

    instance.submit_callback(callback)
    instance.run()

    imu_orientations = np.array(imu_orientations)
    imu_orientations = (
        np.mod(
            to_eulerian_angles_vectorized(
                imu_orientations[3],
                imu_orientations[0],
                imu_orientations[1],
                imu_orientations[2],
            )[2]
            + np.pi,
            2 * np.pi,
        )
        - np.pi
    )
    rel_cones_positions = {
        "rel_cones_positions_" + str(i): rel_cones_positions[i]
        for i in range(len(rel_cones_positions))
    }
    np.savez_compressed(
        f"data/localization_dataset/{track_name}_{v_x_max}.npz",
        global_cones_positions=np.vstack((track.right_cones, track.left_cones)),
        states=instance.states,
        controls=instance.controls,
        control_derivatives=instance.control_derivatives,
        imu_orientations=imu_orientations,
        **rel_cones_positions,
    )


if __name__ == "__main__":
    hey = [
        (Mission.SHORT_SKIDPAD, "short_skidpad"),
        (Mission.TRACKDRIVE, "fsds_competition_1"),
        (Mission.TRACKDRIVE, "fsds_competition_2"),
        (Mission.TRACKDRIVE, "fsds_competition_3"),
        (Mission.TRACKDRIVE, "fsds_default"),
    ]
    mission, track = hey[2]
    for v_x_max in max_longitudinal_speeds[: -1 if track == "short_skidpad" else None]:
        bruh(mission, track, v_x_max)
