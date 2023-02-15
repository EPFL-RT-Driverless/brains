import argparse
from time import perf_counter

import numpy as np

from brains_python import Mission
from brains_python.control import (
    IHMAcadosParams,
    fsds_ihm_acados_params,
    MotionPlannerController,
    fsds_car_params,
    CarParams,
    MotionPlannerParams,
    StanleyParams,
    stanley_params_from_mission,
)
from common import max_longitudinal_speeds
from brains_python.control.utils import SimulationMode, ClosedLoopRun
from track_database import load_track


def bruh(mission: Mission, track_name: str, v_x_max: float, pitch: int):
    track = load_track(track_name)
    car_params = CarParams(**fsds_car_params)
    car_params.v_x_max = v_x_max

    motion_planner_controller_instance = MotionPlannerController(
        car_params=car_params,
        racing_controller_params=IHMAcadosParams(**fsds_ihm_acados_params),
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
        sampling_time=0.05,
        simulation_mode=SimulationMode.SIMIL,
        max_time=50.0,
        delay=0.0,
        verbosity_level=0,
    )
    camera_images = []
    lidar_point_clouds = []
    lidar_positions = []
    lidar_orientations = []
    rel_cones_positions = []

    def callback(s: ClosedLoopRun):
        if s.iteration % 5 != 0:
            return
        start = perf_counter()
        camera_images.append(instance.fsds_client.get_image())
        lidar_data = instance.fsds_client.low_level_client.getLidarData()
        lidar_point_clouds.append(lidar_data.point_cloud)
        lidar_positions.append(lidar_data.pose.position.to_numpy_array())
        lidar_orientations.append(lidar_data.pose.orientation.to_numpy_array())
        rel_cones_positions.append(instance.fsds_client.find_cones(s.states[-1]))
        print(f"Callback took {1000*(perf_counter() - start)} ms")

    instance.submit_callback(callback)
    instance.run()

    camera_images = np.array(camera_images)
    lidar_point_clouds = {
        "lidar_point_clouds_" + str(i): lidar_point_clouds[i]
        for i in range(len(lidar_point_clouds))
    }
    rel_cones_positions = {
        "rel_cones_positions_" + str(i): rel_cones_positions[i]
        for i in range(len(rel_cones_positions))
    }

    np.savez_compressed(
        f"data/vision_dataset/{track_name}_{v_x_max}_{pitch}.npz",
        states=instance.states[::5],
        camera_images=camera_images,
        **lidar_point_clouds,
        **rel_cones_positions,
    )


if __name__ == "__main__":
    # add argparse here
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mission",
        type=str,
        help="Mission to run",
    )
    parser.add_argument(
        "--track",
        type=str,
        help="Track to run",
    )
    parser.add_argument(
        "--pitch",
        type=int,
        help="Pitch of camera and lidar",
    )
    parser.add_argument(
        "--v_x_max",
        type=float,
        help="Max longitudinal speed",
        required=False,
    )
    args = parser.parse_args()
    mission = Mission[args.mission]
    track = args.track
    pitch = args.pitch
    v_x_max = args.v_x_max
    if v_x_max is not None:
        bruh(mission, track, v_x_max, pitch)
    else:
        for v_x_max in max_longitudinal_speeds[:]:
            bruh(mission, track, v_x_max, pitch)
