import argparse

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


def bruh(mission: Mission, track_name: str, v_x_max: float):
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
            psi_s=np.pi / 2,
            psi_e=np.pi / 2,
            closed=mission == Mission.TRACKDRIVE,
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
        verbosity_level=127,
    )
    instance.run()

    np.savez_compressed(
        f"data/sysid_dataset/{track_name}_{v_x_max}.npz",
        states=instance.states,
        controls=instance.controls,
        control_derivatives=instance.control_derivatives,
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

    args = parser.parse_args()
    mission = Mission[args.mission]
    track = args.track

    for v_x_max in max_longitudinal_speeds[:]:
        bruh(mission, track, v_x_max)
