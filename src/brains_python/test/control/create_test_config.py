#  Copyright (c) 2022. Tudor Oancea EPFL Racing Team Driverless
import numpy as np
from data_visualization import PlotMode
from brains_python.common import Mission
from brains_python.control import (
    CarParams,
    MotionPlannerController,
    MotionPlannerParams,
    fsds_car_params,
    ControllerParams,
    MotionPlanner,
    StanleyParams,
    stanley_params_from_mission,
)
import track_database as td

from brains_python.control.utils import ClosedLoopRun, SimulationMode


def create_test_config(
    controller_name: str,
    mission: Mission,
    track_name: str,
    racing_controller_params: ControllerParams,
):
    track = td.load_track(track_name)
    car_params = CarParams(**fsds_car_params)
    if mission == Mission.AUTOCROSS:
        autox_motion_planner_params = MotionPlannerParams(
            mission=mission,
            center_points=track.center_line,
            widths=track.track_widths,
            psi_s=np.pi / 2,
            psi_e=np.pi / 2,
            closed=True,
            min_curve=False,
            additional_attributes=["right_width_vs_time", "left_width_vs_time"],
        )
        autox_motion_planner = MotionPlanner(
            car_params=car_params,
            motion_planner_params=autox_motion_planner_params,
        )
    else:
        autox_motion_planner_params = None
        autox_motion_planner = None

    motion_planner_controller_instance = MotionPlannerController(
        car_params=car_params,
        racing_controller_params=racing_controller_params,
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
        )
        if mission != Mission.AUTOCROSS
        else autox_motion_planner_params,
        autox_start_point=autox_motion_planner.reference_points[0]
        if mission == Mission.AUTOCROSS
        else None,
        max_lap_count=1,
    )

    return ClosedLoopRun(
        mission=mission,
        track=track,
        car_params=car_params,
        motion_planner_controller=motion_planner_controller_instance,
        autox_motion_planner=autox_motion_planner,
        sampling_time=0.01,
        simulation_mode=SimulationMode.MIL_DYN_6,
        max_time=50.0,
        delay=0.0,
        plot_mode=PlotMode.STATIC,
        # plot_mode=PlotMode.DYNAMIC,
        plot_save_path="logs/{}_{}_{}.png".format(controller_name, mission, track_name),
    )
