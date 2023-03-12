import numpy as np

import track_database as tdb
from brains_python.common import Mission
from brains_python.control import (
    ControllerParams,
    MotionPlannerParams,
    MotionPlanner,
    MotionPlannerController,
    StanleyParams,
    stanley_params_from_mission,
    fsds_car_params,
    CarParams,
    IHMAcadosParams,
    fsds_ihm_acados_params,
)
from brains_python.control.utils import *
from data_visualization import PlotMode


def create_test_config(
    controller_name: str,
    mission: Mission,
    track_name: str,
    racing_controller_params: ControllerParams,
    car_params: CarParams = CarParams(**fsds_car_params),
    simulation_mode: SimulationMode = SimulationMode.MIL_KIN_6,
    plot_mode: PlotMode = PlotMode.STATIC,
):
    track = tdb.load_track(track_name)
    if mission == Mission.AUTOCROSS:
        autox_motion_planner_params = MotionPlannerParams(
            mission=Mission.TRACKDRIVE,
            center_points=track.center_line,
            widths=track.track_widths,
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
        simulation_mode=simulation_mode,
        max_time=50.0,
        delay=0.0,
        plot_mode=plot_mode,
        plot_save_path="logs/{}_{}_{}.png".format(controller_name, mission, track_name),
    )


def main():
    mission, track_name = AVAILABLE_MISSION_TRACK_TUPLES[4]
    controller_params = IHMAcadosParams(**fsds_ihm_acados_params)
    car_params = CarParams(**fsds_car_params)
    car_params.a_y_max = 10.0
    car_params.v_x_max = 15.0
    run_instance = create_test_config(
        "ihm_acados",
        mission,
        track_name,
        controller_params,
        car_params=car_params,
        plot_mode=PlotMode.STATIC,
        simulation_mode=SimulationMode.SIMIL,
    )
    run_instance.run()


if __name__ == "__main__":
    main()
