import numpy as np

import track_database as td
from brains_python.common import Mission
from brains_python.control import (
    CarParams,
    MotionPlannerController,
    MotionPlannerParams,
    fsds_car_params,
    ControllerParams,
    StanleyParams,
    stanley_params_from_mission,
)
from brains_python.control import fsds_ihm_acados_params, IHMAcadosParams
from brains_python.control.utils import ClosedLoopRun, SimulationMode
from data_visualization import PlotMode

missions_tracks = [
    (Mission.ACCELERATION, "acceleration"),
    (Mission.SHORT_SKIDPAD, "short_skidpad"),
    (Mission.TRACKDRIVE, "fsds_competition_1"),
    (Mission.TRACKDRIVE, "fsds_competition_2"),
    (Mission.TRACKDRIVE, "fsds_competition_3"),
    (Mission.TRACKDRIVE, "fsds_default"),
]
pitches = [0, 3, 6, 9]
max_longitudinal_speeds = [2.5, 5.0, 10.0, 15.0]


def create_test_config(
    mission: Mission,
    track_name: str,
    racing_controller_params: ControllerParams = IHMAcadosParams(
        **fsds_ihm_acados_params
    ),
    simulation_mode: SimulationMode = SimulationMode.MIL_KIN_6,
    plot_mode: PlotMode = PlotMode.STATIC,
):
    track = td.load_track(track_name)
    car_params = CarParams(**fsds_car_params)

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
        ),
        max_lap_count=1,
    )

    return ClosedLoopRun(
        mission=mission,
        track=track,
        car_params=car_params,
        motion_planner_controller=motion_planner_controller_instance,
        sampling_time=0.01,
        simulation_mode=simulation_mode,
        max_time=50.0,
        delay=0.0,
        plot_mode=plot_mode,
    )
