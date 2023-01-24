# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
from typing import Callable

import numpy as np
import track_database as td
from data_visualization import PlotMode
from pyGLIS import GLIS

from brains_python.common.mission import Mission
from brains_python.control.utils.closed_loop_run import ClosedLoopRun, SimulationMode
from brains_python.control.constants import fsds_car_params
from brains_python.control.controller import (
    CarParams,
    ControllerParams,
)
from brains_python.control.motion_planner import MotionPlannerParams
from brains_python.control.motion_planner_controller import MotionPlannerController
from brains_python.control.stanley import StanleyParams, stanley_params_from_mission

__all__ = ["calibrate"]


def calibrate(
    mission: Mission,
    track_name: str,
    initial_controller_params: ControllerParams,
    var_to_params_mapping: Callable[[np.ndarray, ControllerParams], ControllerParams],
    # GLIS options
    f: Callable[[np.ndarray], float],
    nvar: int,
    ub: np.ndarray,
    lb: np.ndarray,
    nsamp: int = 10,
    maxevals: int = 1300,
    alpha: float = 1.0,
    delta: float = 1.0,
    csv_dump_file: str = "calibration.csv",
    load_previous_results: bool = True,
    # simulation options
    sampling_time: float = 0.01,
    simulation_mode: SimulationMode = SimulationMode.MIL_DYN_6,
    clock_speed: int = 1,
) -> np.ndarray:
    """
    Calibrates a controller using GLIS and closed loop simulation as defined in closed_loop_run.py (using either MIL or
    SimIL). You can define custom controllers, metrics to optimize, tracks on which to run the simulation, etc.
    This function is specified in each controller implementation sub package (e.g. control_module/stanley/).

    :param mission: Mission to run the simulation on.
    :param track_name: Name of the track to run the simulation on.
    :param initial_controller_params: Initial controller parameters to start the calibration with.
    :param var_to_params_mapping: Function that maps the GLIS variables to the controller parameters.
    :param f: Function that maps the metrics the controller outputted during a closed loop simulation to a single value
        to minimize.
    :param nvar: Number of variables to optimize (total number of parameters).
    :param ub: Upper bounds of the variables to optimize.
    :param lb: Lower bounds of the variables to optimize.
    :param nsamp: Number of initial samples to generate in GLIS.
    :param maxevals: Maximum number of evaluations in GLIS.
    :param alpha: Alpha parameter to use for GLIS (see documentation for more details).
    :param delta: Delta parameter to use for GLIS (see documentation for more details).
    :param csv_dump_file: File to dump the GLIS results to.
    :param load_previous_results: Whether to load previous results from the csv_dump_file.
    :param sampling_time: Sampling time to use for the closed loop simulation.
    :param simulation_mode: Simulation mode to use for the closed loop simulation.
    :param clock_speed: Clock speed to use for the closed loop simulation.
    :return: The best controller parameters found by GLIS.
    """
    track = td.load_track(track_name)
    car_params = CarParams(**fsds_car_params)
    max_time = 50.0 if mission in {Mission.TRACKDRIVE, Mission.AUTOCROSS} else 16.0
    if simulation_mode == SimulationMode.SIMIL:
        sampling_time /= clock_speed
        max_time /= clock_speed

    CP = initial_controller_params
    CP.sampling_time = sampling_time
    motion_planner_controller_instance = MotionPlannerController(
        car_params=car_params,
        racing_controller_params=CP,
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
    cached_motion_planner = motion_planner_controller_instance.motion_planner
    run_instance = ClosedLoopRun(
        mission=mission,
        track=track,
        car_params=car_params,
        motion_planner_controller=motion_planner_controller_instance,
        sampling_time=0.01,
        simulation_mode=SimulationMode.MIL_DYN_6,
        max_time=50.0,
        delay=0.0,
    )

    # Define GLIS function to minimize (i.e. the objective function + the simulation)
    def fun(x: np.ndarray) -> float:
        run_instance.motion_planner_controller = MotionPlannerController(
            car_params=car_params,
            racing_controller_params=var_to_params_mapping(x, CP),
            stopping_controller_params=StanleyParams(
                **stanley_params_from_mission(mission)
            ),
            motion_planner=cached_motion_planner,
            max_lap_count=1,
        )
        run_instance.run()
        return f(run_instance.metrics)

    # declare GLIS instance and run the optimization
    glis_instance = GLIS(
        f=fun,
        nvar=nvar,
        lb=lb,
        ub=ub,
        nsamp=nsamp,
        maxevals=maxevals,
        alpha=alpha,
        delta=delta,
        rbf_function=lambda x1, x2: np.exp(-np.sum(np.square(x1 - x2), axis=-1)),
        scaling=True,
        verbose=True,
        load_previous_results=load_previous_results,
        csv_dump_file=csv_dump_file,
    )
    res = glis_instance.run()

    # select the best solution and run the simulation again to visualize the behavior of the controller
    run_instance.motion_planner_controller = MotionPlannerController(
        car_params=car_params,
        racing_controller_params=var_to_params_mapping(res["opt"], CP),
        stopping_controller_params=StanleyParams(
            **stanley_params_from_mission(mission)
        ),
        motion_planner=cached_motion_planner,
        max_lap_count=1,
    )
    run_instance.plot_mode = PlotMode.STATIC
    run_instance.run()
    if run_instance.successful_run:
        print("Optimal mission time = {} s".format(run_instance.mission_time))

    return res["opt"]
