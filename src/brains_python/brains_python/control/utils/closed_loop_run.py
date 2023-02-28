# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import sys
from enum import Enum
from time import perf_counter
from typing import Optional, Union, Callable

import numpy as np
import track_database as td
from data_visualization import *

# from fsds_client import *
from fsds_client.new_client import FSDSClient

from brains_python.common import Mission, sleep
from brains_python.control.constants import *
from brains_python.control.controller import *
from brains_python.control.motion_planner import *
from brains_python.control.motion_planner_controller import *

__all__ = ["ClosedLoopRun", "SimulationMode", "AVAILABLE_MISSION_TRACK_TUPLES"]

AVAILABLE_MISSION_TRACK_TUPLES = [
    (Mission.ACCELERATION, "acceleration"),
    (Mission.SKIDPAD, "skidpad"),
    (Mission.SHORT_SKIDPAD, "short_skidpad"),
    (Mission.TRACKDRIVE, "fsds_competition_1"),
    (Mission.TRACKDRIVE, "fsds_competition_2"),
    (Mission.TRACKDRIVE, "fsds_competition_3"),
    (Mission.TRACKDRIVE, "fsds_default"),
    (Mission.AUTOCROSS, "fsds_competition_1"),
    (Mission.AUTOCROSS, "fsds_competition_2"),
    (Mission.AUTOCROSS, "fsds_competition_3"),
    (Mission.AUTOCROSS, "fsds_default"),
]


class SimulationMode(Enum):
    SIMIL = 0
    MIL = 1  # defaults to MIL_KIN_6 for the moment
    MIL_KIN_6 = 2
    MIL_DYN_6 = 3


class ClosedLoopRun:
    mission: Mission
    track: td.Track
    motion_planner_controller: MotionPlannerController
    autox_motion_planner: MotionPlanner
    fsds_client: Optional[FSDSClient]  # only for SIMIL mode
    first_fsds_run: Optional[bool]

    # run params
    sampling_time: float
    simulation_mode: SimulationMode
    max_time: float
    delay: float  # always sleep delay
    plot_mode: PlotMode  # if None, no plot
    plot_save_path: Optional[str]
    verbosity_level: int

    # simulation results
    SimResults = Union[np.ndarray, list[np.ndarray], list[float]]
    states: SimResults
    controls: SimResults
    control_derivatives: SimResults
    metrics: SimResults
    references: SimResults
    predictions: SimResults
    compute_control_times: SimResults
    successful_run: bool
    exit_reason: Optional[str]
    mission_time: Optional[float]
    iteration: int
    callbacks: list[Callable]

    def __init__(
        self,
        mission: Mission,
        track: td.Track,
        car_params: CarParams,
        motion_planner_controller: MotionPlannerController,
        autox_motion_planner: Optional[MotionPlanner] = None,
        # run params
        sampling_time: float = 0.03,
        simulation_mode: SimulationMode = SimulationMode.MIL,
        max_time: float = 50.0,
        delay: float = 0.0,
        plot_mode: Optional[PlotMode] = None,
        plot_save_path: Optional[str] = None,
        verbosity_level: int = 0,
    ):
        # check that mission and track are compatible
        assert (
            mission,
            track.name,
        ) in AVAILABLE_MISSION_TRACK_TUPLES, "Mission {} and track with name {} are not compatible".format(
            mission, track.name
        )
        self.mission = mission
        self.track = track
        # declare the FSDS client end check that it corresponds to the mission
        if simulation_mode == SimulationMode.SIMIL:
            self.fsds_client = FSDSClient()
            fsds_map_name = self.fsds_client.map_name
            assert (
                fsds_map_name == track.name
            ), "FSDS client is not connected to the right track: {} instead of {}".format(
                fsds_map_name, track.name
            )
            self.first_fsds_run = True
        else:
            self.fsds_client = None
            self.first_fsds_run = None

        # import the other run params
        self.sampling_time = sampling_time
        self.simulation_mode = simulation_mode
        self.max_time = max_time
        self.delay = delay
        self.plot_mode = plot_mode
        self.plot_save_path = plot_save_path
        if plot_save_path == PlotMode.LIVE_DYNAMIC:
            raise NotImplementedError("Live dynamic plot not implemented")
        self.verbosity_level = verbosity_level

        self.motion_planner_controller = motion_planner_controller
        self.autox_motion_planner = autox_motion_planner
        self.car_params = car_params
        self.states = []
        self.controls = []
        self.control_derivatives = []
        self.metrics = []
        self.references = []
        self.predictions = []
        self.compute_control_times = []
        self.successful_run = False
        self.exit_reason = None
        self.mission_time = None
        self.iteration = 0
        self.callbacks = []

    def submit_callback(self, callback: Callable):
        self.callbacks.append(callback)

    def _compute_new_motion_planner_params(
        self,
        pos: np.ndarray,
        last_s: float,
    ) -> tuple[float, MotionPlannerParams]:
        if self.autox_motion_planner is not None:
            reference_arc_lengths = (
                self.autox_motion_planner.extract_horizon_arc_lengths(
                    horizon_size=20 - 1,
                    sampling_time=5.0 / 20,
                    pos=pos,
                    guess=last_s,
                )
            )
            horizon_points = np.array(
                [
                    self.autox_motion_planner.X_ref_vs_arc_length(
                        reference_arc_lengths
                    ),
                    self.autox_motion_planner.Y_ref_vs_arc_length(
                        reference_arc_lengths
                    ),
                ]
            ).T
            psi_s = float(self.autox_motion_planner.phi_ref_vs_arc_length(last_s))
            psi_e = float(
                self.autox_motion_planner.phi_ref_vs_arc_length(
                    reference_arc_lengths[-1]
                )
            )
            t0 = self.autox_motion_planner.time_vs_arc_length(last_s)
            time_horizon = np.linspace(t0, t0 + 5.0, 20)
            horizon_widths = np.array(
                [
                    self.autox_motion_planner.additional_attributes[
                        "right_width_vs_time"
                    ](time_horizon),
                    self.autox_motion_planner.additional_attributes[
                        "left_width_vs_time"
                    ](time_horizon),
                ]
            ).T
            # set new MotionPlanner in motion_planner_controller
            return reference_arc_lengths[0], MotionPlannerParams(
                mission=Mission.AUTOCROSS,
                center_points=horizon_points,
                widths=horizon_widths,
                psi_s=psi_s,
                psi_e=psi_e,
                additional_attributes=[],
            )
        else:
            return 0.0, MotionPlannerParams()

    @property
    def mil(self) -> bool:
        return self.simulation_mode in [
            SimulationMode.MIL,
            SimulationMode.MIL_KIN_6,
            SimulationMode.MIL_DYN_6,
        ]

    @property
    def simil(self) -> bool:
        return self.simulation_mode == SimulationMode.SIMIL

    def _status_message(self, msg: str, verbosity_level: int = 1):
        if self.verbosity_level > 0 and verbosity_level <= self.verbosity_level:
            if not (msg.endswith("\n") or msg.endswith("\r")):
                msg += "\n"
            sys.stdout.write(msg)

    def run(self):
        # STEP 1 : initialize simulation data ================================================
        self.states = [np.array([0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0])]
        self.controls = [np.zeros(global_control_dim)]
        self.control_derivatives = [np.zeros(global_control_dim)]
        self.metrics = []
        self.compute_control_times = []
        self.references = []
        self.predictions = []
        self.successful_run = False
        self.exit_reason = None
        self.mission_time = None

        # initialize FSDS client in SimIL mode ===================================================
        if (
            self.first_fsds_run is not None
            and not self.first_fsds_run
            and self.simulation_mode == SimulationMode.SIMIL
        ):
            self.fsds_client.restart()
            self.first_fsds_run = False

        # initialize MotionPlannerController =================================================
        last_s = 0.0 if self.mission == Mission.AUTOCROSS else None
        if self.mission == Mission.AUTOCROSS:
            (
                last_s,
                initial_motion_planner_params,
            ) = self._compute_new_motion_planner_params(self.states[0][:2], last_s)
            self.motion_planner_controller.set_motion_planner(
                initial_motion_planner_params, self.car_params
            )
        # initialize Publisher in live dynamic mode ===========================================
        if self.plot_mode == PlotMode.LIVE_DYNAMIC:
            publisher_instance = Publisher()
        else:
            publisher_instance = None

        # STEP 3 : run simulation =========================================================
        self.iteration = 0
        last_cross_track_error = 0.0  # we could technically not be at 0.0 at the beginning tho, but we don't care
        intermediate_state = self.states[-1]
        # this value does not actually matter since it is redefined in Step 3.3
        while True:
            start_iteration = perf_counter()
            for callback in self.callbacks:
                callback(self)

            # STEP 3.1: get current state ================================================
            # at first iteration, we have already
            if self.iteration > 0:
                if self.mil:
                    self.states.append(intermediate_state)
                    self.states[-1][2] = (
                        np.mod(self.states[-1][2] + np.pi, 2 * np.pi) - np.pi
                    )
                else:
                    self.states.append(self.fsds_client.get_state()[0])

            # STEP 3.2: check termination conditions ======================================
            if self.motion_planner_controller.stopping and (
                np.abs(self.states[-1][3])
                if False
                # if self.simulation_mode is SimulationMode.SIMIL
                else self.states[-1][3] < 1.0e-3
            ):
                self.successful_run = True
                break
            if last_cross_track_error > 3.0:
                self.successful_run = False
                self.exit_reason = "Cross track error too high, aborting "
                sys.stdout.write(self.exit_reason)
                break
            if self.iteration * self.sampling_time > self.max_time:
                self.successful_run = False
                self.exit_reason = "Max time reached, aborting "
                sys.stdout.write(self.exit_reason)
                break

            # STEP 3.3: [in autoX] compute new center points ==============================
            if self.mission == Mission.AUTOCROSS:
                # set new MotionPlanner in motion_planner_controller
                start_path_planning = perf_counter()
                (
                    last_s,
                    new_motion_planner_params,
                ) = self._compute_new_motion_planner_params(
                    self.states[-1][:2],
                    last_s,
                )
                self.motion_planner_controller.set_motion_planner(
                    motion_planner_params=new_motion_planner_params,
                    car_params=self.car_params,
                )
                stop_path_planning = perf_counter()
                self._status_message(
                    "iteration {} : path planning time = {} ms".format(
                        self.iteration,
                        1000 * (stop_path_planning - start_path_planning),
                    ),
                    verbosity_level=1,
                )
                if self.mil:
                    intermediate_state = ClosedLoopRun._next_state_mil(
                        self.states[-1],
                        self.controls[-1],
                        self.control_derivatives[-1],
                        stop_path_planning - start_iteration,
                        self.car_params,
                    )

            # STEP 3.4: compute control ===================================================
            start_compute_control = perf_counter()
            try:
                res: ControlReturnDev = (
                    self.motion_planner_controller.compute_control_dev(
                        self.states[-1], self.controls[-1]
                    )
                )
            except RuntimeError as e:
                sys.stdout.write("RuntimeError in compute_control_dev: {}\n".format(e))
                self.successful_run = False
                break

            end_compute_control = perf_counter()
            compute_control_time = end_compute_control - start_compute_control
            self._status_message(
                f"iteration {self.iteration} : solving time = {1000 * compute_control_time} ms",
                verbosity_level=1,
            )
            if self.mil:
                intermediate_state = ClosedLoopRun._next_state_mil(
                    intermediate_state,
                    self.controls[-1],
                    self.control_derivatives[-1],
                    compute_control_time,
                    self.car_params,
                )
            self.compute_control_times.append(compute_control_time)
            self.metrics.append(res.metric)
            self.references.append(res.reference_horizon)
            self.predictions.append(res.prediction)
            last_cross_track_error = (
                res.metric[0] if isinstance(res.metric, np.ndarray) else res.metric
            )

            # STEP 3.5: [MIL mode] delays ============================================================
            # the delay here is artificial time added only in MIL mode to simulate the
            # time it takes to actually implement the computed control (i.e. actually
            # turn the steering wheel, press the gas pedal, etc.)
            # It is ignored in SimIL mode since there are already delays in the simulation
            if self.mil:
                if self.delay > 0.0:
                    intermediate_state = ClosedLoopRun._next_state_mil(
                        intermediate_state,
                        self.controls[-1],
                        self.control_derivatives[-1],
                        self.delay,
                        self.car_params,
                    )
            else:
                sleep(self.delay)

            # STEP 3.6: send control ======================================================
            # store control
            self.controls.append(np.copy(res.control))
            self.control_derivatives.append(np.copy(res.control_derivative))

            if self.simil:
                start_send_control = perf_counter()
                self.fsds_client.set_control(res.control)
                end_send_control = perf_counter()
                send_control_time = end_send_control - start_send_control
                self._status_message(
                    f"iteration {self.iteration} : send self.controls time = {1000 * send_control_time} ms",
                    verbosity_level=2,
                )

            # STEP 3.7: [live dynamic mode] send data to publisher ========================
            if self.plot_mode == PlotMode.LIVE_DYNAMIC and self.iteration % 10 == 0:
                start_publish_live_dynamic = perf_counter()
                publisher_instance.publish_msg(
                    {
                        "map": {
                            "trajectory": self.states[-1][:2],
                        },
                        "phi": {
                            "phi": np.rad2deg(self.states[-1][2]),
                        },
                        "v_x": {
                            "v_x": self.states[-1][3],
                        },
                        "T": {
                            "T": res.control[0],
                        },
                        "delta": {
                            "delta": np.rad2deg(res.control[1]),
                        },
                        "r": {
                            "r": np.rad2deg(self.states[-1][5]),
                        },
                        "v_y": {
                            "v_y": self.states[-1][4],
                        },
                        "dT": {
                            "dT": res.control_derivative[0],
                        },
                        "ddelta": {
                            "ddelta": np.rad2deg(res.control_derivative[1]),
                        },
                    }
                )
                end_publish_live_dynamic = perf_counter()
                publish_live_dynamic_time = (
                    end_publish_live_dynamic - start_publish_live_dynamic
                )
                self._status_message(
                    f"iteration {self.iteration} : send self.controls time = {1000 * publish_live_dynamic_time} ms",
                    verbosity_level=2,
                )
                if self.mil:
                    intermediate_state = ClosedLoopRun._next_state_mil(
                        intermediate_state,
                        self.controls[-1],
                        self.control_derivatives[-1],
                        publish_live_dynamic_time,
                        self.car_params,
                    )

            # STEP 3.8: [live dynamic mode or SimIL] wait for the end of the iteration ====
            if self.mil:
                intermediate_state = ClosedLoopRun._next_state_mil(
                    intermediate_state,
                    self.controls[-1],
                    self.control_derivatives[-1],
                    max(
                        self.sampling_time - compute_control_time - self.delay,
                        0.0,
                    ),
                    self.car_params,
                )
            end_iteration = perf_counter()
            if self.simil or self.plot_mode == PlotMode.LIVE_DYNAMIC:
                to_sleep = self.sampling_time - (end_iteration - start_iteration)
                if to_sleep > 0:
                    sleep(to_sleep)
                else:
                    self._status_message(
                        f"iteration {self.iteration} : iteration took too long to execute, skipping sleep\n",
                        verbosity_level=2,
                    )

            self._status_message(
                f"iteration {self.iteration} : iteration time = {1000 * (perf_counter()- start_iteration)} ms",
                verbosity_level=2,
            )

            self.iteration += 1

        # convert all the sim data to numpy arrays
        self.states = np.array(self.states)
        self.controls = np.array(self.controls)
        self.control_derivatives = np.array(self.control_derivatives)
        # find first index where the metric changes dimension
        bruh = len(self.metrics)
        for i in range(len(self.metrics)):
            if len(self.metrics[i]) != len(self.metrics[0]):
                bruh = i
                break
        self.references = np.concatenate(
            [np.expand_dims(r, 0) for r in self.references[:bruh]], axis=0
        )
        # bruh = len(self.predictions)
        # for i in range(len(self.predictions)):
        #     if len(self.predictions[i]) != len(self.predictions[0]):
        #         bruh = i
        #         break
        #
        # self.predictions = np.concatenate(
        #     [np.expand_dims(p, 0) for p in self.predictions[:bruh]], axis=0
        # )

        self.compute_control_times = np.array(self.compute_control_times)
        self.mission_time = self.iteration * self.sampling_time

        # visualize simulation ===================================================
        if self.plot_mode is not None and self.plot_mode != PlotMode.LIVE_DYNAMIC:
            self._internal_plot_everything(
                states=self.states,
                controls=self.controls,
                control_derivatives=self.control_derivatives,
            )

        # STEP 5 : Terminate publisher when the simulation is finished =====================
        if self.plot_mode == PlotMode.LIVE_DYNAMIC:
            publisher_instance.terminate()

    @staticmethod
    def _next_state_mil(
        state: np.ndarray,
        control: np.ndarray,
        control_derivative: np.ndarray,
        sampling_time: float,
        car_params: CarParams,
        rk4_nodes: int = 10,
        simulation_mode: SimulationMode = SimulationMode.MIL,
    ) -> np.ndarray:
        if (
            simulation_mode == SimulationMode.MIL
            or simulation_mode == SimulationMode.MIL_KIN_6
        ):
            model = lambda x: ClosedLoopRun.KIN_6_cont_dynamics(
                x, control, control_derivative, car_params
            )
        elif simulation_mode == SimulationMode.MIL_DYN_6:
            model = lambda x: ClosedLoopRun.DYN_6_cont_dynamics(x, control, car_params)
        elif simulation_mode == SimulationMode.SIMIL:
            return state
        else:
            raise ValueError("Unknown model")

        assert rk4_nodes > 0
        dt = sampling_time / rk4_nodes
        new_state = state
        for i in range(rk4_nodes):
            k_1 = model(new_state)
            k_2 = model(new_state + 0.5 * dt * k_1)
            k_3 = model(new_state + 0.5 * dt * k_2)
            k_4 = model(new_state + dt * k_3)
            new_state = new_state + (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt / 6

        return new_state

    @staticmethod
    def KIN_6_cont_dynamics(
        x: np.ndarray, u: np.ndarray, du: np.ndarray, car_params: CarParams
    ) -> np.ndarray:
        """
        state x = [X, Y, phi, v_x, v_y, r]
        control u = [T, delta]
        control derivative du = [dT, ddelta]
        """
        F_x = car_params.C_m * u[0] + (
            -car_params.C_r2 * x[3] ** 2 * np.sign(x[3]) - car_params.C_r0
        )
        return np.array(
            [
                x[3] * np.cos(x[2]) - x[4] * np.sin(x[2]),
                x[3] * np.sin(x[2]) + x[4] * np.cos(x[2]),
                x[5],
                F_x / car_params.m,
                (du[1] * x[3] + u[1] * F_x / car_params.m)
                * car_params.l_r
                / (car_params.l_r + car_params.l_f),
                (du[1] * x[3] + u[1] * F_x / car_params.m)
                / (car_params.l_r + car_params.l_f),
            ]
        )

    @staticmethod
    def DYN_6_cont_dynamics(
        x: np.ndarray, u: np.ndarray, car_params: CarParams
    ) -> np.ndarray:
        """
        state x = [X, Y, phi, v_x, v_y, r]
        control u = [T, delta]
        """
        F_r_x = (
            car_params.C_m * u[0]
            - float(np.abs(x[3]) >= 1.0e-3) * car_params.C_r0
            - car_params.C_r2 * x[3] ** 2 * np.sign(x[3])
        )
        F_f_x = F_r_x
        F_r_y = (
            car_params.D_r
            * np.sin(
                car_params.C_r
                * np.arctan(
                    car_params.B_r * np.arctan((x[4] - car_params.l_r * x[5]) / x[3])
                )
            )
            if np.abs(x[3]) >= 1.0e-3
            else 0.0
        )

        F_f_y = (
            car_params.D_f
            * np.sin(
                car_params.C_f
                * np.arctan(
                    car_params.B_f * np.arctan((x[4] + car_params.l_f * x[5]) / x[3])
                    - u[1]
                )
            )
            if np.abs(x[3]) >= 1.0e-3
            else 0.0
        )

        # compute dx/dt
        return np.array(
            [
                x[3] * np.cos(x[2]) - x[4] * np.sin(x[2]),
                x[3] * np.sin(x[2]) + x[4] * np.cos(x[2]),
                x[5],
                (
                    F_r_x
                    - F_f_y * np.sin(u[1])
                    + F_f_x * np.cos(u[1])
                    + car_params.m * x[4] * x[5]
                )
                / car_params.m,
                (
                    F_r_y
                    + F_f_y * np.cos(u[1])
                    - car_params.m
                    + F_f_x * np.sin(u[1]) * x[3] * x[5]
                )
                / car_params.m,
                (
                    F_f_y * car_params.l_f * np.cos(u[1])
                    + car_params.l_f * F_f_x * np.sin(u[1])
                    - F_r_y * car_params.l_r
                )
                / car_params.I_z,
            ]
        )

    def _internal_plot_everything(
        self,
        states: Optional[np.ndarray] = None,
        controls: Optional[np.ndarray] = None,
        control_derivatives: Optional[np.ndarray] = None,
        # predictions: Optional[
        #     np.ndarray
        # ] = None,  # (N, 8) with N>=1  and each row is [X, Y, phi, v_x, T, delta, dT, ddelta]
        # reference_values: Optional[np.ndarray] = None,  # [X,Y,phi,v_x]
    ):
        ClosedLoopRun.plot_everything(
            sampling_time=self.sampling_time,
            max_time=self.max_time,
            plot_mode=self.plot_mode,
            car_params=self.car_params,
            save_path=self.plot_save_path,
            center_points=self.motion_planner_controller.motion_planner.reference_points,
            right_cones=self.track.right_cones,
            left_cones=self.track.left_cones,
            states=states,
            controls=controls,
            control_derivatives=control_derivatives,
            # predictions=predictions,
            # reference_values=reference_values,
        )

    @staticmethod
    def plot_everything(
        sampling_time: float,
        max_time: float,
        plot_mode: PlotMode,
        car_params: CarParams,
        center_points: np.ndarray,
        right_cones: np.ndarray,
        left_cones: np.ndarray,
        save_path: Optional[str] = None,
        states: Optional[np.ndarray] = None,
        controls: Optional[np.ndarray] = None,
        control_derivatives: Optional[np.ndarray] = None,
        # predictions: Optional[
        #     np.ndarray
        # ] = None,  # (N, 8) with N>=1  and each row is [X, Y, phi, v_x, T, delta, dT, ddelta]
        # reference_values: Optional[np.ndarray] = None,  # [X,Y,phi,v_x]
    ):
        live_data = plot_mode == PlotMode.LIVE_DYNAMIC
        if not live_data:
            assert (
                states is not None
            ), "states must be provided for static and dynamic plots"
            assert (
                controls is not None
            ), "controls must be provided for static and dynamic plots"
            assert (
                control_derivatives is not None
            ), "control_derivatives must be provided for static and dynamic plots"

        bounds_options = {"color": "red", "linestyle": ":"}
        bounds_x = np.ones(int(max_time / sampling_time) + 1)
        simulation_plot = Plot(
            row_nbr=4,
            col_nbr=3,
            mode=plot_mode,
            sampling_time=sampling_time,
            interval=1,
            figsize=(15, 8),
            # port=5002,
            show_car=False,
        )
        simulation_plot.add_subplot(
            row_idx=range(4),
            col_idx=0,
            subplot_name="map",
            subplot_type=SubplotType.SPATIAL,
            unit="m",
            show_unit=True,
            curves={
                "center_line": {
                    "data": center_points,
                    "curve_type": CurveType.STATIC,
                    "curve_style": CurvePlotStyle.PLOT,
                    "options": {"color": "black"},
                },
                "left_cones": {
                    "data": left_cones,
                    "curve_style": CurvePlotStyle.SCATTER,
                    "curve_type": CurveType.STATIC,
                    "mpl_options": {"color": "blue", "marker": "^"},
                },
                "right_cones": {
                    "data": right_cones,
                    "curve_type": CurveType.STATIC,
                    "curve_style": CurvePlotStyle.SCATTER,
                    "mpl_options": {"color": "yellow", "marker": "^"},
                },
                "trajectory": {
                    "data": None if live_data else states[:, :2],
                    "curve_type": CurveType.REGULAR,
                    "curve_style": CurvePlotStyle.PLOT,
                    "mpl_options": {"color": "blue", "linewidth": 2},
                },
            },
            # | (
            #     {
            #         "prediction": {
            #             "data": None if live_data else predictions[:, :2],
            #             "curve_type": CurveType.PREDICTION,
            #             "curve_style": CurvePlotStyle.PLOT,
            #             "mpl_options": {"color": "red", "linewidth": 2},
            #         },
            #         "reference": {
            #             "data": None if live_data else reference_values[:, :2],
            #             "curve_type": CurveType.PREDICTION,
            #             "curve_style": CurvePlotStyle.PLOT,
            #             "mpl_options": {"color": "green", "linewidth": 1},
            #         },
            #     }
            #     if live_data
            #     else {}
            # ),
        )
        simulation_plot.add_subplot(
            row_idx=0,
            col_idx=1,
            subplot_name="phi",
            subplot_type=SubplotType.TEMPORAL,
            unit="°",
            show_unit=True,
            curves={
                "phi": {
                    "data": np.rad2deg(states[:, 2]) if not live_data else None,
                    "curve_type": CurveType.REGULAR,
                    "curve_style": CurvePlotStyle.PLOT,
                    "mpl_options": {"color": "blue", "linewidth": 2},
                },
            },
        )
        simulation_plot.add_subplot(
            row_idx=1,
            col_idx=1,
            subplot_name="v_x",
            subplot_type=SubplotType.TEMPORAL,
            unit="m/s",
            show_unit=True,
            curves={
                "v_x": {
                    "data": states[:, 3] if not live_data else None,
                    "curve_type": CurveType.REGULAR,
                    "curve_style": CurvePlotStyle.PLOT,
                    "mpl_options": {"color": "blue", "linewidth": 2},
                },
                "v_x_max": {
                    "data": bounds_x * car_params.v_x_max,
                    "curve_type": CurveType.STATIC,
                    "curve_style": CurvePlotStyle.PLOT,
                    "mpl_options": bounds_options,
                },
                "v_x_min": {
                    "data": bounds_x * 0.0,
                    "curve_type": CurveType.STATIC,
                    "curve_style": CurvePlotStyle.PLOT,
                    "mpl_options": bounds_options,
                },
            },
        )
        simulation_plot.add_subplot(
            row_idx=2,
            col_idx=1,
            subplot_name="T",
            subplot_type=SubplotType.TEMPORAL,
            unit="1",
            show_unit=False,
            curves={
                "T": {
                    "data": controls[:, 0] if not live_data else None,
                    "curve_type": CurveType.REGULAR,
                    "curve_style": CurvePlotStyle.PLOT,
                    "mpl_options": {"color": "blue", "linewidth": 2},
                },
                "T_max": {
                    "data": 1 * bounds_x,
                    "curve_type": CurveType.STATIC,
                    "curve_style": CurvePlotStyle.PLOT,
                    "mpl_options": bounds_options,
                },
                "T_min": {
                    "data": -1 * bounds_x,
                    "curve_type": CurveType.STATIC,
                    "curve_style": CurvePlotStyle.PLOT,
                    "mpl_options": bounds_options,
                },
            },
        )
        simulation_plot.add_subplot(
            row_idx=3,
            col_idx=1,
            subplot_name="delta",
            subplot_type=SubplotType.TEMPORAL,
            unit="°",
            show_unit=True,
            curves={
                "delta": {
                    "data": np.rad2deg(controls[:, 1]) if not live_data else None,
                    "curve_type": CurveType.REGULAR,
                    "curve_style": CurvePlotStyle.PLOT,
                    "mpl_options": {"color": "blue", "linewidth": 2},
                },
                "delta_max": {
                    "data": np.rad2deg(car_params.delta_max) * bounds_x,
                    "curve_type": CurveType.STATIC,
                    "curve_style": CurvePlotStyle.PLOT,
                    "mpl_options": bounds_options,
                },
                "delta_min": {
                    "data": -np.rad2deg(car_params.delta_max) * bounds_x,
                    "curve_type": CurveType.STATIC,
                    "curve_style": CurvePlotStyle.PLOT,
                    "mpl_options": bounds_options,
                },
            },
        )
        simulation_plot.add_subplot(
            row_idx=0,
            col_idx=2,
            subplot_name="r",
            subplot_type=SubplotType.TEMPORAL,
            unit="°/s",
            show_unit=True,
            curves={
                "r": {
                    "data": np.rad2deg(states[:, 5]) if not live_data else None,
                    "curve_type": CurveType.REGULAR,
                    "curve_style": CurvePlotStyle.PLOT,
                    "mpl_options": {"color": "blue", "linewidth": 2},
                },
            },
        )
        simulation_plot.add_subplot(
            row_idx=1,
            col_idx=2,
            subplot_name="v_y",
            subplot_type=SubplotType.TEMPORAL,
            unit="m/s",
            show_unit=True,
            curves={
                "v_y": {
                    "data": states[:, 4] if not live_data else None,
                    "curve_type": CurveType.REGULAR,
                    "curve_style": CurvePlotStyle.PLOT,
                    "mpl_options": {"color": "blue", "linewidth": 2},
                },
            },
        )
        simulation_plot.add_subplot(
            row_idx=2,
            col_idx=2,
            subplot_name="dT",
            subplot_type=SubplotType.TEMPORAL,
            unit="1/s",
            show_unit=False,
            curves={
                "dT": {
                    "data": control_derivatives[:, 0] if not live_data else None,
                    "curve_type": CurveType.REGULAR,
                    "curve_style": CurvePlotStyle.PLOT,
                    "mpl_options": {"color": "blue", "linewidth": 2},
                },
                "dT_max": {
                    "data": car_params.dT_max * bounds_x,
                    "curve_type": CurveType.STATIC,
                    "curve_style": CurvePlotStyle.PLOT,
                    "mpl_options": bounds_options,
                },
                "dT_min": {
                    "data": -car_params.dT_max * bounds_x,
                    "curve_type": CurveType.STATIC,
                    "curve_style": CurvePlotStyle.PLOT,
                    "mpl_options": bounds_options,
                },
            },
        )
        simulation_plot.add_subplot(
            row_idx=3,
            col_idx=2,
            subplot_name="ddelta",
            subplot_type=SubplotType.TEMPORAL,
            unit="°/s",
            show_unit=True,
            curves={
                "ddelta": {
                    "data": np.rad2deg(control_derivatives[:, 1])
                    if not live_data
                    else None,
                    "curve_type": CurveType.REGULAR,
                    "curve_style": CurvePlotStyle.PLOT,
                    "mpl_options": {"color": "blue", "linewidth": 2},
                },
                "ddelta_max": {
                    "data": np.rad2deg(car_params.ddelta_max) * bounds_x,
                    "curve_type": CurveType.STATIC,
                    "curve_style": CurvePlotStyle.PLOT,
                    "mpl_options": bounds_options,
                },
                "ddelta_min": {
                    "data": np.rad2deg(-car_params.ddelta_max) * bounds_x,
                    "curve_type": CurveType.STATIC,
                    "curve_style": CurvePlotStyle.PLOT,
                    "mpl_options": bounds_options,
                },
            },
        )
        simulation_plot.plot(
            show=plot_mode == PlotMode.LIVE_DYNAMIC,
            save_path=save_path,
        )
