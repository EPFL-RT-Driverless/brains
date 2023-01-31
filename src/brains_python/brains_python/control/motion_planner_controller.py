# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
from typing import Optional

import numpy as np

from brains_python.control.controller import *
from brains_python.control.motion_planner import *
from brains_python.common import Mission

__all__ = ["MotionPlannerController"]


class MotionPlannerController:
    """
    The most useful class of the control module. It encapsulates the motion planner and
    the controller. Actually two controllers, one for the racing on the track and one
    for stopping the car.
    It can be configured with any MotionPlanner and any (pair of) Controller(s).
    """

    car_params: CarParams
    motion_planner: MotionPlanner
    racing_controller: Controller
    stopping_controller: Controller

    stopping: bool
    last_arc_length_localization: float

    # variables used to stop the car
    old_y: float
    lap_count: int
    max_lap_count: int
    autox_start_point: Optional[np.ndarray]

    def __init__(
        self,
        car_params: CarParams,
        racing_controller_params: ControllerParams,
        stopping_controller_params: ControllerParams,
        motion_planner_params: Optional[MotionPlannerParams] = None,
        autox_start_point: Optional[np.ndarray] = None,
        **kwargs,
    ):
        self.stopping = False
        self.car_params = car_params
        if "motion_planner" in kwargs and kwargs["motion_planner"] is not None:
            self.motion_planner = kwargs["motion_planner"]
            self.last_arc_length_localization = 0.0
        else:
            self.set_motion_planner(motion_planner_params, car_params)

        racing_controller_type = racing_controller_params.controller_type
        self.racing_controller = racing_controller_type(
            car_params=car_params, controller_params=racing_controller_params
        )
        stopping_controller_type = stopping_controller_params.controller_type
        self.stopping_controller = stopping_controller_type(
            car_params=car_params, controller_params=stopping_controller_params
        )
        self.old_y = 0.0
        self.lap_count = -1
        self.autox_start_point = autox_start_point
        if self.motion_planner.mission == Mission.AUTOCROSS:
            assert (
                self.autox_start_point is not None
            ), "start point is not set for AutoX mission"
        self.max_lap_count = kwargs.get("max_lap_count", 10)

    def set_motion_planner(
        self, motion_planner_params: MotionPlannerParams, car_params: CarParams
    ):
        self.motion_planner = MotionPlanner(
            motion_planner_params=motion_planner_params, car_params=car_params
        )
        self.last_arc_length_localization = 0.0

    def compute_control(
        self, current_state: np.ndarray, current_control: np.ndarray
    ) -> ControlReturn:
        """
        Compute the new control input (and its derivative) for the current state and the
        current control input.

        :param current_state: state in the form [X, Y, phi, v_x, v_y, r]
        :param current_control: control in the form [T, delta]
        :return: an instance of ControlReturn containing the control [T, delta] and its
         derivative [dT, ddelta]
        """
        res_dev = self.compute_control_dev(current_state, current_control)
        return ControlReturn(res_dev.control, res_dev.control_derivative)

    def compute_control_dev(
        self, current_state: np.ndarray, current_control: np.ndarray
    ) -> ControlReturnDev:
        """
        Compute the new control input (and its derivative) for the current state and the
        current control input.

        :param current_state: state in the form [X, Y, phi, v_x, v_y, r]
        :param current_control: control in the form [T, delta]
        :return: an instance of ControlReturn containing the control [T, delta] and its
         derivative [dT, ddelta]
        """
        # select controller
        actual_controller = (
            self.racing_controller if not self.stopping else self.stopping_controller
        )

        # localize current position on reference path
        reference_arc_lengths = self.motion_planner.extract_horizon_arc_lengths(
            horizon_size=actual_controller.horizon_size,
            sampling_time=actual_controller.controller_params.sampling_time,
            pos=current_state[:2],
            guess=self.last_arc_length_localization,
        )
        self.last_arc_length_localization = (
            reference_arc_lengths[0]
            if isinstance(reference_arc_lengths, np.ndarray)
            else reference_arc_lengths
        )
        XY_ref = np.array(
            [
                self.motion_planner.X_ref_vs_arc_length(reference_arc_lengths),
                self.motion_planner.Y_ref_vs_arc_length(reference_arc_lengths),
            ]
        ).T
        phi_ref = self.motion_planner.phi_ref_vs_arc_length(reference_arc_lengths)
        v_x_ref = (
            self.motion_planner.v_x_ref_vs_arc_length(reference_arc_lengths)
            if not self.stopping
            else np.zeros(actual_controller.horizon_size + 1)
        )

        # see if we have to stop the car
        if not self.stopping:
            if self.mission in {
                Mission.ACCELERATION,
                Mission.SKIDPAD,
                Mission.SHORT_SKIDPAD,
            }:
                if (
                    self.last_arc_length_localization
                    >= self.motion_planner.key_points[-1][0]
                ):
                    self.stop_car()
                    return self.compute_control_dev(current_state, current_control)
            elif self.mission == Mission.TRACKDRIVE:
                if (
                    current_state[1]
                    >= self.motion_planner.reference_points[0, 1]
                    >= self.old_y
                    and self.motion_planner.reference_points[0, 0]
                    - self.motion_planner.widths[0, 1]
                    <= current_state[0]
                    <= self.motion_planner.reference_points[0, 0]
                    + self.motion_planner.widths[0, 0]
                ):
                    self.lap_count += 1

                if self.lap_count >= self.max_lap_count:
                    self.stop_car()
                    return self.compute_control_dev(current_state, current_control)

            elif self.mission == Mission.AUTOCROSS:
                if (
                    current_state[1] >= self.autox_start_point[1] > self.old_y
                    and np.linalg.norm(current_state[:2] - self.autox_start_point)
                    <= self.motion_planner.widths[0, 0]
                ):
                    self.lap_count += 1
                    if self.lap_count >= 1:
                        self.stop_car()
                        return self.compute_control_dev(current_state, current_control)

            self.old_y = current_state[1]

        # call control
        res = actual_controller.compute_control_dev(
            current_state=current_state,
            current_control=current_control,
            XY_ref=XY_ref,
            phi_ref=phi_ref,
            v_x_ref=v_x_ref,
            optimized=True,
        )
        # if self.stopping and current_state[3] < 0.0:
        #     res.control[0] = 0.0
        #     res.control_derivative[0] = 0.0

        return res

    def stop_car(self):
        """
        Tell the car to stop (and switch to the stopping controller).
        """
        self.stopping = True

    @property
    def mission(self) -> Mission:
        return self.motion_planner.mission
