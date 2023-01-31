# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import numpy as np

from brains_python.control.controller import (
    Controller,
    ControlReturnDev,
    ControllerParams,
    CarParams,
)

__all__ = ["Stanley", "StanleyParams"]


class StanleyParams(ControllerParams):
    k_P: float
    k_I: float
    k_psi: float
    k_e: float
    k_s: float

    def __init__(self, **kwargs):
        current_params, remaining_params = StanleyParams._transform_dict(kwargs)
        for key, val in current_params.items():
            setattr(self, key, val)
        super().__init__(**remaining_params)

    @property
    def controller_type(self) -> type:
        return Stanley


class Stanley(Controller):
    controller_params_type = StanleyParams
    controller_params: StanleyParams

    last_epsilon: float  # last velocity error

    def __init__(self, car_params: CarParams, controller_params: StanleyParams):
        super().__init__(
            car_params=car_params,
            controller_params=controller_params,
            state_dim=4,
            control_dim=2,
            horizon_size=0,
        )
        self.last_epsilon = 0.0

    def compute_control_dev(
        self,
        current_state: np.ndarray,
        current_control: np.ndarray,
        **kwargs,
    ) -> ControlReturnDev:
        XY_ref = kwargs["XY_ref"]
        phi_ref = kwargs["phi_ref"]
        v_x_ref = kwargs["v_x_ref"]

        # check the dimensions of the input data
        assert current_state.shape == (
            6,
        ), "wrong current_state shape have {} but expected (6,)".format(
            current_state.shape
        )
        assert current_control.shape == (
            2,
        ), "wrong current_control shape have {} but expected (2,)".format(
            current_control.shape
        )
        assert XY_ref.shape == (self.horizon_size, 2,) or XY_ref.shape == (
            2,
        ), "wrong reference_points shape have {} but expected ({},2) or (2,)".format(
            XY_ref.shape, self.horizon_size
        )
        if isinstance(v_x_ref, np.ndarray):
            v_x_ref = float(v_x_ref)
        if isinstance(phi_ref, np.ndarray):
            phi_ref = float(phi_ref)

        XY_ref = (
            XY_ref[0]
            if XY_ref.shape == (self.horizon_size, self.control_dim)
            else XY_ref
        )

        # declare new control variable
        control = np.zeros(self.control_dim)

        # STEP 1: Longitudinal control =================================================
        epsilon = v_x_ref - current_state[3]
        control[
            0
        ] = self.controller_params.k_P * epsilon + self.controller_params.k_I * (
            (epsilon + self.last_epsilon) * 0.5 * self.controller_params.sampling_time
        )
        control[0] = np.clip(control[0], -1.0, 1.0)
        control[0] = np.clip(
            control[0],
            current_control[0]
            - self.controller_params.sampling_time * self.car_params.dT_max,
            current_control[0]
            + self.controller_params.sampling_time * self.car_params.dT_max,
        )
        # if np.abs(control[0]) <= 10e-3:
        #     control[0] = 0.0

        self.last_epsilon = epsilon

        # STEP 3: Lateral control ======================================================
        rho = np.mod(phi_ref + np.pi, 2 * np.pi) - np.pi
        psi = np.mod(rho - current_state[2] + np.pi, 2 * np.pi) - np.pi
        theta = np.arctan2(
            XY_ref[1] - current_state[1],
            XY_ref[0] - current_state[0],
        )
        e = np.linalg.norm(XY_ref - current_state[:2])

        control[1] = self.controller_params.k_psi * psi + (
            np.arctan(
                self.controller_params.k_e
                * e
                / (self.controller_params.k_s + current_state[3])
            )
            * np.sign(theta - rho)
            * (-1.0 if np.abs(theta - rho) > np.pi else 1.0)
        )
        control[1] = np.clip(
            control[1],
            -self.car_params.delta_max,
            self.car_params.delta_max,
        )
        control[1] = np.clip(
            control[1],
            current_control[1]
            - self.controller_params.sampling_time * self.car_params.ddelta_max,
            current_control[1]
            + self.controller_params.sampling_time * self.car_params.ddelta_max,
        )

        # STEP 4: Compute the derivative ==============================================
        derivative = (control - current_control) / self.controller_params.sampling_time

        return ControlReturnDev(
            control=control,
            control_derivative=derivative,
            prediction=None,
            reference_horizon=XY_ref,
            metric=np.array([e, psi, epsilon]),
        )
