# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
from collections import deque
from typing import Callable

import casadi as ca
import numpy as np
import numpy.linalg
from scipy.linalg import solve

from brains_python.control.controller import *

__all__ = ["LQRParams", "LQR"]


class LQRParams(ControllerParams):
    horizon_size: int
    Q: np.ndarray
    R: np.ndarray
    R_tilde: np.ndarray

    def __init__(self, **kwargs):
        current_params, remaining_params = LQRParams._transform_dict(kwargs)
        for key, val in current_params.items():
            setattr(self, key, val)
        super().__init__(**remaining_params)

    @property
    def controller_type(self) -> type:
        return LQR


class LQR(Controller):
    R_tilde: np.ndarray
    Q_tilde: np.ndarray
    A: Callable[[np.ndarray], np.ndarray]
    B: Callable[[np.ndarray], np.ndarray]

    def __init__(self, car_params: CarParams, controller_params: LQRParams):
        super().__init__(
            controller_params=controller_params,
            car_params=car_params,
            state_dim=4,
            control_dim=2,
            horizon_size=controller_params.horizon_size,
        )
        # costs ================================================================
        Q = (
            controller_params.Q
            if len(controller_params.Q.shape) == 2
            else np.diag(controller_params.Q)
        )
        assert Q.shape == (
            self.state_dim,
            self.state_dim,
        ), "Q must be a square matrix with shape ({}, {})".format(
            self.state_dim, self.state_dim
        )
        R = (
            controller_params.R
            if len(controller_params.R.shape) == 2
            else np.diag(controller_params.R)
        )
        assert R.shape == (
            self.control_dim,
            self.control_dim,
        ), "R must be a square matrix with shape ({}, {})".format(
            self.control_dim, self.control_dim
        )
        self.R_tilde = (
            controller_params.R_tilde
            if len(controller_params.R_tilde.shape) == 2
            else np.diag(controller_params.R_tilde)
        )
        assert self.R_tilde.shape == (
            self.control_dim,
            self.control_dim,
        ), "R_tilde must be a square matrix with shape ({}, {})".format(
            self.control_dim, self.control_dim
        )
        self.Q_tilde = np.block(
            [
                [
                    Q,
                    np.zeros((self.state_dim, self.control_dim)),
                ],
                [
                    np.zeros((self.control_dim, self.state_dim)),
                    R,
                ],
            ]
        )

        # dynamics =============================================================
        X = ca.SX.sym("X")
        Y = ca.SX.sym("Y")
        phi = ca.SX.sym("phi")
        v_x = ca.SX.sym("v_x")
        T = ca.SX.sym("T")
        delta = ca.SX.sym("delta")
        x = ca.vertcat(X, Y, phi, v_x)
        u = ca.vertcat(T, delta)
        F_x = car_params.C_m * T - car_params.C_r0 - car_params.C_r2 * v_x**2
        beta = ca.arctan(
            car_params.l_r * ca.tan(delta) / (car_params.l_f + car_params.l_r)
        )
        xdot = ca.vertcat(
            v_x * ca.cos(phi + beta),
            v_x * ca.sin(phi + beta),
            v_x / car_params.l_r * ca.tan(beta),
            F_x / car_params.m,
        )
        f_cont = ca.Function("f_cont", [x, u], [xdot])
        # discretize f_cont with RK4
        x_new = x
        dt = controller_params.sampling_time / 6
        for _ in range(6):
            k1 = f_cont(x_new, u)
            k2 = f_cont(x_new + dt / 2 * k1, u)
            k3 = f_cont(x_new + dt / 2 * k2, u)
            k4 = f_cont(x_new + dt * k3, u)
            x_new += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        f_disc_jac_x = ca.Function("f_disc_jac_x", [x, u], [ca.jacobian(x_new, x)])
        f_disc_jac_u = ca.Function("f_disc_jac_u", [x, u], [ca.jacobian(x_new, u)])
        # if not os.path.exists("f_disc_jac_x.c") or not os.path.exists("f_disc_jac_u.c"):
        #     # generate C-code for f_disc_jac_x and f_disc_jac_u
        #     f_disc_jac_x.generate("f_disc_jac_x.c")
        #     f_disc_jac_u.generate("f_disc_jac_u.c")
        # else:
        #     f_disc_jac_x = ca.external("f_disc_jac_x", "f_disc_jac_x.so")
        #     f_disc_jac_u = ca.external("f_disc_jac_u", "f_disc_jac_u.so")

        self.A = lambda x_ref: f_disc_jac_x(x_ref, np.zeros(self.control_dim)).full()
        self.B = lambda x_ref: f_disc_jac_u(x_ref, np.zeros(self.control_dim)).full()
        self.A_tilde: Callable[[np.ndarray], np.ndarray] = lambda x_ref: np.block(
            [
                [self.A(x_ref), self.B(x_ref)],
                [
                    np.zeros((self.control_dim, self.state_dim)),
                    np.eye(self.control_dim),
                ],
            ]
        )
        self.B_tilde: Callable[[np.ndarray], np.ndarray] = lambda x_ref: np.vstack(
            (self.B(x_ref), np.eye(self.control_dim))
        )

    def compute_control_dev(
        self,
        current_state: np.ndarray,
        current_control: np.ndarray,
        **kwargs,
    ) -> ControlReturnDev:
        # retrieve reference data
        XY_ref = np.asarray(kwargs["XY_ref"])
        phi_ref = np.asarray(kwargs["phi_ref"])
        v_x_ref = np.asarray(kwargs["v_x_ref"])

        # recast phi_ref to a value in the same range as current_state[2] (that is not
        # necessarily between -pi and pi or 0 and 2pi)
        while np.abs(phi_ref[0] - current_state[2]) >= 1.5 * np.pi:
            phi_ref += 2 * np.pi * np.sign(current_state[2] - phi_ref)

        # compute linear dynamics
        x_ref = np.hstack(
            (
                XY_ref,
                phi_ref.reshape(-1, 1),
                v_x_ref.reshape(-1, 1),
            )
        )
        A_tilde = [self.A_tilde(x_re) for x_re in x_ref]
        B_tilde = [self.B_tilde(x_re) for x_re in x_ref]

        # compute cost-to-go and feedback gain
        K_tilde = deque()
        P_tilde = deque()
        P_tilde.appendleft(self.Q_tilde)
        for t in range(self.horizon_size - 1, -1, -1):
            K_tilde.appendleft(
                -solve(
                    np.copy(self.R_tilde + B_tilde[t].T @ P_tilde[-1] @ B_tilde[t]),
                    np.copy(B_tilde[t].T @ P_tilde[-1] @ A_tilde[t]),
                    overwrite_a=True,
                    overwrite_b=True,
                    assume_a="pos",
                )
            )
            P_tilde.appendleft(
                self.Q_tilde
                + A_tilde[t].T @ P_tilde[-1] @ A_tilde[t]
                + A_tilde[t].T @ P_tilde[-1] @ B_tilde[t] @ K_tilde[-1]
            )

        x_tilde_prime_1 = (A_tilde[0] + B_tilde[0] @ K_tilde[0]) @ np.append(
            current_state[: self.state_dim] - x_ref[0], current_control
        )
        u_0 = x_tilde_prime_1[self.state_dim :]
        x_1 = x_ref[1] + x_tilde_prime_1[: self.state_dim]
        u_0 = np.clip(
            u_0,
            np.array([-1.0, -self.car_params.delta_max]),
            np.array([1.0, self.car_params.delta_max]),
        )
        u_0 = np.clip(
            u_0,
            current_control
            - self.controller_params.sampling_time
            * np.array([self.car_params.dT_max, self.car_params.ddelta_max]),
            current_control
            + self.controller_params.sampling_time
            * np.array([self.car_params.dT_max, self.car_params.ddelta_max]),
        )

        return ControlReturnDev(
            control=u_0,
            control_derivative=(u_0 - current_control)
            / self.controller_params.sampling_time,
            prediction=np.nan
            * np.ones((self.horizon_size, self.state_dim + self.control_dim)),
            reference_horizon=XY_ref,
            metric=np.array(
                [
                    np.linalg.norm(current_state[:2] - XY_ref[0]),
                    np.linalg.norm(x_1[:2] - XY_ref[1]),
                ]
            ),
        )
