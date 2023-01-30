# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import warnings

import casadi as ca
import numpy as np
from scipy.linalg import block_diag

try:
    from acados_template import (
        AcadosModel,
        AcadosOcp,
        AcadosOcpSolver,
    )
except ImportError:
    warnings.warn(
        "acados_template is not installed in the current python interpreter, please install it if you really want to use IHMAcados."
    )
    AcadosModel = None
    AcadosOcp = None
    AcadosOcpSolver = None

from ..controller import ControlReturnDev, ControllerParams, Controller, CarParams


# parameters p=[q_d, q_dT, q_ddelta, l_r, l_f, m, C_m, C_r0, C_r2]
# Acados status codes:
# 0 – success
# 1 – failure
# 2 – maximum number of iterations reached
# 3 – minimum step size in QP solver reached
# 4 – qp solver failed
# https://github.com/acados/acados/blob/dbc1c306ddaeefcb755d2afb289a25495446e008/acados/utils/types.h#L66

# hpipm exit codes:
# enum hpipm_status
# 0 - SUCCESS, // found solution satisfying accuracy tolerance
# 1 - MAX_ITER, // maximum iteration number reached
# 2 - MIN_STEP, // minimum step length reached
# 3 - NAN_SOL, // NaN in solution detected
# 4 - INCONS_EQ, // unconsistent equality constraints
class IHMAcadosParams(ControllerParams):
    horizon_size: int
    # MPC costs
    Q: np.ndarray  # costs on the state deviation
    R: np.ndarray  # costs on the control
    R_tilde: np.ndarray  # costs on the control derivative
    Zl: np.ndarray  # lower bound on the slack variables
    Zu: np.ndarray  # upper bound on the slack variables
    zu: np.ndarray
    zl: np.ndarray

    def __init__(self, **kwargs):
        current_params, remaining_params = IHMAcadosParams._transform_dict(kwargs)
        for key, val in current_params.items():
            setattr(self, key, val)
        super().__init__(**remaining_params)

    @property
    def controller_type(self) -> type:
        return IHMAcados


class IHMAcados(Controller):
    model: AcadosModel
    ocp: AcadosOcp
    solver: AcadosOcpSolver
    first_call: bool
    last_prediction: np.ndarray

    def __init__(self, car_params: CarParams, controller_params: IHMAcadosParams):
        # initialize Controller superclass ==========================================
        super().__init__(
            car_params=car_params,
            controller_params=controller_params,
            state_dim=4,
            control_dim=2,
            horizon_size=controller_params.horizon_size,
        )
        self.first_call = True

        # create model ===============================================================
        model = AcadosModel()
        self.model = model
        model.name = "ihm_acados"

        # variables
        X = ca.SX.sym("X")
        Y = ca.SX.sym("Y")
        phi = ca.SX.sym("phi")
        v_x = ca.SX.sym("v_x")
        T = ca.SX.sym("T")
        delta = ca.SX.sym("delta")
        x = ca.vertcat(X, Y, phi, v_x, T, delta)

        dT = ca.SX.sym("dT")
        ddelta = ca.SX.sym("ddelta")
        # s_v = ca.SX.sym("s_v")
        # s_dT = ca.SX.sym("s_dT")
        # s_ddelta = ca.SX.sym("s_ddelta")
        # u = ca.vertcat(dT, ddelta, s_v, s_dT, s_ddelta)
        u = ca.vertcat(dT, ddelta)

        # car physical parameters (we leave them as parameters for now to see if we could
        # leverage pre-compiled Cython solver in the future)
        l_r = ca.SX.sym("l_r")
        l_f = ca.SX.sym("l_f")
        m = ca.SX.sym("m")
        C_m = ca.SX.sym("C_m")
        C_r0 = ca.SX.sym("C_r0")
        C_r2 = ca.SX.sym("C_r2")
        p = ca.vertcat(l_r, l_f, m, C_m, C_r0, C_r2)

        # dynamics
        beta = ca.arctan(l_r / (l_f + l_r) * ca.tan(delta))
        F_x = C_m * T - C_r0 - C_r2 * v_x**2
        f_cont = ca.Function(
            "f_cont",
            [x, u],
            [
                ca.vertcat(
                    v_x * ca.cos(phi + beta),
                    v_x * ca.sin(phi + beta),
                    v_x / l_r * ca.sin(beta),
                    F_x / m,
                    dT,
                    ddelta,
                )
            ],
        )
        # integrate using RK4
        x_new = x
        for _ in range(4):
            k1 = f_cont(x_new, u)
            k2 = f_cont(x_new + 0.5 * self.controller_params.sampling_time / 4 * k1, u)
            k3 = f_cont(x_new + 0.5 * self.controller_params.sampling_time / 4 * k2, u)
            k4 = f_cont(x_new + self.controller_params.sampling_time / 4 * k3, u)
            x_new += (
                self.controller_params.sampling_time
                / 4
                / 6
                * (k1 + 2 * k2 + 2 * k3 + k4)
            )

        xdot = ca.SX.sym("xdot", 6)

        model.x = x
        model.f_expl_expr = f_cont(x, u)
        model.xdot = xdot
        model.f_impl_expr = xdot - f_cont(x, u)
        model.disc_dyn_expr = x_new
        model.u = u
        model.p = p

        # create ocp ================================================================
        ocp = AcadosOcp()
        self.ocp = ocp
        ocp.model = model
        ocp.dims.nx = x.size()[0]
        ocp.dims.nu = u.size()[0]
        ocp.dims.ny = self.nx + self.nu
        ocp.dims.ny_e = self.nx
        ocp.dims.np = p.size()[0]
        ocp.dims.N = controller_params.horizon_size
        ocp.solver_options.tf = controller_params.sampling_time * ocp.dims.N
        ocp.parameter_values = np.array(
            [
                car_params.l_r,
                car_params.l_f,
                car_params.m,
                car_params.C_m,
                car_params.C_r0,
                car_params.C_r2,
            ]
        )
        self.last_prediction = np.zeros((controller_params.horizon_size + 1, self.ny))

        # costs
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.yref = np.zeros(self.ny)
        ocp.cost.yref_e = np.zeros(self.ny_e)
        ocp.cost.Vx = np.vstack(
            (
                np.eye(self.nx),
                np.zeros((self.ny - self.nx, self.nx)),
            )
        )
        ocp.cost.Vu = np.vstack(
            (
                np.zeros((self.ny - self.nu, self.nu)),
                np.eye(self.nu),
            )
        )
        ocp.cost.Vx_e = np.eye(self.nx)
        ocp.cost.W = block_diag(
            controller_params.Q
            if len(controller_params.Q.shape) == 2
            else np.diag(controller_params.Q),
            controller_params.R
            if len(controller_params.R.shape) == 2
            else np.diag(controller_params.R),
            controller_params.R_tilde
            if controller_params.R_tilde is not None
            else np.zeros((self.control_dim, self.control_dim)),
            # controller_params.S
            # if len(controller_params.S.shape) == 2
            # else np.diag(controller_params.S),
        )
        ocp.cost.W_e = block_diag(
            controller_params.Q
            if len(controller_params.Q.shape) == 2
            else np.diag(controller_params.Q),
            controller_params.R
            if len(controller_params.R.shape) == 2
            else np.diag(controller_params.R),
        )
        ocp.cost.Zl = controller_params.Zl
        ocp.cost.Zu = controller_params.Zu
        ocp.cost.zl = controller_params.zl
        ocp.cost.zu = controller_params.zu

        ocp.constraints.x0 = np.zeros(self.nx)
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.lbu = np.array([-car_params.dT_max, -car_params.ddelta_max])
        ocp.constraints.ubu = np.array([car_params.dT_max, car_params.ddelta_max])
        ocp.constraints.idxbx = np.array([3, 4, 5])
        ocp.constraints.lbx = np.array([0.0, -1.0, -car_params.delta_max])
        ocp.constraints.ubx = np.array(
            [car_params.v_x_max + 5.0, 1.0, car_params.delta_max]
        )
        ocp.constraints.idxbx_e = np.array([3, 4, 5])
        ocp.constraints.lbx_e = np.array([0.0, -1.0, -car_params.delta_max])
        ocp.constraints.ubx_e = np.array(
            [car_params.v_x_max, 1.0, car_params.delta_max]
        )
        ocp.constraints.idxsbx = np.array([3])

        # solver
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_max_iter = 200
        ocp.solver_options.qp_solver_iter_max = 100
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.sim_method_num_steps = 6
        ocp.solver_options.print_level = 0
        ocp.solver_options.qp_solver_warm_start = 1
        ocp.code_export_directory = model.name + "_gen_code"

        self.ocp = ocp

        self.solver = AcadosOcpSolver(ocp, json_file=model.name + "_ocp.json")

    def compute_control_dev(
        self,
        current_state: np.ndarray,
        current_control: np.ndarray,
        **kwargs,
    ) -> ControlReturnDev:
        """Compute the control deviation."""
        # retrieve reference data
        XY_ref = np.asarray(kwargs["XY_ref"])
        phi_ref = np.asarray(kwargs["phi_ref"])
        v_x_ref = np.asarray(kwargs["v_x_ref"])

        offset = np.pi - current_state[2]
        phi_ref = np.mod(phi_ref + offset, 2 * np.pi) - offset
        self.last_prediction[:, 2] = (
            np.mod(self.last_prediction[:, 2] + offset, 2 * np.pi) - offset
        )

        # set initial guess
        x0 = np.append(current_state[: self.state_dim], current_control)
        if self.first_call:
            # apply the reference control ur to the current state xcurrent
            for j in range(self.horizon_size):
                self.solver.set(j, "x", x0)
                self.solver.set(j, "u", np.zeros(self.nu))
            self.solver.set(self.horizon_size, "x", x0)
            self.first_call = False
        else:
            # shift the last prediction
            self.solver.set(0, "x", x0)
            for j in range(self.horizon_size - 1):
                self.solver.set(
                    j + 1,
                    "x",
                    self.last_prediction[j + 2, : self.nx],
                )
                self.solver.set(j, "u", (self.last_prediction[j + 1, -self.nu :]))
            self.solver.set(
                self.horizon_size,
                "x",
                self.last_prediction[-1, : self.nx],  # we append the same last state
            )

        # set yref at each stage
        for i in range(self.horizon_size):
            self.solver.cost_set(
                i,
                "yref",
                np.array(
                    [
                        XY_ref[i, 0],
                        XY_ref[i, 1],
                        phi_ref[i],
                        v_x_ref[i],
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
            )
        self.solver.cost_set(
            self.horizon_size,
            "yref",
            np.array(
                [
                    XY_ref[self.horizon_size, 0],
                    XY_ref[self.horizon_size, 1],
                    phi_ref[self.horizon_size],
                    v_x_ref[self.horizon_size],
                    0.0,
                    0.0,
                ]
            ),
        )

        # set x0
        self.solver.constraints_set(0, "lbx", x0)
        self.solver.constraints_set(0, "ubx", x0)

        # solve
        status = self.solver.solve()
        if status != 0:
            # 1 – failure
            # 2 – maximum number of iterations reached
            # 3 – minimum step size in QP solver reached
            # 4 – qp solver failed
            # get slack constraints and check if any are nonzero
            for i in range(1, self.horizon_size):
                sl = self.solver.get(i, "sl")
                print("sl", sl)
                if np.any(sl > 1e-3):
                    print("stage {}, sl = {}".format(i, sl))
                su = self.solver.get(i, "su")
                print("su", su)
                if np.any(su > 1e-3):
                    print("stage {}, sl = {}".format(i, su))
            raise RuntimeError(
                "acados solver failed to solve the problem! exitflag {}, i.e. {} ".format(
                    status,
                    {
                        1: "failure",
                        2: "maximum number of iterations reached",
                        3: "minimum step size in QP solver reached",
                        4: "QP solver failed",
                    }[status],
                )
            )

        # extract prediction
        self.last_prediction[0, : self.state_dim + self.control_dim] = x0
        for i in range(self.horizon_size):
            self.last_prediction[i + 1, : self.nx] = self.solver.get(i + 1, "x")
            self.last_prediction[i, -self.nu :] = self.solver.get(i, "u")

        # return control input
        metrics = np.linalg.norm(XY_ref - self.last_prediction[:, :2], axis=1)
        np.insert(metrics, 0, metrics[0])
        return ControlReturnDev(
            control=self.last_prediction[
                1, self.state_dim : self.state_dim + self.control_dim
            ],
            control_derivative=self.last_prediction[
                0,
                self.state_dim
                + self.control_dim : self.state_dim
                + 2 * self.control_dim,
            ],
            prediction=self.last_prediction,
            reference_horizon=XY_ref,
            metric=metrics,
        )

    @property
    def nx(self) -> int:
        return self.ocp.dims.nx

    @property
    def nu(self) -> int:
        return self.ocp.dims.nu

    @property
    def ny(self) -> int:
        return self.ocp.dims.ny

    @property
    def np(self) -> int:
        return self.ocp.dims.np

    @property
    def ny_e(self) -> int:
        return self.ocp.dims.ny_e

    @property
    def nh(self) -> int:
        return self.ocp.dims.nh

    @property
    def ns(self) -> int:
        return self.ocp.dims.ns
