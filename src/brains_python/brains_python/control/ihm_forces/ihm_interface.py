# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import warnings

try:
    from forcespro import CodeOptions
    from forcespro.nlp import SymbolicModel, Solver
except ImportError:
    warnings.warn(
        "forcespro wasn't found in the PYTHONPATH env variable, please add it if you really want to use IHM."
    )
    CodeOptions = None
    Solver = None
    SymbolicModel = None

import numpy as np

from .ihm_model import PhysicalModel, load_solver
from ..controller import *

__all__ = ["IHMForces", "IHMForcesParams"]


class IHMForcesParams(ControllerParams):
    solver_path: str
    physical_model: PhysicalModel
    horizon_size: int
    # MPC costs
    q_d: float
    q_dT: float
    q_ddelta: float

    def __init__(self, **kwargs):
        current_params, remaining_params = IHMForcesParams._transform_dict(kwargs)
        for key, val in current_params.items():
            setattr(self, key, val)
        super().__init__(**remaining_params)

    @property
    def controller_type(self) -> type:
        return IHMForces


class IHMForces(Controller):
    controller_params_type = IHMForcesParams
    controller_params: IHMForcesParams

    solver: Solver
    model: SymbolicModel
    reinitialize_solver: bool

    def __init__(
        self, car_params: CarParams, controller_params: IHMForcesParams, **kwargs
    ):
        super().__init__(
            car_params=car_params,
            controller_params=controller_params,
            state_dim=int(controller_params.physical_model) + 2,
            control_dim=2,
            horizon_size=controller_params.horizon_size,
        )
        # import the solver
        (
            self.model,
            self.solver,
            horizon_size,
            sampling_time,
            physical_model,
        ) = load_solver(controller_params.solver_path)

        # verify that the specified controller params correspond to the loaded solver
        assert (
            controller_params.physical_model == physical_model
        ), "physical model mismatch"
        assert controller_params.horizon_size == horizon_size, "horizon size mismatch"
        assert (
            controller_params.sampling_time == sampling_time
        ), "sampling time mismatch"
        assert self.model.neq == self.state_dim, "state dim mismatch"
        assert (
            self.model.nvar - self.model.neq == self.control_dim
        ), "control dim mismatch"

        # bruh
        self.reinitialize_solver = True

    def compute_control_dev(
        self,
        current_state: np.ndarray,
        current_control: np.ndarray,
        **kwargs,
    ) -> ControlReturnDev:
        XY_ref = kwargs["XY_ref"]

        # check the dimensions of the data
        assert current_state.shape == (
            6,
        ), "wrong current_state shape have {} but expected (6,)".format(
            current_state.shape
        )
        assert current_control.shape == (
            2,
        ), "wrong current_control shape have {} but expected (6,)".format(
            current_control.shape
        )
        assert XY_ref.shape == (
            self.horizon_size,
            2,
        ) or XY_ref.shape == (
            2,
        ), "wrong XY_ref shape have {} but expected ({},2) or (2,)".format(
            XY_ref.shape, self.horizon_size
        )

        state = np.append(
            current_state[: int(self.controller_params.physical_model)], current_control
        )

        problem = {
            # "reinitialize": self._reinitialize,
            "x0": np.tile(
                np.concatenate((np.zeros(self.control_dim), state), axis=0),
                self.horizon_size,
            ),
            "xinit": state,
            "all_parameters": self.compute_parameters(XY_ref),
            "lb": np.tile(
                [
                    -self.car_params.dT_max,
                    -self.car_params.ddelta_max,
                    0.0,
                    -1.0,
                    -self.car_params.delta_max,
                ],
                (self.horizon_size,),
            ),
            "ub": np.tile(
                [
                    self.car_params.dT_max,
                    self.car_params.ddelta_max,
                    self.car_params.v_x_max,
                    1.0,
                    self.car_params.delta_max,
                ],
                (self.horizon_size,),
            ),
        }
        # once we have solved the problem once, we don't reinitialize anymore
        if self.reinitialize_solver:
            self.reinitialize_solver = False

        # Time to solve the MPC!
        output, exitflag, info = self.solver.solve(problem)

        # Make sure the solver has exited properly.
        try:
            assert exitflag == 1, "bad exitflag : {}".format(exitflag)
        except AssertionError as e:
            print(e)

        # Extract MPC prediction
        prediction = np.zeros((self.state_dim + self.control_dim, self.horizon_size))
        for i in range(0, self.horizon_size):
            prediction[:, i] = output["x{0:02d}".format(i + 1)]

        return ControlReturnDev(
            control=prediction[-2:, 1],
            control_derivative=prediction[: self.control_dim, 0],
            prediction=prediction.T,
            reference_horizon=XY_ref,
            metric=np.linalg.norm(
                prediction[self.control_dim : self.control_dim + 2, 1] - XY_ref[:, 1]
            ),
        )

    def compute_parameters(self, next_path_points: np.ndarray):
        params_list = [
            self.controller_params.q_d,
            self.controller_params.q_dT,
            self.controller_params.q_ddelta,
            self.car_params.m,
            self.car_params.l_r,
            self.car_params.l_f,
            self.car_params.C_m,
            self.car_params.C_r0,
            self.car_params.C_r2,
        ]
        if (
            self.controller_params.physical_model == PhysicalModel.DYN_6
            or self.controller_params.physical_model == PhysicalModel.KIN_DYN_6
        ):
            params_list.append(self.car_params.B)
            params_list.append(self.car_params.C)
            params_list.append(self.car_params.D)
            params_list.append(self.car_params.I_z)

        return (
            np.reshape(
                np.concatenate(
                    (
                        next_path_points,
                        np.tile(
                            np.array(params_list).reshape(-1, 1),
                            (1, self.horizon_size),
                        ),
                    ),
                    axis=0,
                ),
                (self.model.npar * self.horizon_size, 1),
                order="F",
            ),
        )
