# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

from brains_python.common.params import Params


__all__ = [
    "Controller",
    "ControlReturn",
    "ControlReturnDev",
    "CarParams",
    "ControllerParams",
]


class ControlReturn:
    """
    PODS (Plain Old Data Structure) class that contains the control [T,delta] and its
    derivative [dT, ddelta].
    Is returned by the methods Controller.compute_control() and
    MotionPlannerController.compute_control() .
    """

    control: np.ndarray
    control_derivative: np.ndarray

    def __init__(self, control: np.ndarray, control_derivative: np.ndarray):
        self.control = control
        self.control_derivative = control_derivative


class ControlReturnDev(ControlReturn):
    """
    PODS (Plain Old Data Structure) class that contains the control [T,delta], its
    derivative [dT, ddelta] and other useful measures for debugging and development
    purposes: the prediction given returned by an MPC, the closest point on the
    reference trajectory given the current car position and some internal metrics
    used to calibrate the controller.
    Is returned by the methods Controller.compute_control_dev() and
    MotionPlannerController.compute_control_dev() .
    IMPORTANT: metric should always contain at least the cross track error
    """

    prediction: Optional[np.ndarray]
    reference_horizon: np.ndarray
    metric: Union[float, np.ndarray]

    def __init__(
        self,
        control: np.ndarray,
        control_derivative: np.ndarray,
        prediction: Optional[np.ndarray],
        reference_horizon: np.ndarray,
        metric: Union[float, np.ndarray],
    ):
        super().__init__(control, control_derivative)
        self.prediction = prediction
        self.reference_horizon = reference_horizon
        self.metric = metric


class CarParams(Params):
    """
    PODS (Plain Old Data Structure) class that contains all the useful physical
    parameters of a car.
    - m [kg]: mass of the car
    - I_z [kg.m^2]: moment of inertia of the car (around the z axis)
    - l_f [m]: distance between the front axle and the center of mass
    - l_r [m]: distance between the rear axle and the center of mass
    - L [m]: length of the car
    - W [m]: width of the car
    - a_y_max [m.s^-2]: maximum lateral acceleration of the car
    - v_x_max [m.s^-1]: maximum longitudinal velocity of the car
    - delta_max [rad]: maximum steering angle of the car
    - ddelta_max [rad.s^-1]: maximum steering angle rate of the car
    - dT_max [1/s]: maximum rate of change of the throttle
    - C_m, C_r0, C_r2: parameters of the longitudinal dynamics model
    - B, C, D: parameters of the Pacejka model for the lateral forces on the tyres
    """

    m: float
    I_z: float
    l_f: float
    l_r: float
    L: float
    W: float
    # car dynamics
    a_y_max: float
    v_x_max: float
    delta_max: float
    ddelta_max: float
    dT_max: float
    C_m: float
    C_r0: float
    C_r2: float
    # Pacejka coefficients
    B_f: float
    C_f: float
    D_f: float
    B_r: float
    C_r: float
    D_r: float

    def __init__(self, **kwargs):
        current_params, remaining_params = CarParams._transform_dict(kwargs)
        for key, val in current_params.items():
            setattr(self, key, val)
        super().__init__(**remaining_params)


class ControllerParams(Params):
    """
    PODS (Plain Old Data Structure) class that contains all the useful parameters of a
    Controller. Only contains one field common to all Controllers and is subclassed by
    other Controllers to add their own parameters.
    Example:
         StanleyParams add the parameters k_e, k_delta and many others.
    """

    sampling_time: float

    def __init__(self, **kwargs):
        current_params, remaining_params = ControllerParams._transform_dict(kwargs)
        for key, val in current_params.items():
            setattr(self, key, val)
        super().__init__(**remaining_params)

    @property
    def controller_type(self) -> type:
        return Controller


class Controller(ABC):
    """
    Abstract class that defines the interface for all controllers.
    Is subclassed by Stanley and IHM that implement the compute_control_dev() method.
    """

    car_params: CarParams
    controller_params: ControllerParams
    controller_params_type = ControllerParams

    state_dim: int
    control_dim: int

    horizon_size: int

    def __init__(
        self,
        controller_params: ControllerParams,
        car_params: CarParams,
        state_dim: int,
        control_dim: int,
        horizon_size: int,
    ):
        self.controller_params = controller_params
        self.car_params = car_params
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.horizon_size = horizon_size

    @abstractmethod
    def compute_control_dev(
        self,
        current_state: np.ndarray,
        current_control: np.ndarray,
        **kwargs,
    ) -> ControlReturnDev:
        """
        :param current_state: state in the form [X, Y, phi, v_x, v_y, r]
        :param current_control: control in the form [T, delta]
        """
        pass
