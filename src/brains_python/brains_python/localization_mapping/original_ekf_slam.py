from collections import deque
from typing import Dict, Union, Deque

import numpy as np

from .localizer import Localizer

__all__ = ["Localizer"]

# global constants
state_size = 5  # [X, Y, phi, v_x, r]
landmark_size = 2  # [X_cone, Y_cone]


class EKFLocalizer(Localizer):
    """
    Perform localization based on Extended Kalman Filter Slam algorithm :
    Attributes :
    - self.mu : [state , landmarks], shape = [[x], [y], [yaw], [x1], [y1], [x2], [y2], ...]
    - self.cov : covariance matrix of mu, shape = (len(mu), len(mu))
    General concept :
    - Initialize cone map in mu (give high confidence map is perfect, otherwise give low confidence
        to allow mapping adjustments from slam algorithm
    - Give high confidence in initial state
    - Do not allow new landmarks, size of mu is fixed
    """

    _slam_mode: bool
    _depth_view_limit: float
    _angle_view_limit: float
    _use_torch: bool
    _velocity_from_history: bool
    _velocity_movmean_size: int

    _map: np.ndarray
    _nbr_landmarks: int

    mu: np.ndarray
    cov: np.ndarray
    dead_reckoning: np.ndarray
    history: Deque[np.ndarray]
    time_history: Deque[float]

    # State and Observation covariance for ekf gain
    Q = (
        np.diag([0.2, 0.2, np.deg2rad(3.0), 0.1, np.deg2rad(1.0)]) ** 2
    )  # predict state covariance
    R = np.diag([5, 5]) ** 2  # Observation x,y position covariance
    new_landmark_sensitivity = 1000.0  # minimal distance two cones
    new_landmark_covariance = 5.0  # initial covariance of a new observed cone

    def __init__(
        self,
        config: Dict[str, Union[str, float, bool, list]],
        initial_landmark_positions: np.ndarray,
    ):
        self._slam_mode = config["slam_mode"]
        self._depth_view_limit = config["depth_view_limit"]
        self._angle_view_limit = config["angle_view_limit"]
        self._use_torch = config["use_torch"]
        self._velocity_from_history = config["velocity_from_history"]
        self._velocity_movmean_size = config["nbr_states"]
        assert self._velocity_movmean_size >= 2

        # load the track
        self._map = initial_landmark_positions
        self._nbr_landmarks = len(self._map)

        # EKF initialization
        self.mu_size = self._velocity_movmean_size + self._nbr_landmarks * landmark_size
        self.mu = np.append(
            (np.array([0.0, 0.0, np.pi / 2, 0.0, 0.0])), self._map.flatten()
        )

        self.cov = np.diag(
            np.append(
                (
                    np.array(
                        [0.1, 0.1, np.deg2rad(0.1), 0.001, np.deg2rad(0.1)],
                    ),  # initial state covariance
                    0.01
                    * np.ones(
                        self._nbr_landmarks * landmark_size
                    ),  # initial landmark covariance
                )
            )
        )
        self.dead_reck = [self.mu[:state_size].copy()]
        self.history = deque(maxlen=self._velocity_movmean_size)
        self.time_history = deque(maxlen=self._velocity_movmean_size - 1)

    def localize(
        self, cones: np.ndarray, motion_data: np.ndarray = None, dt: float = None
    ):
        """
        Localize the car from observations and motion data with the tuned EKF SLAM
        @param cones: observed cones
        @param motion_data: [v_x, r]
        @param dt: delta time between iteration
        @return: [X, Y, phi, v_x, r]
        """
        # verify inputs
        assert len(cones.shape) == 2 and cones.shape[1] == 2

        # PREDICTION STEP =========================================================
        self.mu[:state_size] = EKFLocalizer.motion_model(
            self.mu[:state_size], motion_data, dt
        )
        G = EKFLocalizer.jacob_motion(
            self.mu[:state_size], velocity=motion_data[0], dt=dt
        )

        self.cov[:state_size, :state_size] = (
            G.T @ self.cov[:state_size, :state_size] @ G + EKFLocalizer.Q
        )

        self.dead_reck.append(
            EKFLocalizer.motion_model(self.dead_reck[-1], motion_data, dt)
        )

        # UPDATE STEP =============================================================
        z = cart2polar(cones)
        for iz in range(z.shape[0]):  # for each observation
            min_id = self.search_correspond_landmark_id(z[iz, :])
            if min_id == self._nbr_landmarks:
                print("New cone detected")
                # Extend state and covariance matrix
                self.mu = np.vstack(
                    (self.mu, calc_landmark_position(self.mu, z[iz, :]))
                )
                self.cov = np.bmat(
                    [
                        [self.cov, None],
                        [None, EKFLocalizer.new_landmark_covariance * np.eye(2)],
                    ]
                )
                self._nbr_landmarks += 1

            lm = self.get_landmark_position_from_state(min_id)
            y, S, H = self.calc_innovation(lm, z[iz, 0:2], min_id)

            K = (self.cov @ H.T) @ np.linalg.inv(S)
            self.mu = self.mu + (K @ y)

            self.cov = (
                np.eye(state_size + 2 * self._nbr_landmarks) - (K @ H)
            ) @ self.cov

        self.mu[2] = wrap_to_pi(self.mu[2])
        state = self.mu[:state_size]

        # Update history
        self.history.append(state[:2])
        self.time_history.append(dt)
        if len(self.time_history) > self._velocity_movmean_size - 2:
            history = np.array(self.history)
            delta = history[1:] - history[:-1]
            velocity = np.sum(np.linalg.norm(delta, axis=1))
            time_sum = np.sum(self.time_history)
            velocity /= time_sum
            state[3] = velocity

        return state

    @staticmethod
    def motion_model(x, u, dt):
        """
        Linearized motion model

        :param x: state [X, Y, phi, v_x, r]
        :param u: input [v_x, r]
        :param dt: sampling time
        :returns: predicted state
        """
        F = np.array(
            [
                [1.0, 0.0, 0.0, dt * np.cos(x[2, 0]), 0.0],
                [0.0, 1.0, 0.0, dt * np.sin(x[2, 0]), 0.0],
                [0.0, 0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        B = np.array(
            [
                [0.0, 0.0],
                [0, 0, 0.0],
                [0.0, 0, 0],
                [0.1, 0, 0],
                [0.0, 1.0],
            ]
        )

        return F @ x + B @ u

    @staticmethod
    def jacob_motion(state: np.ndarray, velocity: float, dt: float):
        """
        Get the jacobian matrix of the motion dynamics

        :param state: state [X, Y, phi]
        :param velocity: (longitudinal) velocity
        :param dt: delta time between iterations
        :return: jacobian matrix (size=(state_size, STATE:SIZE))
        """
        jF = np.array(
            [
                [
                    0.0,
                    0.0,
                    -dt * velocity * np.sin(state[2]),
                    dt * np.cos(state[2]),
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    dt * velocity * np.cos(state[2]),
                    dt * np.sin(state[2]),
                    0.0,
                ],
                [0.0, 0.0, 0.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        G = np.eye(state_size) + jF

        return G

    def search_correspond_landmark_id(self, observed_cone: np.ndarray):
        """
        Find the closest landmark to the observed cone

        :param observed_cone: an observed cone in polar coordinates
        :return: id (int) of the closest cone in the reference map
        """
        landmarks = self.mu[state_size:].reshape(-1, 2)
        xyi = calc_landmark_position(self.mu, observed_cone).squeeze()
        dist = np.linalg.norm(landmarks - xyi, axis=1)
        argmin = np.argmin(dist)
        if dist[argmin] < EKFLocalizer.new_landmark_sensitivity:
            return argmin
        else:
            # new landmark
            return self._nbr_landmarks

    def calc_innovation(self, lm, z, lm_id):
        """
        Calculate the new confidence in the landmark's position
        :param lm: landmark == a cone
        :param z: observed cone
        :param lm_id: id of the closest landmark in the reference map
        :return: confidence in position
        """
        delta = lm - self.mu[:2]
        q = delta.T @ delta
        z_angle = np.arctan2(delta[1], delta[0]) - self.mu[2, 0]
        zp = np.array([[np.sqrt(q), wrap_to_pi(z_angle)]])
        y = (z - zp).T
        y[1] = wrap_to_pi(y[1])
        H = self.jacobian_observation(q, delta, lm_id + 1)
        S = H @ self.cov @ H.T + EKFLocalizer.R
        return y, S, H

    def get_landmark_position_from_state(self, index: int) -> np.ndarray:
        """
        Get the landmark position in self.mu array from landmark index
        @param index: (int)
        @return: [x, y]
        """
        return self.mu[
            state_size
            + landmark_size * index : state_size
            + landmark_size * (index + 1),
            :,
        ]

    def jacobian_observation(self, q, delta, i) -> np.ndarray:
        """
        Get the jacobian matrix from the observation deltas
        @param q: norm of delta ** 2
        @param delta: delta between observed lm and hard coded lm
        @param i: lm id + 1
        @return:
        """
        sq = np.sqrt(q)  # norm of delta vector
        G = np.array(
            [
                [-sq * delta[0], -sq * delta[1], 0, 0, 0, sq * delta[0], sq * delta[1]],
                [delta[1], -delta[0], -q, 0, 0, -delta[1], delta[0]],
            ]
        )

        G = G / q
        F1 = np.hstack((np.eye(state_size), np.zeros((state_size, 2 * self.n_lm))))
        F2 = np.hstack(
            (
                np.zeros((2, state_size)),
                np.zeros((2, 2 * (i - 1))),
                np.eye(2),
                np.zeros((2, 2 * self._nbr_landmarks - 2 * i)),
            )
        )

        F = np.vstack((F1, F2))
        H = G @ F
        return H

    # def get_landmark_covariance(self, index):
    #     """
    #     Get the landmark position in self.mu array from landmark index
    #     @param index: (int)
    #     @return: [x, y]
    #     """
    #     index = state_size + landmark_size * index
    #     slice_index = slice(index, index + 2)
    #     return self.cov[slice_index, slice_index]


def calc_landmark_position(x, z):
    """
    Calculate the landmark global cartesian ccoordinates [X, Y] from its local (relative to the car) polar coordinates
    [r, theta] and the car's pose [X, Y, phi]
    :param x: estimated car pose [X, Y, phi]
    :param z: observed landmark polar coordinates [r, theta]
    """
    return np.array(
        [x[0] + z[0] * np.cos(x[2] + z[1]), x[1] + z[0] * np.sin(x[2] + z[1])]
    )


def wrap_to_pi(angle: Union[float, np.ndarray]):
    """Normalize angles between [-pi, pi]"""
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def cart2polar(cones: np.ndarray) -> np.ndarray:
    """
    From cartesian coordinate to polar with theta = 0 :: straight foward
    @param cones: array of cones in cartesian coordinates
    @return: array of cones in polar coordinates
    """
    x, y = cones[:, 0], cones[:, 1]
    r = np.hypot(x, y)
    theta = wrap_to_pi(np.arctan2(y, x) - np.pi / 2)
    z = np.vstack((r, theta)).T
    return z


if __name__ == "__main__":
    EKFLocalizer()
    print("hello")
