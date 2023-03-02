from enum import Enum
from typing import Union

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
from scipy.stats.distributions import chi2

from brains_python.common import wrapToPi

__all__ = ["EKFSLAM", "EKFSLAMMode"]


class EKFSLAMMode(Enum):
    LOCALIZATION = 0
    MAPPING = 1


class EKFSLAM:
    """
    Extended Kalman Filter Localization with use of landmarks to update the beliefs.
    """

    mode: EKFSLAMMode

    nx: int
    nu: int = 3
    threshold_1 = chi2.ppf(0.95, df=2)
    threshold_2 = chi2.ppf(0.99, df=2)
    # threshold_2 = 15.0

    mu: np.ndarray
    Sigma: np.ndarray
    sampling_time: float
    data_association_statistics: list[Union[int, float]]

    def __init__(
        self,
        mode: EKFSLAMMode,
        initial_state: np.ndarray,
        initial_state_uncertainty: np.ndarray,
        initial_map: np.ndarray,
        initial_landmark_uncertainty: np.ndarray,
        sampling_time: float,
    ):
        if mode == EKFSLAMMode.MAPPING:
            initial_map = np.empty((0, 2), dtype=float)
        assert len(initial_map.shape) == 2
        assert initial_state.shape == (3,)
        assert initial_state_uncertainty.shape == (3, 3)
        assert initial_landmark_uncertainty.shape == (2, 2)
        assert len(initial_map.shape) == 2
        if mode == EKFSLAMMode.LOCALIZATION:
            assert initial_map.shape[0] > 0

        self.mode = mode
        self.mu = np.hstack((initial_state, np.ravel(initial_map)))
        self.nx = self.mu.shape[0]
        self.Sigma = block_diag(
            *(
                [initial_state_uncertainty]
                + [initial_landmark_uncertainty] * initial_map.shape[0]
            )
        )
        self.sampling_time = sampling_time
        self.data_association_statistics = []

    @property
    def state(self) -> np.ndarray:
        return self.mu[:3]

    @property
    def map(self) -> np.ndarray:
        return self.mu[3:].reshape((-1, 2))

    @property
    def nlandmarks(self) -> int:
        return (self.nx - 3) // 2

    def odometry_update(
        self,
        odometry: np.ndarray,
        odometry_uncertainty: np.ndarray,
        sampling_time: float = None,
    ):
        assert odometry.shape == (self.nu,)
        assert odometry_uncertainty.shape == (
            self.nu,
        ) or odometry_uncertainty.shape == (self.nu, self.nu)
        if len(odometry_uncertainty.shape) == 1:
            odometry_uncertainty = np.diag(odometry_uncertainty)

        dt = sampling_time if sampling_time is not None else self.sampling_time
        phi = self.mu[2]
        v_x = odometry[0]
        v_y = odometry[1]
        r = odometry[2]
        G = np.array(
            [
                [1.0, 0.0, (-v_x * np.sin(phi) - v_y * np.cos(phi)) * dt],
                [0.0, 1.0, (v_x * np.cos(phi) - v_y * np.sin(phi)) * dt],
                [0.0, 0.0, dt],
            ]
        )
        V = np.array(
            [
                [np.cos(phi) * dt, -np.sin(phi) * dt, 0.0],
                [np.sin(phi) * dt, np.cos(phi) * dt, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.mu[:3] += np.array(
            [
                (v_x * np.cos(phi) - v_y * np.sin(phi)) * dt,
                (v_x * np.sin(phi) + v_y * np.cos(phi)) * dt,
                r * dt,
            ]
        )
        self.mu[2] = wrapToPi(self.mu[2])
        tpr = G @ self.Sigma[:3, 3:]
        self.Sigma[:3, :3] = (
            G @ self.Sigma[:3, :3] @ G.T + V @ odometry_uncertainty @ V.T
        )
        self.Sigma[:3, 3:] = tpr
        self.Sigma[3:, :3] = tpr.T

    def yaw_update(self, yaw: float, yaw_uncertainty: float = 2e-2):
        H = np.zeros(self.nx)
        H[2] = 1.0
        K = self.Sigma[:, 2] / (self.Sigma[2, 2] + yaw_uncertainty)  # Kalman gain
        self.mu = self.mu + K * wrapToPi(yaw - self.mu[2])
        self.mu[2] = wrapToPi(self.mu[2])
        self.Sigma = self.Sigma - K[:, None] @ (K[None, :] * (self.Sigma[2, 2] + 1e-3))

    def cones_update(
        self,
        observations: np.ndarray,
        observations_uncertainties: np.ndarray,
        cones_coordinates: str = "polar",
    ):
        """
        :param observations: shape (n, 2) array of cones positions in polar coordinates
        :param observations_uncertainties: shape (2,) or (2,2), uncertainty of the observations
        :param cones_coordinates: how the observations are specified, in "polar" or "cartesian" coordinates (always in the car frame)
        """
        # INPUT VALIDATION ============================================================
        assert len(observations.shape) == 2 and observations.shape[1] == 2
        if len(observations_uncertainties.shape) == 1:
            observations_uncertainties = np.diag(observations_uncertainties)
        assert observations_uncertainties.shape == (2, 2)

        # DATA ASSOCIATION STEP =========================================================
        if cones_coordinates == "cartesian":
            observations = np.array(
                [
                    np.hypot(observations[:, 0], observations[:, 1]),
                    wrapToPi(np.arctan2(observations[:, 1], observations[:, 0])),
                ]
            ).T

        I = self.nlandmarks  # number of known landmarks
        J = observations.shape[0]  # number of observations

        # filter potential landmarks
        potential_landmarks_idx = self._filter_potential_landmarks()
        Iprime = len(
            potential_landmarks_idx
        )  # number of known landmarks after filtering
        if Iprime > 0:
            delta = self.map - self.state[:2]  # shape (I', 2)
            q = np.sum(delta**2, axis=1)  # shape (I',)
            z_hat = np.array(
                [
                    np.sqrt(q),
                    wrapToPi(np.arctan2(delta[:, 1], delta[:, 0]) - self.mu[2]),
                ]
            ).T  # shape (I', 2)
            Hs = np.array(
                [
                    [
                        [
                            -np.sqrt(q[i]) * delta[i, 0],
                            -np.sqrt(q[i]) * delta[i, 1],
                            0.0,
                            np.sqrt(q[i]) * delta[i, 0],
                            np.sqrt(q[i]) * delta[i, 1],
                        ],
                        [
                            delta[i, 1],
                            -delta[i, 0],
                            -q[i],
                            -delta[i, 1],
                            delta[i, 0],
                        ],
                    ]
                    for i in potential_landmarks_idx
                ]
            )  # shape (I', 2, 5)
            S = np.array(
                [
                    Hs[i]
                    @ np.block(
                        [
                            [
                                self.Sigma[:3, :3],
                                self.Sigma[:3, 2 * i + 3 : 2 * i + 5],
                            ],
                            [
                                self.Sigma[2 * i + 3 : 2 * i + 5, :3],
                                self.Sigma[
                                    2 * i + 3 : 2 * i + 5, 2 * i + 3 : 2 * i + 5
                                ],
                            ],
                        ]
                    )
                    @ Hs[i].T
                    + observations_uncertainties
                    for i in range(Iprime)
                ]
            )  # shape (I', 2, 2)
            Sinv = np.linalg.inv(S)  # shape (I', 2, 2)
            delta_z = np.transpose(
                observations[:, np.newaxis, :]
                - z_hat[np.newaxis, potential_landmarks_idx, :],
                (1, 0, 2),
            )  # shape (I', J, 2)
            delta_z[:, :, 1] = wrapToPi(delta_z[:, :, 1])  # shape (I', J, 2)

            # compute the mahalanobis distance: D[j, i] = delta_z[i, j] @ Sinv[i] @ delta_z[i, j]
            D = np.einsum("ijk,ikl,ijl->ij", delta_z, Sinv, delta_z).T  # shape (J, I')
            if J > Iprime:
                D = np.hstack(
                    (D, 2 * self.threshold_2 * np.ones((J, J - Iprime)))
                )  # shape (J,J)
            observation_id, landmark_id = linear_sum_assignment(D)

            associations = []
            self.data_association_statistics = [0, 0, 0]
            for j, i in zip(observation_id, landmark_id):
                if D[j, i] < self.threshold_1:
                    # observation j has been associated with landmark i
                    self.data_association_statistics[0] += 1
                    associations.append((j, i))
                elif self.mode == EKFSLAMMode.MAPPING and D[j, i] > self.threshold_2:
                    # observation j has not been associated with any landmark, so we create a new one
                    self.data_association_statistics[2] += 1
                    self._add_landmark(observations[j, :], observations_uncertainties)
                else:
                    # observation i has been discarded because it is not a good enough match
                    self.data_association_statistics[1] += 1

            # CONES UPDATE STEP ======================================================================
            for j, i in associations:
                K = (
                    np.hstack((self.Sigma[:, :3], self.Sigma[:, 2 * i + 3 : 2 * i + 5]))
                    @ Hs[i].T
                    @ Sinv[i]
                )
                self.mu += K @ delta_z[i, j]
                self.mu[2] = wrapToPi(self.mu[2])
                self.Sigma -= K @ S[i] @ K.T
        else:
            # we don't have any landmark yet
            if self.mode == EKFSLAMMode.LOCALIZATION:
                raise RuntimeError(
                    "Something went terribly wrong, we have more observations that landmarks"
                )
            for j in range(J):
                self._add_landmark(observations[j], observations_uncertainties)
            self.data_association_statistics = [0, 0, J]

        self.data_association_statistics = [
            x / J for x in self.data_association_statistics
        ]

    def _filter_potential_landmarks(self):
        return np.arange(self.nlandmarks)

    def _add_landmark(
        self,
        observation: np.ndarray,
        observations_uncertainties: np.ndarray = 1e-2 * np.eye(2),
    ):
        """Add a new landmark to the state vector and covariance matrix."""
        self.mu = np.append(
            self.mu,
            [
                self.mu[0] + observation[0] * np.cos(self.mu[2] + observation[1]),
                self.mu[1] + observation[0] * np.sin(self.mu[2] + observation[1]),
            ],
        )
        H_inv_vehicle = np.array(
            [
                [
                    1,
                    0,
                    -observation[0] * np.sin(self.mu[2] + observation[1]),
                ],
                [
                    0,
                    1,
                    observation[0] * np.cos(self.mu[2] + observation[1]),
                ],
            ]
        )
        H_inv_landmark = np.array(
            [
                [
                    np.cos(self.mu[2] + observation[1]),
                    -observation[0] * np.sin(self.mu[2] + observation[1]),
                ],
                [
                    np.sin(self.mu[2] + observation[1]),
                    -observation[0] * np.cos(self.mu[2] + observation[1]),
                ],
            ]
        )
        tpr = H_inv_vehicle @ self.Sigma[:3, :]
        self.Sigma = np.block(
            [
                [
                    self.Sigma,
                    tpr.T,
                ],
                [
                    tpr,
                    H_inv_vehicle @ self.Sigma[:3, :3] @ H_inv_vehicle.T
                    + H_inv_landmark @ observations_uncertainties @ H_inv_landmark.T,
                ],
            ]
        )
        self.nx += 2
