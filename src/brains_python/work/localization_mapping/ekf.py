import os.path
from typing import Tuple, Union, List

from numpy import ndarray
from scipy.linalg import block_diag
from scipy.stats.distributions import chi2
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import track_database as tdb
from track_database.utils import plot_cones
from fsds_client import HighLevelClient
from fsds_client.utils import *

np.random.seed(127)
bruh = True


def mod(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


class EKFLocalizer:
    """
    Extended Kalman Filter Localization with use of landmarks to update the beliefs.
    """

    nx: int = 3
    nu: int = 3
    chi2_95 = chi2.ppf(0.95, df=2)

    map: np.ndarray
    x: np.ndarray
    Sigma: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    association_sensitivity: float
    dt: float

    def __init__(
        self,
        map: np.ndarray,
        x0: np.ndarray,
        Sigma0: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        dt: float,
        association_sensitivity: float,
    ):
        self.map = map
        self.x = x0
        self.Sigma = Sigma0
        self.Q = Q
        self.R = R
        self.association_sensitivity = association_sensitivity
        self.dt = dt

    def localize(
        self,
        cones: np.ndarray,
        u: np.ndarray,
        phi_obs: float,
        dt: float = None,
        cones_coordinates: str = "polar",
    ) -> np.array:
        """
        :param cones: shape (n, 2) array of cones positions in polar coordinates
        :param u: shape (3,), u=[v_x, v_y, r]
        :param phi_obs: float, orientation of the robot observed with IMU

        """
        # verify inputs
        assert len(cones.shape) == 2 and cones.shape[1] == 2
        assert u.shape == (self.nu,)
        if dt is None:
            dt = self.dt

        # PREDICTION STEP =========================================================
        new_x_hat, G, V = self.motion_model(self.x, u, dt)
        new_x_hat[2] = mod(new_x_hat[2])
        new_Sigma_hat = G @ self.Sigma @ G.T + V @ self.Q @ V.T

        # UPDATE STEP =========================================================
        # if necessary, convert cones positions in polar coordinates
        if cones_coordinates == "cartesian":
            local_polar_coordinates = np.array(
                [
                    np.hypot(cones[:, 0], cones[:, 1]),
                    np.mod(np.arctan2(cones[:, 1], cones[:, 0]) + np.pi, 2 * np.pi)
                    - np.pi,
                ]
            ).T
        else:
            local_polar_coordinates = cones

        new_x_hat, new_Sigma_hat = self.yaw_update(new_x_hat, new_Sigma_hat, phi_obs)

        # associate each cone to a landmark, i.e. create a list of tuples (cone, landmark)
        # where cone is in local polar coordinates and landmark is in global cartesian coordinates
        if bruh:
            associated_landmarks = self.associate_landmarks(local_polar_coordinates)
        else:
            associated_landmarks = self.associate_landmarks_2(
                local_polar_coordinates, new_x_hat, new_Sigma_hat
            )

        # update the state and covariance using the cones observations
        for z, lm in associated_landmarks:
            tpr = self.x
            tpr[2] = new_x_hat[2]
            z_hat, H = self.cone_observation_model(lm, tpr)
            K = (
                new_Sigma_hat @ H.T @ np.linalg.inv(H @ new_Sigma_hat @ H.T + self.R)
            )  # Kalman gain for this observation
            new_x_hat = new_x_hat + K @ (z - z_hat)
            new_Sigma_hat = (np.eye(self.nx) - K @ H) @ new_Sigma_hat

        # correct using yaw observation from IMU (considered precise)
        new_x_hat, new_Sigma_hat = self.yaw_update(new_x_hat, new_Sigma_hat, phi_obs)

        self.x = new_x_hat
        self.Sigma = new_Sigma_hat

        return self.x, self.Sigma

    def motion_model(self, x, u, dt):
        phi = x[2]
        v_x = u[0]
        v_y = u[1]
        r = u[2]
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
        new_x_hat = self.x + np.array(
            [
                (v_x * np.cos(phi) - v_y * np.sin(phi)) * dt,
                (v_x * np.sin(phi) + v_y * np.cos(phi)) * dt,
                r * dt,
            ]
        )
        return new_x_hat, G, V

    def cone_observation_model(self, landmark, x):
        delta = landmark - x[:2]
        q = np.dot(delta, delta)
        z_hat = np.array([np.sqrt(q), mod(np.arctan2(delta[1], delta[0]) - x[2])])
        H = np.array(
            [
                [-np.sqrt(q) * delta[0], -np.sqrt(q) * delta[1], 0.0],
                [delta[1], -delta[0], -q],
            ]
        )  # Jacobian of observation model h(x)
        return z_hat, H

    def yaw_update(self, x, Sigma, phi_obs):
        H = np.array([0.0, 0.0, 1.0])  # Jacobian of observation model h(x)
        K = Sigma @ H.T / (float(H @ Sigma @ H.T) + 0.0000001)  # Kalman gain
        return x + K * (phi_obs - x[2]), (np.eye(self.nx) - K * H) @ Sigma

    def associate_landmarks(
        self, cones: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Find the closest landmark to each observed cone

        :param cones: an array of observed cones in local polar coordinates
        :returns: a list of tuples (cone, landmark) where cone is the observed cone in
         local polar coordinates and landmark is in global cartesian coordinates
        """
        result = []
        global_cartesian_coords = np.array(
            (
                self.x[0] + cones[:, 0] * np.cos(cones[:, 1] + self.x[2]),
                self.x[1] + cones[:, 0] * np.sin(cones[:, 1] + self.x[2]),
            )
        ).T
        dist_mat = distance_matrix(global_cartesian_coords, self.map)
        for i, cone in enumerate(cones):
            j = dist_mat[i, :].argmin()
            if dist_mat[i, j] < self.association_sensitivity:
                result.append((cone, self.map[j]))

        return result

    def associate_landmarks_2(
        self, observations: np.ndarray, new_x_hat: np.ndarray, new_Sigma_hat: np.ndarray
    ):
        # first construct the matrix with all mahalanobis distances
        # start = perf_counter()
        I = self.map.shape[0]
        J = observations.shape[0]
        delta = self.map - self.x[:2]  # shape (I, 2)
        q = np.sum(delta**2, axis=1)  # shape (I,)
        z_hat = np.array(
            [np.sqrt(q), mod(np.arctan2(delta[:, 1], delta[:, 0]) - new_x_hat[2])]
        ).T  # shape (I, 2)
        H = np.moveaxis(
            np.array(
                [
                    [-np.sqrt(q) * delta[:, 0], -np.sqrt(q) * delta[:, 1], np.zeros(I)],
                    [delta[:, 1], -delta[:, 0], -q],
                ]
            ),
            2,
            0,
        )  # shape (I, 2, 3)
        Sinv = np.linalg.inv(
            H @ new_Sigma_hat @ H.transpose((0, 2, 1)) + self.R
        )  # shape (I, 2, 2)
        tpr = np.transpose(
            observations[:, None, :] - z_hat[None, :, :], (1, 0, 2)
        )  # shape (I, J, 2)
        tpr[:, :, 1] = mod(tpr[:, :, 1])  # shape (I, J, 2)
        # remove along the first axis the elements with bearing difference greater than pi/2 or distance greater than 10
        # potential_landmarks_idx = np.where(
        #     (-np.pi / 2 - np.sqrt(self.R[1, 1]) <= z_hat[:, 1])
        #     & (z_hat[:, 1] <= np.pi / 2 + np.sqrt(self.R[1, 1]))
        #     & (z_hat[:, 0] <= 25.0)
        # )
        potential_landmarks_idx = np.arange(I)
        tpr = tpr[potential_landmarks_idx]  # shape (I', J, 2)
        Sinv = Sinv[potential_landmarks_idx]  # shape (I', 2, 2)
        Iprime = tpr.shape[0]

        # compute the mahalanobis distance D of shape (I', J), i.e. D[i, j] = tpr[i, j] @ Sinv[i] @ tpr[i, j]
        D = np.einsum("ijk,ikl,ijl->ij", tpr, Sinv, tpr).T
        # print(f"Mahalanobis time: {1000*(perf_counter() - start):.3f} ms")
        row_id, col_id = linear_sum_assignment(D)
        associations = [
            (observations[row_id[k]], self.map[col_id[k]])
            for k in range(J)
            if D[row_id[k], col_id[k]] < self.chi2_95
        ]
        return associations


class EKFSLAM:
    """
    Extended Kalman Filter Localization with use of landmarks to update the beliefs.
    """

    nx: int
    nu: int = 3
    chi2_95 = chi2.ppf(0.95, df=2)
    chi2_99 = chi2.ppf(0.99, df=2)

    mu: np.ndarray
    Sigma: np.ndarray
    sampling_time: float
    data_association_statistics: list[Union[int, float]]

    def __init__(
        self,
        initial_state: np.ndarray,
        initial_state_uncertainty: np.ndarray,
        sampling_time: float,
        initial_landmark_uncertainty: np.ndarray,
        map: np.ndarray = np.empty((0, 2)),
    ):
        self.mu = np.hstack((initial_state, np.ravel(map)))
        self.nx = self.mu.shape[0]
        self.Sigma = block_diag(
            *(
                [initial_state_uncertainty]
                + [initial_landmark_uncertainty] * map.shape[0]
            )
        )
        self.sampling_time = sampling_time
        self.data_association_statistics = []

    @property
    def map(self) -> np.ndarray:
        return self.mu[3:].reshape((-1, 2))

    def prediction_step(
        self, odometry: np.ndarray, odometry_uncertainty: np.ndarray, dt: float
    ):
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
        self.mu[2] = mod(self.mu[2])
        tpr = G @ self.Sigma[:3, 3:]
        self.Sigma[:3, :3] = (
            G @ self.Sigma[:3, :3] @ G.T + V @ odometry_uncertainty @ V.T
        )
        self.Sigma[:3, 3:] = tpr
        self.Sigma[3:, :3] = tpr.T

    def localize(
        self,
        observations: np.ndarray,
        odometry: np.ndarray,
        phi_obs: float,
        odometry_uncertainty: np.ndarray,
        observations_uncertainties: np.ndarray,
        sampling_time: float = None,
        cones_coordinates: str = "polar",
    ) -> tuple[Union[ndarray, ndarray], Union[ndarray, ndarray], list[int]]:
        """
        :param observations: shape (n, 2) array of cones positions in polar coordinates
        :param odometry: shape (3,), u=[v_x, v_y, r]
        :param phi_obs: float, orientation of the robot observed with IMU
        :param odometry_uncertainty: shape (3,) or (3,3), uncertainty of the odometry
        :param observations_uncertainties: shape (2,) or (2,2), uncertainty of the observations
        """
        # verify inputs
        assert len(observations.shape) == 2 and observations.shape[1] == 2
        assert odometry.shape == (self.nu,)
        if len(odometry_uncertainty.shape) == 1:
            odometry_uncertainty = np.diag(odometry_uncertainty)
        assert odometry_uncertainty.shape == (self.nu, self.nu)
        if len(observations_uncertainties.shape) == 1:
            observations_uncertainties = np.diag(observations_uncertainties)
        assert observations_uncertainties.shape == (2, 2)
        if sampling_time is None:
            sampling_time = self.sampling_time

        # PREDICTION STEP =========================================================
        self.prediction_step(odometry, odometry_uncertainty, sampling_time)

        # PRELIMINARY YAW UPDATE =========================================================
        self.mu, self.Sigma = self.yaw_update(self.mu, self.Sigma, phi_obs)
        self.mu[2] = mod(self.mu[2])

        # DATA ASSOCIATION STEP =========================================================
        if cones_coordinates == "cartesian":
            observations = np.array(
                [
                    np.hypot(observations[:, 0], observations[:, 1]),
                    mod(np.arctan2(observations[:, 1], observations[:, 0])),
                ]
            ).T

        I = (self.nx - 3) // 2  # number of known landmarks
        J = observations.shape[0]  # number of observations

        if I < J:
            # we are still initalizing the map
            for j in range(J):
                self.add_landmark(observations[j])
            self.data_association_statistics = [0.0, 0.0, 1.0]
        else:
            # TODO: filter the potential landmarks with a better criterion
            potential_landmarks_idx = np.arange(I)
            # angles = mod(
            #     np.arctan2(
            #         self.mu[3:].reshape(-1, 2)[:, 1] - self.mu[1],
            #         self.mu[3:].reshape(-1, 2)[:, 0] - self.mu[0],
            #     )
            # )
            # potential_landmarks_idx = np.where(
            #     (
            #         np.linalg.norm(self.mu[3:].reshape(-1, 2) - self.mu[:2], axis=1)
            #         < 20.0 + 3 * observations_uncertainties[0, 0]
            #     )
            #     # & (
            #     #     angles
            #     #     > max(
            #     #         -np.pi, -np.pi / 2 - 3 * np.sqrt(observations_uncertainties[1, 1])
            #     #     )
            #     # )
            #     # & (
            #     #     angles
            #     #     < min(np.pi, np.pi / 2 + 3 * np.sqrt(observations_uncertainties[1, 1]))
            #     # )
            # )[0]
            Iprime = len(
                potential_landmarks_idx
            )  # number of known landmarks after filtering
            if Iprime < J:
                # we have removed too many landmarks to properly perform the data association so we consider that all
                # observations correspond to new landmarks
                for j in range(J):
                    self.add_landmark(observations[j])
                self.data_association_statistics = [0.0, 0.0, 1.0]
            else:
                delta = self.mu[3:].reshape(-1, 2) - self.mu[:2]  # shape (I, 2)
                q = np.sum(delta**2, axis=1)  # shape (I,)
                z_hat = np.array(
                    [np.sqrt(q), mod(np.arctan2(delta[:, 1], delta[:, 0]) - self.mu[2])]
                ).T  # shape (I, 2)
                Hs = [
                    np.array(
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
                    )
                    for i in potential_landmarks_idx
                ]  # list of shape (Iprime, 2, 5)
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
                    observations[:, None, :] - z_hat[None, potential_landmarks_idx, :],
                    (1, 0, 2),
                )  # shape (I', J, 2)
                delta_z[:, :, 1] = mod(delta_z[:, :, 1])  # shape (I', J, 2)

                # compute the mahalanobis distance: D[j, i] = delta_z[i, j] @ Sinv[i] @ delta_z[i, j]
                D = np.einsum(
                    "ijk,ikl,ijl->ij", delta_z, Sinv, delta_z
                ).T  # shape (J, I')
                observation_id, landmark_id = linear_sum_assignment(D)

                associations = []
                self.data_association_statistics = [0, 0, 0]
                for j, i in zip(observation_id, landmark_id):
                    if D[j, i] < self.chi2_95:
                        # observation i has been associated with landmark j
                        self.data_association_statistics[0] += 1
                        associations.append((j, i))
                    elif D[j, i] > self.chi2_99:
                        # observation i has not been associated with any landmark so we create a new one
                        self.data_association_statistics[2] += 1
                        self.add_landmark(
                            observations[j, :], observations_uncertainties
                        )
                    else:
                        # observation i has been discarded because it is not a good enough match
                        self.data_association_statistics[1] += 1

                print(
                    "{}/{} accepted, {}/{} discarded, {}/{} new".format(
                        self.data_association_statistics[0],
                        J,
                        self.data_association_statistics[1],
                        J,
                        self.data_association_statistics[2],
                        J,
                    )
                )
                self.data_association_statistics = [
                    x / J for x in self.data_association_statistics
                ]

                # CONES UPDATE STEP ======================================================================
                for j, i in associations:
                    K = (
                        np.hstack(
                            (self.Sigma[:, :3], self.Sigma[:, 2 * i + 3 : 2 * i + 5])
                        )
                        @ Hs[i].T
                        @ Sinv[i]
                    )
                    self.mu += K @ delta_z[i, j]
                    self.mu[2] = mod(self.mu[2])
                    self.Sigma -= K @ S[i] @ K.T

        # YAW UPDATE STEP ======================================================================
        self.mu, self.Sigma = self.yaw_update(self.mu, self.Sigma, phi_obs)
        self.mu[2] = mod(self.mu[2])

        return self.mu, self.Sigma, self.data_association_statistics

    def yaw_update(self, x, Sigma, phi_obs):
        H = np.zeros(self.nx)
        H[2] = 1.0
        K = Sigma[:, 2] / (Sigma[2, 2] + 1e-3)  # Kalman gain
        return x + K * mod(phi_obs - x[2]), Sigma - K[:, None] @ (
            K[None, :] * (Sigma[2, 2] + 1e-3)
        )

    def add_landmark(
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
        H_inv_v = np.array(
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
        H_inv_j = np.array(
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
        tpr = H_inv_v @ self.Sigma[:3, :]
        self.Sigma = np.block(
            [
                [
                    self.Sigma,
                    tpr.T,
                ],
                [
                    tpr,
                    H_inv_v @ self.Sigma[:3, :3] @ H_inv_v.T
                    + H_inv_j @ observations_uncertainties @ H_inv_j.T,
                ],
            ]
        )
        self.nx += 2


def run():
    dt = 0.01
    filename = "fsds_competition_1_10.0.npz"
    # noinspection PyTypeChecker
    data: np.lib.npyio.NpzFile = np.load(
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../datasets/data/localization_dataset",
                filename,
            )
        )
    )
    print(data.files[:6])
    localizer = EKFLocalizer(
        map=data["global_cones_positions"],
        x0=np.array([0.0, 0.0, np.pi / 2]),
        Sigma0=np.diag([0.01, 0.01, 0.01]),
        Q=np.diag([0.1, 0.1, 0.01]) ** 2,
        R=np.diag([0.1, 0.1]) ** 2,
        association_sensitivity=10.0,
        dt=dt,
    )
    runtimes = []
    states = []
    true_states = []
    for file_id in range(0, (len(data.files) - 5) // 2, int(dt / 0.01)):
        true_state = data["states"][file_id]
        cones = data[f"rel_cones_positions_{file_id}"]
        start = perf_counter()
        state, _ = localizer.localize(
            cones=cones
            + np.random.multivariate_normal(np.zeros(2), localizer.R, cones.shape[0]),
            u=true_state[-3:] + np.random.multivariate_normal(np.zeros(3), localizer.Q),
            phi_obs=true_state[2],
        )
        end = perf_counter()
        runtimes.append(end - start)
        states.append(state)
        true_states.append(true_state)
        # sleep_sub_ms(dt)

    states = np.array(states)
    true_states = np.array(true_states)
    runtimes = np.array(runtimes)
    print("Average runtime: {} ms".format(np.mean(runtimes) * 1000))
    localization_error = np.linalg.norm(states[:, :2] - true_states[:, :2], axis=1)
    orientation_error = np.abs(states[:, 2] - true_states[:, 2])
    print(
        "position error: {:.3f} ± {:.3f} m".format(
            np.mean(localization_error), np.std(localization_error)
        ),
        "orientation error: {:.3f} ± {:.3f} rad".format(
            np.mean(orientation_error), np.std(orientation_error)
        ),
    )
    track = tdb.load_track("fsds_competition_1")
    plot_cones(
        track.blue_cones,
        track.yellow_cones,
        track.big_orange_cones,
        track.small_orange_cones,
        show=False,
    )
    plt.plot(states[:, 0], states[:, 1], "b-", label="Estimated trajectory")
    plt.plot(true_states[:, 0], true_states[:, 1], "g-", label="True trajectory")
    plt.legend()
    plt.tight_layout()
    plt.axis("equal")
    # plt.show()


def run_slam():
    dt = 0.01
    track_name = "fsds_competition_1"
    filename = f"{track_name}_5.0.npz"
    # noinspection PyTypeChecker
    data: np.lib.npyio.NpzFile = np.load(
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../datasets/data/localization_dataset",
                filename,
            )
        )
    )
    print(data.files[:6])
    localizer = EKFSLAM(
        # map=data["global_cones_positions"],
        initial_state=np.array([0.0, 0.0, np.pi / 2]),
        initial_state_uncertainty=0.0 * np.eye(3),
        initial_landmark_uncertainty=1e-1 * np.eye(2),
        sampling_time=dt,
    )
    tpr = data[f"rel_cones_positions_{0}"]
    [localizer.add_landmark(tpr[i]) for i in range(tpr.shape[0])]
    runtimes = []
    states = []
    true_states = []
    landmark_types_counts = []
    for file_id in range(0, (len(data.files) - 5), int(dt / 0.01)):
        true_state = data["states"][file_id]
        cones = data[f"rel_cones_positions_{file_id}"]
        start = perf_counter()
        R = np.array([0.0, 0.1]) ** 2
        Q = np.array([0.1, 0.1, 0.01]) ** 2
        state, _, counts = localizer.localize(
            observations=cones
            + np.random.multivariate_normal(np.zeros(2), np.diag(R), cones.shape[0]),
            odometry=true_state[-3:]
            + np.random.multivariate_normal(np.zeros(3), np.diag(Q)),
            phi_obs=true_state[2],
            observations_uncertainties=R,
            odometry_uncertainty=Q,
            sampling_time=dt,
        )

        end = perf_counter()
        runtimes.append(end - start)
        print("slam runtime: {} ms".format((end - start) * 1000))
        landmark_types_counts.append(counts)
        states.append(state[:3])
        true_states.append(true_state)
        if np.linalg.norm(state[:2] - true_state[:2]) > 3:
            print("Localization error too high, aborting")
            break

    states = np.array(states)
    true_states = np.array(true_states)
    runtimes = np.array(runtimes)
    landmark_types_counts = np.array(landmark_types_counts)

    print("Average runtime: {} ms".format(np.mean(runtimes) * 1000))
    localization_error = np.linalg.norm(states[:, :2] - true_states[:, :2], axis=1)
    orientation_error = np.abs(states[:, 2] - true_states[:, 2])
    print(
        "position error: {:.3f} ± {:.3f} m".format(
            np.mean(localization_error), np.std(localization_error)
        ),
        "orientation error: {:.3f} ± {:.3f} rad".format(
            np.mean(orientation_error), np.std(orientation_error)
        ),
    )
    print(
        "Average landmark types counts: {:.3f} % associated, {:.3f}% discarded, {:.3f}% new".format(
            *np.mean(landmark_types_counts, axis=0)
        )
    )
    track = tdb.load_track(track_name)
    plot_cones(
        track.blue_cones,
        track.yellow_cones,
        track.big_orange_cones,
        track.small_orange_cones,
        show=False,
    )
    plt.scatter(localizer.map[:, 0], localizer.map[:, 1], c="r", label="Map")
    plt.plot(states[:, 0], states[:, 1], "b-", label="Estimated trajectory")
    plt.plot(true_states[:, 0], true_states[:, 1], "g-", label="True trajectory")
    plt.legend()
    plt.tight_layout()
    plt.axis("equal")
    plt.show()


def main():
    global bruh
    plt.figure()
    run()
    # bruh = False
    # plt.figure()
    # run()
    plt.show()


def main_slam():
    run_slam()


if __name__ == "__main__":
    main_slam()
