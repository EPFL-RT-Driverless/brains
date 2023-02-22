import os.path
from copy import copy
from time import perf_counter
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import track_database as tdb
from numpy import ndarray
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
from scipy.stats.distributions import chi2
from track_database.utils import plot_cones

np.random.seed(127)
bruh = True


def mod(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


class EKFSLAM:
    """
    Extended Kalman Filter Localization with use of landmarks to update the beliefs.
    """

    nx: int
    nu: int = 3
    chi2_95 = chi2.ppf(0.95, df=2)
    chi2_99 = chi2.ppf(0.99, df=2)
    threshold_1 = chi2.ppf(0.95, df=2)
    threshold_2 = chi2.ppf(0.99, df=2)

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

    @property
    def nlandmarks(self) -> int:
        return (self.nx - 3) // 2

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

        # filter potential landmarks
        potential_landmarks_idx = self.filter_potential_landmarks()
        Iprime = len(
            potential_landmarks_idx
        )  # number of known landmarks after filtering
        if Iprime > 0:
            delta = self.mu[3:].reshape(-1, 2) - self.mu[:2]  # shape (I', 2)
            q = np.sum(delta**2, axis=1)  # shape (I',)
            z_hat = np.array(
                [np.sqrt(q), mod(np.arctan2(delta[:, 1], delta[:, 0]) - self.mu[2])]
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
            delta_z[:, :, 1] = mod(delta_z[:, :, 1])  # shape (I', J, 2)

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
                elif D[j, i] > self.threshold_2:
                    # observation j has not been associated with any landmark so we create a new one
                    self.data_association_statistics[2] += 1
                    self.add_landmark(observations[j, :], observations_uncertainties)
                else:
                    # observation i has been discarded because it is not a good enough match
                    self.data_association_statistics[1] += 1
        else:
            # we don't have any landmark yet
            for j in range(J):
                self.add_landmark(observations[j], observations_uncertainties)
            self.data_association_statistics = [0, 0, J]
            associations = []

        # print(
        #     "{}/{} accepted, {}/{} discarded, {}/{} new".format(
        #         self.data_association_statistics[0],
        #         J,
        #         self.data_association_statistics[1],
        #         J,
        #         self.data_association_statistics[2],
        #         J,
        #     )
        # )
        self.data_association_statistics = [
            x / J for x in self.data_association_statistics
        ]

        # CONES UPDATE STEP ======================================================================
        for j, i in associations:
            K = (
                np.hstack((self.Sigma[:, :3], self.Sigma[:, 2 * i + 3 : 2 * i + 5]))
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

    def filter_potential_landmarks(self):
        return np.arange(self.nlandmarks)

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


def run():
    dt = 0.01
    track_name = "fsds_competition_1"
    filename = f"{track_name}_10.0.npz"
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
    added_landmarks_idx = []
    runtimes = []
    states = []
    true_states = []
    data_association_statistics = []
    nx = copy(localizer.nx)
    for file_id in tqdm(range(0, (len(data.files) - 5), int(dt / 0.01))):
        true_state = data["states"][file_id]
        cones = data[f"rel_cones_positions_{file_id}"]
        Q = np.array([0.1, 0.1, 0.01]) ** 2
        R = np.array([1.0, 0.1]) ** 2
        start = perf_counter()
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
        if localizer.nx - nx > 0:
            added_landmarks_idx.append(
                list(range((nx - 3) // 2, (localizer.nx - 3) // 2))
            )
            nx = copy(localizer.nx)
        else:
            added_landmarks_idx.append([])

        runtimes.append(end - start)
        # print("slam runtime: {} ms".format((end - start) * 1000))
        data_association_statistics.append(counts)
        states.append(state[:3])
        true_states.append(true_state)
        # if np.linalg.norm(state[:2] - true_state[:2]) > 3:
        #     print("Localization error too high, aborting")
        #     break

    states = np.array(states)
    true_states = np.array(true_states)
    runtimes = np.array(runtimes)
    data_association_statistics = np.array(data_association_statistics)

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
        "Average data association statistics: {:.3f} % associated, {:.3f}% discarded, {:.3f}% new".format(
            *np.mean(data_association_statistics, axis=0)
        )
    )
    track = tdb.load_track(track_name)

    fig = plt.figure(figsize=(10, 10))
    plot_cones(
        track.blue_cones,
        track.yellow_cones,
        track.big_orange_cones,
        track.small_orange_cones,
        show=False,
    )
    plt.axis("equal")
    plt.scatter(localizer.map[:, 0], localizer.map[:, 1], c="r", label="Map")
    plt.plot(states[:, 0], states[:, 1], "b-", label="Estimated trajectory")
    plt.plot(true_states[:, 0], true_states[:, 1], "g-", label="True trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ekfslam.png", dpi=300)
    # for it, idx in enumerate(added_landmarks_idx):
    #     print(
    #         "iteration {} state {} added landmarks: {}".format(
    #             it, true_states[it, :3], [localizer.map[i, :] for i in idx]
    #         )
    #     )

    plt.show()


if __name__ == "__main__":
    run()
