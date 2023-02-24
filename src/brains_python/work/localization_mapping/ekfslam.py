import os.path
from time import perf_counter
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import track_database as tdb
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
from scipy.stats.distributions import chi2
from track_database.utils import plot_cones

np.random.seed(127)

__all__ = ["EKFSLAM"]


def wrapToPi(x):
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
    # threshold_2 = chi2.ppf(0.99, df=2)
    threshold_2 = 15.0

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

    def yaw_update(self, yaw: float, yaw_uncertainty: float = 1e-3):
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
                elif D[j, i] > self.threshold_2:
                    # observation j has not been associated with any landmark so we create a new one
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


def run():
    dt = 0.01
    track_name = "fsds_competition_1"
    vxmax = 10.0
    filename = f"{track_name}_{vxmax}.npz"
    # mode = "localization"
    mode = "mapping"
    live_plot = False
    pause_time = 0.1

    track = tdb.load_track(track_name)
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
    slamer = EKFSLAM(
        map=data["global_cones_positions"]
        if mode == "localization"
        else np.empty((0, 2)),
        initial_state=np.array([0.0, 0.0, np.pi / 2]),
        initial_state_uncertainty=0.0 * np.eye(3),
        initial_landmark_uncertainty=1e-1 * np.eye(2),
        sampling_time=dt,
    )
    # tpr = data[f"rel_cones_positions_{0}"]
    # [slamer._add_landmark(tpr[i]) for i in range(tpr.shape[0])]
    odometry_update_runtimes = []
    yaw_update_runtimes = []
    cones_update_runtimes = []

    poses = np.empty((0, 3), dtype=float)
    true_poses = np.empty((0, 3), dtype=float)
    data_association_statistics = np.empty((0, 3), dtype=float)
    if live_plot:
        plt.figure(figsize=(8, 8))
        plt.ion()
        plt.show(block=False)
    for file_id in tqdm(range(0, (len(data.files) - 5), int(dt / 0.01))):
        # odometry update
        true_state = data["states"][file_id]
        Q1 = (
            np.array([0.04 * true_state[3], 0.04 * true_state[4], 0.04 * true_state[5]])
            ** 2
        )
        Q2 = (
            np.array([0.04 * true_state[3], 0.04 * true_state[4], 0.04 * true_state[5]])
            * 2
        ) ** 2
        odometry_measurement = true_state[-3:] + np.random.multivariate_normal(
            np.zeros(3), np.diag(Q1)
        )
        start = perf_counter()
        slamer.odometry_update(
            odometry=odometry_measurement,
            odometry_uncertainty=Q2,
            sampling_time=dt,
        )
        stop = perf_counter()
        odometry_update_runtimes.append(1000 * (stop - start))

        # yaw update
        start = perf_counter()
        slamer.yaw_update(yaw=true_state[2], yaw_uncertainty=1e-3)
        stop = perf_counter()
        yaw_update_runtimes.append(1000 * (stop - start))

        # cones update
        if file_id % 5 == 0:
            cones = data[f"rel_cones_positions_{file_id}"]
            e_rho = 0.1
            e_theta = 0.1
            R1 = np.array([e_rho, e_theta]) ** 2
            R2 = (np.array([e_rho, e_theta]) * 1) ** 2
            observations = cones + np.random.multivariate_normal(
                np.zeros(2), np.diag(R1), cones.shape[0]
            )
            inverse_observations = np.array(
                [
                    slamer.mu[0]
                    + observations[:, 0] * np.cos(observations[:, 1] + slamer.mu[2]),
                    slamer.mu[1]
                    + observations[:, 0] * np.sin(observations[:, 1] + slamer.mu[2]),
                ]
            ).T
            start = perf_counter()
            slamer.cones_update(
                observations=observations,
                observations_uncertainties=R2,
            )
            stop = perf_counter()
            cones_update_runtimes.append(1000 * (stop - start))
            # data_association_statistics.append(slamer.data_association_statistics)
            data_association_statistics = np.vstack(
                (data_association_statistics, slamer.data_association_statistics)
            )
            if live_plot:
                # plot stuff
                plt.clf()
                plot_cones(
                    track.blue_cones,
                    track.yellow_cones,
                    track.big_orange_cones,
                    track.small_orange_cones,
                    show=False,
                )
                plt.axis("equal")
                plt.tight_layout()
                plt.plot(
                    true_poses[:, 0], true_poses[:, 1], "g-", label="True trajectory"
                )
                plt.plot(poses[:, 0], poses[:, 1], "b-", label="Estimated trajectory")
                plt.scatter(slamer.map[:, 0], slamer.map[:, 1], c="r", label="Map")
                plt.scatter(
                    inverse_observations[:, 0],
                    inverse_observations[:, 1],
                    c="g",
                    label="Reconstructed landmarks",
                )
                plt.legend()
                plt.xlim(
                    -(np.max(np.abs(inverse_observations[:, 0])) + 1),
                    np.max(np.abs(inverse_observations[:, 0])) + 1,
                )
                plt.ylim(
                    0.0,
                    np.max(inverse_observations[:, 1]) + 1,
                )

        # states.append(slamer.state)
        poses = np.vstack((poses, slamer.state))
        # true_states.append(true_state)
        true_poses = np.vstack((true_poses, true_state[:3]))
        if live_plot:
            plt.pause(pause_time)

        if np.linalg.norm(slamer.state[:2] - true_state[:2]) > 3:
            print("Localization error too high, aborting")
            break

    # states = np.array(states)
    # true_states = np.array(true_states)
    # data_association_statistics = np.array(data_association_statistics)
    odometry_update_runtimes = np.array(odometry_update_runtimes)
    yaw_update_runtimes = np.array(yaw_update_runtimes)
    cones_update_runtimes = np.array(cones_update_runtimes)

    def print_stat(arr, name):
        return "{:<15}: {:<6.3f} ± {:>6.3f}".format(name, np.mean(arr), np.std(arr))

    print(
        "Runtimes:\n",
        "\t" + print_stat(odometry_update_runtimes, "odometry update") + " ms\n",
        "\t" + print_stat(yaw_update_runtimes, "yaw update") + " ms\n",
        "\t" + print_stat(cones_update_runtimes, "cones update") + " ms",
    )

    localization_error = np.linalg.norm(poses[:, :2] - true_poses[:, :2], axis=1)
    orientation_error = np.abs(poses[:, 2] - true_poses[:, 2])
    print(
        "Error statistics:\n",
        "\t" + print_stat(localization_error, "position error") + " m\n",
        "\t" + print_stat(orientation_error, "orientation error") + " rad",
    )

    print(
        "Average data association statistics:\n{:.3f} % associated, {:.3f}% discarded, {:.3f}% new".format(
            *np.mean(data_association_statistics, axis=0)
        )
    )

    if not live_plot:
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
        plt.tight_layout()
        plt.plot(true_poses[:, 0], true_poses[:, 1], "g-", label="True trajectory")
        plt.plot(poses[:, 0], poses[:, 1], "b-", label="Estimated trajectory")
        plt.scatter(slamer.map[:, 0], slamer.map[:, 1], c="r", label="Map")
        plt.legend()
        plt.savefig("ekfslam.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    run()
