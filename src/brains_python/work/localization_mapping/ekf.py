from time import perf_counter
from typing import Union

import matplotlib.pyplot as plt
import track_database as tdb
import numpy as np
from fsds_client import HighLevelClient
from fsds_client.utils import *
from scipy.spatial import distance_matrix


class EKFLocalizer:
    """
    Extended Kalman Filter Localization with use of landmarks to update the beliefs.
    """

    nx: int = 3
    nu: int = 3

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
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        association_sensitivity: float,
        sampling_time: float,
    ):
        self.map = map
        self.x = initial_state
        self.Sigma = initial_covariance
        self.Q = Q
        self.R = R
        self.association_sensitivity = association_sensitivity
        self.dt = sampling_time

    def localize(
        self,
        cones: np.ndarray,
        u: np.ndarray,
        phi_obs: float,
        sampling_time: float,
        cones_coordinates: str = "polar",
    ) -> np.array:
        # verify inputs
        assert len(cones.shape) == 2 and cones.shape[1] == 2

        # PREDICTION STEP =========================================================
        phi = self.x[2]
        v_x = u[0]
        v_y = u[1]
        r = u[2]
        G = np.array(
            [
                [1.0, 0.0, (v_x * np.sin(phi) - v_y * np.cos(phi)) * sampling_time],
                [0.0, 1.0, (v_x * np.cos(phi) + v_y * np.sin(phi)) * sampling_time],
                [0.0, 0.0, sampling_time],
            ]
        )
        V = np.array(
            [
                [np.cos(phi) * sampling_time, -np.sin(phi) * sampling_time, 0.0],
                [np.sin(phi) * sampling_time, np.cos(phi) * sampling_time, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        Q = V @ self.Q @ V.T
        predicted_state = self.x + np.array(
            [
                (v_x * np.cos(phi) - v_y * np.sin(phi)) * sampling_time,
                (v_x * np.sin(phi) + v_y * np.cos(phi)) * sampling_time,
                r * sampling_time,
            ]
        )
        predicted_state[2] = np.mod(predicted_state[2] + np.pi, 2 * np.pi) - np.pi
        predicted_covariance = G @ self.Sigma @ G.T + Q

        # UPDATE STEP =========================================================
        R = self.R
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

        # associate each cone to a landmark
        associated_landmarks = self.associate_landmarks(local_polar_coordinates)

        # update the state and covariance
        for z, lm in associated_landmarks:
            delta = lm - self.x[:2]
            q = (delta**2).sum(axis=0)
            z_tilde = np.array(
                [
                    np.sqrt(q),
                    np.mod(
                        np.arctan2(delta[1], delta[0]) - predicted_state[2] + np.pi,
                        2 * np.pi,
                    )
                    - np.pi,
                ]
            )
            H = np.array(
                [
                    [-np.sqrt(q) * delta[0], -np.sqrt(q) * delta[1], 0.0],
                    [delta[1], -delta[0], -q],
                ]
            )
            K = (
                predicted_covariance
                @ H.T
                @ np.linalg.inv(H @ predicted_covariance @ H.T + R)
            )
            predicted_state = predicted_state + K @ (z - z_tilde)
            predicted_covariance = (np.eye(self.nx) - K @ H) @ predicted_covariance

        # correct using phi_obs
        H = np.array([0.0, 0.0, 1.0])
        K = (
            predicted_covariance
            @ H.T
            / (float(H @ predicted_covariance @ H.T) + 0.0000001)
        )
        predicted_state = predicted_state + K * (phi_obs - predicted_state[2])
        predicted_covariance = (np.eye(self.nx) - K * H) @ predicted_covariance

        # self.x[2] = phi_obs
        self.x = predicted_state
        self.Sigma = predicted_covariance

        return self.x

    def associate_landmarks(
        self, cones: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Find the closest landmark to each observed cone

        :param cones: an array of observed cones in local polar coordinates
        :returns: a list of tuples (cone, landmark) where cone is the observed cone in
         local polar coordinates and the corresponding landmark (in global cartesian
         coordinates) in the known map
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


def main():
    sampling_time = 0.05
    client = HighLevelClient()
    client.low_level_client.enableApiControl(False)
    track = tdb.Track(client.get_map_name().removesuffix("_cones"))
    map = np.vstack((track.right_cones, track.left_cones))
    localizer = EKFLocalizer(
        map=map,
        initial_state=np.array([0.0, 0.0, np.pi / 2]),
        initial_covariance=np.diag([0.001, 0.001, 0.01]),
        Q=np.diag([0.5, 0.5, 0.01]),
        R=np.diag([1.0e-8, 1.0e-8]),
        association_sensitivity=10.0,
        sampling_time=sampling_time,
    )
    runtimes = []
    states = np.expand_dims(localizer.x, 0)
    for _ in range(300):
        true_state = client.get_state()
        cones = client.find_cones(state=true_state, coords_type="polar")
        start = perf_counter()
        state = localizer.localize(
            cones=cones
            + np.random.multivariate_normal(np.zeros(2), localizer.R, cones.shape[0]),
            u=true_state[-3:]
            + np.random.multivariate_normal(np.zeros(3), np.diag([0.1, 0.1, 0.01])),
            phi_obs=true_state[2],
            sampling_time=sampling_time,
            cones_coordinates="polar",
        )
        end = perf_counter()
        runtimes.append(end - start)
        states = np.vstack((states, state))
        # print(states[-1])
        sleep_sub_ms(sampling_time)

    print("Average runtime: {} ms".format(np.mean(runtimes) * 1000))
    plt.scatter(map[:, 0], map[:, 1], s=3, c="r", marker="^")
    plt.plot(states[:, 0], states[:, 1], "b-")
    plt.tight_layout()
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
