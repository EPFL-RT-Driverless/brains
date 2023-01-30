from typing import Union

import numpy as np

from .localizer import SLAM


class EKFSLAM(SLAM):
    """
    Extended Kalman Filter SLAM
    """

    state_dim: int  # current dimension of the state vector (3 + 2 * nbr_landmarks)
    nbr_landmarks: int  # number of landmarks in the currents map
    current_map: np.ndarray
    state: np.ndarray  # [X, Y, phi]
    covariance: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    association_sensitivity: float
    new_landmark_covariance: np.ndarray

    def __init__(
        self,
        config: dict[str, Union[str, float, bool, list]],
    ):
        """
        Initialize the localizer
        :param config: configuration dictionary containing the following keys:
            - initial_state: list, initial state of the car in global cartesian coordinates [X, Y, phi]
            - initial_covariance: list, initial covariance of the state (diagonal elements)
            - Q: process noise covariance matrix (diagonal elements)
            - R: measurement noise covariance matrix (diagonal elements)
            - association_sensitivity: maximum distance between a cone and a landmark to be considered as the same landmark
            - new_landmark_covariance: covariance of a new landmark (diagonal elements)
        """
        self.state_dim = 3
        self.nbr_landmarks = 0
        self.current_map = np.ndarray(shape=(0, 2))
        self.state = np.array(config.get("initial_state", [0.0, 0.0, np.pi / 2]))
        self.covariance = np.diag(
            config.get("initial_covariance", [0.1, 0.1, np.deg2rad(0.1)])
        )
        self.Q = config.get("Q", [0.2**2, 0.2**2, np.deg2rad(3.0) ** 2])
        self.R = config.get("R", [25.0, 25.0])
        self.association_sensitivity = config.get("association_sensitivity", 1000.0)
        self.new_landmark_covariance = np.diag(
            config.get("new_landmark_covariance", [5.0, 5.0])
        )

    def localize(
        self,
        cones: np.ndarray,
        motion_data: np.ndarray = None,
        sampling_time: float = None,
        cones_coordinates: str = "cartesian",
    ) -> np.array:
        """
        Localize the car on the track
        :param cones: detected cones in local (in the car's reference frame) coordinates
        :param motion_data: motion data [v_x, r]
        :param sampling_time: sampling time
        :param cones_coordinates: how the cones coordinates are specified ("cartesian" or "polar")
        :return: state of the car in global cartesian coordinates
        """
        # verify inputs
        assert len(cones.shape) == 2 and cones.shape[1] == 2

        # PREDICTION STEP =========================================================
        phi = self.state[2]
        v_x = motion_data[0]
        r = motion_data[1]
        G = np.array(
            [
                [1.0, 0.0, v_x * np.sin(phi) * sampling_time],
                [0.0, 1.0, v_x * np.cos(phi) * sampling_time],
                [0.0, 0.0, sampling_time],
            ]
        )
        Q = self.Q
        V = np.array(
            [
                [np.cos(phi) * sampling_time, -np.sin(phi) * sampling_time, 0.0],
                [np.sin(phi) * sampling_time, np.cos(phi) * sampling_time, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        Q = V @ Q @ V.T
        predicted_state = self.state
        predicted_state[:3] += np.array(
            [
                v_x * np.cos(phi) * sampling_time,
                v_x * np.sin(phi) * sampling_time,
                r * sampling_time,
            ]
        )
        if self.nbr_landmarks == 0:
            predicted_covariance = G @ self.covariance @ G.T + Q
        else:
            tpr = G @ self.covariance[:3, 3:]
            predicted_covariance = np.bmat(
                [
                    [G @ self.covariance[:3, :3] @ G.T + Q, tpr],
                    [tpr.T, self.covariance[3:, 3:]],
                ]
            )

        # UPDATE STEP =========================================================
        # TODO: re-implement the update step
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

        for z, lm in associated_landmarks:
            if lm != None:
                # if we matched a landmark, update the state and the covariance
                delta = lm - self.state[:2]
                q = (delta**2).sum(axis=0)
                z_tilde = np.array(
                    [np.sqrt(q), np.arctan2(delta[1], delta[0]) - predicted_state[2]]
                    # [np.sqrt(q), np.arctan2(delta[1], delta[0])]
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
                self.state = predicted_state + K @ (z - z_tilde)
                self.covariance = (
                    np.eye(self.state_dim) - K @ H
                ) @ predicted_covariance
            else:
                self.state = np.append(
                    self.state,
                    np.array(
                        [
                            self.state[0] + z[0] * np.cos(self.state[2] + z[1]),
                            self.state[1] + z[0] * np.sin(self.state[2] + z[1]),
                        ]
                    ),
                )
                self.covariance = np.bmat(
                    [
                        [self.covariance, np.zeros((self.state_dim, 2))],
                        [np.zeros((2, self.state_dim)), self.new_landmark_covariance],
                    ]
                )
                self.nbr_landmarks += 1
                self.state_dim += 2

        return self.state[:3]

    # def associate_landmark(self, z: np.ndarray) -> int:
    #     """
    #     Find the closest landmark to the observed cone
    #
    #     :param z: an observed cone in local polar coordinates
    #     :returns: id (int) of the corresponding cone in the reference map or len(self.known_map) if no match
    #     """
    #     cone_global_cartesian_coords = np.array(
    #         [
    #             self.state[0] + z[0] * np.cos(self.state[2] + z[1]),
    #             self.state[1] + z[0] * np.sin(self.state[2] + z[1]),
    #         ]
    #     )
    #     dists = np.linalg.norm(self.current_map - cone_global_cartesian_coords, axis=1)
    #     argmin = np.argmin(dists)
    #     if dists[argmin] < self.association_sensitivity:
    #         return argmin
    #     else:
    #         # unknown landmark
    #         return len(self.current_map)

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
        for cone in cones:
            if self.current_map.size == 0:
                result.append((cone, None))
            else:
                cone_global_cartesian_coords = np.array(
                    [
                        self.state[0] + cone[0] * np.cos(self.state[2] + cone[1]),
                        self.state[1] + cone[0] * np.sin(self.state[2] + cone[1]),
                    ]
                )
                dists = np.linalg.norm(
                    self.current_map - cone_global_cartesian_coords, axis=1
                )
                argmin = np.argmin(dists)
                if dists[argmin] < self.association_sensitivity:
                    result.append((cone, self.current_map[argmin]))
                else:
                    # unknown landmark
                    result.append((cone, None))

        return result
