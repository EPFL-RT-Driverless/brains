import numpy as np
import cv2
from brains_python.common.math import cart2pol

__all__ = ["PnP"]

CMAERA_MATRICES = {
    "Standard": np.array(
        [
            [454.85865806, 0.0, 523.31033627],
            [0.0, 454.85417328, 270.88621438],
            [0.0, 0.0, 1.0],
        ]
    ),
    "NewCamera": np.array(
        [
            [409.89981079, 0.0, 523.11943881],
            [0.0, 409.12664795, 270.29835373],
            [0.0, 0.0, 1.0],
        ]
    ),
    "ComputedMatrix": np.array(
        [
            [869.5, 0.00000000e00, 1.01209980e03],
            [0.00000000e00, 869.5, 5.12126793e02],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
    "RawMatrix": np.array(
        [
            [434.75, 0.00000000e00, 5.12e02],
            [0.00000000e00, 434.75, 2.72e02],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
    "SimuMatrixReg": np.array([[461, 0.0, 512], [0.0, 520, 272], [0.0, 0.0, 1.0]]),
    "SimuMatrix": np.array([[461, 0.0, 512], [0.0, 461, 272], [0.0, 0.0, 1.0]]),
    "SimuMatrixSym": np.array([[461, 0.0, 512], [0.0, 461, 272], [0.0, 0.0, 1.0]]),
}
AVAILABLE_CAMERA_MATRICES = list(CMAERA_MATRICES.keys())


class PnP:
    camera_matrix_name: str
    camera_matrix: np.ndarray
    dist: np.ndarray

    small_cones = np.array(
        [
            (0, -30.7, 0),
            (-3.75, -21.5, 0),
            (3.75, -21.5, 0),
            (-5.25, -13, 0),
            (5.25, -13, 0),
            (-7.85, -2.9, 0),
            (7.85, -2.9, 0),
        ]
    )
    bigCones7 = np.array(
        [
            (0, -50.5, 0),
            (-3.75, -37.5, 0),
            (3.75, -37.5, 0),
            (-5.25, 0, 0),
            (5.25, -13, 0),
            (-7.85, -2.9, 0),
            (7.85, -2.9, 0),
        ]
    )
    bigCones11 = np.array(
        [
            (0, -50.5, 0),
            (-3.75, -37.5, 0),
            (3.75, -37.5, 0),
            (-5.25, 0, 0),
            (5.25, -13, 0),
            (-7.85, -2.9, 0),
            (7.85, -2.9, 0),
            (5.25, -13, 0),
            (-7.85, -2.9, 0),
            (7.85, -2.9, 0),
            (7.85, -2.9, 0),
        ]
    )

    def __init__(self, camera_matrix_name: str):
        assert camera_matrix_name in AVAILABLE_CAMERA_MATRICES, (
            f"Camera matrix name {camera_matrix_name} is not available. "
            f"Available camera matrices are {AVAILABLE_CAMERA_MATRICES}"
        )
        self.camera_matrix_name = camera_matrix_name
        self.camera_matrix = CMAERA_MATRICES[camera_matrix_name]
        # self.dist = np.array([-1.49901205e-01, 9.01283812e-02, -3.49855610e-05, -3.39499075e-04, -1.98075283e-02])
        self.dist = np.zeros(5)
        # self.ObjectPoints = np.array([(0, -30.7, 0),
        #                          (3.75, -21.5, 0),
        #                          (5.25, -13, 0),
        #                          (7.85, -2.9, 0),
        #                          (-3.75, -21.5, 0),
        #                          (-5.25, -13, 0),
        #                          (-7.85, -2.9, 0)], dtype="double")

    def get2Dpoint(self, keypoints):
        _, _, tvecs = cv2.solvePnP(
            self.small_cones,
            keypoints,
            self.camera_matrix,
            self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        return [tvecs[0][0] / 100, tvecs[2][0] / 100]

    def get2DpointPol(self, keypoints):
        _, _, tvecs = cv2.solvePnP(
            self.small_cones,
            keypoints,
            self.camera_matrix,
            self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        # _, _, tvecs, inliers = cv2.solvePnPRansac(self.smallCones, keypoints, self.camera_matrix, distCoeffs=self.dist, flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.9999)
        # print(inliers)
        return cart2pol(tvecs[0][0] / 100, tvecs[2][0] / 100)

    def get2DpointPolCam(self, keypoints, camMat):
        _, _, tvecs = cv2.solvePnP(
            self.small_cones, keypoints, camMat, self.dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        # _, _, tvecs, inliers = cv2.solvePnPRansac(self.smallCones, keypoints, self.camera_matrix, distCoeffs=self.dist, flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.9999)
        # print(inliers)
        return cart2pol(tvecs[0][0] / 100, tvecs[2][0] / 100)
