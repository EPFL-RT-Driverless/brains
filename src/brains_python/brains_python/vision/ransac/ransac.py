import time
import numpy as np
import pandas as pd
import random
import math
from sklearn.cluster import DBSCAN
import scipy


class RANSAC:
    """
    RANSAC Class
    """

    def __init__(self, point_cloud, sim, max_iterations, distance_ratio_threshold):
        self.point_cloud = self.filter_manually(point_cloud)
        self.max_iterations = max_iterations
        self.distance_ratio_threshold = distance_ratio_threshold
        self.sim = sim

    def filter_manually(self, pcd_array):
        norm = np.array(np.linalg.norm(pcd_array, axis=1)).reshape(-1, 1)
        np_pcd_norm = np.append(pcd_array, norm, axis=1)

        mask = np_pcd_norm[:, 3] > 0.5
        np_pcd_norm = np_pcd_norm[mask]

        mask = np_pcd_norm[:, 3] < 20
        np_pcd_norm = np_pcd_norm[mask]

        mask = np_pcd_norm[:, 2] < -0.5
        np_pcd_norm = np_pcd_norm[mask]

        mask = np_pcd_norm[:, 1] > -15
        np_pcd_norm = np_pcd_norm[mask]

        return np_pcd_norm[:, 0:3]

    def remove_2d_points(self, scaled_2d, scaling_factor, top, bottom):
        mask_photo = scaled_2d[:, 0] <= bottom[0]
        scaled_2d = scaled_2d[mask_photo]
        scaling_factor = scaling_factor[mask_photo]

        mask_photo = scaled_2d[:, 0] >= top[0]
        scaled_2d = scaled_2d[mask_photo]
        scaling_factor = scaling_factor[mask_photo]

        mask_photo = scaled_2d[:, 1] <= bottom[1]
        scaled_2d = scaled_2d[mask_photo]
        scaling_factor = scaling_factor[mask_photo]

        mask_photo = scaled_2d[:, 1] >= top[1]
        scaled_2d = scaled_2d[mask_photo]
        scaling_factor = scaling_factor[mask_photo]

        return scaled_2d, scaling_factor

    def run(self):
        """
        method for running the class directly
        :return:
        """
        inliers, outliers, final_plan = self._ransac_algorithm(
            self.max_iterations, self.distance_ratio_threshold
        )

        clustering_one_cone = DBSCAN(eps=0.5, min_samples=3, algorithm="ball_tree")
        clustering_one_cone.fit_predict(outliers)
        labels = clustering_one_cone.labels_
        to_keep = []
        for l in np.unique(labels):
            np_points = outliers[labels == l]
            if len(np_points) <= 100:
                dist_mtx = scipy.spatial.distance.cdist(np_points, np_points)
                if dist_mtx.max() <= 0.4:
                    to_keep.append(np.mean(np_points, axis=1))
        print(to_keep)

    def _find_biggest_plane(self, max_iterations, distance_ratio_threshold):
        inliers_result = np.ones((1, 3))
        final_a, final_b, final_c, final_d = 0, 0, 0, 0
        for _ in range(max_iterations):
            start = time.perf_counter()
            max_iterations -= 1
            # Add 3 random indexes
            random.seed()
            below_points = np.argwhere(
                self.point_cloud[:, 2] < -0.75
            )  # If consider that all ground points should be at least lower than that
            three_seeds = np.random.choice(
                below_points.squeeze(), size=3, replace=False
            )

            x1, y1, z1 = self.point_cloud[three_seeds[0]]
            x2, y2, z2 = self.point_cloud[three_seeds[1]]
            x3, y3, z3 = self.point_cloud[three_seeds[2]]

            # Plane Equation --> ax + by + cz + d = 0
            # Value of Constants for inlier plane
            a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
            b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
            c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            d = -(a * x1 + b * y1 + c * z1)
            # plane_length = max(0.1, math.sqrt(a*a + b*b + c*c))
            plane_length = math.sqrt(a * a + b * b + c * c)

            distances = (
                np.absolute(
                    a * self.point_cloud[:, 0]
                    + b * self.point_cloud[:, 1]
                    + c * self.point_cloud[:, 2]
                    + d
                )
                / plane_length
            )
            inliers = np.argwhere(distances <= distance_ratio_threshold)

            if len(inliers) > len(inliers_result):
                inliers_result = inliers
                final_a, final_b, final_c, final_d = a, b, c, d

            end = time.perf_counter()
            # print(F"One interation took {(end-start)*1000} ms")

        return inliers_result, [final_a, final_b, final_c, final_d]

    def _ransac_algorithm(self, max_iterations, distance_ratio_threshold):
        inliers_result, [final_a, final_b, final_c, final_d] = self._find_biggest_plane(
            max_iterations, distance_ratio_threshold
        )
        mask_in = np.zeros(len(self.point_cloud), bool)
        mask_in[inliers_result] = 1

        mask_out = np.ones(len(self.point_cloud), bool)
        mask_out[inliers_result] = 0

        inlier_points = self.point_cloud[mask_in]
        outlier_points = self.point_cloud[mask_out]

        return inlier_points, outlier_points, [final_a, final_b, final_c, final_d]


def read_simu(path):
    sim = np.load(path)
    pt_cld = sim.f.lidar_point_clouds_100
    x, y, z = [], [], []
    for idx, i in enumerate(pt_cld):
        if idx % 3 == 0:
            x.append(i)
        if idx % 3 == 1:
            y.append(i)
        if idx % 3 == 2:
            z.append(i)
    np_pcd_raw = np.array([x, y, z]).T
    return np_pcd_raw, sim


def read_real_data(path):
    point_cloud = pd.read_csv(path)
    point_cloud = point_cloud.drop(point_cloud.columns[[3, 4]], axis=1)
    np_pcd_raw = point_cloud.to_numpy()
    return np_pcd_raw, None


if __name__ == "__main__":
    np_pcd_raw, sim = read_real_data(
        "/home/joe/Desktop/Lidar/lidar_angle/December_06_2022_15_52_03/1.csv"
    )

    APPLICATION = RANSAC(
        np_pcd_raw, sim, max_iterations=1, distance_ratio_threshold=0.01
    )
    APPLICATION.run()
