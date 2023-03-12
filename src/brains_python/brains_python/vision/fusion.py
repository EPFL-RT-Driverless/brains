from sklearn.cluster import DBSCAN
from statistics import mode
import numpy as np

__all__ = ["ConeObserverFusion"]


class ConeObserverFusion:
    def __init__(self, point_cloud, bounding_boxes):
        self.point_cloud = self.filter_manually(point_cloud)
        self.bounding_boxes = bounding_boxes
        self.newcameramtx = np.array(
            [[461, 0.0, 512], [0.0, 461, 272], [0.0, 0.0, 1.0]]
        )  # For the simu
        self.lidar_cam_translation = [0, -0.107, 0]  # For the simu

    def filter_manually(self, pcd_array):
        norm = np.array(np.linalg.norm(pcd_array, axis=1)).reshape(-1, 1)
        np_pcd_norm = np.append(pcd_array, norm, axis=1)

        mask = np_pcd_norm[:, 3] > 0.5
        np_pcd_norm = np_pcd_norm[mask]

        mask = np_pcd_norm[:, 3] < 20
        np_pcd_norm = np_pcd_norm[mask]

        mask = np_pcd_norm[:, 2] < -0.5
        np_pcd_norm = np_pcd_norm[mask]

        mask = np_pcd_norm[:, 0] < 15
        np_pcd_norm = np_pcd_norm[mask]

        return np_pcd_norm

    def world_to_cam(self, coords, translation):
        # Add translation
        coords[:, 0] += translation[0]
        coords[:, 1] += translation[1]
        coords[:, 2] += translation[2]

        Xc = -coords[:, 1]
        Yc = -coords[:, 2]
        Zc = coords[:, 0]

        # Points in camera referential with translation
        np_new_ref = np.hstack(
            (Xc.reshape(-1, 1), Yc.reshape(-1, 1), Zc.reshape(-1, 1))
        )

        return np_new_ref

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

    def cam_to_world(self, coords, translation):
        Xr = coords[:, 2]
        Yr = -coords[:, 0]
        Zr = -coords[:, 1]
        # Add translation

        Xr -= translation[0]
        Yr -= translation[1]
        Zr -= translation[2]

        # Points in camera referential with translation
        reconstructed_world = np.hstack(
            (Xr.reshape(-1, 1), Yr.reshape(-1, 1), Zr.reshape(-1, 1))
        )
        return reconstructed_world

    def check_ground(self, cone, threshold, lowest):
        z = cone[:, 2]
        # print(lowest)
        if (abs(max(z) - min(z))) < threshold:
            # if (abs(max(z)-lowest) < threshold) and (abs(min(z)-lowest) < threshold):
            # print(f"low point with min {min(z)} and max {max(z)}")
            return True
        else:
            return False

    def apply_dbscan(self, point_cloud):
        lowest_point = min(point_cloud[:, 2])
        clustering_one_cone = DBSCAN(eps=0.25, min_samples=3, algorithm="ball_tree")
        clustering_one_cone.fit_predict(point_cloud)
        labels = clustering_one_cone.labels_
        labels = list(labels)
        while len(labels) > 0:
            giant_component_label = mode(labels)
            mask_gc = labels == giant_component_label
            new_point_cloud = point_cloud[
                mask_gc
            ]  # points in giant component for 1 cone
            if self.check_ground(new_point_cloud, 1.5e-2, lowest_point):
                labels = list(filter((giant_component_label).__ne__, labels))
                point_cloud = point_cloud[mask_gc == False]
            else:
                break

        return new_point_cloud

    def run(self):
        np_pcd_norm = self.point_cloud
        preds = self.bounding_boxes

        np_new_ref = self.world_to_cam(np_pcd_norm, self.lidar_cam_translation)
        n_uv = (self.newcameramtx @ np_new_ref.T).T
        scaling_values = n_uv[:, 2]
        scaled_n_uv = np.divide(n_uv, n_uv[:, 2][:, None])
        scaled_n_uv, scaling_values = self.remove_2d_points(
            scaled_n_uv, scaling_values, (0, 0), (1024, 544)
        )
        all_cones = []
        avg_cones = []
        classes = []
        for pred in preds:
            eps_h = int(0.1 * (pred[3] - pred[1]))
            eps_w = int(0.1 * (pred[2] - pred[0]))
            scaled_n_uv_cone, scaling_values_cone = self.remove_2d_points(
                scaled_n_uv,
                scaling_values,
                (pred[0] - eps_w, pred[1] - eps_h),
                (pred[2] + eps_w, pred[3] + eps_h),
            )
            if len(scaled_n_uv_cone) > 0:
                reconstructed = (
                    np.linalg.inv(self.newcameramtx)
                    @ ((np.multiply(scaled_n_uv_cone, scaling_values_cone[:, None])).T)
                ).T
                reconstructed_world = self.cam_to_world(
                    reconstructed, self.lidar_cam_translation
                )
                reconstructed_world = self.apply_dbscan(reconstructed_world)
                all_cones.append(reconstructed_world)
                avg_cones.append(np.mean(reconstructed_world, axis=0).reshape(1, 3))
                classes.append(pred[-1])

        avg_cones = np.concatenate(avg_cones, axis=0)
        final_prediction = np.hstack((avg_cones, np.array(classes).reshape(-1, 1)))

        return final_prediction
