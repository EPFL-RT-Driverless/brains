import numpy as np
import torch

from brains_python.vision.pnp import PnP
from brains_python.vision.yolo.yolo_pose import YoloPose
from brains_python.common import ConeColor


class ConeObserverCameraOnly:
    yolo: YoloPose
    distance_kpt: PnP
    line_thickness = 3
    window_name = "streaming"

    def __init__(self, camera_matrix_name: str = "SimuMatrixReg"):
        self.yolo = YoloPose()
        self.distance_kpt = PnP(camera_matrix_name)

    def get_cones_observations(
        self, image
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies YOLO to the image to find the bounding boxes and the keypoints of the
        cones, and then estimates the distance using PnP.
        :return: (cones_ranges, cones_bearings, cones_colors)
        """
        # TODO implement this
        # NOTE: use the ConeColor enum to get the int corresponding to the color of the cone
        pass

    # def compute_keypointsPnP(self, im0):
    #     """

    #     Args:
    #     - image: Image for which

    #     Returns:
    #     - The [x, y] position of the cone
    #     """

    #     #t_before = time.perf_counter()

    #     # pred = self.yolo_pose.detect(im0)
    #     pred = self.yolo.detect(im0)
    #     cone_positions = []

    #     # Iterate over predicted cones in the image
    #     for pred_index, (*xyxy, conf, cls) in enumerate(pred[:,:6]):
    #         # Compute distance
    #         pkpt_x = pred[pred_index, (6)::3]
    #         pkpt_y = pred[pred_index, (6 + 1)::3]
    #         kpts = torch.cat((pkpt_x, pkpt_y)).reshape(2, self.yolo.opt.get('nkpt')).t()
    #         r, a = self.distance_kpt.get2DpointPol(kpts.numpy())
    #         if (r > 15) or (cls.item() == 3) or (((xyxy[3] - xyxy[1])/(xyxy[2] - xyxy[0])).item() > 1.5) or (pkpt_y[0].item() > im0.shape[0]-75):
    #             # Skip cones that are predicted further away than 18m
    #             # OR are predicted as big orange cones
    #             # OR where the ratio between height and width is bigger than 2
    #             # OR where the top point is too close to the lower border (at least 50 pixel above)
    #             continue
    #         cone_positions.append([r, a])

    #         # Stream result
    #         # if SHOW_BOXES:
    #         #     c = int(cls)  # integer class
    #         #     # label = f'{self.yolo_pose.names[c]} {conf:.2f} ({r:.2f},{a:.2f})'
    #         #     label = f'{self.yolo.names[c]} {conf:.2f} ({r:.2f},{a:.2f})'
    #         #     kpts = pred[pred_index, 6:]
    #         #     plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=self.line_thickness, kpt_label=True, kpts=kpts, steps=3, orig_shape=im0.shape[:2])

    #     #t_after = time.perf_counter()
    #     #print(f"Yolo prediction with keypoints: {t_after - t_before}")

    #     # if SHOW_BOXES:
    #     #     cv2.imshow(self.window_name, im0)
    #     #     cv2.waitKey(1)

    #     return pred[:, :6], cone_positions
