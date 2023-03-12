import os

import numpy as np
import torch

from brains_python.vision.yolo.models.yolo import DetectMultiBackend
from brains_python.vision.yolo.utils.datasets import letterbox
from brains_python.vision.yolo.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
)


class YoloPose:
    @torch.no_grad()
    def __init__(
        self,
        imgsz=[1024, 544],
        model_type="tiny",
        conf_thresh=0.01,
        iou_thresh=0.25,
        nc=4,
        nkpt=7,
        trt=False,
    ):
        # Load model
        ext = "engine" if trt else "pt"
        weights_path = os.path.join(
            os.path.dirname(__file__), f"weights/yolov7-{model_type}-cones-1200.{ext}"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DetectMultiBackend(
            weights=weights_path, device=torch.device(self.device)
        )  # load FP32 model
        self.stride, self.names, self.pt = (
            self.model.stride,
            self.model.names,
            self.model.pt,
        )
        if not isinstance(imgsz, (list, tuple)):
            imgsz = list(imgsz)
        if len(imgsz) == 1:
            imgsz[0] = check_img_size(imgsz[0], s=self.stride)
            imgsz.append(imgsz[0])
        else:
            assert len(imgsz) == 2
            "height and width of image has to be specified"
            imgsz[0] = check_img_size(imgsz[0], s=self.stride)
            imgsz[1] = check_img_size(imgsz[1], s=self.stride)
        self.imgsz = imgsz
        # imgsz[1] = 576 # Imgsz must be dividable by 64 for the w6 model

        self.model.half()  # to FP16
        self.opt = {"conf": conf_thresh, "iou": iou_thresh, "nc": nc, "nkpt": nkpt}
        self.conf = conf_thresh
        self.iou = iou_thresh
        self.nc = nc
        self.nkpt = nkpt
        print(self.imgsz)
        print(self.opt)

        # cudnn.benchmark = True  # set True to speed up constant image size inference, TODO:Check speedup

        # Warmup
        if self.device != "cpu":
            self.model.warmup(imgsz=(1, 3, *imgsz))  # warmup

    # Run inference on an image retrieved inside this function
    def detect(self, im0):
        img = letterbox(im0, self.imgsz, stride=self.stride, auto=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=False)

        # Apply NMS
        # pred = non_max_suppression(pred, self.opt.get('conf'), self.opt.get('iou'), nc=self.opt.get('nc'), nkpt=self.opt.get('nkpt'))[0]
        pred = non_max_suppression(
            pred, self.conf, self.iou, nc=self.nc, nkpt=self.nkpt
        )[0]
        # Rescale boxes from img_size to im0 size
        scale_coords(img.shape[2:], pred[:, :4], im0.shape, kpt_label=False)
        scale_coords(img.shape[2:], pred[:, 6:], im0.shape, kpt_label=True, step=3)
        pred = pred.detach().cpu()

        return pred
