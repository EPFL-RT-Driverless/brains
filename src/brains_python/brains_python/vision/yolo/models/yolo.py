# YOLOv5 YOLO-specific modules
import argparse
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn

from brains_python.vision.yolo.utils.autoanchor import check_anchor_order
from brains_python.vision.yolo.utils.general import (
    make_divisible,
    check_file,
)
from brains_python.vision.yolo.utils.torch_utils import (
    time_synchronized,
    fuse_conv_and_bn,
    model_info,
    scale_img,
    initialize_weights,
    select_device,
    copy_attr,
)

# This file contains modules common to various models

import math
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from brains_python.vision.yolo.utils.datasets import letterbox
from brains_python.vision.yolo.utils.general import (
    check_suffix,
    increment_path,
    make_divisible,
    non_max_suppression,
    non_max_suppression_export,
    save_one_box,
    scale_coords,
    xyxy2xywh,
)
from brains_python.vision.yolo.utils.plots import colors, plot_one_box
from brains_python.vision.yolo.utils.torch_utils import time_synchronized


__all__ = ["DetectMultiBackend"]


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)


class SPF(nn.Module):
    def __init__(self, k=3, s=1):
        super(SPF, self).__init__()
        self.n = (k - 1) // 2
        self.m = nn.Sequential(
            *[nn.MaxPool2d(kernel_size=3, stride=s, padding=1) for _ in range(self.n)]
        )

    def forward(self, x):
        return self.m(x)


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class ImplicitA(nn.Module):
    def __init__(self, channel):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, std=0.02)

    def forward(self, x):
        return self.implicit.expand_as(x) + x


class ImplicitM(nn.Module):
    def __init__(self, channel):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=1.0, std=0.02)

    def forward(self, x):
        return self.implicit.expand_as(x) * x


class ReOrg(nn.Module):
    def __init__(self):
        super(ReOrg, self).__init__()

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return torch.cat(
            [
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2],
            ],
            1,
        )


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if act != "ReLU":
            self.act = (
                nn.SiLU()
                if act is True
                else (act if isinstance(act, nn.Module) else nn.Identity())
            )
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(
            *[TransformerLayer(c2, num_heads) for _ in range(num_layers)]
        )
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self, c1, c2, shortcut=True, g=1, e=0.5, act=True
    ):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_, c2, 3, 1, g=g, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


def export_formats():
    # YOLOv5 export formats
    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorRT", "engine", ".engine", False, True],
        ["CoreML", "coreml", ".mlmodel", True, False],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
        ["TensorFlow GraphDef", "pb", ".pb", True, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", False, False],
        ["TensorFlow.js", "tfjs", "_web_model", False, False],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True],
    ]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])


class BottleneckCSPF(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSPF, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        # self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class BottleneckCSP2(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=False, g=1, e=0.5
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP2, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU()
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act=True
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c1, c_, 1, 1, act=act)
        self.cv3 = Conv(2 * c_, c2, 1, act=act)  # act=FReLU(c2)
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, e=1.0, act=act) for _ in range(n)]
        )
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(3, 3, 3)):
        print(k)
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        num_3x3_maxpool = []
        max_pool_module_list = []
        for pool_kernel in k:
            assert (pool_kernel - 3) % 2 == 0
            "Required Kernel size cannot be implemented with kernel_size of 3"
            num_3x3_maxpool = 1 + (pool_kernel - 3) // 2
            max_pool_module_list.append(
                nn.Sequential(
                    *num_3x3_maxpool
                    * [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
                )
            )
            # max_pool_module_list[-1] = nn.ModuleList(max_pool_module_list[-1])
        self.m = nn.ModuleList(max_pool_module_list)

        # self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPCSP(nn.Module):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSP, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU()
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class SPPCSPC(nn.Module):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.contract = Contract(gain=2)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        if hasattr(self, "contract"):
            x = self.contract(x)
        elif hasattr(self, "conv_slice"):
            x = self.conv_slice(x)
        else:
            x = torch.cat(
                [
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2],
                ],
                1,
            )
        return self.conv(x)


class ConvFocus(nn.Module):
    # Focus wh information into c-space
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvFocus, self).__init__()
        slice_kernel = 3
        slice_stride = 2
        self.conv_slice = Conv(c1, c1 * 4, slice_kernel, slice_stride, p, g, act)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        if hasattr(self, "conv_slice"):
            x = self.conv_slice(x)
        else:
            x = torch.cat(
                [
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2],
                ],
                1,
            )
        x = self.conv(x)
        return x


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        (
            N,
            C,
            H,
            W,
        ) = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s**2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s**2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, conf=0.25, kpt_label=False, nc=4, nkpt=7):
        super(NMS, self).__init__()
        self.conf = conf
        self.kpt_label = kpt_label
        self.nc = nc
        self.nkpt = nkpt

    def forward(self, x):
        return non_max_suppression(
            x[0],
            conf_thres=self.conf,
            iou_thres=self.iou,
            classes=self.classes,
            kpt_label=self.kpt_label,
            nc=self.nc,
            nkpt=self.nkpt,
        )


class NMS_Export(nn.Module):
    # Non-Maximum Suppression (NMS) module used while exporting ONNX model
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, conf=0.001, kpt_label=False, nc=4, nkpt=7):
        super(NMS_Export, self).__init__()
        self.conf = conf
        self.kpt_label = kpt_label
        self.nc = nc
        self.nkpt = nkpt

    def forward(self, x):
        return non_max_suppression_export(
            x[0],
            conf_thres=self.conf,
            iou_thres=self.iou,
            classes=self.classes,
            kpt_label=self.kpt_label,
            nc=self.nc,
            nkpt=self.nkpt,
        )


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print(
            "autoShape already enabled, skipping... "
        )  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/images/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != "cpu"):
                return self.model(
                    imgs.to(p.device).type_as(p), augment, profile
                )  # inference

        # Pre-process
        n, imgs = (
            (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])
        )  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f"image{i}"  # filename
            if isinstance(im, str):  # filename or uri
                im, f = (
                    np.asarray(
                        Image.open(
                            requests.get(im, stream=True).raw
                            if im.startswith("http")
                            else im
                        )
                    ),
                    im,
                )
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, "filename", f) or f
            files.append(Path(f).with_suffix(".jpg").name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = (
                im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)
            )  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = size / max(s)  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [
            make_divisible(x, int(self.stride.max()))
            for x in np.stack(shape1, 0).max(0)
        ]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.0  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != "cpu"):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(
                y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes
            )  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [
            torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1.0, 1.0], device=d)
            for im in imgs
        ]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple(
            (times[i + 1] - times[i]) * 1000 / self.n for i in range(3)
        )  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(
        self,
        pprint=False,
        show=False,
        save=False,
        crop=False,
        render=False,
        save_dir=Path(""),
    ):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f"image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} "
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f"{self.names[int(cls)]} {conf:.2f}"
                        if crop:
                            save_one_box(
                                box,
                                im,
                                file=save_dir
                                / "crops"
                                / self.names[int(cls)]
                                / self.files[i],
                            )
                        else:  # all others
                            plot_one_box(box, im, label=label, color=colors(cls))

            im = (
                Image.fromarray(im.astype(np.uint8))
                if isinstance(im, np.ndarray)
                else im
            )  # from np
            if pprint:
                print(str.rstrip(", "))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                print(
                    f"{'Saved' * (i == 0)} {f}",
                    end="," if i < self.n - 1 else f" to {save_dir}\n",
                )
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        print(
            f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}"
            % self.t
        )

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir="runs/hub/exp"):
        save_dir = increment_path(
            save_dir, exist_ok=save_dir != "runs/hub/exp", mkdir=True
        )  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir="runs/hub/exp"):
        save_dir = increment_path(
            save_dir, exist_ok=save_dir != "runs/hub/exp", mkdir=True
        )  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        print(f"Saved results to {save_dir}\n")

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = (
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "confidence",
            "class",
            "name",
        )  # xyxy columns
        cb = (
            "xcenter",
            "ycenter",
            "width",
            "height",
            "confidence",
            "class",
            "name",
        )  # xywh columns
        for k, c in zip(["xyxy", "xyxyn", "xywh", "xywhn"], [ca, ca, cb, cb]):
            a = [
                [x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()]
                for x in getattr(self, k)
            ]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [
            Detections([self.imgs[i]], [self.pred[i]], self.names, self.s)
            for i in range(self.n)
        ]
        for d in x:
            for k in ["imgs", "pred", "xyxy", "xyxyn", "xywh", "xywhn"]:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat(
            [self.aap(y) for y in (x if isinstance(x, list) else [x])], 1
        )  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(
        self,
        weights="yolov5s.pt",
        device=torch.device("cpu"),
        dnn=False,
        data=None,
        fp16=True,
    ):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   TensorRT:                       *.engine
        super().__init__()

        self.nc = 4  # number of classes
        self.nkpt = 7
        self.no_det = self.nc + 5  # number of outputs per anchor for box and class
        self.no_kpt = 3 * self.nkpt  ## number of outputs per anchor for keypoints
        self.no = self.no_det + self.no_kpt
        self.nl = 3  # number of detection layers
        self.grid = [torch.zeros(1)] * self.nl
        self.device = device
        self.anchor_grid = torch.tensor(
            [
                [[[[[4, 5]]], [[[6, 8]]], [[[10, 12]]]]],  # P3/8
                [[[[[15, 19]]], [[[23, 30]]], [[[39, 52]]]]],  # P4/16
                [[[[[72, 97]]], [[[123, 164]]], [[[209, 297]]]]],
            ],  # P5/32
            dtype=torch.float16,
            device=device,
        )
        self.na = 3
        self.stride_f = torch.tensor([8, 16, 32], device=device)

        w = str(weights[0] if isinstance(weights, list) else weights)
        (
            pt,
            jit,
            onnx,
            xml,
            engine,
            coreml,
            saved_model,
            pb,
            tflite,
            edgetpu,
            tfjs,
        ) = self.model_type(
            w
        )  # get backend
        stride, names = 32, [f"class{i}" for i in range(1000)]  # assign defaults
        fp16 &= (pt or jit or onnx or engine) and device.type != "cpu"  # FP16

        if pt:  # PyTorch
            print("attempt_load pt")
            model = attempt_load(
                weights if isinstance(weights, list) else w, map_location=device
            )
            stride = max(int(model.stride.max()), 32)  # model stride
            names = (
                model.module.names if hasattr(model, "module") else model.names
            )  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif engine:  # TensorRT
            print("Load engine")
            # trt.init_libnvinfer_plugins(None,'')
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            bindings = OrderedDict()
            print(model)
            print(model.num_bindings)
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                print(index, name)
                dtype = trt.nptype(model.get_binding_dtype(index))
                shape = tuple(model.get_binding_shape(index))
                data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(
                    device
                )
                bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
                if model.binding_is_input(index) and dtype == np.float16:
                    fp16 = True
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            context = model.create_execution_context()
            batch_size = bindings["images"].shape[0]
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.pt:  # PyTorch
            y = self.model(im, augment=augment)[0]
        elif self.engine:  # TensorRT
            assert im.shape == self.bindings["images"].shape, (
                im.shape,
                self.bindings["images"].shape,
            )
            self.binding_addrs["images"] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            # y = self.bindings['output0'].data
            # y0 = self.bindings['411'].data
            y0 = self.bindings["onnx::Slice_411"].data
            print(0, y0.shape)

            # y1 = self.bindings['477'].data
            y1 = self.bindings["onnx::Slice_967"].data
            print(1, y1.shape)

            # y2 = self.bindings['543'].data
            y2 = self.bindings["onnx::Slice_1523"].data
            print(2, y2.shape)

            y = self.concat_output([y0, y1, y2])
            print(y.shape)

            y = self.bindings["2014"].data
            print(y.shape)

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 1024, 544)):
        # Warmup model by running inference once
        if any(
            (self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb)
        ):  # warmup types
            if self.device.type != "cpu":  # only warmup GPU models
                im = torch.zeros(
                    *imgsz,
                    dtype=torch.half if self.fp16 else torch.float,
                    device=self.device,
                )  # input
                for _ in range(2 if self.jit else 1):  #
                    self.forward(im)  # warmup

    @staticmethod
    def model_type(p="path/to/model.pt"):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx

        suffixes = list(export_formats().Suffix) + [".xml"]  # export suffixes
        check_suffix(p, suffixes)  # checks
        p = Path(p).name  # eliminate trailing separators
        print(suffixes)
        (
            pt,
            jit,
            onnx,
            xml,
            engine,
            coreml,
            saved_model,
            pb,
            tflite,
            edgetpu,
            tfjs,
            paddle,
            xml2,
        ) = (s in p for s in suffixes)
        xml |= xml2  # *_openvino_model or *.xml
        tflite &= not edgetpu  # *.tflite
        return (
            pt,
            jit,
            onnx,
            xml,
            engine,
            coreml,
            saved_model,
            pb,
            tflite,
            edgetpu,
            tfjs,
        )

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def concat_output(self, out_ls):
        z = []
        for i in range(self.nl):
            x_det = out_ls[i][..., : self.no_det]
            x_kpt = out_ls[i][..., self.no_det :]
            print(i, len(out_ls[i][out_ls[i][..., 4] > 0.1]))

            if self.grid[i].shape[2:4] != out_ls[i].shape[2:4]:
                _, self.na, ny, nx, _ = out_ls[i].shape
                self.grid[i] = self._make_grid(nx, ny).to(out_ls[i].device)

            kpt_grid_x = self.grid[i][..., 0:1]
            kpt_grid_y = self.grid[i][..., 1:2]

            y = x_det.sigmoid()

            xy = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride_f[i]  # xy
            wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(
                1, self.na, 1, 1, 2
            )  # wh
            x_kpt[..., 0::3] = (
                x_kpt[..., ::3] * 2.0 - 0.5 + kpt_grid_x.repeat(1, 1, 1, 1, self.nkpt)
            ) * self.stride_f[
                i
            ]  # xy
            x_kpt[..., 1::3] = (
                x_kpt[..., 1::3] * 2.0 - 0.5 + kpt_grid_y.repeat(1, 1, 1, 1, self.nkpt)
            ) * self.stride_f[
                i
            ]  # xy
            # x_kpt[..., 0::3] = (x_kpt[..., ::3] + kpt_grid_x.repeat(1,1,1,1,self.nkpt)) * self.stride[i]  # xy
            # x_kpt[..., 1::3] = (x_kpt[..., 1::3] + kpt_grid_y.repeat(1,1,1,1,self.nkpt)) * self.stride[i]  # xy
            # print('=============')
            # print(self.anchor_grid[i].shape)
            # print(self.anchor_grid[i][...,0].unsqueeze(4).shape)
            # print(x_kpt[..., 0::3].shape)
            # x_kpt[..., 0::3] = ((x_kpt[..., 0::3].tanh() * 2.) ** 3 * self.anchor_grid[i][...,0].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_x.repeat(1,1,1,1,self.nkpt) * self.stride[i]  # xy
            # x_kpt[..., 1::3] = ((x_kpt[..., 1::3].tanh() * 2.) ** 3 * self.anchor_grid[i][...,1].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_y.repeat(1,1,1,1,self.nkpt) * self.stride[i]  # xy
            # x_kpt[..., 0::3] = (((x_kpt[..., 0::3].sigmoid() * 4.) ** 2 - 8.) * self.anchor_grid[i][...,0].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_x.repeat(1,1,1,1,self.nkpt) * self.stride[i]  # xy
            # x_kpt[..., 1::3] = (((x_kpt[..., 1::3].sigmoid() * 4.) ** 2 - 8.) * self.anchor_grid[i][...,1].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_y.repeat(1,1,1,1,self.nkpt) * self.stride[i]  # xy
            x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()

            y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim=-1)
            z.append(y.view(1, -1, self.no))

        return torch.cat(z, 1)


def attempt_load(weights, map_location=None, inplace=True):
    from brains_python.vision.yolo.models.yolo import Detect, Model

    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(
            ckpt["ema" if ckpt.get("ema") else "model"].float().fuse().eval()
        )  # FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [
            nn.Hardswish,
            nn.LeakyReLU,
            nn.ReLU,
            nn.ReLU6,
            nn.SiLU,
            Detect,
            Model,
        ]:
            m.inplace = inplace  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print("Ensemble created with %s\n" % weights)
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super(Sum, self).__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(
                -torch.arange(1.0, n) / 2, requires_grad=True
            )  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(
        self, c1, c2, k=1, s=1, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv, self).__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super(GhostBottleneck, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),
        )  # pw-linear
        self.shortcut = (
            nn.Sequential(
                DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)
            )
            if s == 2
            else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class MixConv2d(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1e-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[
                0
            ].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [
                nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False)
                for g in range(groups)
            ]
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(
        self, nc=80, anchors=(), nkpt=None, ch=(), inplace=True, dw_conv_kpt=False
    ):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.nkpt = nkpt
        self.dw_conv_kpt = dw_conv_kpt
        self.no_det = nc + 5  # number of outputs per anchor for box and class
        self.no_kpt = 3 * self.nkpt  ## number of outputs per anchor for keypoints
        self.no = self.no_det + self.no_kpt
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.flip_test = False
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer(
            "anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2)
        )  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no_det * self.na, 1) for x in ch
        )  # output conv
        if self.nkpt is not None:
            if self.dw_conv_kpt:  # keypoint head is slightly more complex
                self.m_kpt = nn.ModuleList(
                    nn.Sequential(
                        DWConv(x, x, k=3),
                        Conv(x, x),
                        DWConv(x, x, k=3),
                        Conv(x, x),
                        DWConv(x, x, k=3),
                        Conv(x, x),
                        DWConv(x, x, k=3),
                        Conv(x, x),
                        DWConv(x, x, k=3),
                        Conv(x, x),
                        DWConv(x, x, k=3),
                        nn.Conv2d(x, self.no_kpt * self.na, 1),
                    )
                    for x in ch
                )
            else:  # keypoint head is a single convolution
                self.m_kpt = nn.ModuleList(
                    nn.Conv2d(x, self.no_kpt * self.na, 1) for x in ch
                )

        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            if self.nkpt is None or self.nkpt == 0:
                x[i] = self.m[i](x[i])
            else:
                x[i] = torch.cat((self.m[i](x[i]), self.m_kpt[i](x[i])), axis=1)

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
            x_det = x[i][..., :6]
            x_kpt = x[i][..., 6:]

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                kpt_grid_x = self.grid[i][..., 0:1]
                kpt_grid_y = self.grid[i][..., 1:2]

                if self.nkpt == 0:
                    y = x[i].sigmoid()
                else:
                    y = x_det.sigmoid()

                if self.inplace:
                    xy = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(
                        1, self.na, 1, 1, 2
                    )  # wh
                    if self.nkpt != 0:
                        x_kpt[..., 0::3] = (
                            x_kpt[..., ::3] * 2.0
                            - 0.5
                            + kpt_grid_x.repeat(1, 1, 1, 1, self.nkpt)
                        ) * self.stride[
                            i
                        ]  # xy
                        x_kpt[..., 1::3] = (
                            x_kpt[..., 1::3] * 2.0
                            - 0.5
                            + kpt_grid_y.repeat(1, 1, 1, 1, self.nkpt)
                        ) * self.stride[
                            i
                        ]  # xy
                        # x_kpt[..., 0::3] = ((x_kpt[..., 0::3].tanh() * 2.) ** 3 * self.anchor_grid[i][:,0].repeat(self.nkpt,1).permute(1,0).view(1, self.na, 1, 1, self.nkpt)) + kpt_grid_x.repeat(1,1,1,1,self.nkpt) * self.stride[i]  # xy
                        # x_kpt[..., 1::3] = ((x_kpt[..., 1::3].tanh() * 2.) ** 3 * self.anchor_grid[i][:,0].repeat(self.nkpt,1).permute(1,0).view(1, self.na, 1, 1, self.nkpt)) + kpt_grid_y.repeat(1,1,1,1,self.nkpt) * self.stride[i]  # xy
                        x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()

                    y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim=-1)

                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    if self.nkpt != 0:
                        y[..., 6:] = (
                            y[..., 6:] * 2.0
                            - 0.5
                            + self.grid[i].repeat((1, 1, 1, 1, self.nkpt))
                        ) * self.stride[
                            i
                        ]  # xy
                    y = torch.cat((xy, wh, y[..., 4:]), -1)

                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class IDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(
        self, nc=80, anchors=(), nkpt=None, ch=(), inplace=True, dw_conv_kpt=False
    ):  # detection layer
        super(IDetect, self).__init__()
        self.nc = nc  # number of classes
        self.nkpt = nkpt
        self.dw_conv_kpt = dw_conv_kpt
        self.no_det = nc + 5  # number of outputs per anchor for box and class
        self.no_kpt = 3 * self.nkpt  ## number of outputs per anchor for keypoints
        self.no = self.no_det + self.no_kpt
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.flip_test = False
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer(
            "anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2)
        )  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no_det * self.na, 1) for x in ch
        )  # output conv

        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no_det * self.na) for _ in ch)

        if self.nkpt is not None:
            if self.dw_conv_kpt:  # keypoint head is slightly more complex
                self.m_kpt = nn.ModuleList(
                    nn.Sequential(
                        DWConv(x, x, k=3),
                        Conv(x, x),
                        DWConv(x, x, k=3),
                        Conv(x, x),
                        DWConv(x, x, k=3),
                        Conv(x, x),
                        DWConv(x, x, k=3),
                        Conv(x, x),
                        DWConv(x, x, k=3),
                        Conv(x, x),
                        DWConv(x, x, k=3),
                        nn.Conv2d(x, self.no_kpt * self.na, 1),
                    )
                    for x in ch
                )
            else:  # keypoint head is a single convolution
                self.m_kpt = nn.ModuleList(
                    nn.Conv2d(x, self.no_kpt * self.na, 1) for x in ch
                )

        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            if self.nkpt is None or self.nkpt == 0:
                x[i] = self.im[i](self.m[i](self.ia[i](x[i])))  # conv
            else:
                x[i] = torch.cat(
                    (self.im[i](self.m[i](self.ia[i](x[i]))), self.m_kpt[i](x[i])),
                    axis=1,
                )

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
            x_det = x[i][..., :6]
            x_kpt = x[i][..., 6:]

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                kpt_grid_x = self.grid[i][..., 0:1]
                kpt_grid_y = self.grid[i][..., 1:2]

                if self.nkpt == 0:
                    y = x[i].sigmoid()
                else:
                    y = x_det.sigmoid()

                if self.inplace:
                    xy = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(
                        1, self.na, 1, 1, 2
                    )  # wh
                    if self.nkpt != 0:
                        x_kpt[..., 0::3] = (
                            x_kpt[..., ::3] * 2.0
                            - 0.5
                            + kpt_grid_x.repeat(1, 1, 1, 1, self.nkpt)
                        ) * self.stride[
                            i
                        ]  # xy
                        x_kpt[..., 1::3] = (
                            x_kpt[..., 1::3] * 2.0
                            - 0.5
                            + kpt_grid_y.repeat(1, 1, 1, 1, self.nkpt)
                        ) * self.stride[
                            i
                        ]  # xy
                        # x_kpt[..., 0::3] = (x_kpt[..., ::3] + kpt_grid_x.repeat(1,1,1,1,self.nkpt)) * self.stride[i]  # xy
                        # x_kpt[..., 1::3] = (x_kpt[..., 1::3] + kpt_grid_y.repeat(1,1,1,1,self.nkpt)) * self.stride[i]  # xy
                        # print('=============')
                        # print(self.anchor_grid[i].shape)
                        # print(self.anchor_grid[i][...,0].unsqueeze(4).shape)
                        # print(x_kpt[..., 0::3].shape)
                        # x_kpt[..., 0::3] = ((x_kpt[..., 0::3].tanh() * 2.) ** 3 * self.anchor_grid[i][...,0].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_x.repeat(1,1,1,1,self.nkpt) * self.stride[i]  # xy
                        # x_kpt[..., 1::3] = ((x_kpt[..., 1::3].tanh() * 2.) ** 3 * self.anchor_grid[i][...,1].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_y.repeat(1,1,1,1,self.nkpt) * self.stride[i]  # xy
                        # x_kpt[..., 0::3] = (((x_kpt[..., 0::3].sigmoid() * 4.) ** 2 - 8.) * self.anchor_grid[i][...,0].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_x.repeat(1,1,1,1,self.nkpt) * self.stride[i]  # xy
                        # x_kpt[..., 1::3] = (((x_kpt[..., 1::3].sigmoid() * 4.) ** 2 - 8.) * self.anchor_grid[i][...,1].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_y.repeat(1,1,1,1,self.nkpt) * self.stride[i]  # xy
                        x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()

                    y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim=-1)

                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    if self.nkpt != 0:
                        y[..., 6:] = (
                            y[..., 6:] * 2.0
                            - 0.5
                            + self.grid[i].repeat((1, 1, 1, 1, self.nkpt))
                        ) * self.stride[
                            i
                        ]  # xy
                    y = torch.cat((xy, wh, y[..., 4:]), -1)

                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class IKeypoint(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(
        self, nc=4, anchors=(), nkpt=7, ch=(), inplace=True, dw_conv_kpt=False
    ):  # detection layer
        super(IKeypoint, self).__init__()
        self.nc = nc  # number of classes
        self.nkpt = nkpt
        self.dw_conv_kpt = dw_conv_kpt
        self.no_det = nc + 5  # number of outputs per anchor for box and class
        self.no_kpt = 3 * self.nkpt  ## number of outputs per anchor for keypoints
        self.no = self.no_det + self.no_kpt
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors

        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.flip_test = False
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer(
            "anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2)
        )  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no_det * self.na, 1) for x in ch
        )  # output conv

        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no_det * self.na) for _ in ch)

        if self.nkpt is not None:
            if self.dw_conv_kpt:  # keypoint head is slightly more complex
                self.m_kpt = nn.ModuleList(
                    nn.Sequential(
                        DWConv(x, x, k=3),
                        Conv(x, x),
                        DWConv(x, x, k=3),
                        Conv(x, x),
                        DWConv(x, x, k=3),
                        Conv(x, x),
                        DWConv(x, x, k=3),
                        Conv(x, x),
                        DWConv(x, x, k=3),
                        Conv(x, x),
                        DWConv(x, x, k=3),
                        nn.Conv2d(x, self.no_kpt * self.na, 1),
                    )
                    for x in ch
                )
            else:  # keypoint head is a single convolution
                self.m_kpt = nn.ModuleList(
                    nn.Conv2d(x, self.no_kpt * self.na, 1) for x in ch
                )

        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        print("Forward IKeypoint")
        self.training |= self.export
        for i in range(self.nl):
            if self.nkpt is None or self.nkpt == 0:
                x[i] = self.im[i](self.m[i](self.ia[i](x[i])))  # conv
            else:
                x[i] = torch.cat(
                    (self.im[i](self.m[i](self.ia[i](x[i]))), self.m_kpt[i](x[i])),
                    axis=1,
                )

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
            # print(i, x[i].shape)
            # print(i, len(x[i][x[i][..., 4]>0.6]))
            x_det = x[i][..., : self.no_det]
            x_kpt = x[i][..., self.no_det :]

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                kpt_grid_x = self.grid[i][..., 0:1]
                kpt_grid_y = self.grid[i][..., 1:2]

                if self.nkpt == 0:
                    y = x[i].sigmoid()
                else:
                    y = x_det.sigmoid()

                # print(self.inplace)
                if self.inplace:
                    xy = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(
                        1, self.na, 1, 1, 2
                    )  # wh
                    if self.nkpt != 0:
                        x_kpt[..., 0::3] = (
                            x_kpt[..., ::3] * 2.0
                            - 0.5
                            + kpt_grid_x.repeat(1, 1, 1, 1, self.nkpt)
                        ) * self.stride[
                            i
                        ]  # xy
                        x_kpt[..., 1::3] = (
                            x_kpt[..., 1::3] * 2.0
                            - 0.5
                            + kpt_grid_y.repeat(1, 1, 1, 1, self.nkpt)
                        ) * self.stride[
                            i
                        ]  # xy
                        # x_kpt[..., 0::3] = (x_kpt[..., ::3] + kpt_grid_x.repeat(1,1,1,1,self.nkpt)) * self.stride[i]  # xy
                        # x_kpt[..., 1::3] = (x_kpt[..., 1::3] + kpt_grid_y.repeat(1,1,1,1,self.nkpt)) * self.stride[i]  # xy
                        # print('=============')
                        # print(self.anchor_grid[i].shape)
                        # print(self.anchor_grid[i][...,0].unsqueeze(4).shape)
                        # print(x_kpt[..., 0::3].shape)
                        # x_kpt[..., 0::3] = ((x_kpt[..., 0::3].tanh() * 2.) ** 3 * self.anchor_grid[i][...,0].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_x.repeat(1,1,1,1,self.nkpt) * self.stride[i]  # xy
                        # x_kpt[..., 1::3] = ((x_kpt[..., 1::3].tanh() * 2.) ** 3 * self.anchor_grid[i][...,1].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_y.repeat(1,1,1,1,self.nkpt) * self.stride[i]  # xy
                        # x_kpt[..., 0::3] = (((x_kpt[..., 0::3].sigmoid() * 4.) ** 2 - 8.) * self.anchor_grid[i][...,0].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_x.repeat(1,1,1,1,self.nkpt) * self.stride[i]  # xy
                        # x_kpt[..., 1::3] = (((x_kpt[..., 1::3].sigmoid() * 4.) ** 2 - 8.) * self.anchor_grid[i][...,1].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_y.repeat(1,1,1,1,self.nkpt) * self.stride[i]  # xy
                        x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()

                    y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim=-1)

                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    if self.nkpt != 0:
                        y[..., 6:] = (
                            y[..., 6:] * 2.0
                            - 0.5
                            + self.grid[i].repeat((1, 1, 1, 1, self.nkpt))
                        ) * self.stride[
                            i
                        ]  # xy
                    y = torch.cat((xy, wh, y[..., 4:]), -1)

                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(
        self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None
    ):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            self.yaml["nc"] = nc  # override yaml value
        self.model, self.save = parse_model(
            deepcopy(self.yaml), ch=[ch]
        )  # model, savelist
        self.inplace = self.yaml.get("inplace", True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect) or isinstance(m, IDetect) or isinstance(m, IKeypoint):
            s = 256  # 2x min stride
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))]
            )  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()

    def forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        return torch.cat(y, 1), None  # augmented inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            # print(m)
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers

            if isinstance(m, nn.Upsample):
                m.recompute_scale_factor = False

            if profile:
                o = 0.0
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    print(
                        f"{'time (ms)':>10s} {'GFLOPS':>10s} {'params':>10s}  {'module'}"
                    )
                print(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")

            y.append(x if m.i in self.save else None)  # save output

        return x

        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
            x, y, wh = (
                p[..., 0:1] / scale,
                p[..., 1:2] / scale,
                p[..., 2:4] / scale,
            )  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _initialize_biases(
        self, cf=None
    ):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(
                8 / (640 / s) ** 2
            )  # obj (8 objects per 640 image)
            b.data[:, 5:] += (
                math.log(0.6 / (m.nc - 0.99))
                if cf is None
                else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             logger.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS
        if mode and not present:
            m = NMS()  # module
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name="%s" % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        m = autoShape(self)  # wrap model
        copy_attr()  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    anchors, nc, nkpt, gd, gw = (
        d["nc"],
        d["nkpt"],
        d["depth_multiple"],
        d["width_multiple"],
    )
    na = (
        (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    )  # number of anchors
    no = na * (nc + 5 + 2 * nkpt)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate():  # from, number, module, args
        args_dict = {}
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            DWConv,
            MixConv2d,
            Focus,
            ConvFocus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3TR,
            BottleneckCSPF,
            BottleneckCSP2,
            SPPCSP,
            SPPCSPC,
        ]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [
                BottleneckCSP,
                C3,
                C3TR,
                BottleneckCSPF,
                BottleneckCSP2,
                SPPCSP,
                SPPCSPC,
            ]:
                args.insert(2, n)  # number of repeats
                n = 1
            if m in [
                Conv,
                GhostConv,
                Bottleneck,
                GhostBottleneck,
                DWConv,
                MixConv2d,
                Focus,
                ConvFocus,
                CrossConv,
                BottleneckCSP,
                C3,
                C3TR,
            ]:
                if "act" in d.keys():
                    args_dict = {"act": d["act"]}
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m in [Detect, IDetect, IKeypoint]:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if "dw_conv_kpt" in d.keys():
                args_dict = {"dw_conv_kpt": d["dw_conv_kpt"]}
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        m_ = (
            nn.Sequential(*[m(*args, **args_dict) for _ in range(n)])
            if n > 1
            else m(*args, **args_dict)
        )  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = (
            i,
            f,
            t,
            np,
        )  # attach index, 'from' index, type, number params
        save.extend(
            x % i for x in ([f] if isinstance(f, int) else f) if x != -1
        )  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="model.yaml")
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 320, 320).to(device)
    # y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # logger.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
