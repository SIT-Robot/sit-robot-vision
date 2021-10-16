import random

import numpy as np

from yolov5.utils.datasets import letterbox

import torch
from torch.backends import cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.torch_utils import select_device

import sys


# sys.path.insert(0, './yolov5')


class _DetectOpt:
    def __init__(self):
        self.output = 'inference/output'
        self.source = 0
        # self.deep_sort_weights = 'deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'
        self.yolo_weights = 'yolov5/weights/yolov5s.pt'
        self.imgsz = 640
        self.evaluate = False
        self.device = '0'
        # self.config_deepsort = "deep_sort_pytorch/configs/deep_sort.yaml"
        self.conf_thres = 0.5
        self.iou_thres = 0.7
        self.classes = 0
        self.agnostic_nms = False


class MyDetect:
    def __init__(self):
        self.opt = _DetectOpt()
        self._load_model()

    def _load_model(self):
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = attempt_load(self.opt.yolo_weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.opt.imgsz, s=self.stride)  # check img_size
        if self.half:
            self.model.half()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        pass

    def _process_image(self, source):
        pro_image = letterbox(source, new_shape=self.opt.imgsz)[0]
        pro_image = pro_image[:, :, ::-1].transpose(2, 0, 1)
        pro_image = np.ascontiguousarray(pro_image)
        pro_image = torch.from_numpy(pro_image).to(self.device)
        pro_image = pro_image.half()
        pro_image /= 255
        if pro_image.ndimension() == 3:
            pro_image = pro_image.unsqueeze(0)
        return pro_image

    def detect(self, source):
        """
        返回xywhs,confs,cls_s 这些都是tensor
        """
        xywhs = confs = clss = None

        processed_image = self._process_image(source)
        pred = self.model(processed_image, augment=False)[0]
        pred = non_max_suppression(
            pred, self.opt.conf_thres, self.opt.iou_thres)
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    processed_image.shape[2:], det[:, :4], source.shape).round()
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
        return xywhs, confs, clss
