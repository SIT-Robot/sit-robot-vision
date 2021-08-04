from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config


class _DeepSortOpt:
    def __init__(self):
        self.config_deepsort = "deep_sort_pytorch/configs/deep_sort.yaml"


class MyDeepSort:
    def __init__(self):
        self.opt = _DeepSortOpt()
        self.cfg = get_config()
        self.cfg.merge_from_file(self.opt.config_deepsort)
        self.deepsort = DeepSort(self.cfg.DEEPSORT.REID_CKPT, max_dist=self.cfg.DEEPSORT.MAX_DIST,
                                 min_confidence=self.cfg.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=self.cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=self.cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=self.cfg.DEEPSORT.MAX_AGE, n_init=self.cfg.DEEPSORT.N_INIT,
                                 nn_budget=self.cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)

    def update(self, xywh_s, conf_s, cls_s, image):
        return self.deepsort.update(xywh_s.cpu(), conf_s.cpu(), cls_s, image)
