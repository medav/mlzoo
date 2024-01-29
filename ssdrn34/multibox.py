import torch
from dataclasses import dataclass
import numpy as np

#
# Mostly a copy of
#    https://github.com/kuangliu/pytorch-ssd/blob/master/multibox_layer.py
#
# With the added abstraction of Detector which makes dealing with
# default boxes a bit easier / more explicit (IMO).
#

def non_max_suppression(bboxes, scores, threshold=0.5, mode='union'):
    '''Non maximum suppression.

    Args:
        bboxes: (tensor) bounding boxes, sized [N,4].
        scores: (tensor) bbox scores, sized [N,].
        threshold: (float) overlap threshold.
        mode: (str) 'union' or 'min'.

    Returns:
        keep: (tensor) selected indices.

    Ref:
        https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    areas = (x2-x1) * (y2-y1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)

class Detector(torch.nn.Module):
    @dataclass
    class Params:
        size : tuple[int, int]
        in_chan : int
        stride : int
        ratios : list[int]
        minmax_size : tuple[int, int]
        step : tuple[int, int]

    def __init__(self, params : Params, num_classes : int):
        super().__init__()
        self.params = params
        self.num_classes = num_classes

        gm_size = np.sqrt(self.min_size * self.max_size)

        self.box_sizes = [
            (self.min_size, self.min_size),
            (gm_size, gm_size)
        ] + [
            (self.min_size * np.sqrt(r), self.min_size / np.sqrt(r))
            for r in self.params.ratios
        ] + [
            (self.min_size / np.sqrt(r), self.min_size * np.sqrt(r))
            for r in self.params.ratios
        ]

        self.dboxes = self.register_buffer('dboxes', torch.tensor([
            (
                (x + 0.5) * self.step_w / self.w,
                (y + 0.5) * self.step_h / self.h,
                bw,
                bh
            )
            for bw, bh in self.box_sizes
            for y in range(self.h)
            for x in range(self.w)
        ]), persistent=False)

        self.loc = torch.nn.Conv2d(
            self.params.in_chan,
            len(self.box_sizes) * 4,
            kernel_size=3,
            padding=1,
            stride=self.params.stride
        )

        self.conf = torch.nn.Conv2d(
            self.params.in_chan,
            len(self.box_sizes) * self.num_classes,
            kernel_size=3,
            padding=1,
            stride=self.params.stride
        )

    @property
    def h(self): return self.params.size[0]

    @property
    def w(self): return self.params.size[1]

    @property
    def step_h(self): return self.params.step[0]

    @property
    def step_w(self): return self.params.step[1]

    @property
    def min_size(self): return self.params.minmax_size[0]

    @property
    def max_size(self): return self.params.minmax_size[1]

    def forward(self, x : torch.Tensor):
        loc = self.loc(x)
        conf = self.conf(x)
        return loc, conf

    def detect(self, x : torch.Tensor):
        raise NotImplementedError()
        scale_xy = 0.1 # TODO: These should be parameters of the Detector
        scale_wh = 0.2 # TODO: These should be parameters of the Detector

        loc, conf = self.forward(x)
        cxcy = loc[:, :2] * scale_xy * self.dboxes[:, 2:] + self.dboxes[:, :2]

        wh = torch.exp(loc[:, 2:] * scale_wh) * self.dboxes[:, 2:]

        boxes = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 1)  # [8732,4]

        max_conf, labels = conf.max(1)  # [8732,1]
        ids = labels.squeeze(1).nonzero().squeeze(1)  # [#boxes,]

        keep = self.nms(boxes[ids], max_conf[ids].squeeze(1))
        return boxes[ids][keep], labels[ids][keep], max_conf[ids][keep]

class MultiboxDetector(torch.nn.Module):
    def __init__(
        self,
        num_classes : int,
        dparams : list[Detector.Params]
    ):
        super().__init__()
        self.num_classes = num_classes

        self.dets = torch.nn.ModuleList([
            Detector(p, num_classes) for p in dparams
        ])

    def forward(self, xs : list[torch.Tensor]):
        locs = []
        confs = []
        for x, det in zip(xs, self.dets):
            loc, conf = det(x)
            locs.append(loc)
            confs.append(conf)

        return locs, confs

    def detect(self, xs : list[torch.Tensor]):
        raise NotImplementedError()


