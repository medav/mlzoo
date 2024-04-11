import torch
from dataclasses import dataclass, replace
from enum import Enum
import numpy as np

#
# Mostly a copy of
#    https://github.com/kuangliu/pytorch-ssd/blob/master/multibox_layer.py
#
# With the added abstraction of Detector which makes dealing with
# default boxes a bit easier / more explicit (IMO).
#


@dataclass
class SsdDetection:
    xywh : tuple[float, float, float, float]
    label : int

    @property
    def ltrb(self):
        return (
            self.xywh[0],
            self.xywh[1],
            self.xywh[0] + self.xywh[2],
            self.xywh[1] + self.xywh[3]
        )


    def flip_lr(self, img_hw : tuple[int, int]):
        x, y, w, h = self.xywh
        return SsdDetection(
            (img_hw[1] - x - w, y, w, h),
            self.label
        )

    def flip_tb(self, img_hw : tuple[int, int]):
        x, y, w, h = self.xywh
        return SsdDetection(
            (x, img_hw[0] - y - h, w, h),
            self.label
        )

    def resize(self, orig_hw : tuple[int, int], new_hw : tuple[int, int]):
        oh, ow = orig_hw
        nh, nw = new_hw
        sw = nw / ow
        sh = nh / oh

        x, y, w, h = self.xywh
        return SsdDetection(
            (
                x * sw,
                y * sh,
                w * sw,
                h * sh
            ),
            self.label
        )

    def resized_crop(self, new_top, new_left, new_w, new_h, orig_hw : tuple[int, int]):
        oh, ow = orig_hw
        sw = new_w / ow
        sh = new_h / oh

        x, y, w, h = self.xywh
        return SsdDetection(
            (
                (x - new_left) / sw,
                (y - new_top) / sh,
                w / sw,
                h / sh
            ),
            self.label
        )

def xywh_to_cxy(xywh : torch.Tensor) -> torch.Tensor:
    return xywh[:, :2] + xywh[:, 2:] / 2

def cxywh_to_ltrb(xywh : torch.Tensor) -> torch.Tensor:
    return torch.cat([
        xywh[..., :2] - xywh[..., 2:] / 2,
        xywh[..., :2] + xywh[..., 2:] / 2
    ], -1)

def xywh_to_ltrb(xywh : torch.Tensor) -> torch.Tensor:
    return torch.cat([
        xywh[..., :2],
        xywh[..., :2] + xywh[..., 2:]
    ], -1)

def iou_ltrb(
    ltrb_a : torch.Tensor, # [N, 4]
    ltrb_b : torch.Tensor  # [M, 4]
) -> torch.Tensor:
    '''Calculate intersection over union for two sets of boxes.

    This code was copied and adapted from the following source:
        https://github.com/kuangliu/pytorch-ssd/blob/master/encoder.py#L38
        See licenses/LICENSE.kuangliu.ssd for details.
    '''
    N = ltrb_a.size(0)
    M = ltrb_b.size(0)

    lt = torch.max(
        ltrb_a[:, :2].unsqueeze(1).expand(N, M, 2),
        ltrb_b[:, :2].unsqueeze(0).expand(N, M, 2),
    )

    rb = torch.min(
        ltrb_a[:, 2:].unsqueeze(1).expand(N, M, 2),
        ltrb_b[:, 2:].unsqueeze(0).expand(N, M, 2),
    )

    wh = rb - lt

    # Clip at 0
    wh[wh < 0] = 0

    intersection = wh[:, :, 0] * wh[:, :, 1]

    area1 = ((ltrb_a[:, 2] - ltrb_a[:, 0]) * (ltrb_a[:, 3] - ltrb_a[:, 1])) \
        .unsqueeze(1).expand_as(intersection)

    area2 = ((ltrb_b[:, 2] - ltrb_b[:, 0]) * (ltrb_b[:, 3] - ltrb_b[:, 1])) \
        .unsqueeze(0).expand_as(intersection)

    union = area1 + area2 - intersection
    return intersection / union


def non_max_suppression(
    bboxes_ltrb : torch.Tensor,
    scores : torch.Tensor,
    threshold : float = 0.1,
    max_dets : int = 100,
    mode : str = 'union'
) -> torch.LongTensor:
    '''Non maximum suppression.

    Args:
        bboxes: (tensor) bounding boxes, sized [N,4] in LTRB format.
        scores: (tensor) bbox scores, sized [N,].
        threshold: (float) overlap threshold.
        mode: (str) 'union' or 'min'.

    Returns:
        keep: (tensor) selected indices.

    Ref:
        https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    x1 = bboxes_ltrb[:, 0]
    y1 = bboxes_ltrb[:, 1]
    x2 = bboxes_ltrb[:, 2]
    y2 = bboxes_ltrb[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)
    order = order.squeeze()

    keep = []
    while order.numel() > 0 and len(keep) < max_dets:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break

        i = order[0]
        keep.append(i)

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break

        order = order[ids + 1].reshape((-1,))

    return torch.LongTensor(keep)


class Detector(torch.nn.Module):
    @dataclass
    class Config:
        in_chan : int
        res_hw : tuple[int, int]
        aspect_ratios : list[int]
        min_size : float = None
        max_size : float = None
        num_classes : int = None
        scale_xy : float = 0.1
        scale_wh : float = 0.2
        stride : int = 1

    def __init__(self, config : Config):
        super().__init__()
        self.config = config

        min_size = self.config.min_size
        max_size = self.config.max_size

        gm_size = np.sqrt(min_size * max_size)

        self.box_sizes = [
            (min_size, min_size),
            (gm_size, gm_size)
        ] + [
            (min_size * np.sqrt(r), min_size / np.sqrt(r))
            for r in self.config.aspect_ratios
        ] + [
            (min_size / np.sqrt(r), min_size * np.sqrt(r))
            for r in self.config.aspect_ratios
        ]

        self.register_buffer(
            'dboxes_cxywh',
            torch.tensor([
                (
                    (x + 0.5) / self.config.res_hw[1],
                    (y + 0.5) / self.config.res_hw[0],
                    bw,
                    bh
                )
                for bw, bh in self.box_sizes
                for y in range(self.config.res_hw[0])
                for x in range(self.config.res_hw[1])
            ]).clamp(0.0, 1.0),
            persistent=False
        )

        self.dboxes_cxywh : torch.Tensor

        self.register_buffer(
            'dboxes_ltrb',
            cxywh_to_ltrb(self.dboxes_cxywh),
            persistent=False
        )

        self.dboxes_ltrb : torch.Tensor

        self.loc = torch.nn.Conv2d(
            self.config.in_chan,
            len(self.box_sizes) * 4,
            kernel_size=3,
            padding=1,
            stride=self.config.stride
        )

        self.conf = torch.nn.Conv2d(
            self.config.in_chan,
            len(self.box_sizes) * (self.config.num_classes + 1),
            kernel_size=3,
            padding=1,
            stride=self.config.stride
        )

    @property
    def num_boxes(self): return self.dboxes_cxywh.shape[0]

    def forward(self, act : torch.Tensor):
        batch_size = act.size(0)
        loc = self.loc(act).view(batch_size, 4, -1).permute(0, 2, 1)
        conf = self.conf(act) \
            .view(batch_size, self.config.num_classes + 1, -1).permute(0, 2, 1)
        return loc, conf

    def encode(
        self,
        detss : list[list[SsdDetection]],
        img_hw : tuple[int, int],
        iou_threshold : float = 0.5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dev = self.dboxes_ltrb.device
        locs = []
        clss = []

        dboxes_cxy = self.dboxes_cxywh[:, :2]
        dboxes_wh = self.dboxes_cxywh[:, 2:]

        loc_scale = torch.tensor([
            img_hw[1], img_hw[0], img_hw[1], img_hw[0]
        ], device=dev)

        for dets in detss:
            if len(dets) == 0:
                locs.append(torch.zeros((self.num_boxes, 4), device=dev))
                clss.append(torch.zeros(
                    self.num_boxes, device=dev, dtype=torch.long))
                continue

            ltrb = torch.tensor([
                det.ltrb for det in dets
            ], device=dev) / loc_scale # [ND, 4]

            ious = iou_ltrb(self.dboxes_ltrb, ltrb) # [NB, ND]
            max_iou, max_iou_idx = ious.max(1)

            # max_iou_idx : [NB]

            xywh = torch.tensor([
                det.xywh for det in dets
            ], device=dev)[max_iou_idx] / loc_scale

            cls  = torch.tensor([
                det.label + 1 for det in dets
            ], device=dev)[max_iou_idx]

            cxy = xywh[:, :2] + xywh[:, 2:] / 2
            wh = xywh[:, 2:]

            loc = torch.cat([
                (cxy - dboxes_cxy) / (self.config.scale_xy * dboxes_wh),
                torch.log(wh / dboxes_wh) / self.config.scale_wh
            ], 1)

            cls[max_iou < iou_threshold] = 0

            locs.append(loc)
            clss.append(cls)

        return torch.stack(locs), torch.stack(clss)

    def detect(self, x : torch.Tensor):
        pred_deltas, pred_conf = self.forward(x)
        # pred_deltas : [B, NB, 4]
        # pred_conf : [B, NB, C]

        scale_xy = self.config.scale_xy
        scale_wh = self.config.scale_wh

        dboxes_cxy = self.dboxes_cxywh[:, :2]
        dboxes_wh = self.dboxes_cxywh[:, 2:]

        wh = torch.exp(pred_deltas[..., 2:] * scale_wh) * dboxes_cxy
        cxy = pred_deltas[..., :2] * scale_xy * dboxes_wh + dboxes_cxy

        boxes_ltrb = torch.cat([cxy - wh / 2, cxy + wh / 2], -1)
        # boxes_ltrb : [B, NB, 4]

        conf, labels = pred_conf.max(dim=-1)
        # conf : [B, NB]
        # labels : [B, NB]

        return boxes_ltrb, conf, labels


class DboxSizes(Enum):
    COMPUTE = 0
    EXPLICIT = 1


class MultiboxDetector(torch.nn.Module):
    @dataclass
    class Config:
        num_classes : int
        dbox_sizes : DboxSizes
        dconfig : list[Detector.Config]
        s_min : float = None
        s_max : float = None

    def __init__(self, config : Config):
        super().__init__()
        self.config = config


        if self.config.dbox_sizes == DboxSizes.COMPUTE:
            assert self.config.s_min is not None
            assert self.config.s_max is not None

            k = np.array(range(len(self.config.dconfig) + 1)) + 1

            s_min = self.config.s_min
            s_max = self.config.s_max

            sk = s_min + (s_max - s_min) / (len(k) - 2) * (k - 1)

            min_sizes = sk[:-1]
            max_sizes = sk[1:]

        elif self.config.dbox_sizes == DboxSizes.EXPLICIT:
            min_sizes = [dc.min_size for dc in self.config.dconfig]
            max_sizes = [dc.max_size for dc in self.config.dconfig]

        else:
            raise ValueError('Invalid dbox_sizes')


        self.dets = torch.nn.ModuleList([
            Detector(replace(
                dc,
                num_classes=self.config.num_classes,
                min_size=min_sizes[i],
                max_size=max_sizes[i]
            ))
            for i, dc in enumerate(self.config.dconfig)
        ])

        self.dets : list[Detector]

    def forward(self, xs : list[torch.Tensor]):
        pred = [det.forward(x) for det, x in zip(self.dets, xs)]

        pred_locs = torch.cat([p[0] for p in pred], dim=1)
        pred_labels = torch.cat([p[1] for p in pred], dim=1)

        return pred_locs, pred_labels

    def encode(self, detss : list[list[SsdDetection]]):
        encoded = [det.encode(detss) for det in self.dets]

        target_locs = torch.cat([e[0] for e in encoded], dim=1)
        target_labels = torch.cat([e[1] for e in encoded], dim=1)

        return target_locs, target_labels

    def detect(self, xs : list[torch.Tensor], img_hw : tuple[int, int]):
        det_outs = [det.detect(x) for x, det in zip(xs, self.dets)]
        boxes_ltrbss = torch.cat([do[0] for do in det_outs], dim=1)
        confss = torch.cat([do[1] for do in det_outs], dim=1)
        labelss = torch.cat([do[2] for do in det_outs], dim=1)

        ltrb_scale = torch.tensor([
            img_hw[1], img_hw[0], img_hw[1], img_hw[0]
        ], device='cpu')

        detss = []

        for bi in range(boxes_ltrbss.size(0)):
            dets = []
            boxes_ltrbs = boxes_ltrbss[bi]
            confs = confss[bi]
            labels = labelss[bi]

            idx = labels.nonzero()
            if idx.numel() == 0:
                detss.append(dets)
                continue

            boxes_ltrbs = boxes_ltrbs[idx].squeeze(1)
            confs = confs[idx]
            labels = labels[idx]

            keep = non_max_suppression(boxes_ltrbs, confs)

            for i in keep:
                ltrb = (boxes_ltrbs[i].cpu().detach().clamp(0.0, 1.0) * ltrb_scale).numpy()
                dets.append(SsdDetection(
                    (ltrb[0], ltrb[1], ltrb[2] - ltrb[0], ltrb[3] - ltrb[1]),
                    labels[i] - 1
                ))

            detss.append(dets)

        return detss

    def loc_loss(
        self,
        pred_locs   : torch.Tensor, # [B, NB,  4]
        target_locs : torch.Tensor, # [B, NB,  4]
        pos_dets    : torch.Tensor  # [B, NB]
    ):
        mask = pos_dets.unsqueeze(2).expand_as(pred_locs)
        return torch.nn.functional.smooth_l1_loss(
            pred_locs[mask].view(-1, 4),
            target_locs[mask].view(-1, 4),
            reduction='sum'
        )

    def conf_loss(
        self,
        pred_logits   : torch.Tensor, # [B, NB, NC]
        target_labels : torch.Tensor, # [B, NB]
        pos_dets      : torch.Tensor  # [B, NB]
    ):
        assert pred_logits.size(-1) == self.num_classes

        conf_loss = torch.nn.functional.cross_entropy(
            pred_logits.view(-1, pred_logits.size(-1)),
            target_labels.view(-1),
            reduction='none'
        ).view_as(target_labels)

        num_boxes = pos_dets.size(-1)
        conf_loss[pos_dets] = 0

        _, idx = conf_loss.sort(dim=-1, descending=True)  # sort by neg conf_loss
        _, rank = idx.sort(dim=1)  # [B, NB]

        num_pos = pos_dets.long().sum(1)  # [B]

        num_neg = torch.clamp(3 * num_pos, max=num_boxes - 1)  # [B]
        neg_dets = rank < num_neg.unsqueeze(1).expand_as(rank)  # [B, NB]

        pos_mask = pos_dets.unsqueeze(2).expand_as(pred_logits)  # [B, NB, NC]
        neg_mask = neg_dets.unsqueeze(2).expand_as(pred_logits)  # [B, NB, NC]

        pred_mask = torch.logical_or(pos_mask, neg_mask)
        preds = pred_logits[pred_mask].view(-1, self.num_classes)
        # preds : [#pos+#neg, NC]

        target_mask = torch.logical_or(pos_dets, neg_dets)
        targets = target_labels[target_mask]
        # targets : [#pos+#neg,]

        return torch.nn.functional.cross_entropy(
            preds, targets, reduction='sum')

    def loss(
        self,
        xs : list[torch.Tensor],
        detss : list[list[SsdDetection]]
    ):
        pred_locs, pred_logitss = self.forward(xs)
        target_locs, target_labelss = self.encode_dets(detss)

        # pred_locs      : [B, NB,  4]
        # target_locs    : [B, NB,  4]

        # pred_logitss   : [B, NB, NC]
        # target_labelss : [B, NB]

        pos_dets = target_labelss > 0 # [B, NB]
        num_matched_boxes = pos_dets.sum() # [B]

        if num_matched_boxes == 0:
            return torch.tensor(0.0, device=pred_locs.device)

        loc_loss = self.loc_loss(pred_locs, target_locs, pos_dets)
        conf_loss = self.conf_loss(pred_logitss, target_labelss, pos_dets)

        return (loc_loss + conf_loss) / num_matched_boxes

