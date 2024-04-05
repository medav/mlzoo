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
):
    '''Calculate intersection over union for two sets of boxes.

    This adapted was copied from:
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


def non_max_suppression(bboxes_ltrb, scores, threshold=0.1, max_dets=100, mode='union'):
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
        img_size : tuple[int, int]
        det_size : tuple[int, int]
        in_chan : int
        ratios : list[int]
        minmax_size : tuple[int, int]
        stride : int = 1
        scale_xy : float = 0.1
        scale_wh : float = 0.2

    def __init__(self, config : Config, num_classes : int):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        gm_size = np.sqrt(self.min_size * self.max_size)

        self.box_sizes = [
            (self.min_size, self.min_size),
            (gm_size, gm_size)
        ] + [
            (self.min_size * np.sqrt(r), self.min_size / np.sqrt(r))
            for r in self.config.ratios
        ] + [
            (self.min_size / np.sqrt(r), self.min_size * np.sqrt(r))
            for r in self.config.ratios
        ]

        self.register_buffer('dboxes_cxywh', torch.tensor([
            (
                (x + 0.5) / self.det_w,
                (y + 0.5) / self.det_h,
                bw,
                bh
            )
            for bw, bh in self.box_sizes
            for y in range(self.det_h)
            for x in range(self.det_w)
        ]).clamp(0.0, 1.0), persistent=False)

        # print(self.dboxes_cxywh.shape)
        # print(self.dboxes_cxywh[:50, :])

        self.dboxes_cxywh : torch.Tensor

        self.register_buffer(
            'dboxes_ltrb', cxywh_to_ltrb(self.dboxes_cxywh), persistent=False)
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
            len(self.box_sizes) * self.num_classes,
            kernel_size=3,
            padding=1,
            stride=self.config.stride
        )

    @property
    def img_h(self): return self.config.img_size[0]

    @property
    def img_w(self): return self.config.img_size[1]

    @property
    def det_h(self): return self.config.det_size[0]

    @property
    def det_w(self): return self.config.det_size[1]

    # @property
    # def scale_h(self): return self.img_h / self.det_h

    # @property
    # def scale_w(self): return self.img_w / self.det_w

    @property
    def min_size(self): return self.config.minmax_size[0]

    @property
    def max_size(self): return self.config.minmax_size[1]

    @property
    def num_boxes(self): return self.dboxes_cxywh.shape[0]

    def forward(self, act : torch.Tensor):
        batch_size = act.size(0)
        loc = self.loc(act).view(batch_size, 4, -1).permute(0, 2, 1)
        conf = self.conf(act).view(batch_size, self.num_classes, -1).permute(0, 2, 1)
        return loc, conf

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


    def encode_dets(self, detss : list[list[SsdDetection]], iou_threshold : float = 0.5):
        dev = self.dboxes_ltrb.device
        locs = []
        clss = []

        dboxes_cxy = self.dboxes_cxywh[:, :2]
        dboxes_wh = self.dboxes_cxywh[:, 2:]

        loc_scale = torch.tensor([
            self.img_w, self.img_h, self.img_w, self.img_h
        ], device=dev)

        for dets in detss:
            if len(dets) == 0:
                locs.append(torch.zeros((self.num_boxes, 4), device=dev))
                clss.append(torch.zeros(self.num_boxes, device=dev, dtype=torch.long))
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


class MultiboxDetector(torch.nn.Module):

    def __init__(
        self,
        num_classes : int,
        dconfig : list[Detector.Config]
    ):
        super().__init__()
        self.num_classes = num_classes

        self.dets = torch.nn.ModuleList([
            Detector(p, num_classes) for p in dconfig
        ])

        self.dets : list[Detector]

    def forward(self, xs : list[torch.Tensor]):
        pred_locs = []
        pred_logitss = []

        for detector, x in zip(self.dets, xs):
            pred_loc, pred_logits = detector.forward(x)
            pred_locs.append(pred_loc)
            pred_logitss.append(pred_logits)

        pred_locs = torch.cat(pred_locs, dim=1)           # [B, NB,  4]
        pred_logitss = torch.cat(pred_logitss, dim=1)     # [B, NB, NC]

        return pred_locs, pred_logitss

    def detect(self, xs : list[torch.Tensor], img_hw : tuple[int, int]):
        boxes_ltrbss = []
        confss = []
        labelss = []

        ltrb_scale = torch.tensor([
            img_hw[1], img_hw[0], img_hw[1], img_hw[0]
        ], device='cpu')

        for x, det in zip(xs, self.dets):
            boxes_ltrbs, confs, labels = det.detect(x)
            boxes_ltrbss.append(boxes_ltrbs)
            confss.append(confs)
            labelss.append(labels)

        boxes_ltrbss = torch.cat(boxes_ltrbss, dim=1)
        confss = torch.cat(confss, dim=1)
        labelss = torch.cat(labelss, dim=1)

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
                dets.append(SsdDetection(
                    (boxes_ltrbs[i].cpu().detach() * ltrb_scale).numpy(),
                    labels[i] - 1
                ))

            detss.append(dets)

        return detss

    def encode_dets(self, detss : list[list[SsdDetection]]):
        target_locs = []
        target_labelss = []

        for det in self.dets:
            target_loc, target_labels = det.encode_dets(detss)
            target_locs.append(target_loc)
            target_labelss.append(target_labels)

        return torch.cat(target_locs, dim=1), torch.cat(target_labelss, dim=1)

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



