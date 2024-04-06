import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time

import torch

import torch.amp
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from .. import multibox as MB
from .. import dataset
from .. import ssdrn34



def main():
    writer = SummaryWriter()
    batch_size = 128

    if os.path.exists('coco.pkl'):
        coco = dataset.CocoDataset.from_pkl('coco.pkl', train=False)
    else:
        coco = dataset.CocoDataset.from_annotations(
            '/home/medavies/coco/annotations/instances_val2017.json',
            '/home/medavies/coco/val2017',
            train=False)

        coco.save_pkl('coco.pkl')

    net = ssdrn34.SsdResNet34()
    net.load_state_dict(torch.load('/research/data/mlmodels/ssdrn34.pth'))
    net.cuda().eval()

    with torch.no_grad():
        for i in range(len(coco)):
            img, dets = coco[i]
            print(f'[{i}]: {len(dets)} ground-truth detections')

            dets = net.detect(img.unsqueeze(0).cuda())[0]

            for det in dets:
                print(coco.categories[det.label].name, det.ltrb)

            writer.add_image_with_boxes(
                f'coco_val2017[{i}]',
                coco.get_img_raw(i),
                torch.tensor([det.ltrb for det in dets]),
                0,
                labels=[coco.categories[det.label].name for det in dets])

    writer.close()

if __name__ == "__main__":
    main()

