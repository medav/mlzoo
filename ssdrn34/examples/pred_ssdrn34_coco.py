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
            '/home/medavies/coco/annotations/instances_train2014.json',
            '/home/medavies/coco/train2014',
            train=True)

        coco.save_pkl('coco.pkl')

    net = ssdrn34.SsdResNet34()
    if os.path.exists('ssdrn34_coco.pth'):
        print('Loading ssdrn34_coco.pth...')
        net.load_state_dict(torch.load('ssdrn34_coco.pth'))
    else:
        net.load_pretrained_resnet34()

    net.cuda()
    net.freeze_backbone()
    net.eval()

    with torch.no_grad():
        for i in range(1):
            img, dets = coco[i]
            print(f'[{i}]: {len(dets)} ground-truth detections')

            dets = net.detect(img.unsqueeze(0).cuda())[0]

            for det in dets:
                print(coco.categories[det.label].name, det.ltrb)

            writer.add_image_with_boxes(
                'example_imgs',
                img,
                torch.tensor([det.ltrb for det in dets]),
                i,
                labels=[coco.categories[det.label].name for det in dets])

    writer.close()

if __name__ == "__main__":
    main()

