import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time

import torch

import torch.amp
import torch.utils.data

from .. import multibox as MB
from .. import dataset
from .. import ssdrn34



def main():

    if os.path.exists('coco.pkl'):
        coco = dataset.CocoDataset.from_pkl('coco.pkl', train=False)
    else:
        coco = dataset.CocoDataset.from_annotations(
            '/home/medavies/coco/annotations/instances_train2017.json',
            '/home/medavies/coco/train2017',
            train=True)

        coco.save_pkl('coco.pkl')

    for i, cat in enumerate(coco.categories):
        print(f'[{i}] {cat.name}')

if __name__ == "__main__":
    main()

