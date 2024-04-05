
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import pickle as pkl
import functools
from dataclasses import dataclass

import torch
import torchvision
from torchvision import transforms as TVT
import torchvision.transforms.functional

from . import multibox as MB

@dataclass
class CocoCategory:
    id : int
    name : str
    supercategory : str

@dataclass
class CocoImageRecord:
    img_id : int
    file : str
    dets : list[MB.SsdDetection]

class CustomRandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x : tuple[torch.Tensor, list[MB.SsdDetection]]):
        img, dets = x
        flipped = torch.rand(1) < self.p

        if flipped:
            img = torch.flip(img, dims=(-1,))

            dets = [
                det.flip_lr(img.shape[-2:])
                for det in dets
            ]

        return (img, dets)

class CustomRandomResizedCrop(TVT.RandomResizedCrop):
    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=TVT.InterpolationMode.BILINEAR,
        antialias = True,
    ):
        super().__init__(size, scale, ratio, interpolation, antialias)

    def forward(self, x : tuple[torch.Tensor, list[MB.SsdDetection]]):
        img, dets = x
        t, l, h, w = self.get_params(img, self.scale, self.ratio)
        img = torchvision.transforms.functional.resized_crop(
            img, t, l, h, w, self.size, self.interpolation, antialias=self.antialias)

        return (img, [det.resized_crop(t, l, w, h, self.size) for det in dets])

class CustomColorJitter(TVT.ColorJitter):
    def __init__(
        self,
        brightness = 0,
        contrast= 0,
        saturation = 0,
        hue = 0,
    ) -> None:
        super().__init__(brightness, contrast, saturation, hue)

    def forward(self, x : tuple[torch.Tensor, list[MB.SsdDetection]]):
        img, dets = x
        img = super().forward(img)
        return (img, dets)

class CustomResize(TVT.Resize):
    def __init__(self, size, interpolation=TVT.InterpolationMode.BILINEAR):
        super().__init__(size, interpolation, antialias=True)

    def forward(self, x : tuple[torch.Tensor, list[MB.SsdDetection]]):
        img, dets = x
        orig_size = img.shape[-2:]
        img = super().forward(img)
        return (img, [det.resize(orig_size, img.shape[-2:]) for det in dets])

class CustomNormalize(TVT.Normalize):
    def __init__(self, mean, std) -> None:
        super().__init__(mean, std)

    def forward(self, x : tuple[torch.Tensor, list[MB.SsdDetection]]):
        img, dets = x
        img = super().forward(img)
        return (img, dets)

class CocoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records : list[CocoImageRecord],
        categories : list[CocoCategory],
        img_dir : str,
        size : tuple[int, int] = (300, 300),
        train : bool = False
    ):
        self.records = records
        self.categories = categories
        self.img_dir = img_dir
        self.size = size
        self.train = train
        if train:
            self.transform = TVT.Compose([
                CustomResize(size),
                CustomRandomHorizontalFlip(),
                CustomRandomResizedCrop(size, scale=(0.08, 1.0)),
                CustomColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
                CustomNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        else:
            self.transform = TVT.Compose([
                CustomResize(size),
                CustomNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    @staticmethod
    def from_annotations(
        anno_file : str,
        img_dir : str,
        size : tuple[int, int] = (300, 300),
        train : bool = False
    ):
        annotations = json.load(open(anno_file, 'r'))

        categories = [
            CocoCategory(
                id=cat['id'],
                name=cat['name'],
                supercategory=cat['supercategory']
            )
            for cat in annotations['categories']
        ]

        cat_id_to_idx = \
            {cat.id: idx for idx, cat in enumerate(categories)}

        img_ids = [img['id'] for img in annotations['images']]
        img_files = [img['file_name'] for img in annotations['images']]
        img_dets = [[] for _ in img_ids]
        img_id_to_idx = {img_id: idx for idx, img_id in enumerate(img_ids)}


        for anno in annotations['annotations']:
            img_dets[img_id_to_idx[anno['image_id']]].append(
                MB.SsdDetection(
                    xywh=anno['bbox'],
                    label=cat_id_to_idx[anno['category_id']]
                )
            )

        records = [
            CocoImageRecord(
                img_id=img_id,
                file=img_file,
                dets=dets
            )
            for img_id, img_file, dets
            in zip(img_ids, img_files, img_dets)
        ]


        return CocoDataset(records, categories, img_dir, size)

    @staticmethod
    def from_pkl(
        pkl_file : str,
        size : tuple[int, int] = (300, 300),
        train : bool = False
    ):
        with open(pkl_file, 'rb') as f:
            data = pkl.load(f)
            return CocoDataset(
                records=data['records'],
                categories=data['categories'],
                img_dir=data['img_dir'],
                size=size,
                train=train
            )

    def save_pkl(self, pkl_file : str):
        with open(pkl_file, 'wb') as f:
            pkl.dump({
                'records': self.records,
                'categories': self.categories,
                'img_dir': self.img_dir
            }, f)

    def __len__(self):
        return len(self.records)


    @functools.lru_cache(maxsize=1000)
    def __getitem__(self, idx):
        record : CocoImageRecord = self.records[idx]
        img_file = os.path.join(self.img_dir, record.file)
        img = torchvision.io.read_image(img_file)
        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)

        img = img.float() / 255.0

        img, dets = self.transform((img, record.dets))
        return img, dets


class CocoDatasetInMem(torch.utils.data.Dataset):
    def __init__(
        self,
        records : list[CocoImageRecord],
        categories : list[CocoCategory],
        img_dir : str,
        size : tuple[int, int] = (300, 300),
        train : bool = False
    ):
        self.records = records
        self.categories = categories
        self.img_dir = img_dir
        self.size = size
        self.train = train

        if train:
            self.transform = TVT.Compose([
                CustomRandomHorizontalFlip(),
                CustomRandomResizedCrop(size, scale=(0.08, 1.0)),
                CustomColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
                CustomNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        else:
            self.transform = TVT.Compose([
                CustomNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        resize = TVT.Resize(size, antialias=True)

        self.data = torch.zeros(
            (len(self.records), 3, *size), dtype=torch.uint8, device='cpu')

        print(f'Loading {len(self.records)} images into memory...')
        for i, r in enumerate(self.records):
            img = torchvision.io.read_image(os.path.join(img_dir, r.file))
            if img.shape[0] == 1:
                img = img.expand(3, -1, -1)

            r.dets = [det.resize(img.shape[-2:], size) for det in r.dets]
            self.data[i, :, :, :] = resize(img)
        print('Done loading images.')

    @staticmethod
    def from_annotations(
        anno_file : str,
        img_dir : str,
        size : tuple[int, int] = (300, 300),
        train : bool = False
    ):
        annotations = json.load(open(anno_file, 'r'))

        categories = [
            CocoCategory(
                id=cat['id'],
                name=cat['name'],
                supercategory=cat['supercategory']
            )
            for cat in annotations['categories']
        ]

        cat_id_to_idx = \
            {cat.id: idx for idx, cat in enumerate(categories)}

        img_ids = [img['id'] for img in annotations['images']]
        img_files = [img['file_name'] for img in annotations['images']]
        img_dets = [[] for _ in img_ids]
        img_id_to_idx = {img_id: idx for idx, img_id in enumerate(img_ids)}


        for anno in annotations['annotations']:
            img_dets[img_id_to_idx[anno['image_id']]].append(
                MB.SsdDetection(
                    xywh=anno['bbox'],
                    label=cat_id_to_idx[anno['category_id']]
                )
            )

        records = [
            CocoImageRecord(
                img_id=img_id,
                file=img_file,
                dets=dets
            )
            for img_id, img_file, dets
            in zip(img_ids, img_files, img_dets)
        ]

        return CocoDatasetInMem(records, categories, img_dir, size, train)

    def __len__(self):
        return len(self.records)


    def __getitem__(self, idx):
        record = self.records[idx]
        img = self.data[idx].float() / 255.0
        img, dets = self.transform((img, record.dets))
        return img, dets

def coco_collate_fn(batch):
    imgs, dets = zip(*batch)
    return torch.stack(imgs), dets

if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    if os.path.exists('coco.pkl'):
        coco = CocoDataset.from_pkl('coco.pkl')
    else:
        coco = CocoDataset.from_annotations(
            '/home/medavies/coco/annotations/instances_train2014.json',
            '/home/medavies/coco/train2014')

        coco.save_pkl('coco.pkl')


    img, dets = coco[0]

    for det in dets:
        print(det.ltrb)

    writer.add_image_with_boxes(
        'example_imgs',
        img,
        torch.tensor([det.ltrb for det in dets]),
        0,
        labels=[coco.categories[det.label].name for det in dets])

    writer.close()
