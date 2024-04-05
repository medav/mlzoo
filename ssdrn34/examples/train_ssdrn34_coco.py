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

class RollingAvg:
    def __init__(self, n):
        self.n_max = n
        self.n = 0
        self.data = [0] * n
        self.i = 0

    def add(self, x):
        self.data[self.i] = x
        self.i = (self.i + 1) % self.n_max
        self.n = min(self.n + 1, self.n_max)

    @property
    def avg(self):
        if self.n == 0: return 0
        return sum(self.data) / self.n


def main():
    writer = SummaryWriter()
    batch_size = 32

    if os.path.exists('coco.pkl'):
        coco = dataset.CocoDataset.from_pkl('coco.pkl', train=True)
    else:
        coco = dataset.CocoDataset.from_annotations(
            '/research/data/mldata/coco/annotations/instances_train2017.json',
            '/research/data/mldata/coco/train2017',
            train=True)

        coco.save_pkl('coco.pkl')


    coco_val = dataset.CocoDataset.from_annotations(
        '/research/data/mldata/coco/annotations/instances_val2017.json',
        '/research/data/mldata/coco/val2017',
        train=False)

    dl = torch.utils.data.DataLoader(
        coco,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=dataset.coco_collate_fn,
        pin_memory=True,
        num_workers=4)

    net = ssdrn34.SsdResNet34()
    if os.path.exists('ssdrn34_coco.pth'):
        print('Loading ssdrn34_coco.pth...')
        net.load_state_dict(torch.load('ssdrn34_coco.pth'))
    else:
        net.load_pretrained_resnet34()

    net.cuda()

    opt = torch.optim.SGD(
        net.parameters(),
        lr=1e-4,
        momentum=0.9,
        weight_decay=5e-4
    )

    global_step = 0
    avg_time = RollingAvg(100)

    for i in range(100):
        print(f'Epoch {i}')
        imgs : torch.Tensor
        dets : list[list[MB.SsdDetection]]
        for imgs, dets in dl:
            ti = time.perf_counter()
            opt.zero_grad()
            loss = net.loss(imgs.cuda(), dets)
            writer.add_scalar('loss', loss.item(), global_step)
            loss.backward()
            opt.step()
            global_step += 1
            tf = time.perf_counter() - ti
            avg_time.add(tf)

            print(f'{i:3d}/{global_step:5d}: {batch_size / avg_time.avg:5.2f} img/sec, loss: {loss.item():8.4f}')
            if global_step % 100 == 0:
                print('Saving ssdrn34_coco.pth...')
                try:
                    torch.save(net.state_dict(), 'ssdrn34_coco.pth')
                except: pass

                print('Testing...')
                with torch.no_grad():
                    img, dets = coco_val[0]
                    print(f'[{i}]: {len(dets)} ground-truth detections')

                    dets = net.detect(img.unsqueeze(0).cuda())[0]

                    for det in dets:
                        print('    ', coco.categories[det.label].name, det.ltrb)

                    writer.add_image_with_boxes(
                        'coco[0]',
                        img,
                        torch.tensor([det.ltrb for det in dets]),
                        global_step,
                        labels=[coco.categories[det.label].name for det in dets])


                writer.flush()

        print('Saving ssdrn34_coco.pth...')
        try:
            torch.save(net.state_dict(), 'ssdrn34_coco.pth')
        except: pass

    writer.close()

if __name__ == "__main__":
    main()

