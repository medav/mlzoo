import torch

from .. import model
from .. import dataset

net = model.Nerf().half().cuda()

print('Loading dataset...')
ds = dataset.BlenderDataset('/research/data/nerf/lego', split='train')
print('Done!')

rays, target = ds[0:65536]

rays = rays.half().cuda()

for _ in range(1000):
    net.render_rays(rays[0], rays[1], ds.focal, 8, near=2, far=6)
