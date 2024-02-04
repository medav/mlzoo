import numpy
import torch
from .. import model

net = model.Unet3D(model.Unet3D.mlperf_config).cuda()

bs = 1

x = torch.randn((bs, 1, 128, 128, 128)).cuda()

net(x)
