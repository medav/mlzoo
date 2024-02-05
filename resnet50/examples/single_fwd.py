import numpy

import torch
from .. import model

batch_size = 64
dtype = torch.float16
dev = torch.device('cuda:0')

net = model.ResNet(model.ResNet.config_resnet50_mlperf).to(dtype).to(dev)
x = torch.randn(batch_size, 3, 224, 224, device=dev, dtype=dtype)

net(x)

