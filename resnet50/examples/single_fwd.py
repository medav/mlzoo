import numpy

import torch
import torch.fx
import torchvision


batch_size = 64
dtype = torch.float16
dev = torch.device('cuda:0')

net = torchvision.models.resnet50(pretrained=True).to(dtype).to(dev)
x = torch.randn(batch_size, 3, 224, 224, device=dev, dtype=dtype)

net(x)

