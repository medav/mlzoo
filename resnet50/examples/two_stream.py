import numpy

import torch
import torch.fx
import torchvision


batch_size = 32
dtype = torch.float16
dev = torch.device('cuda:0')

net = torchvision.models.resnet50(pretrained=True).to(dtype).to(dev)

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

x1 = torch.randn(batch_size, 3, 224, 224, device=dev, dtype=dtype)
x2 = torch.randn(batch_size, 3, 224, 224, device=dev, dtype=dtype)

with torch.cuda.stream(s1):
    net(x1)

with torch.cuda.stream(s2):
    net(x2)

torch.cuda.synchronize()

