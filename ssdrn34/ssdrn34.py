import torch
from torchvision.models.resnet import resnet34

from . import multibox as MB

def _patch_rn34_layer(layer : torch.nn.Module):
    layer[0].conv1.stride = (1, 1)
    layer[0].downsample[0].stride = (1, 1)
    layer[1].conv1.stride = (1, 1)
    layer[2].conv1.stride = (1, 1)
    layer[3].conv1.stride = (1, 1)
    layer[4].conv1.stride = (1, 1)
    layer[5].conv1.stride = (1, 1)
    return layer

def _patch_rn34_for_ssd(rn34 : torch.nn.Module):
    return torch.nn.Sequential(*[
        rn34.conv1,
        rn34.bn1,
        rn34.relu,
        rn34.maxpool,
        rn34.layer1,
        rn34.layer2,
        _patch_rn34_layer(rn34.layer3)
    ])

def conv_block(
    in_chan : int,
    out_chan : int,
    int_chan : int,
    ksize1 : int,
    ksize2 : int,
    stride2 : int,
    pad2 : int = 1
):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_chan, int_chan, kernel_size=ksize1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(int_chan, out_chan, kernel_size=ksize2, padding=pad2, stride=stride2),
        torch.nn.ReLU(inplace=True)
    )

class SsdRn34(torch.nn.Module):
    def __init__(
        self,
        input_size : tuple[int, int] = (300, 300), # TODO: Make this configurable
        num_classes : int = 81,
        chans : list[int] = [512, 512, 256, 256, 256],
        strides : list[int] = [2, 2, 2, 1, 1],
        dstride : int = 1
    ):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        # The output of the patched resnet is always 256 channels so we don't
        # let that be configurable.
        self.chans = [256] + chans

        self.strides = strides

        self.rn34 = _patch_rn34_for_ssd(resnet34())
        self.blocks = torch.nn.ModuleList([
            self.rn34,
            conv_block(self.chans[0], self.chans[1], 256, 1, 3, self.strides[0], 1),
            conv_block(self.chans[1], self.chans[2], 256, 1, 3, self.strides[1], 1),
            conv_block(self.chans[2], self.chans[3], 128, 1, 3, self.strides[2], 1),
            conv_block(self.chans[3], self.chans[4], 128, 1, 3, self.strides[3], 0),
            conv_block(self.chans[4], self.chans[5], 128, 1, 3, self.strides[4], 0)
        ])

        self.det = MB.MultiboxDetector(
            num_classes,
            [
                MB.Detector.Params((50, 50), self.chans[0], dstride, [2],    (21, 45),   (6, 6)),
                MB.Detector.Params((25, 25), self.chans[1], dstride, [2, 3], (45, 99),   (12, 12)),
                MB.Detector.Params((13, 13), self.chans[2], dstride, [2, 3], (99, 153),  (23, 23)),
                MB.Detector.Params((7, 7),   self.chans[3], dstride, [2, 3], (153, 207), (42, 42)),
                MB.Detector.Params((3, 3),   self.chans[4], dstride, [2],    (207, 261), (100, 100)),
                MB.Detector.Params((3, 3),   self.chans[5], dstride, [2],    (261, 315), (100, 100))
            ])


    def forward(self, x):
        activations = []
        for l in self.blocks:
            x = l(x)
            activations.append(x)

        return self.det(activations)


if __name__ == '__main__':
    model = SsdRn34()
    print(model)

    x = torch.randn(1, 3, 300, 300)
    loc, conf = model(x)

