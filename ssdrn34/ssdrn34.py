import torch
from dataclasses import dataclass
from typing import Optional, Callable, Type, Union, List
from . import multibox as MB

# Note: This is an adaptation from torchvision.models.resnet
# See LICENSE.meta.torchvision for licensing details. I mostly wanted to remove
# the extra cruft that torchvision has to accomodate all the variations of
# ResNet. I also wanted to simplify the code around ResNetLayer (which used to
# be the _make_layer member function) to reduce the amount of easy-to-miss state
# that was being passed around.


class ResNetDownsample(torch.nn.Module):
    @dataclass
    class Config:
        in_planes : int
        out_planes : int
        stride : int
        norm : Type | callable

    def __init__(self, config : Config):
        super().__init__()

        self.layers = torch.nn.Sequential(*[
            torch.nn.Conv2d(
                config.in_planes,
                config.out_planes,
                kernel_size=1,
                stride=config.stride,
                bias=False),
            config.norm(config.out_planes)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResNetBasicBlock(torch.nn.Module):
    @dataclass
    class Config:
        in_planes : int
        hid_planes : int
        out_planes : int
        downsample_by : int = 1
        conv_groups : int = 1
        norm : Type | callable = torch.nn.BatchNorm2d

    def __init__(self, config: Config):
        super().__init__()

        self.layers = torch.nn.Sequential(*[
            torch.nn.Conv2d(
                config.in_planes,
                config.hid_planes,
                kernel_size=3,
                stride=config.downsample_by \
                    if config.downsample_by is not None else 1,
                padding=1,
                bias=False
            ),

            config.norm(config.hid_planes),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(
                config.hid_planes,
                config.out_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=config.conv_groups,
                bias=False
            ),

            config.norm(config.out_planes)
        ])

        self.relu = torch.nn.ReLU(inplace=True)

        if config.downsample_by is None:
            self.downsample = torch.nn.Identity()
        else:
            self.downsample = ResNetDownsample(
                ResNetDownsample.Config(
                    config.in_planes,
                    config.out_planes,
                    config.downsample_by,
                    config.norm
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.layers(x) + self.downsample(x))


class ResNetLayer(torch.nn.Module):
    @dataclass
    class Config:
        in_planes : int
        hid_planes : int
        out_planes : int
        num_blocks : int
        downsample_by : int = None
        conv_groups : int = 1
        norm : Type | callable = torch.nn.Identity

    def __init__(self, config: Config):
        super().__init__()

        block_config = ResNetBasicBlock.Config(
            config.out_planes,
            config.hid_planes,
            config.out_planes,
            downsample_by=None,
            conv_groups=config.conv_groups
        )

        self.layers = torch.nn.Sequential(*[
            ResNetBasicBlock(
                ResNetBasicBlock.Config(
                    config.in_planes,
                    config.hid_planes,
                    config.out_planes,
                    downsample_by=config.downsample_by,
                    conv_groups=config.conv_groups,
                    norm=config.norm
                )
            )
        ] + [
            ResNetBasicBlock(block_config)
            for _ in range(config.num_blocks - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

def det_conv_block(
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


class SsdResNet34(torch.nn.Module):
    """
    This is configured to match the MLPerf 0.5 SSD training model. The following
    link has a description of the model which calls out the modifications
    performed on the original ResNet34 model:

    https://github.com/mlcommons/training/tree/v0.5/single_stage_detector/ssd#publicationattribution
    """

    def __init__(
        self,
        input_size : tuple[int, int] = (300, 300), # TODO: Make this configurable
        num_classes : int = 81,
        chans : list[int] = [512, 512, 256, 256, 256],
        strides : list[int] = [2, 2, 2, 2, 1]
    ):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        # The output of the patched resnet is always 256 channels so we don't
        # let that be configurable.
        self.chans = [256] + chans

        self.strides = strides
        rn34_norm = torch.nn.BatchNorm2d

        rn34 = torch.nn.Sequential(*[
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            rn34_norm(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResNetLayer(
                ResNetLayer.Config(
                    in_planes=64,
                    hid_planes=64,
                    out_planes=64,
                    num_blocks=3,
                    downsample_by=None,
                    norm=rn34_norm
                )
            ),
            ResNetLayer(
                ResNetLayer.Config(
                    in_planes=64,
                    hid_planes=128,
                    out_planes=128,
                    num_blocks=4,
                    downsample_by=2,
                    norm=rn34_norm
                )
            ),
            ResNetLayer(
                ResNetLayer.Config(
                    in_planes=128,
                    hid_planes=256,
                    out_planes=256,
                    num_blocks=6,
                    downsample_by=1,
                    norm=rn34_norm
                )
            )
        ])

        self.det_blocks = torch.nn.ModuleList([
            rn34,
            det_conv_block(self.chans[0], self.chans[1], 256, 1, 3, self.strides[0], 1),
            det_conv_block(self.chans[1], self.chans[2], 256, 1, 3, self.strides[1], 1),
            det_conv_block(self.chans[2], self.chans[3], 128, 1, 3, self.strides[2], 1),
            det_conv_block(self.chans[3], self.chans[4], 128, 1, 3, self.strides[3], 1),
            det_conv_block(self.chans[4], self.chans[5], 128, 1, 3, self.strides[4], 0)
        ])

        self.det = MB.MultiboxDetector(
            num_classes,
            [
                MB.Detector.Config(input_size, (38, 38), self.chans[0], [2],    ( 21 / 300,  45 / 300)),
                MB.Detector.Config(input_size, (19, 19), self.chans[1], [2, 3], ( 45 / 300,  99 / 300)),
                MB.Detector.Config(input_size, (10, 10), self.chans[2], [2, 3], ( 99 / 300, 153 / 300)),
                MB.Detector.Config(input_size, (5, 5),   self.chans[3], [2, 3], (153 / 300, 207 / 300)),
                MB.Detector.Config(input_size, (3, 3),   self.chans[4], [2],    (207 / 300, 261 / 300)),
                MB.Detector.Config(input_size, (1, 1),   self.chans[5], [2],    (261 / 300, 315 / 300))
            ])


    def forward(self, x):
        activations = []
        print(x.shape)
        for l in self.det_blocks:
            x = l(x)
            print(x.shape)
            activations.append(x)

        return self.det(activations)


    def loss(self, x : torch.Tensor, detss : list[list[MB.SsdDetection]]) -> torch.Tensor:
        activations = []
        for l in self.det_blocks:
            x = l(x)
            activations.append(x)

        return self.det.loss(activations, detss)

    def detect(self, x : torch.Tensor) -> list[MB.SsdDetection]:
        h, w = x.shape[-2:]
        activations = []
        for l in self.det_blocks:
            x = l(x)
            activations.append(x)

        return self.det.detect(activations, (h, w))

    def load_pretrained_resnet34(self):
        from torchvision.models import resnet34
        from torchvision.models.resnet import BasicBlock

        rn34 = resnet34(pretrained=True)

        patched_layer3 = rn34.layer3
        patched_layer3[0].conv1.stride = (1, 1)
        patched_layer3[0].downsample[0].stride = (1, 1)
        patched_layer3[1].conv1.stride = (1, 1)
        patched_layer3[2].conv1.stride = (1, 1)
        patched_layer3[3].conv1.stride = (1, 1)
        patched_layer3[4].conv1.stride = (1, 1)
        patched_layer3[5].conv1.stride = (1, 1)

        patched = torch.nn.Sequential(*[
            rn34.conv1,
            rn34.bn1,
            rn34.relu,
            rn34.maxpool,
            rn34.layer1,
            rn34.layer2,
            patched_layer3
        ])


        def hookup_conv(conv : torch.nn.Conv2d, patched : torch.nn.Conv2d):
            conv.weight = torch.nn.Parameter(patched.weight.data)
            if patched.bias is not None:
                conv.bias = torch.nn.Parameter(patched.bias.data)

        def hookup_bn(bn : torch.nn.BatchNorm2d, patched : torch.nn.BatchNorm2d):
            bn.weight = torch.nn.Parameter(patched.weight.data)
            bn.bias = torch.nn.Parameter(patched.bias.data)
            bn.running_mean = patched.running_mean
            bn.running_var = patched.running_var

        def hookup_block(block : ResNetBasicBlock, patched : 'BasicBlock'):
            hookup_conv(block.layers[0], patched.conv1)
            hookup_bn(block.layers[1], patched.bn1)
            hookup_conv(block.layers[3], patched.conv2)
            hookup_bn(block.layers[4], patched.bn2)

        def hookup_layer(layer : ResNetLayer, patched : torch.nn.Sequential):
            hookup_block(layer.layers[0], patched[0])
            hookup_block(layer.layers[1], patched[1])
            hookup_block(layer.layers[2], patched[2])

        hookup_conv(self.det_blocks[0][0], patched[0])
        hookup_bn(self.det_blocks[0][1], patched[1])

        hookup_layer(self.det_blocks[0][4], patched[4])
        hookup_layer(self.det_blocks[0][5], patched[5])
        hookup_layer(self.det_blocks[0][6], patched[6])

    def freeze_backbone(self):
        for p in self.det_blocks[0].parameters():
            p.requires_grad = False

if __name__ == '__main__':
    model = SsdResNet34()
    model.load_pretrained_resnet34()

    x = torch.randn(1, 3, 300, 300)
    loc, conf = model(x)

