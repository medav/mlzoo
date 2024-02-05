import torch
from dataclasses import dataclass
from typing import Optional, Callable, Type, Union, List

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

class Bottleneck(torch.nn.Module):
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
                kernel_size=1,
                stride=1,
                bias=False
            ),

            config.norm(config.hid_planes),

            torch.nn.Conv2d(
                config.hid_planes,
                config.hid_planes,
                kernel_size=3,
                stride=config.downsample_by \
                    if config.downsample_by is not None else 1,
                padding=1,
                groups=config.conv_groups,
                bias=False
            ),

            config.norm(config.hid_planes),

            torch.nn.Conv2d(
                config.hid_planes,
                config.out_planes,
                kernel_size=1,
                stride=1,
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

        block_config = Bottleneck.Config(
            config.out_planes,
            config.hid_planes,
            config.out_planes,
            downsample_by=None,
            conv_groups=config.conv_groups
        )

        self.layers = torch.nn.Sequential(*[
            Bottleneck(
                Bottleneck.Config(
                    config.in_planes,
                    config.hid_planes,
                    config.out_planes,
                    downsample_by=config.downsample_by,
                    conv_groups=config.conv_groups,
                    norm=config.norm
                )
            )
        ] + [
            Bottleneck(block_config)
            for _ in range(config.num_blocks - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResNet(torch.nn.Module):
    @dataclass
    class Config:
        num_classes : int
        layers : list[int]
        zero_init_residual : bool
        groups: int
        width_per_group : int
        norm : Type | callable

    config_resnet50_mlperf = Config(
        num_classes=1000,
        layers=[3, 4, 6, 3],
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        norm=torch.nn.BatchNorm2d
    )

    def __init__(self, config : Config):
        super().__init__()
        inplanes = 64

        self.layers = torch.nn.Sequential(*[
            torch.nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            config.norm(inplanes),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResNetLayer(
                ResNetLayer.Config(
                    in_planes=64,
                    hid_planes=64,
                    out_planes=256,
                    num_blocks=config.layers[0],
                    downsample_by=1,
                    norm=config.norm
                )
            ),
            ResNetLayer(
                ResNetLayer.Config(
                    in_planes=256,
                    hid_planes=128,
                    out_planes=512,
                    num_blocks=config.layers[1],
                    downsample_by=2,
                    norm=config.norm
                )
            ),
            ResNetLayer(
                ResNetLayer.Config(
                    in_planes=512,
                    hid_planes=256,
                    out_planes=1024,
                    num_blocks=config.layers[2],
                    downsample_by=2,
                    norm=config.norm
                )
            ),
            ResNetLayer(
                ResNetLayer.Config(
                    in_planes=1024,
                    hid_planes=512,
                    out_planes=2048,
                    num_blocks=config.layers[3],
                    downsample_by=2,
                    norm=config.norm
                )
            ),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        ])

        self.classifier = torch.nn.Linear(512 * 4, config.num_classes)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.layers(x), 1))

