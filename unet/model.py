
from dataclasses import dataclass
import torch

class ConvBlock(torch.nn.Module):
    @dataclass
    class Config:
        in_channels : int
        out_channels : int
        kernel_size : int = 3
        stride : int = 1
        padding : int = 1
        conv : callable = torch.nn.Conv3d
        norm : callable = lambda _: torch.nn.Identity()
        act : callable = torch.nn.ReLU

    def __init__(self, config : Config):
        super().__init__()
        self.conv = config.conv(
            config.in_channels,
            config.out_channels,
            config.kernel_size,
            config.stride,
            config.padding)

        self.norm = config.norm(config.out_channels)
        self.act = config.act()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

@dataclass
class BlockConfig:
    in_channels : int
    out_channels : int
    norm : callable = lambda _: torch.nn.Identity()
    act : callable = torch.nn.ReLU

class DownsampleBlock(torch.nn.Module):
    def __init__(self, config : BlockConfig):
        super().__init__()
        self.conv1 = ConvBlock(
            ConvBlock.Config(
                config.in_channels,
                config.out_channels,
                stride=2,
                norm=config.norm,
                act=config.act
            )
        )

        self.conv2 = ConvBlock(
            ConvBlock.Config(
                config.out_channels,
                config.out_channels,
                norm=config.norm,
                act=config.act
            )
        )

    def forward(self, x : torch.Tensor):
        return self.conv2(self.conv1(x))


class UpsampleBlock(torch.nn.Module):
    def __init__(self, config : BlockConfig):
        super().__init__()
        self.upsample_conv = ConvBlock(
            ConvBlock.Config(
                config.in_channels,
                config.out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                conv=torch.nn.ConvTranspose3d,
                norm=lambda _: torch.nn.Identity(),
                act=lambda: torch.nn.Identity()
            )
        )

        self.conv1 = ConvBlock(
            ConvBlock.Config(
                2 * config.out_channels,
                config.out_channels,
                norm=config.norm,
                act=config.act
            )
        )

        self.conv2 = ConvBlock(
            ConvBlock.Config(
                config.out_channels,
                config.out_channels,
                norm=config.norm,
                act=config.act
            )
        )

    def forward(self, x : torch.Tensor, skip : torch.Tensor):
        return self.conv2(self.conv1(
            torch.cat((self.upsample_conv(x), skip), dim=1)))

class InputBlock(torch.nn.Module):
    def __init__(self, config : BlockConfig):
        super().__init__()
        self.conv1 = ConvBlock(
            ConvBlock.Config(
                config.in_channels,
                config.out_channels,
                norm=config.norm,
                act=config.act
            )
        )

        self.conv2 = ConvBlock(
            ConvBlock.Config(
                config.out_channels,
                config.out_channels,
                norm=config.norm,
                act=config.act
            )
        )


    def forward(self, x):
        return self.conv2(self.conv1(x))


class OutputLayer(torch.nn.Module):
    def __init__(self, in_channels : int, n_class : int):
        super().__init__()
        self.conv = torch.nn.Conv3d(
            in_channels, n_class, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x): return self.conv(x)


class Unet3D(torch.nn.Module):
    @dataclass
    class Config:
        in_channels : int
        n_class : int
        norm : callable
        act : callable
        filters : list[int]
        weights_init_scale : float = 1.0

    mlperf_config = Config(
        in_channels=1,
        n_class=3,
        norm=torch.nn.InstanceNorm3d,
        act=torch.nn.ReLU,
        filters=[32, 64, 128, 256, 320],
        weights_init_scale=1.0
    )

    def __init__(self, config : Config):
        super().__init__()

        in_channels = config.filters[:-1]
        out_channels = config.filters[1:]

        self.input_block = InputBlock(
            BlockConfig(
                config.in_channels,
                config.filters[0],
                config.norm,
                config.act
            )
        )

        self.downsample = torch.nn.ModuleList([
            DownsampleBlock(BlockConfig(ic, oc, config.norm, config.act))
             for ic, oc in zip(in_channels, out_channels)
        ])

        self.bottleneck = DownsampleBlock(
            BlockConfig(
                config.filters[-1],
                config.filters[-1],
                config.norm,
                config.act
            )
        )

        self.upsample = torch.nn.ModuleList([
            UpsampleBlock(
                BlockConfig(
                    config.filters[-1],
                    config.filters[-1],
                    config.norm,
                    config.act
                )
            )
        ] + [
            UpsampleBlock(BlockConfig(ic, oc, config.norm, config.act))
            for ic, oc in reversed(list(zip(out_channels, in_channels)))
        ])

        self.output = OutputLayer(config.filters[0], config.n_class)

        for name, v in self.named_parameters():
            if 'weight' in name or 'bias' in name:
                v.data *= float(config.weights_init_scale)

    def forward(self, x):
        x = self.input_block(x)
        outputs = [x]

        for downsample in self.downsample:
            x = downsample(x)
            outputs.append(x)

        x = self.bottleneck(x)

        for upsample, skip in zip(self.upsample, reversed(outputs)):
            x = upsample(x, skip)

        x = self.output(x)

        return x
