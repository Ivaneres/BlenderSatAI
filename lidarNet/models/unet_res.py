import torch
from torch import nn

from lidarNet.utils.ml_utils import crop_masks


def conv2d_layer(in_channels: int, out_channels: int, stride=1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, padding_mode="replicate")


class ResBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, downsample=False):
        super().__init__()

        init_layer_stride = 2 if downsample else 1

        self.conv1 = conv2d_layer(in_channels, out_channels, stride=init_layer_stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.PReLU()
        self.conv2 = conv2d_layer(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = conv2d_layer(in_channels, out_channels, stride=init_layer_stride)
        self.relu2 = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu1(res)
        res = self.conv2(res)

        res += self.conv3(x)
        res = self.bn2(res)
        res = self.relu2(res)
        return res


class UpsamplingResBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = conv2d_layer(in_channels, out_channels)
        self.upsample1 = nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.PReLU()
        self.conv2 = conv2d_layer(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = conv2d_layer(in_channels, out_channels)
        self.relu2 = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv1(x)
        res = self.upsample1(res)
        res = self.bn1(res)
        res = self.relu1(res)
        res = self.conv2(res)

        concat = self.conv3(x)
        concat = self.upsample2(concat)

        res += concat
        res = self.bn2(res)
        res = self.relu2(res)
        return res


class UNetRes(nn.Module):

    def __init__(self):
        super().__init__()
        encoder_blocks = []
        decoder_blocks = []

        channels = [3, 64, 128, 256, 512, 1024]

        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.enc_bn_1 = nn.BatchNorm2d(channels[0])
        self.enc_bn_2 = nn.BatchNorm2d(channels[1])
        self.dec_bn = nn.BatchNorm2d(channels[1])

        encoder_blocks.append(
            nn.Sequential(
                self.enc_bn_1,
                ResBlock(channels[0], channels[0], downsample=True),
                conv2d_layer(channels[0], channels[1]),
                self.enc_bn_2,
                self.relu1,
                ResBlock(channels[1], channels[1]),
                ResBlock(channels[1], channels[1], downsample=True),
                ResBlock(channels[1], channels[1]),
                ResBlock(channels[1], channels[2], downsample=True),
                ResBlock(channels[2], channels[2]),
            )
        )
        encoder_blocks.append(
            nn.Sequential(
                ResBlock(channels[2], channels[3], downsample=True),
                ResBlock(channels[3], channels[3]),
            )
        )
        encoder_blocks.append(
            nn.Sequential(
                ResBlock(channels[3], channels[4], downsample=True),
                ResBlock(channels[4], channels[4]),
            )
        )
        encoder_blocks.append(
            nn.Sequential(
                ResBlock(channels[4], channels[5])
            )
        )

        decoder_blocks.append(
            nn.Sequential(
                ResBlock(channels[5], channels[5])
            )
        )
        decoder_blocks.append(
            nn.Sequential(
                UpsamplingResBlock(channels[5] + channels[4], channels[4]),
                ResBlock(channels[4], channels[4])
            )
        )
        decoder_blocks.append(
            nn.Sequential(
                UpsamplingResBlock(channels[4] + channels[3], channels[3]),
                ResBlock(channels[3], channels[3])
            )
        )
        decoder_blocks.append(
            nn.Sequential(
                UpsamplingResBlock(channels[3] + channels[2], channels[2]),
                ResBlock(channels[2], channels[2]),
                UpsamplingResBlock(channels[2], channels[1]),
                conv2d_layer(channels[1], channels[1]),
                self.dec_bn,
                self.relu2,
                conv2d_layer(channels[1], 1)
            )
        )

        self.encoder = nn.Sequential(*encoder_blocks)
        self.decoder = nn.Sequential(*decoder_blocks)

    def encode(self, x):
        xs = []
        for block in self.encoder:
            x = block(x)
            xs.append(x)
        return xs

    @staticmethod
    def concat_layer(z, x):
        # z is smaller one, x is larger
        # x_crop = crop_masks(x, z.shape[-2:])
        return torch.cat((z, x), dim=1)

    def decode(self, xs):
        z = xs[-1]
        for i, block in enumerate(self.decoder[:-1]):
            z = block(z)
            z = self.concat_layer(z, xs[-(i + 2)])
        z = self.decoder[-1](z)
        return z

    def forward(self, x):
        xs = self.encode(x)
        m = self.decode(xs)
        return m
