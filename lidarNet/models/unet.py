import torch
from torch import nn

from lidarNet.utils.ml_utils import crop_masks


class UNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        encoder_layers = []
        decoder_layers = []

        # channels = [3, 64, 128, 256, 512, 1024]
        channels = [3]
        for i in range(layers + 1):
            channels.append(2 ** (i + 6))
        for prev_c, c in zip(channels[:-1], channels[1:-1]):
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(prev_c, c, 3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c, c, 3),
                    nn.ReLU(inplace=True),
                )
            )
            encoder_layers.append(
                nn.MaxPool2d(kernel_size=2)
            )

        self.bottom_layer = nn.Sequential(
            nn.Conv2d(channels[-2], channels[-1], 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[-1], channels[-1], 3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[-1], channels[-2], kernel_size=2, stride=2)
        )

        for i in reversed(range(3, len(channels))):
            layer = nn.Sequential(
                nn.Conv2d(channels[i], channels[i - 1], 3),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[i - 1], channels[i - 1], 3),
                nn.ReLU(inplace=True),
            )
            layer.append(nn.ConvTranspose2d(channels[i - 1], channels[i - 2], kernel_size=2, stride=2))
            decoder_layers.append(layer)
        decoder_layers.append(nn.Sequential(
            nn.Conv2d(channels[2], channels[1], 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[1], channels[1], 3),
            nn.ReLU(inplace=True),
        ))

        self.final_layer = nn.Sequential(
            nn.Conv2d(channels[1], 1, 1)
        )
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        self.downscale = nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=2, padding=1)
        self.upscale = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)

    def encode(self, x):
        xs = []
        layers = self.encoder.children()
        for convs in layers:
            maxpool = next(layers)
            x = convs(x)
            xs.append(x)
            x = maxpool(x)
        return x, xs

    @staticmethod
    def concat_layer(z, x):
        # z is smaller one, x is larger
        x_crop = crop_masks(x, z.shape[-2:])
        return torch.cat((z, x_crop), dim=1)

    def decode(self, z, xs):
        z = self.bottom_layer(z)
        for i, l in enumerate(self.decoder.children()):
            z = self.concat_layer(z, xs[-(i + 1)])
            z = l(z)
        return self.final_layer(z)

    def forward(self, x):
        x = self.downscale(x)
        z, xs = self.encode(x)
        m = self.decode(z, xs)
        m = self.upscale(m)
        return m
