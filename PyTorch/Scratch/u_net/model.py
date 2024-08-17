import torch
import torch.nn as nn

from model_builder import DoubleConv, DownSample, UpSample


class UNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()

        # Down-sampling
        self.down1 = DownSample(in_channels=in_channels, out_channels=64)
        self.down2 = DownSample(in_channels=64, out_channels=128)
        self.down3 = DownSample(in_channels=128, out_channels=256)
        self.down4 = DownSample(in_channels=256, out_channels=512)

        # Bottleneck
        self.bottle_neck = DoubleConv(in_channels=512, out_channels=1024)

        # Up-sampling
        self.up1 = UpSample(in_channels=1024, out_channels=512)
        self.up2 = UpSample(in_channels=512, out_channels=256)
        self.up3 = UpSample(in_channels=256, out_channels=128)
        self.up4 = UpSample(in_channels=128, out_channels=64)

        # Output layer
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down1, p1 = self.down1(x)
        down2, p2 = self.down2(p1)
        down3, p3 = self.down3(p2)
        down4, p4 = self.down4(p3)

        bottle_neck = self.bottle_neck(p4)

        up1 = self.up1(bottle_neck, down4)
        up2 = self.up2(up1, down3)
        up3 = self.up3(up2, down2)
        up4 = self.up4(up3, down1)

        return self.out(up4)
