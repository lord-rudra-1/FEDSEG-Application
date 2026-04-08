"""
ResNet-UNet for segmentation (matches final-fedrated-pro.ipynb).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetUNet(nn.Module):
    def __init__(self, pretrained_encoder=False):
        super().__init__()
        base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained_encoder else None)

        self.encoder0 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.pool = base.maxpool
        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv4 = nn.Conv2d(256 + 256, 256, 3, padding=1)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = nn.Conv2d(128 + 128, 128, 3, padding=1)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = nn.Conv2d(64 + 64, 64, 3, padding=1)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv1 = nn.Conv2d(64 + 64, 64, 3, padding=1)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        input_size = x.shape[2:]

        e0 = self.encoder0(x)
        p = self.pool(e0)
        e1 = self.encoder1(p)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.up4(e4)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = F.relu(self.conv4(d4))

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = F.relu(self.conv3(d3))

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = F.relu(self.conv2(d2))

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e0], dim=1)
        d1 = F.relu(self.conv1(d1))

        out = self.final(d1)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(out)
