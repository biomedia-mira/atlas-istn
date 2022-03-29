import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet2D(nn.Module):

    def __init__(self, num_classes):
        super(UNet2D, self).__init__()
        use_bias = True
        self.conv11 = nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=use_bias)
        self.conv12 = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=use_bias)
        self.down1 = nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=use_bias)
        self.down3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=use_bias)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv32 = nn.Conv2d(96, 32, kernel_size=3, padding=1, bias=use_bias)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv22 = nn.Conv2d(48, 16, kernel_size=3, padding=1, bias=use_bias)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv13 = nn.Conv2d(24, 8, kernel_size=3, padding=1, bias=use_bias)
        self.conv14 = nn.Conv2d(8, num_classes, kernel_size=1, padding=0, bias=use_bias)

    def forward(self, x):
        x1 = F.relu(self.conv11(x))
        x1 = F.relu(self.conv12(x1))
        x2 = self.down1(x1)
        x2 = F.relu(self.conv21(x2))
        x3 = self.down2(x2)
        x3 = F.relu(self.conv31(x3))
        x4 = self.down3(x3)
        x4 = F.relu(self.conv41(x4))
        x3 = torch.cat([self.up3(x4), x3], dim=1)
        x3 = F.relu(self.conv32(x3))
        x2 = torch.cat([self.up2(x3), x2], dim=1)
        x2 = F.relu(self.conv22(x2))
        x1 = torch.cat([self.up1(x2), x1], dim=1)
        x1 = F.relu(self.conv13(x1))
        x = self.conv14(x1)

        return x

class UNet3D(nn.Module):

    def __init__(self, num_classes):
        super(UNet3D, self).__init__()
        use_bias = True
        self.conv11 = nn.Conv3d(1, 8, kernel_size=3, padding=1, bias=use_bias)
        self.conv12 = nn.Conv3d(8, 8, kernel_size=3, padding=1, bias=use_bias)
        self.down1 = nn.Conv3d(8, 16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down2 = nn.Conv3d(16, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=use_bias)
        self.down3 = nn.Conv3d(32, 64, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=use_bias)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv32 = nn.Conv3d(96, 32, kernel_size=3, padding=1, bias=use_bias)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv22 = nn.Conv3d(48, 16, kernel_size=3, padding=1, bias=use_bias)
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv13 = nn.Conv3d(24, 8, kernel_size=3, padding=1, bias=use_bias)
        self.conv14 = nn.Conv3d(8, num_classes, kernel_size=1, padding=0, bias=use_bias)

    def forward(self, x):
        x1 = F.relu(self.conv11(x))
        x1 = F.relu(self.conv12(x1))
        x2 = self.down1(x1)
        x2 = F.relu(self.conv21(x2))
        x3 = self.down2(x2)
        x3 = F.relu(self.conv31(x3))
        x4 = self.down3(x3)
        x4 = F.relu(self.conv41(x4))
        x3 = torch.cat([self.up3(x4), x3], dim=1)
        x3 = F.relu(self.conv32(x3))
        x2 = torch.cat([self.up2(x3), x2], dim=1)
        x2 = F.relu(self.conv22(x2))
        x1 = torch.cat([self.up1(x2), x1], dim=1)
        x1 = F.relu(self.conv13(x1))
        x = self.conv14(x1)

        return x
