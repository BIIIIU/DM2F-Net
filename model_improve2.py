import torch
import torch.nn.functional as F
from torch import nn

import torchvision.models as models
from resnext import ResNeXt101

from model_improve import Base, Base_OHAZE, BaseA, BaseITS

class ImprovedAttention_wo(nn.Module):
    def __init__(self, dim, dimout):
        super(ImprovedAttention_wo, self).__init__()

        self.norm = nn.BatchNorm2d(dim)
        # Simple Channel Attention
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1),
            nn.GELU(),
            nn.Conv2d(dim // 8, 1, 1),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dimout, 1)
        )
        self.attention_before = nn.Sequential(
            
            nn.Conv2d(dim, dim // 4, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(dim // 4, dim // 4, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(dim // 4, dimout, kernel_size=1)
        )

    def forward(self, x):

        previous = x
        x = self.norm(x)
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp(x)

        x += self.attention_before(previous)
        return x

class DM2FNet_plus_attention(Base):
    def __init__(self, num_features=128, arch='resnext101_32x8d'):
        super(DM2FNet_plus_attention, self).__init__()
        self.num_features = num_features

        # resnext = ResNeXt101()
        #
        # self.layer0 = resnext.layer0
        # self.layer1 = resnext.layer1
        # self.layer2 = resnext.layer2
        # self.layer3 = resnext.layer3
        # self.layer4 = resnext.layer4

        assert arch in ['resnet50', 'resnet101',
                        'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']
        backbone = models.__dict__[arch](pretrained=True)
        del backbone.fc
        self.backbone = backbone

        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )

        self.t = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 1, kernel_size=1), nn.Sigmoid()
        )
        self.a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, 1, kernel_size=1), nn.Sigmoid()
        )

        # self.attention_phy = nn.Sequential(
        #     nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
        #     nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
        #     nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        # )
        self.attention_phy = ImprovedAttention_wo(num_features * 4, num_features * 4)

        # self.attention1 = nn.Sequential(
        #     nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
        #     nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
        #     nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        # )
        # self.attention2 = nn.Sequential(
        #     nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
        #     nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
        #     nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        # )
        # self.attention3 = nn.Sequential(
        #     nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
        #     nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
        #     nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        # )
        # self.attention4 = nn.Sequential(
        #     nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
        #     nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
        #     nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        # )
        self.attention1 = ImprovedAttention_wo(num_features * 4, num_features * 4)
        self.attention2 = ImprovedAttention_wo(num_features * 4, num_features * 4)
        self.attention3 = ImprovedAttention_wo(num_features * 4, num_features * 4)
        self.attention4 = ImprovedAttention_wo(num_features * 4, num_features * 4)

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

        self.j1 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.j2 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.j3 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.j4 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )

        # self.attention_fusion = nn.Sequential(
        #     nn.Conv2d(num_features * 4, num_features, kernel_size=1), nn.SELU(),
        #     nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
        #     nn.Conv2d(num_features // 2, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
        #     nn.Conv2d(num_features // 2, 15, kernel_size=1)
        # )
        self.attention_fusion = ImprovedAttention_wo(num_features * 4, 15)

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0, x0_hd=None):
        x = (x0 - self.mean) / self.std

        backbone = self.backbone

        layer0 = backbone.conv1(x)
        layer0 = backbone.bn1(layer0)
        layer0 = backbone.relu(layer0)
        layer0 = backbone.maxpool(layer0)

        layer1 = backbone.layer1(layer0)
        layer2 = backbone.layer2(layer1)
        layer3 = backbone.layer3(layer2)
        layer4 = backbone.layer4(layer3)

        # layer0 = self.layer0(x)
        # layer1 = self.layer1(layer0)
        # layer2 = self.layer2(layer1)
        # layer3 = self.layer3(layer2)
        # layer4 = self.layer4(layer3)

        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')

        concat = torch.cat((down1, down2, down3, down4), 1)

        n, c, h, w = down1.size()

        attention_phy = self.attention_phy(concat)
        attention_phy = F.softmax(attention_phy.view(n, 4, c, h, w), 1)
        f_phy = down1 * attention_phy[:, 0, :, :, :] + down2 * attention_phy[:, 1, :, :, :] + \
                down3 * attention_phy[:, 2, :, :, :] + down4 * attention_phy[:, 3, :, :, :]
        f_phy = self.refine(f_phy) + f_phy

        attention1 = self.attention1(concat)
        attention1 = F.softmax(attention1.view(n, 4, c, h, w), 1)
        f1 = down1 * attention1[:, 0, :, :, :] + down2 * attention1[:, 1, :, :, :] + \
             down3 * attention1[:, 2, :, :, :] + down4 * attention1[:, 3, :, :, :]
        f1 = self.refine(f1) + f1

        attention2 = self.attention2(concat)
        attention2 = F.softmax(attention2.view(n, 4, c, h, w), 1)
        f2 = down1 * attention2[:, 0, :, :, :] + down2 * attention2[:, 1, :, :, :] + \
             down3 * attention2[:, 2, :, :, :] + down4 * attention2[:, 3, :, :, :]
        f2 = self.refine(f2) + f2

        attention3 = self.attention3(concat)
        attention3 = F.softmax(attention3.view(n, 4, c, h, w), 1)
        f3 = down1 * attention3[:, 0, :, :, :] + down2 * attention3[:, 1, :, :, :] + \
             down3 * attention3[:, 2, :, :, :] + down4 * attention3[:, 3, :, :, :]
        f3 = self.refine(f3) + f3

        attention4 = self.attention4(concat)
        attention4 = F.softmax(attention4.view(n, 4, c, h, w), 1)
        f4 = down1 * attention4[:, 0, :, :, :] + down2 * attention4[:, 1, :, :, :] + \
             down3 * attention4[:, 2, :, :, :] + down4 * attention4[:, 3, :, :, :]
        f4 = self.refine(f4) + f4

        if x0_hd is not None:
            x0 = x0_hd
            x = (x0 - self.mean) / self.std

        log_x0 = torch.log(x0.clamp(min=1e-8))
        log_log_x0_inverse = torch.log(torch.log(1 / x0.clamp(min=1e-8, max=(1 - 1e-8))))

        # J0 = (I - A0 * (1 - T0)) / T0
        a = self.a(f_phy)
        t = F.upsample(self.t(f_phy), size=x0.size()[2:], mode='bilinear')
        x_phy = ((x0 - a * (1 - t)) / t.clamp(min=1e-8)).clamp(min=0., max=1.)

        # J2 = I * exp(R2)
        r1 = F.upsample(self.j1(f1), size=x0.size()[2:], mode='bilinear')
        x_j1 = torch.exp(log_x0 + r1).clamp(min=0., max=1.)

        # J2 = I + R2
        r2 = F.upsample(self.j2(f2), size=x0.size()[2:], mode='bilinear')
        x_j2 = ((x + r2) * self.std + self.mean).clamp(min=0., max=1.)

        #
        r3 = F.upsample(self.j3(f3), size=x0.size()[2:], mode='bilinear')
        x_j3 = torch.exp(-torch.exp(log_log_x0_inverse + r3)).clamp(min=0., max=1.)

        # J4 = log(1 + I * R4)
        r4 = F.upsample(self.j4(f4), size=x0.size()[2:], mode='bilinear')
        # x_j4 = (torch.log(1 + r4 * x0)).clamp(min=0, max=1)
        x_j4 = (torch.log(1 + torch.exp(log_x0 + r4))).clamp(min=0., max=1.)

        attention_fusion = F.upsample(self.attention_fusion(concat), size=x0.size()[2:], mode='bilinear')
        x_f0 = torch.sum(F.softmax(attention_fusion[:, :5, :, :], 1) *
                         torch.stack((x_phy[:, 0, :, :], x_j1[:, 0, :, :], x_j2[:, 0, :, :],
                                      x_j3[:, 0, :, :], x_j4[:, 0, :, :]), 1), 1, True)
        x_f1 = torch.sum(F.softmax(attention_fusion[:, 5: 10, :, :], 1) *
                         torch.stack((x_phy[:, 1, :, :], x_j1[:, 1, :, :], x_j2[:, 1, :, :],
                                      x_j3[:, 1, :, :], x_j4[:, 1, :, :]), 1), 1, True)
        x_f2 = torch.sum(F.softmax(attention_fusion[:, 10:, :, :], 1) *
                         torch.stack((x_phy[:, 2, :, :], x_j1[:, 2, :, :], x_j2[:, 2, :, :],
                                      x_j3[:, 2, :, :], x_j4[:, 2, :, :]), 1), 1, True)
        x_fusion = torch.cat((x_f0, x_f1, x_f2), 1).clamp(min=0., max=1.)

        if self.training:
            return x_fusion, x_phy, x_j1, x_j2, x_j3, x_j4, t, a.view(x.size(0), -1)
        else:
            return x_fusion