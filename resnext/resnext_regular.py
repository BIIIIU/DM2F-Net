import torch
from torch import nn

from . import resnext_101_32x4d_
from .config import resnext_101_32_path
import torchvision.models as models


class ResNeXt101(nn.Module):
    def __init__(self, pretrained=True, mode='ori'):
        super(ResNeXt101, self).__init__()
        # mode_dict = {'ori': resnext_101_32x4d_, 'syn': resnext_101_32x4d_syn}
        mode_dict = {'ori': resnext_101_32x4d_}
        assert mode in mode_dict

        # print(mode)
        net = mode_dict[mode].resnext_101_32x4d

        if pretrained:
            net = models.__dict__['resnext101_32x8d'](pretrained=True)
            # net.load_state_dict(torch.load(resnext_101_32_path))
        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])
        self.layer1 = nn.Sequential(*net[3: 5])
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4
