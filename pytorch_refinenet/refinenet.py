import torch.nn as nn
import torchvision.models as models

from .blocks import (
    RefineNetBlock, RefineNetBottomBlock,
    ResidualConvUnit
)


class RefineNet(nn.Module):

    def __init__(self, input_shape,
                 num_classes=1,
                 resnet_factory=models.resnet101,
                 pretrained=True):
        super().__init__()

        input_channel, input_size = input_shape
        if input_size % 32 != 0:
            raise ValueError(f"{input_shape} not divisble by 32")

        resnet = resnet_factory(pretrained=pretrained)

        self.layer1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )

        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.layer1_rn = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1, bias=False)

        self.refinenet4 = RefineNetBottomBlock(2048, features=512)
        self.refinenet3 = RefineNetBlock(
            256, (512, input_size // 32), (1024, input_size // 16))
        self.refinenet2 = RefineNetBlock(
            256, (256, input_size // 16), (512, input_size // 8))
        self.refinenet1 = RefineNetBlock(
            256, (256, input_size // 8), (256, input_size // 4))

        self.output_conv = nn.Sequential(
            ResidualConvUnit(256),
            ResidualConvUnit(256),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):

        layer_1 = self.layer1(x)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        layer_1_rn = self.layer1_rn(layer_1)
        layer_2_rn = self.layer2_rn(layer_2)
        layer_3_rn = self.layer3_rn(layer_3)

        path_4 = self.refinenet4(layer_4)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)
        out = self.output_conv(path_1)
        return out
