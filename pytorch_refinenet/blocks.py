import torch.nn as nn


class ResidualConvUnit(nn.Module):

    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class MultiResolutionFusion(nn.Module):

    def __init__(self, out_features, in_shape_1, in_shape_2):
        super().__init__()

        in_feat_1, in_size_1 = in_shape_1
        in_feat_2, in_size_2 = in_shape_2

        if 2 * in_size_1 != in_size_2:
            raise ValueError("2 * in_feat_1 must equal in_size_2")

        self.conv1 = nn.Conv2d(
            in_feat_1, out_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(
            in_feat_2, out_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.upsample(x1)
        x2 = self.conv2(x2)

        return x1 + x2


class ChainedResidualPool(nn.Module):

    def __init__(self, feats):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.block1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.block3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        x = self.relu(x)

        out = self.block1(x)
        x += out
        out = self.block2(out)
        x += out
        out = self.block3(out)
        x += out

        return x


class RefineNetBlock(nn.Module):

    def __init__(self, path_shape, layer_shape, features=256):
        super().__init__()

        path_feats, path_size = path_shape
        layer_feats, layer_size = layer_shape

        self.rcu_path = nn.Sequential(
            ResidualConvUnit(path_feats),
            ResidualConvUnit(path_feats)
        )
        self.rcu_layer = nn.Sequential(
            nn.Conv2d(layer_feats, features, kernel_size=3, padding=1, stride=1, bias=False),
            ResidualConvUnit(features),
            ResidualConvUnit(features)
        )

        self.mrf = MultiResolutionFusion(features, path_shape, (features, layer_size))
        self.crp = ChainedResidualPool(features)
        self.output_conv = ResidualConvUnit(features)

    def forward(self, x_path, x_layer):
        x_layer = self.rcu_layer(x_layer)
        x_path = self.rcu_path(x_path)

        out = self.mrf(x_path, x_layer)
        out = self.crp(out)
        return self.output_conv(out)


class RefineNetBottomBlock(nn.Module):

    def __init__(self, layer_feats, features=512):
        super().__init__()

        self.rcu = nn.Sequential(
            nn.Conv2d(layer_feats, features, kernel_size=3, padding=1, stride=1, bias=False),
            ResidualConvUnit(features),
            ResidualConvUnit(features)
        )
        self.crp = ChainedResidualPool(features)
        self.output_conv = ResidualConvUnit(features)

    def forward(self, x):
        x = self.rcu(x)
        x = self.crp(x)
        return self.output_conv(x)
