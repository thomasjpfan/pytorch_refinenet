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

    def __init__(self, out_feats, *shapes):
        super().__init__()

        _, max_size = max(shapes, key=lambda x: x[1])

        for i, shape in enumerate(shapes):
            feat, size = shape
            if max_size % size != 0:
                raise ValueError(f"max_size not divisble by shape {i}")

            scale_factor = max_size // size
            if scale_factor != 1:
                self.add_module(f"resolve{i}", nn.Sequential(
                    nn.Conv2d(feat, out_feats, kernel_size=3,
                              stride=1, padding=1, bias=False),
                    nn.Upsample(scale_factor=scale_factor, mode='bilinear')
                ))
            else:
                self.add_module(
                    f"resolve{i}",
                    nn.Conv2d(feat, out_feats, kernel_size=3,
                              stride=1, padding=1, bias=False)
                )

    def forward(self, *xs):

        output = self.resolve0(xs[0])

        for i, x in enumerate(xs[1:], 1):
            output += self.__getattr__(f"resolve{i}")(x)

        return output


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

    def __init__(self, features, path_shape, layer_shape):
        super().__init__()

        path_feats, path_size = path_shape
        layer_feats, layer_size = layer_shape

        self.rcu_path = nn.Sequential(
            ResidualConvUnit(path_feats),
            ResidualConvUnit(path_feats)
        )
        self.rcu_layer = nn.Sequential(
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
            nn.Conv2d(layer_feats, features, kernel_size=3,
                      padding=1, stride=1, bias=False),
            ResidualConvUnit(features),
            ResidualConvUnit(features)
        )
        self.crp = ChainedResidualPool(features)
        self.output_conv = ResidualConvUnit(features)

    def forward(self, x):
        x = self.rcu(x)
        x = self.crp(x)
        return self.output_conv(x)
