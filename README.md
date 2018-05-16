# Pytorch Implementation of RefineNet

[![Build Status](https://travis-ci.org/thomasjpfan/pytorch_refinenet.svg?branch=master)](https://travis-ci.org/thomasjpfan/pytorch_refinenet)

This is a Pytorch implementation of the Multipath RefineNet architecture from the [paper](https://arxiv.org/abs/1611.06612).

## Installation

Install PyTorch following instructions on their [website](http://pytorch.org). Then install this package:

```bash
pip install git+https://github.com/thomasjpfan/pytorch_refinenet.git
```

## Implemented versions

- Multi-path 4-Cascaded RefineNet: `RefineNet4Cascade`
- Multi-path 4-Cascaded RefineNet With Improved Pooling: `RefineNet4CascadePoolingImproved`

There are diagrams of these two versions in the author's github [repo](https://github.com/guosheng/refinenet/tree/master/net_graphs). The improved pooling version adds an additional pooling/convolution layer and flips the order of the pooling/convolution layers in the *Chained Residual Pooling* block.

## Usage

This implementation of the Multipath RefineNet has the following initialization:

```python
class RefineNet4Cascade(nn.Module):

    def __init__(self,
                 input_shape,
                 num_classes=1,
                 features=256,
                 resnet_factory=models.resnet101,
                 pretrained=True,
                 freeze_resnet=True):
        ...
```

The `input_shape` is a tuple of`(channels, size)` which denotes the number of channels in the
input image and the input width/height. For an input to flow cleanly through the resnet layers, the input size should be divisible by 32. The input size is assumed to be a square image/patch. For
example the `RefineNet4Cascade` can be defined to intake 3x224x224 images:

```python
import torch
from pytorch_refinenet import RefineNet4Cascade

net = RefineNet4Cascade((3, 224), num_classes=10)
x = torch.randn(1, 3, 224, 224)
y = net(x)
y.size()
# torch.Size([1, 10, 56, 56])
```

The number of channels outputed will equal `num_classes` and the size will be 1/4 the size of the
input as described in the paper. You can upscale the to get back to the original resolution.

### Training

The refinenet backbone is frozen by default, which means they will not be updated with gradients during training.
The `parameters` method in `RefineNet4Cascade` was redefined to only return the parameters that require a gradident. Thus this will work for training:

```python
net = RefineNet4Cascade((3, 224), num_classes=10)
opt = optim.Adam(net.parameters())

x = torch.randn(1, 3, 224, 224)
y = net(x)
...
```
