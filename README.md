# Pytorch Implementation of RefineNet

[![Build Status](https://travis-ci.org/thomasjpfan/pytorch_refinenet.svg?branch=master)](https://travis-ci.org/thomasjpfan/pytorch_refinenet)

This is a Pytorch implementation of the Multipath RefineNet architecture from the paper: []().

## Installation

Install PyTorch following instructions on their [website](http://pytorch.org). Then install this package:

```
pip install git+https://github.com/thomasjpfan/pytorch_refinenet.git
```

## Usage

This implementation of the Multipath RefineNet has the following initialization:

```python
class RefineNet(nn.Module):

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
example the `RefineNet` can be defined to intake 3x224x224 images:

```python
import torch
from torch.autograd import Variable

net = RefineNet((3, 224), num_classes=10)
x_var = Variable(torch.randn(1, 3, 224, 224))
y = net(x_var)
y.size()
# torch.Size([1, 10, 56, 56])
```

The number of channels outputed will equal `num_classes` and the size will be 1/4 the size of the
input as described in the paper. You can upscale the to get back to the original resolution.
