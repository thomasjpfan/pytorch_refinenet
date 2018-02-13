import pytest
import torch
import torch.nn as nn
from torch.autograd import Variable

from pytorch_refinenet import RefineNet4Cascade, RefineNet4CascadePoolingImproved


def test_invalid_shape():
    with pytest.raises(ValueError):
        RefineNet4Cascade(input_shape=(3, 225))


def test_refinenet_4cascade():
    net = RefineNet4Cascade(input_shape=(3, 32), num_classes=2, pretrained=False)
    x = torch.randn(10, 3, 32, 32)
    x_var = Variable(x)
    target = Variable(torch.randn(10, 2, 8, 8))

    output = net(x_var)
    output_size = output.size()

    assert output_size[0] == 10
    assert output_size[1] == 2
    assert output_size[2] == 8
    assert output_size[3] == 8

    criterion = nn.MSELoss()
    loss = criterion(output, target)
    net.zero_grad()

    loss.backward()


def test_refinenet_4cascade_pooling_improved():
    net = RefineNet4CascadePoolingImproved(input_shape=(3, 32), num_classes=2, pretrained=False)
    x = torch.randn(10, 3, 32, 32)
    x_var = Variable(x)
    target = Variable(torch.randn(10, 2, 8, 8))

    output = net(x_var)
    output_size = output.size()

    assert output_size[0] == 10
    assert output_size[1] == 2
    assert output_size[2] == 8
    assert output_size[3] == 8

    criterion = nn.MSELoss()
    loss = criterion(output, target)
    net.zero_grad()

    loss.backward()
