import pytest
import torch
from torch.autograd import Variable

from pytorch_refinenet import RefineNet4Cascade


def test_invalid_shape():
    with pytest.raises(ValueError):
        RefineNet4Cascade(input_shape=(3, 225))


def test_refinenet():
    net = RefineNet4Cascade(input_shape=(3, 32), num_classes=2, pretrained=False)
    x = torch.randn(10, 3, 32, 32)
    x_var = Variable(x)

    output = net(x_var)
    output_size = output.size()

    assert output_size[0] == 10
    assert output_size[1] == 2
    assert output_size[2] == 8
    assert output_size[3] == 8
