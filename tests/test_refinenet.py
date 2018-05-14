import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_refinenet import RefineNet4Cascade, RefineNet4CascadePoolingImproved


def test_invalid_shape():
    with pytest.raises(ValueError):
        RefineNet4Cascade(input_shape=(3, 225))


@pytest.mark.parametrize(
    "model",
    [RefineNet4Cascade, RefineNet4CascadePoolingImproved])
def test_refinenet_output_valid_shapes(model):
    net = model(input_shape=(3, 32), num_classes=2, pretrained=False)
    x = torch.randn(10, 3, 32, 32)
    target = torch.randn(10, 2, 8, 8)

    output = net(x)
    output_size = output.size()

    assert output_size[0] == 10
    assert output_size[1] == 2
    assert output_size[2] == 8
    assert output_size[3] == 8

    criterion = nn.MSELoss()
    loss = criterion(output, target)
    net.zero_grad()

    loss.backward()


@pytest.mark.parametrize(
    "model",
    [RefineNet4Cascade, RefineNet4CascadePoolingImproved])
def test_refinenet_optimize_no_error_with_paramaters(model):
    net = model(input_shape=(3, 32), num_classes=2, pretrained=False)
    optim.Adam((net.parameters()))
