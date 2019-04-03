import torch

from dosed.preprocessings import GaussianNoise, RescaleNormal, Invert
from dosed.utils import Compose


def test_regularization():
    x = torch.rand(32, 25, 25)

    regularizer = Compose(
        [GaussianNoise(), RescaleNormal(), Invert()]
    )
    regularizer(x)
