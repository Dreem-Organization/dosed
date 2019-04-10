""" This script contains a set of transformations than can be applied to
input data before feeding it to the model"""

import numpy as np

import torch


class RegularizerBase:
    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            return self.call(x)
        else:
            return x

    def call(self, x):
        raise NotImplementedError


class GaussianNoise(RegularizerBase):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
    """

    def __init__(self, sigma=0.01, p=1):
        super(GaussianNoise, self).__init__(p=p)
        self.sigma = sigma
        self.noise = torch.tensor(0)

    def call(self, x):
        if self.sigma != 0:
            scale = self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x


class RescaleNormal(RegularizerBase):
    def __init__(self, p=0.5, std=0.01):
        super(RescaleNormal, self).__init__(p=p)
        self.std = std

    def call(self, x):
        factor = np.random.normal(loc=1, scale=self.std)
        return x * factor


class Invert(RegularizerBase):
    def __init__(self, p=0.5):
        super(Invert, self).__init__(p=p)

    def call(self, x):
        return x * -1
