import numpy as np

import torch

from .simple_loss import DOSEDSimpleLoss


class DOSEDRandomNegativeMiningLoss(DOSEDSimpleLoss):
    """Loss function inspired from https://github.com/amdegroot/ssd.pytorch"""

    def __init__(self,
                 number_of_classes,
                 device,
                 factor_negative_mining=3,
                 default_negative_mining=10,
                 ):
        super(DOSEDRandomNegativeMiningLoss, self).__init__(
            number_of_classes=number_of_classes,
            device=device)
        self.factor_negative_mining = factor_negative_mining
        self.default_negative_mining = default_negative_mining

    def get_negative_index(self, positive, classifications,
                           classifications_target):
        number_of_default_events = classifications.shape[1]
        number_of_positive = positive.long().sum(1)
        number_of_negative = torch.clamp(
            number_of_positive * self.factor_negative_mining,
            min=self.default_negative_mining)
        number_of_negative = torch.min(
            number_of_negative, (number_of_default_events - number_of_positive))

        def pick_zero_random_index(tensor, size):
            result = torch.zeros_like(tensor)
            for index in np.random.choice(
                    (1 - tensor).nonzero().view(-1), size=size, replace=False):
                result[index] = 1
            return result

        random_negative_index = [
            pick_zero_random_index(line, int(number_of_negative[i]))
            for i, line in enumerate(torch.unbind(positive, dim=0))]
        negative = torch.stack(random_negative_index, dim=0)

        return negative
