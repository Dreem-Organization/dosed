import torch
import torch.nn as nn
from dosed.functions.simple_loss import DOSEDSimpleLoss


class DOSEDWorstNegativeMiningLoss(DOSEDSimpleLoss):
    """Loss function inspired from https://github.com/amdegroot/ssd.pytorch"""

    def __init__(self,
                 number_of_classes,
                 device,
                 factor_negative_mining=3,
                 default_negative_mining=10,
                 ):
        super(DOSEDWorstNegativeMiningLoss, self).__init__(
            number_of_classes=number_of_classes,
            device=device)
        self.factor_negative_mining = factor_negative_mining
        self.default_negative_mining = default_negative_mining

    def get_negative_index(self, positive, classifications, classifications_target):
        batch = classifications.shape[0]
        number_of_default_events = classifications.shape[1]
        number_of_positive = positive.long().sum(1)
        number_of_negative = torch.clamp(number_of_positive * self.factor_negative_mining,
                                         min=self.default_negative_mining)
        number_of_negative = torch.min(number_of_negative,
                                       (number_of_default_events - number_of_positive))
        loss_softmax = -torch.log(nn.Softmax(1)(
            classifications.view(-1, self.number_of_classes)).gather(
            1, classifications_target.view(-1, 1))).view(batch, -1)
        loss_softmax[positive] = 0
        _, loss_softmax_descending_index = loss_softmax.sort(1, descending=True)
        _, loss_softmax_descending_rank = loss_softmax_descending_index.sort(1)
        negative = (loss_softmax_descending_rank <
                    number_of_negative.unsqueeze(1).expand_as(loss_softmax_descending_rank))
        return negative
