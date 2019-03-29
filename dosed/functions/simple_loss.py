import torch.nn as nn
import torch.nn.functional as F


class DOSEDSimpleLoss(nn.Module):
    """Loss function inspired from https://github.com/amdegroot/ssd.pytorch"""

    def __init__(self,
                 number_of_classes,
                 device,
                 ):
        super(DOSEDSimpleLoss, self).__init__()
        self.device = device
        self.number_of_classes = number_of_classes + 1  # eventlessness

    def localization_loss(self, positive, localizations, localizations_target):
        # Localization Loss (Smooth L1)
        positive_expanded = positive.unsqueeze(positive.dim()).expand_as(
            localizations)
        loss_localization = F.smooth_l1_loss(
            localizations[positive_expanded].view(-1, 2),
            localizations_target[positive_expanded].view(-1, 2),
            size_average=False)
        return loss_localization

    def get_negative_index(self, positive, classifications,
                           classifications_target):
        negative = (classifications_target == 0)
        return negative

    def get_classification_loss(self, index, classifications,
                                classifications_target):
        index_expanded = index.unsqueeze(2).expand_as(classifications)

        loss_classification = F.cross_entropy(
            classifications[index_expanded.gt(0)
                            ].view(-1, self.number_of_classes),
            classifications_target[index.gt(0)],
            size_average=False
        )
        return loss_classification

    def forward(self, localizations, classifications, localizations_target,
                classifications_target):

        positive = classifications_target > 0
        negative = self.get_negative_index(positive, classifications,
                                           classifications_target)

        number_of_positive_all = positive.long().sum().float()
        number_of_negative_all = negative.long().sum().float()

        # loc loss
        loss_localization = self.localization_loss(positive, localizations,
                                                   localizations_target)

        # + Classification loss
        loss_classification_positive = 0
        if number_of_positive_all > 0:
            loss_classification_positive = self.get_classification_loss(
                positive, classifications, classifications_target)

        # - Classification loss
        loss_classification_negative = 0
        if number_of_negative_all > 0:
            loss_classification_negative = self.get_classification_loss(
                negative, classifications, classifications_target)

        # Loss: sum
        loss_classification_positive_normalized = (
            loss_classification_positive /
            number_of_positive_all)
        loss_classification_negative_normalized = (
            loss_classification_negative /
            number_of_negative_all)
        loss_localization_normalized = loss_localization / number_of_positive_all

        return (loss_classification_positive_normalized,
                loss_classification_negative_normalized,
                loss_localization_normalized)
