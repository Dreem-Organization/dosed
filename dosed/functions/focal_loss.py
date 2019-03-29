import torch
import torch.nn.functional as F
from dosed.functions.simple_loss import DOSEDSimpleLoss


class DOSEDFocalLoss(DOSEDSimpleLoss):
    """Loss function inspired from https://github.com/amdegroot/ssd.pytorch"""

    def __init__(self,
                 number_of_classes,
                 device,
                 alpha=0.25,
                 gamma=2,
                 ):
        super(DOSEDFocalLoss, self).__init__(
            number_of_classes=number_of_classes,
            device=device)
        self.device = device
        self.number_of_classes = number_of_classes + 1  # eventlessness
        self.alpha = alpha
        self.gamma = gamma

    def get_classification_loss(self, index, classifications,
                                classifications_target):
        index_expanded = index.unsqueeze(2).expand_as(classifications)

        cross_entropy = F.cross_entropy(
            classifications[index_expanded.gt(0)
                            ].view(-1, self.number_of_classes),
            classifications_target[index.gt(0)],
            size_average=False,
            reduce=False
        )
        pt = torch.exp(-cross_entropy)
        loss_classification = (
            self.alpha * ((1 - pt) ** self.gamma) * cross_entropy).sum()
        return loss_classification
