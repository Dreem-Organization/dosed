"""inspired from https://github.com/amdegroot/ssd.pytorch"""
import torch.nn as nn

from ..utils import non_maximum_suppression, decode


class Detection(nn.Module):
    """"""

    def __init__(self,
                 number_of_classes,
                 overlap_non_maximum_suppression,
                 classification_threshold,
                 ):
        super(Detection, self).__init__()
        self.number_of_classes = number_of_classes
        self.overlap_non_maximum_suppression = overlap_non_maximum_suppression
        self.classification_threshold = classification_threshold

    def forward(self, localizations, classifications, localizations_default):
        batch = localizations.size(0)
        scores = nn.Softmax(dim=2)(classifications)
        results = []
        for i in range(batch):
            result = []
            localization_decoded = decode(localizations[i], localizations_default)
            for class_index in range(1, self.number_of_classes):  # we remove class 0
                scores_batch_class = scores[i, :, class_index]
                scores_batch_class_selected = scores_batch_class[
                    scores_batch_class > self.classification_threshold]
                if len(scores_batch_class_selected) == 0:
                    continue
                localizations_decoded_selected = localization_decoded[
                    (scores_batch_class > self.classification_threshold)
                    .unsqueeze(1).expand_as(localization_decoded)].view(-1, 2)

                events = non_maximum_suppression(
                    localizations_decoded_selected,
                    scores_batch_class_selected,
                    overlap=self.overlap_non_maximum_suppression,
                )
                result.extend([(event[0].item(), event[1].item(), class_index - 1)
                               for event in events])
            results.append(result)
        return results
