import torch
from dosed.utils import jaccard_overlap, encode


def match_events_localization_to_default_localizations(localizations_default, events, threshold_overlap):
    batch = len(events)

    # Find localizations_target and classifications_target by matching
    # ground truth localizations to default localizations
    number_of_default_events = localizations_default.size(0)
    localizations_target = torch.Tensor(batch, number_of_default_events, 2)
    classifications_target = torch.LongTensor(batch, number_of_default_events)

    for batch_index in range(batch):

        # If no event add default value to predict (will never be used anyway)
        # And class 0 == backgroung
        if events[batch_index].numel() == 0:
            localizations_target[batch_index][:, :] = torch.FloatTensor(
                [[-1, 1]]).expand_as(localizations_default)
            classifications_target[batch_index] = torch.zeros(localizations_default.size(0))
            continue

        # Else match to most overlapping event and set to background depending on threshold
        localizations_truth = events[batch_index][:, :2]
        classifications_truth = events[batch_index][:, -1]
        localizations_a = localizations_truth
        localizations_b = torch.cat(
            [(localizations_default[:, 0] - localizations_default[:, 1] / 2).unsqueeze(1),
             (localizations_default[:, 0] + localizations_default[:, 1] / 2).unsqueeze(1)],
            1
        )
        overlaps = jaccard_overlap(localizations_a, localizations_b)

        # (Bipartite Matching) https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py
        # might be usefull if an event is included in another
        _, best_prior_index = overlaps.max(1, keepdim=True)
        best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
        best_truth_idx.squeeze_(0)
        best_truth_overlap.squeeze_(0)
        best_prior_index.squeeze_(1)
        # ensure every gt matches with its prior of max overlap
        best_truth_overlap.index_fill_(dim=0, index=best_prior_index, value=2)
        for j in range(best_prior_index.size(0)):
            best_truth_idx[best_prior_index[j]] = j

        localization_match = localizations_truth[best_truth_idx]
        localization_target = encode(localization_match, localizations_default)
        classification_target = classifications_truth[best_truth_idx] + 1  # Add class 0!
        classification_target[best_truth_overlap < threshold_overlap] = 0

        localizations_target[batch_index][:, :] = localization_target
        classifications_target[batch_index] = classification_target.long()

    return localizations_target, classifications_target
