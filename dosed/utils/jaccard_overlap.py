import torch


def jaccard_overlap(localizations_a, localizations_b):
    """Jaccard overlap between two segments A ∩ B / (LENGTH_A + LENGTH_B - A ∩ B)

    localizations_a: tensor of localizations
    localizations_a: tensor of localizations
    """
    A = localizations_a.size(0)
    B = localizations_b.size(0)
    # intersection
    max_min = torch.max(localizations_a[:, 0].unsqueeze(1).expand(A, B),
                        localizations_b[:, 0].unsqueeze(0).expand(A, B))
    min_max = torch.min(localizations_a[:, 1].unsqueeze(1).expand(A, B),
                        localizations_b[:, 1].unsqueeze(0).expand(A, B))
    intersection = torch.clamp((min_max - max_min), min=0)
    lentgh_a = (localizations_a[:, 1] - localizations_a[:, 0]).unsqueeze(1).expand(A, B)
    lentgh_b = (localizations_b[:, 1] - localizations_b[:, 0]).unsqueeze(0).expand(A, B)
    overlaps = intersection / (lentgh_a + lentgh_b - intersection)
    return overlaps
