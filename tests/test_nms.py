import torch

from dosed.utils import non_maximum_suppression


def test_non_maximum_suppression():
    localizations_scores = torch.FloatTensor(
        [
            [20, 50, 0.99],  # 0
            [10, 50, 0.97],  # 1
            [25, 42, 0.6],   # 2
            [30, 60, 0.98],  # 3

            [75, 85, 0.92],  # 4
            [72, 87, 0.90],  # 5
            [76, 85, 0.78],  # 6
            [80, 90, 0.91],  # 7
        ]
    )
    localizations = localizations_scores[:, :2] / 100
    scores = localizations_scores[:, -1]
    overlap = 0.4
    kept = [tuple([int(x * 100) for x in y])
            for y in non_maximum_suppression(localizations, scores, overlap)]
    to_keep = [(20, 50), (75, 85), (80, 90)]
    assert set(kept) == set(to_keep)
