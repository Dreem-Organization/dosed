import torch
from dosed.utils import jaccard_overlap


def precision_function():
    """returns a function to calculate precision"""

    def calculate_precision(prediction, reference, min_iou=0.3):
        """takes 2 event scorings
        (in array format [[start1, end1], [start2, end2], ...])
        and outputs the precision.

        Parameters
        ----------
        min_iou : float
            minimum intersection-over-union with a true event to be considered
            a true positive.
        """

        # Compute precision
        iou = jaccard_overlap(torch.Tensor(prediction),
                              torch.Tensor(reference))
        max_iou, _ = iou.max(1)
        true_positive = (max_iou >= min_iou).sum().item()
        false_positive = len(prediction) - true_positive
        precision = true_positive / (true_positive + false_positive)

        return precision

    return calculate_precision


def recall_function():
    """returns a function to calculate recall"""

    def calculate_recall(prediction, reference, min_iou=0.3):
        """takes 2 event scorings
        (in array format [[start1, end1], [start2, end2], ...])
        and outputs the recall.

        Parameters
        ----------
        min_iou : float
            minimum intersection-over-union with a true event to be considered
            a true positive.
        """

        # Compute recall
        iou = jaccard_overlap(torch.Tensor(prediction),
                              torch.Tensor(reference))
        max_iou, _ = iou.max(1)
        true_positive = (max_iou >= min_iou).sum().item()
        false_negative = len(reference) - true_positive
        recall = true_positive / (true_positive + false_negative)

        return recall

    return calculate_recall


def f1_function():
    """returns a function to calculate f1 score"""

    def calculate_f1_score(prediction, reference, min_iou=0.3):
        """takes 2 event scorings
        (in array format [[start1, end1], [start2, end2], ...])
        and outputs the f1 score.

        Parameters
        ----------
        min_iou : float
            minimum intersection-over-union with a true event to be considered
            a true positive.
        """
        # Compute precision, recall, f1_score
        iou = jaccard_overlap(torch.Tensor(prediction),
                              torch.Tensor(reference))
        max_iou, _ = iou.max(1)
        true_positive = (max_iou >= min_iou).sum().item()
        false_positive = len(prediction) - true_positive
        false_negative = len(reference) - true_positive
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        if precision == 0 or recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * precision * recall / (precision + recall)

        return f1_score

    return calculate_f1_score
