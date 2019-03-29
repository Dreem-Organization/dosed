from .simple_loss import DOSEDSimpleLoss
from .worst_negative_mining_loss import DOSEDWorstNegativeMiningLoss
from .random_negative_mining_loss import DOSEDRandomNegativeMiningLoss
from .focal_loss import DOSEDFocalLoss
from .detection import Detection
from .metrics import precision_function, recall_function, f1_function
from .compute_metrics_dataset import compute_metrics_dataset

loss_functions = {
    "simple": DOSEDSimpleLoss,
    "worst_negative_mining": DOSEDWorstNegativeMiningLoss,
    "focal": DOSEDFocalLoss,
    "random_negative_mining": DOSEDRandomNegativeMiningLoss,

}

available_score_functions = {
    "precision": precision_function(),
    "recall": recall_function(),
    "f1": f1_function(),
}

__all__ = [
    "loss_functions"
    "Detection",
    "available_score_functions",
    "compute_metrics_dataset",
]
