from .base import TrainerBase
from .base_adam import TrainerBaseAdam

__all__ = [
    "TrainerBase",
    "TrainerBaseAdam",
]

trainers = {
    "basic": TrainerBase,
    "adam": TrainerBaseAdam,
}
