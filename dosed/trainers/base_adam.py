""" Trainer class with Adam optimizer """

from torch import device
import torch.optim as optim
from .base import TrainerBase


class TrainerBaseAdam(TrainerBase):
    """ Trainer class with Adam optimizer """

    def __init__(
            self,
            net,
            optimizer_parameters={
                "lr": 0.001,
                "weight_decay": 1e-8,
            },
            loss_specs={
                "type": "focal",
                "parameters": {
                    "number_of_classes": 1,
                    "alpha": 0.25,
                    "gamma": 2,
                    "device": device("cuda"),
                }
            },
            metrics=["precision", "recall", "f1"],
            epochs=100,
            metric_to_maximize="f1",
            patience=None,
            save_folder=None,
            logger_parameters={
                "num_events": 1,
                "output_dir": None,
                "output_fname": 'train_history.json',
                "metrics": ["precision", "recall", "f1"],
                "name_events": ["event_type_1"]
            },
            threshold_space={
                "upper_bound": 0.85,
                "lower_bound": 0.55,
                "num_samples": 10,
                "zoom_in": False,
            },
            matching_overlap=0.5,
    ):
        super(TrainerBaseAdam, self).__init__(
            net=net,
            optimizer_parameters=optimizer_parameters,
            loss_specs=loss_specs,
            metrics=metrics,
            epochs=epochs,
            metric_to_maximize=metric_to_maximize,
            patience=patience,
            save_folder=save_folder,
            logger_parameters=logger_parameters,
            threshold_space=threshold_space,
            matching_overlap=matching_overlap,
        )
        self.optimizer = optim.Adam(net.parameters(), **optimizer_parameters)
