""" Trainer class basic with SGD optimizer """

import copy
import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dosed.datasets import collate
from dosed.functions import (loss_functions, available_score_functions, compute_metrics_dataset)
from dosed.utils import (match_events_localization_to_default_localizations, Logger)


class TrainerBase:
    """Trainer class basic """

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
                    "device": torch.device("cuda"),
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
                "num_samples": 5,
                "zoom_in": False,
            },
            matching_overlap=0.5,
    ):

        self.net = net
        print("Device: ", net.device)
        self.loss_function = loss_functions[loss_specs["type"]](
            **loss_specs["parameters"])
        self.optimizer = optim.SGD(net.parameters(), **optimizer_parameters)
        self.metrics = {
            score: score_function for score, score_function in
            available_score_functions.items()
            if score in metrics + [metric_to_maximize]
        }
        self.epochs = epochs
        self.threshold_space = threshold_space
        self.metric_to_maximize = metric_to_maximize
        self.patience = patience if patience else epochs
        self.save_folder = save_folder
        self.matching_overlap = matching_overlap
        self.matching = match_events_localization_to_default_localizations
        if logger_parameters is not None:
            self.train_logger = Logger(**logger_parameters)

    def on_batch_start(self):
        pass

    def on_epoch_end(self):
        pass

    def validate(self, validation_dataset, threshold_space):
        """
        Compute metrics on validation_dataset net for test_dataset and
        select best classification threshold
        """

        best_thresh = -1
        best_metrics_epoch = {
            metric: -1
            for metric in self.metrics.keys()
        }

        # Compute predicted_events
        thresholds = np.sort(
            np.random.uniform(threshold_space["upper_bound"],
                              threshold_space["lower_bound"],
                              threshold_space["num_samples"]))

        for threshold in thresholds:
            metrics_thresh = compute_metrics_dataset(
                self.net,
                validation_dataset,
                threshold,
            )

            # If 0 events predicted, all superiors thresh's will also predict 0
            if metrics_thresh == -1:
                if best_thresh in (self.threshold_space["upper_bound"],
                                   self.threshold_space["lower_bound"]):
                    print(
                        "Best classification threshold is " +
                        "in the boundary ({})! ".format(best_thresh) +
                        "Consider extending threshold range")
                return best_metrics_epoch, best_thresh

            # Add to logger
            if "train_logger" in vars(self):
                self.train_logger.add_new_metrics((metrics_thresh, threshold))

            # Compute mean metric to maximize across events
            mean_metric_to_maximize = np.nanmean(
                [m[self.metric_to_maximize] for m in metrics_thresh])

            if mean_metric_to_maximize >= best_metrics_epoch[
                    self.metric_to_maximize]:
                best_metrics_epoch = {
                    metric: np.nanmean(
                        [m[self.metric_to_maximize] for m in metrics_thresh])
                    for metric in self.metrics.keys()
                }

                best_thresh = threshold

        if best_thresh in (threshold_space["upper_bound"],
                           threshold_space["lower_bound"]):
            print("Best classification threshold is " +
                  "in the boundary ({})! ".format(best_thresh) +
                  "Consider extending threshold range")

        return best_metrics_epoch, best_thresh

    def get_batch_loss(self, data):
        """ Single forward and backward pass """

        # Get signals and labels
        signals, events = data
        x = signals.to(self.net.device)

        # Forward
        localizations, classifications, localizations_default = self.net.forward(x)

        # Matching
        localizations_target, classifications_target = self.matching(
            localizations_default=localizations_default,
            events=events,
            threshold_overlap=self.matching_overlap)
        localizations_target = localizations_target.to(self.net.device)
        classifications_target = classifications_target.to(self.net.device)

        # Loss
        (loss_classification_positive,
         loss_classification_negative,
         loss_localization) = (
             self.loss_function(localizations,
                                classifications,
                                localizations_target,
                                classifications_target))

        return loss_classification_positive, \
            loss_classification_negative, \
            loss_localization

    def train(self, train_dataset, validation_dataset, batch_size=128):
        """ Metwork training with backprop """

        dataloader_parameters = {
            "num_workers": 0,
            "shuffle": True,
            "collate_fn": collate,
            "pin_memory": True,
            "batch_size": batch_size,
        }
        dataloader_train = DataLoader(train_dataset, **dataloader_parameters)
        dataloader_val = DataLoader(validation_dataset, **dataloader_parameters)

        metrics_final = {
            metric: 0
            for metric in self.metrics.keys()
        }

        best_value = -np.inf
        best_threshold = None
        best_net = None
        counter_patience = 0
        last_update = None
        t = tqdm.tqdm(range(self.epochs,))
        for epoch, _ in enumerate(t):
            if epoch != 0:
                t.set_postfix(
                    best_metric_score=best_value,
                    threshold=best_threshold,
                    last_update=last_update,
                )

            epoch_loss_classification_positive_train = 0.0
            epoch_loss_classification_negative_train = 0.0
            epoch_loss_localization_train = 0.0

            epoch_loss_classification_positive_val = 0.0
            epoch_loss_classification_negative_val = 0.0
            epoch_loss_localization_val = 0.0

            for i, data in enumerate(dataloader_train, 0):

                # On batch start
                self.on_batch_start()

                self.optimizer.zero_grad()

                # Set network to train mode
                self.net.train()

                (loss_classification_positive,
                 loss_classification_negative,
                 loss_localization) = self.get_batch_loss(data)

                epoch_loss_classification_positive_train += \
                    loss_classification_positive
                epoch_loss_classification_negative_train += \
                    loss_classification_negative
                epoch_loss_localization_train += loss_localization

                loss = loss_classification_positive \
                    + loss_classification_negative \
                    + loss_localization
                loss.backward()

                # gradient descent
                self.optimizer.step()

            epoch_loss_classification_positive_train /= (i + 1)
            epoch_loss_classification_negative_train /= (i + 1)
            epoch_loss_localization_train /= (i + 1)

            for i, data in enumerate(dataloader_val, 0):

                (loss_classification_positive,
                 loss_classification_negative,
                 loss_localization) = self.get_batch_loss(data)

                epoch_loss_classification_positive_val += \
                    loss_classification_positive
                epoch_loss_classification_negative_val += \
                    loss_classification_negative
                epoch_loss_localization_val += loss_localization

            epoch_loss_classification_positive_val /= (i + 1)
            epoch_loss_classification_negative_val /= (i + 1)
            epoch_loss_localization_val /= (i + 1)

            metrics_epoch, threshold = self.validate(
                validation_dataset=validation_dataset,
                threshold_space=self.threshold_space,
            )

            if self.threshold_space["zoom_in"] and threshold != -1:
                threshold_space_size = self.threshold_space["upper_bound"] - \
                    self.threshold_space["lower_bound"]
                zoom_metrics_epoch, zoom_threshold = self.validate(
                    validation_dataset=validation_dataset,
                    threshold_space={
                        "upper_bound": threshold + 0.1 * threshold_space_size,
                        "lower_bound": threshold - 0.1 * threshold_space_size,
                        "num_samples": self.threshold_space["num_samples"],
                    })
                if zoom_metrics_epoch[self.metric_to_maximize] > metrics_epoch[
                        self.metric_to_maximize]:
                    metrics_epoch = zoom_metrics_epoch
                    threshold = zoom_threshold

            if self.save_folder:
                self.net.save(self.save_folder + str(epoch) + "_net")

            if metrics_epoch[self.metric_to_maximize] > best_value:
                best_value = metrics_epoch[self.metric_to_maximize]
                best_threshold = threshold
                last_update = epoch
                best_net = copy.deepcopy(self.net)
                metrics_final = {
                    metric: metrics_epoch[metric]
                    for metric in self.metrics.keys()
                }
                counter_patience = 0
            else:
                counter_patience += 1

            if counter_patience > self.patience:
                break

            self.on_epoch_end()
            if "train_logger" in vars(self):
                self.train_logger.add_new_loss(
                    epoch_loss_localization_train.item(),
                    epoch_loss_classification_positive_train.item(),
                    epoch_loss_classification_negative_train.item(),
                    mode="train"
                )
                self.train_logger.add_new_loss(
                    epoch_loss_localization_val.item(),
                    epoch_loss_classification_positive_val.item(),
                    epoch_loss_classification_negative_val.item(),
                    mode="validation"
                )
                self.train_logger.add_current_metrics_to_history()
                self.train_logger.dump_train_history()

        return best_net, metrics_final, best_threshold
