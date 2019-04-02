"""

Some simple logging functionality.

Logs to a tab-separated-values file (./logs/progress.txt)

"""

import os
import json
import time
import os.path as osp
from .colorize import colorize
import tempfile


class Logger:

    """
    A logger to track training parameters.
    """

    def __init__(self,
                 num_events,
                 output_dir=None,
                 output_fname='train_history.json',
                 metrics=["precision", "recall", "f1"],
                 name_events=["event_type_1", "event_type_2"],
                 ):
        """
        Initialize a Logger.
        """

        assert len(name_events) == num_events
        self.name_events = name_events
        self.metrics = metrics
        self.output_fname = output_fname
        self.output_dir = output_dir if output_dir is not None else tempfile.mkdtemp()
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        self.output_file = osp.join(self.output_dir, output_fname)
        print(colorize("Logging data to %s" % self.output_file, 'green',
                       bold=True))

        self.num_events = num_events
        self.history_time = []
        self.history_loc_loss = {"train": [], "validation": []}
        self.history_class_pos_loss = {"train": [], "validation": []}
        self.history_class_neg_loss = {"train": [], "validation": []}
        self.history_metrics = []
        self.current_epoch_metrics = {
            name_event: {metric: [] for metric in self.metrics}
            for name_event in self.name_events
        }

    def log_msg(self, msg, color='green'):
        """ Print a colorized message to stdout. """
        print(colorize(msg, color, bold=True))

    def add_new_loss(self, loc_loss, class_pos_loss, class_neg_loss,
                     mode="validation"):
        """ Adds loss values of a new epoch. Call one time per epoch """
        self.history_loc_loss[mode].append(loc_loss)
        self.history_class_pos_loss[mode].append(class_pos_loss)
        self.history_class_neg_loss[mode].append(class_neg_loss)

    def add_new_metrics(self, metrics):
        """
        Adds metric values to the current epoch metrics.
        Call as many times per epoch as required.
        """
        assert len(metrics[0]) == self.num_events
        for num_event, event in enumerate(self.name_events):
            for metric in self.metrics:
                self.current_epoch_metrics[event][metric].append(
                    (metrics[0][num_event][metric], metrics[1])
                )

    def add_current_metrics_to_history(self):
        """
        Adds current_epoch_metrics to history and resets the variable.
        Call at the end of each epoch
        """
        self.history_metrics.append(self.current_epoch_metrics)
        self.history_time.append(time.time())
        self.current_epoch_metrics = {
            name_event: {metric: [] for metric in self.metrics}
            for name_event in self.name_events
        }

    def dump_train_history(self):
        """ Dump training history into a .json file """

        if len(self.history_loc_loss["train"]) != len(
            self.history_class_pos_loss["train"]) or len(
                self.history_class_pos_loss["train"]) != len(
                self.history_class_neg_loss["train"]) or len(
                self.history_class_neg_loss["train"]) != len(self.history_metrics):
            print(colorize('Warning: length of loss or metrics not consistent',
                           'red'))

        train_history = {}
        train_history["loc_loss"] = self.history_loc_loss
        train_history["class_pos_loss"] = self.history_class_pos_loss
        train_history["class_neg_loss"] = self.history_class_neg_loss
        train_history["metrics"] = self.history_metrics
        train_history["time_stamps"] = self.history_time
        json.dump(train_history,
                  open(self.output_file, 'w'),
                  indent=4)
