import tarfile
import tempfile
import json
import shutil
import numpy as np
import torch
import torch.nn as nn

from ..utils import binary_to_array


class BaseNet(nn.Module):

    def __init__(self):
        super(BaseNet, self).__init__()

    @property
    def device(self):
        try:
            out = next(self.parameters()).device
            return (out if isinstance(out, torch.device)
                    else torch.device('cpu'))
        except Exception:
            return torch.device('cpu')

    def predict(self, x):
        localizations, classifications, localizations_default = self.forward(x)
        localizations_default = localizations_default.to(self.device)
        return self.detector(localizations, classifications, localizations_default)

    def save(self, filename, net_parameters):
        with tarfile.open(filename, "w") as tar:
            temporary_directory = tempfile.mkdtemp()
            name = "{}/net_params.json".format(temporary_directory)
            json.dump(net_parameters, open(name, "w"))
            tar.add(name, arcname="net_params.json")
            name = "{}/state.torch".format(temporary_directory)
            torch.save(self.state_dict(), name)
            tar.add(name, arcname="state.torch")
            shutil.rmtree(temporary_directory)
        return filename

    @classmethod
    def load(cls, filename, use_device=torch.device('cpu')):
        with tarfile.open(filename, "r") as tar:
            net_parameters = json.loads(
                tar.extractfile("net_params.json").read().decode("utf-8"))
            path = tempfile.mkdtemp()
            tar.extract("state.torch", path=path)
            net = cls(**net_parameters)
            net.load_state_dict(
                torch.load(
                    path + "/state.torch",
                    map_location=use_device,
                )
            )
        return net, net_parameters

    def predict_dataset(self,
                        inference_dataset,
                        threshold,
                        overlap_factor=0.5,
                        batch_size=128,
                        ):
        """
        Predicts events in inference_dataset.
        """

        # Set network to eval mode
        self.eval()

        # Set network prediction parameters
        self.detector.classification_threshold = threshold
        window = inference_dataset.window
        overlap = window * overlap_factor

        # List of dicts, to save predictions of each class per record
        predictions = {}
        for record in inference_dataset.records:
            predictions[record] = []

            signal_duration = inference_dataset.signals[record]["duration"]
            result = np.zeros((self.number_of_classes - 1, int(signal_duration)))

            for signals, times in inference_dataset.get_record_batch(
                    record,
                    batch_size=int(batch_size),
                    stride=overlap):
                x = {signal_type: signal.to(self.device) for signal_type, signal in signals.items()}
                batch_predictions = self.predict(x)
                for events, time in zip(batch_predictions, times):
                    for event in events:
                        start = int(round(event[0] * window + time[0]))
                        stop = int(round(event[1] * window + time[0]))
                        result[event[2], start:stop] = 1
            predicted_events = [binary_to_array(k) for k in result]
            assert len(predicted_events) == self.number_of_classes - 1
            for event_num in range(self.number_of_classes - 1):
                predictions[record].append(predicted_events[event_num])

        return predictions

    @property
    def nelement(self):
        cpt = 0
        for p in self.parameters():
            cpt += p.nelement()
        return cpt


def get_overlerapping_default_events(window_size, default_event_sizes, factor_overlap=2):
    window_size = window_size
    default_event_sizes = default_event_sizes
    factor_overlap = factor_overlap
    default_events = []
    for default_event_size in default_event_sizes:
        overlap = default_event_size / factor_overlap
        number_of_default_events = int(window_size / overlap)
        default_events.extend(
            [(overlap * (0.5 + i) / window_size, default_event_size / window_size)
             for i in range(number_of_default_events)]
        )
    return torch.Tensor(default_events)
