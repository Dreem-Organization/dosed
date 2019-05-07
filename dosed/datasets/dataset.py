"""Dataset Class for DOSED training"""

import os
import h5py
import numpy as np
from numpy.lib.stride_tricks import as_strided
from matplotlib import gridspec
from joblib import Memory, Parallel, delayed

import torch
from torch.utils.data import Dataset

from ..utils import get_h5_data, get_h5_events


class EventDataset(Dataset):

    """Extract data and events from h5 files and provide efficient way to retrieve windows with
    their corresponding events.

    args
    ====

    h5_directory:
        Location of the generic h5 files.
    signals:
        The signals from the h5 we want to include together with their normalization
    events:
        The events from the h5 we want to train on
    window:
        Window size in seconds
    downsampling_rate:
        Downsampling rate to apply to signals
    records:
        Use to select subset of records from h5_directory, default is None and uses all available recordings
    n_jobs:
        Number of process used to extract and normalize signals from h5 files.
    cache_data:
        Cache results of extraction and normalization of signals from h5_file in h5_directory + "/.cache"
        We strongly recommend to keep the default value True to avoid memory overhead.
    minimum_overlap:
        For an event on the edge to be considered included in a window
    ratio_positive:
        Sample within a training batch will have a probability of "ratio_positive" to contain at least one spindle

    """

    def __init__(self,
                 h5_directory,
                 signals,
                 window,
                 events=None,
                 downsampling_rate=1,
                 records=None,
                 n_jobs=1,
                 cache_data=True,
                 minimum_overlap=0.5,
                 transformations=None
                 ):

        if events:
            self.number_of_classes = len(events)
        self.transformations = transformations

        # window parameters
        self.window = window

        # records (all of them by default)
        if records is not None:
            for record in records:
                assert record in os.listdir(h5_directory)
            self.records = records
        else:
            self.records = [x for x in os.listdir(h5_directory) if x != ".cache"]

        ###########################
        # Checks on H5
        # Check sampling frequencies
        fs = set(
            [h5py.File("{}/{}".format(h5_directory, record))[signal["h5_path"]].attrs["fs"]
             for record in self.records for signal in signals]
        )
        assert len(fs) == 1
        self.fs = fs.pop() / downsampling_rate

        # check event names
        if events:
            assert len(set([event["name"] for event in events])) == len(events)

        # ### joblib cache
        get_data = get_h5_data
        get_events = get_h5_events
        if cache_data:
            memory = Memory(h5_directory + "/.cache/", mmap_mode="r", verbose=0)
            get_data = memory.cache(get_h5_data)
            get_events = memory.cache(get_h5_events)

        self.window_size = int(self.window * self.fs)
        self.number_of_channels = len(signals)
        # used in network architecture
        self.input_shape = (self.number_of_channels, self.window_size)
        self.minimum_overlap = minimum_overlap  # for events on the edge of window_size

        # Open signals and events
        self.signals = {}
        self.events = {}
        self.index_to_record = []
        self.index_to_record_event = []  # link index to record

        # Preprocess signals from records
        data = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(get_data)(
            filename="{}/{}".format(h5_directory, record),
            signals=signals,
            downsampling_rate=downsampling_rate
        ) for record in self.records)

        for record, data in zip(self.records, data):
            signal_size = data.shape[-1]
            number_of_windows = signal_size // self.window_size

            self.signals[record] = {
                "data": data,
                "size": signal_size,
            }

            self.index_to_record.extend([
                {
                    "record": record,
                    "index": x * self.window_size
                } for x in range(number_of_windows)
            ])

            if events:
                self.events[record] = {}
                number_of_events = 0
                for label, event in enumerate(events):
                    data = get_events(
                        filename="{}/{}".format(h5_directory, record),
                        event=event,
                        fs=self.fs,
                    )

                    number_of_events += data.shape[-1]
                    self.events[record][event["name"]] = {
                        "data": data,
                        "label": label,
                    }

                self.index_to_record_event.extend([
                    {
                        "record": record,
                        "max_index": signal_size - self.window_size
                    } for _ in range(number_of_events)
                ])

    def __len__(self):
        return len(self.index_to_record)

    def __getitem__(self, idx):
        signal, events = self.get_sample(
            record=self.index_to_record[idx]["record"],
            index=self.index_to_record[idx]["index"])

        if self.transformations is not None:
            signal = self.transformations(signal)
        return signal, events

    def get_record_events(self, record):

        events = [[] for _ in range(self.number_of_classes)]

        for event_data in self.events[record].values():
            events[event_data["label"]].extend([
                [start, start + duration]
                for start, duration in event_data["data"].transpose().tolist()
            ])

        return events

    def get_record_batch(self, record, batch_size, stride=None):
        """Return signal data from a specific record as a batch of continuous
           windows. Overlap in seconds allows overlapping among windows in the
           batch. The last data points will be ignored if their length is
           inferior to window_size.
        """

        # stride = overlap_size
        # batch_size = batch

        stride = int((stride if stride is not None else self.window) * self.fs)
        batch_overlap_size = stride * batch_size  # stride at a batch level
        read_size = (batch_size - 1) * stride + self.window_size
        signal_size = self.signals[record]["size"]
        t = np.arange(signal_size)
        number_of_batches_in_record = (signal_size - read_size) // batch_overlap_size + 1

        for batch in range(number_of_batches_in_record):
            start = batch_overlap_size * batch
            stop = batch_overlap_size * batch + read_size
            signal = self.signals[record]["data"][:, start:stop]

            signal_strided = torch.FloatTensor(
                as_strided(
                    x=signal,
                    shape=(batch_size, signal.shape[0], self.window_size),
                    strides=(signal.strides[1] * stride, signal.strides[0],
                             signal.strides[1]),
                )
            )
            time = t[start:stop]
            t_strided = as_strided(
                x=time,
                shape=(batch_size, self.window_size),
                strides=(time.strides[0] * stride, time.strides[0]),
            )

            yield signal_strided, t_strided

        batch_end = (
            signal_size - number_of_batches_in_record * batch_overlap_size - self.window_size
        ) // stride + 1
        if batch_end > 0:

            read_size_end = (batch_end - 1) * stride + self.window_size
            start = batch_overlap_size * number_of_batches_in_record
            end = batch_overlap_size * number_of_batches_in_record + read_size_end
            signal = self.signals[record]["data"][:, start:end]

            signal_strided = torch.FloatTensor(
                as_strided(
                    x=signal,
                    shape=(batch_end, signal.shape[0], self.window_size),
                    strides=(signal.strides[1] * stride, signal.strides[0],
                             signal.strides[1]),
                )
            )
            time = t[start:end]
            t_strided = as_strided(
                x=time,
                shape=(batch_end, self.window_size),
                strides=(time.strides[0] * stride, time.strides[0]),
            )

            yield signal_strided, t_strided

    def plot(self, idx, channels):
        """Plot events and data from channels for record and index found at
           idx"""

        import matplotlib.pyplot as plt
        signal, events = self.extract_balanced_data(
            record=self.index_to_record_event[idx]["record"],
            max_index=self.index_to_record_event[idx]["max_index"])

        non_valid_indexes = np.where(np.array(channels) is None)[0]
        signal = np.delete(signal, non_valid_indexes, axis=0)
        channels = [channel for channel in channels if channel is not None][::-1]

        num_signals = len(channels)
        signal_size = len(signal[0])
        events_numpy = events.numpy()
        plt.figure(figsize=(10 * 4, 2 * num_signals))
        gs = gridspec.GridSpec(num_signals, 1)
        gs.update(wspace=0., hspace=0.)
        for channel_num, channel in enumerate(channels):
            assert signal_size == len(signal[channel_num])
            signal_mean = signal.numpy()[channel_num].mean()
            ax = plt.subplot(gs[channel_num, 0])
            ax.set_ylim(-0.55, 0.55)
            ax.plot(signal.numpy()[channel_num], alpha=0.3)
            for event in events_numpy:
                ax.fill([event[0] * signal_size, event[1] * signal_size],
                        [signal_mean, signal_mean],
                        alpha=0.5,
                        linewidth=30,
                        color="C{}".format(int(event[-1])))
            if channel_num == 0:
                # print(EVENT_DICT[event[2]])
                offset = (1. / num_signals) * 1.1
                step = (1. / num_signals) * 0.78
            plt.gcf().text(0.915, offset + channel_num * step,
                           channel, fontsize=14)
        plt.show()
        plt.close()

    def get_sample(self, record, index):
        """Return a sample [sata, events] from a record at a particularindex"""

        signal_data = self.signals[record]["data"][:, index: index + self.window_size]
        events_data = []

        for event_name, event in self.events[record].items():
            starts, durations = event["data"][0, :], event["data"][1, :]

            # Relative start stop
            starts_relative = (starts - index) / self.window_size
            durations_relative = durations / self.window_size
            stops_relative = starts_relative + durations_relative

            # Find valid start or stop
            valid_starts_index = np.where((starts_relative > 0) *
                                          (starts_relative < 1))[0]
            valid_stops_index = np.where((stops_relative > 0) *
                                         (stops_relative < 1))[0]

            # merge them
            valid_indexes = set(list(valid_starts_index) +
                                list(valid_stops_index))

            # Annotations contains valid index with minimum overlap requirement
            for valid_index in valid_indexes:
                if (valid_index in valid_starts_index) \
                        and (valid_index in valid_stops_index):
                    events_data.append((float(starts_relative[valid_index]),
                                        float(stops_relative[valid_index]),
                                        event["label"]))
                elif valid_index in valid_starts_index:
                    if ((1 - starts_relative[valid_index]) /
                            durations_relative[valid_index]) > self.minimum_overlap:
                        events_data.append((float(starts_relative[valid_index]),
                                            1, event["label"]))

                elif valid_index in valid_stops_index:
                    if ((stops_relative[valid_index]) / durations_relative[valid_index]) \
                            > self.minimum_overlap:
                        events_data.append((0,
                                            float(stops_relative[valid_index]),
                                            event["label"]))

        return torch.FloatTensor(signal_data), torch.FloatTensor(events_data)


class BalancedEventDataset(EventDataset):
    """
    Same as EventDataset but with the possibility to choose the probability to get at least
    one event when retrieving a window.

    """

    def __init__(self,
                 h5_directory,
                 signals,
                 window,
                 events=None,
                 downsampling_rate=1,
                 records=None,
                 minimum_overlap=0.5,
                 transformations=None,
                 ratio_positive=0.5,
                 n_jobs=1,
                 cache_data=True,
                 ):
        super(BalancedEventDataset, self).__init__(
            h5_directory=h5_directory,
            signals=signals,
            events=events,
            window=window,
            downsampling_rate=downsampling_rate,
            records=records,
            minimum_overlap=minimum_overlap,
            transformations=transformations,
            n_jobs=n_jobs,
            cache_data=cache_data,
        )
        self.ratio_positive = ratio_positive

    def __len__(self):
        return len(self.index_to_record_event)

    def __getitem__(self, idx):

        signal, events = self.extract_balanced_data(
            record=self.index_to_record_event[idx]["record"],
            max_index=self.index_to_record_event[idx]["max_index"]
        )

        if self.transformations is not None:
            signal = self.transformations(signal)

        return signal, events

    def extract_balanced_data(self, record, max_index):
        """Extracts an index at random"""
        index = np.random.randint(max_index)
        signal_data, events_data = self.get_sample(record, index)
        choice = np.random.choice([0, 1],
                                  p=[1 - self.ratio_positive,
                                     self.ratio_positive])
        if choice == 0:
            while len(events_data) > 0:
                index = np.random.randint(max_index)
                signal_data, events_data = self.get_sample(record, index)
        else:
            while len(events_data) == 0:
                index = np.random.randint(max_index)
                signal_data, events_data = self.get_sample(record, index)
        return signal_data, events_data
