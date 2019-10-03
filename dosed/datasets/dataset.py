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

import tqdm


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

        # used in network architecture
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
            window=self.window
        ) for record in tqdm.tqdm(self.records))

        ##################
        # Set all variables of the dataset
        data_signals, data_properties = zip(*data)

        self.blocks_names = list(data_signals[0].keys())

        fs = {block_name: set([properties[block_name]["fs"] for properties in data_properties])
              for block_name in self.blocks_names}

        assert np.all([len(fs_block) == 1 for fs_block in fs.values()])
        self.fs = {block_name: fs_block.pop() for block_name, fs_block in fs.items()}

        signal_sizes = [{block_name: properties["signal_size"]
                         for block_name, properties in properties_dict.items()}
                        for properties_dict in data_properties]

        input_shapes = {block_name: set([properties[block_name]["input_shape"]
                                         for properties in data_properties])
                        for block_name in self.blocks_names}

        assert np.all([len(input_shape_block) == 1 for input_shape_block in input_shapes.values()])
        self.input_shapes = {block_name: input_shape_block.pop()
                             for block_name, input_shape_block in input_shapes.items()}

        self.window_sizes = {block_name: int(window * fs) for block_name, fs in self.fs.items()}

        ##################

        for record, data, signal_sizes in zip(self.records, data_signals, signal_sizes):
            number_of_windows = min(set([signal_sizes[block_name] // self.window_sizes[block_name]
                                         for block_name in self.blocks_names]))

            shortest_signal_name = min(sorted({block: signal_sizes[block] / self.fs[block]
                                               for block in self.blocks_names}.items()),
                                       key=lambda x: x[1])[0]

            self.signals[record] = {
                "data": data,
                "size": signal_sizes,
                "duration": int(signal_sizes[shortest_signal_name] / self.fs[shortest_signal_name])
            }

            self.index_to_record.extend([
                {
                    "record": record,
                    "index": x
                } for x in range(number_of_windows)
            ])

            if events:
                self.events[record] = {}
                number_of_events = 0
                events_indexes = set()

                shortest_window_size = self.window_sizes[shortest_signal_name]

                max_index = signal_sizes[shortest_signal_name] - shortest_window_size

                for label, event in enumerate(events):

                    data = get_events(
                        filename="{}/{}".format(h5_directory, record),
                        event_params=event,
                    )

                    number_of_events += data.shape[-1]
                    self.events[record][event["name"]] = {
                        "data": data,
                        "label": label,
                    }

                    for start, duration in zip(*data):
                        start *= self.fs[shortest_signal_name]
                        duration *= self.fs[shortest_signal_name]
                        if shortest_window_size / duration > self.minimum_overlap:
                            stop = start + duration
                            duration_overlap = duration * self.minimum_overlap
                            start_valid_index = int(round(
                                max(0, start + duration_overlap - shortest_window_size + 1)))
                            end_valid_index = int(round(
                                min(max_index + 1, stop - duration_overlap)))

                            indexes = list(range(start_valid_index, end_valid_index))
                            # Check borders
                            if self.get_valid_events_index(start_valid_index - 1,
                                                           [start], [duration],
                                                           shortest_window_size):
                                indexes.append(start_valid_index - 1)
                            if self.get_valid_events_index(end_valid_index + 1,
                                                           [start], [duration],
                                                           shortest_window_size):
                                indexes.append(end_valid_index + 1)
                            events_indexes.update(indexes)

                no_events_indexes = set(range(max_index + 1))
                no_events_indexes = np.array(list(no_events_indexes.difference(events_indexes)))
                events_indexes = np.array(list(events_indexes))
                no_events_indexes = no_events_indexes / shortest_window_size
                events_indexes = events_indexes / shortest_window_size

                if number_of_events > 0:
                    self.index_to_record_event.extend([
                        {
                            "record": record,
                            "max_index": max_index,
                            "events_indexes": events_indexes,
                            "no_events_indexes": no_events_indexes,
                        } for _ in range(number_of_events)
                    ])
                else:
                    print("Record : {} has no event".format(record))

    def __len__(self):
        return len(self.index_to_record)

    def __getitem__(self, idx):
        signals, events = self.get_sample(
            record=self.index_to_record[idx]["record"],
            index=self.index_to_record[idx]["index"])
        if self.transformations is not None:
            for signal_type, signal in signals.items():
                signals[signal_type] = self.transformations(signal)
        return signals, events

    def get_valid_events_index(self, index, starts, durations, window_size):
        """Return the events' indexes that have enough overlap with the given time index
           ex: index = 155
               starts =   [10 140 150 165 2000]
               duration = [4  20  10  10   40]
               minimum_overlap = 0.5
               window_size = 15
           return: [2 3]
        """
        # Relative start stop

        starts = np.array(starts)
        durations = np.array(durations)

        starts_relative = (starts - index) / window_size
        durations_relative = durations / window_size
        stops_relative = starts_relative + durations_relative

        # Find valid start or stop
        valid_starts_index = np.where((starts_relative > 0) *
                                      (starts_relative < 1))[0]
        valid_stops_index = np.where((stops_relative > 0) *
                                     (stops_relative < 1))[0]

        valid_inside_index = np.where((starts_relative <= 0) *
                                      (stops_relative >= 1))[0]

        # merge them
        valid_indexes = set(list(valid_starts_index) +
                            list(valid_stops_index) +
                            list(valid_inside_index))

        # Annotations contains valid index with minimum overlap requirement
        events_indexes = []
        for valid_index in valid_indexes:
            if (valid_index in valid_starts_index) \
                    and (valid_index in valid_stops_index):
                events_indexes.append(valid_index)
            elif valid_index in valid_starts_index:
                if ((1 - starts_relative[valid_index]) /
                        durations_relative[valid_index]) > self.minimum_overlap:
                    events_indexes.append(valid_index)
            elif valid_index in valid_stops_index:
                if ((stops_relative[valid_index]) / durations_relative[valid_index]) \
                        > self.minimum_overlap:
                    events_indexes.append(valid_index)
            elif valid_index in valid_inside_index:
                if window_size / durations[valid_index] > self.minimum_overlap:
                    events_indexes.append(valid_index)

        return events_indexes

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
        stride = stride if stride is not None else self.window
        strides = {block_name: int(stride * fs)
                   for block_name, fs in self.fs.items()}
        # stride at a batch level
        batch_overlap_size = {block_name: stride * batch_size
                              for block_name, stride in strides.items()}

        read_size = {block_name: (batch_size - 1) * strides[block_name] + window_size
                     for block_name, window_size in self.window_sizes.items()}

        duration = self.signals[record]["duration"]
        t = np.arange(duration)

        number_of_batches_in_record = int(
            (duration - (batch_size - 1) * stride + self.window) // (stride * batch_size) + 1)

        for batch in range(number_of_batches_in_record):
            signal_strided = dict()

            for block_name in self.blocks_names:
                start = batch_overlap_size[block_name] * batch
                stop = batch_overlap_size[block_name] * batch + read_size[block_name]

                signal = self.signals[record]["data"][block_name]
                signal = signal[..., start:stop]
                signal_strided[block_name] = torch.FloatTensor(
                    as_strided(
                        x=signal,
                        shape=(batch_size, *signal.shape[:-1], self.window_sizes[block_name]),
                        strides=(signal.strides[-1] * strides[block_name], *signal.strides),
                    )
                )
                time = t[start:stop]
                t_strided = as_strided(
                    x=time,
                    shape=(batch_size, self.window),
                    strides=(int(time.strides[0] * stride), time.strides[0]),
                )

            yield signal_strided, t_strided

        batch_end = int((duration - number_of_batches_in_record *
                         stride * batch_size - self.window) // stride + 1)

        if batch_end > 0:
            signal_strided = dict()

            for block_name in self.blocks_names:
                read_size_end = (batch_end - 1) * \
                    strides[block_name] + self.window_sizes[block_name]
                start = batch_overlap_size[block_name] * number_of_batches_in_record
                end = batch_overlap_size[block_name] * number_of_batches_in_record + read_size_end

                signal = self.signals[record]["data"][block_name]
                signal = signal[..., start:end]
                signal_strided[block_name] = torch.FloatTensor(
                    as_strided(
                        x=signal,
                        shape=(batch_end, *signal.shape[:-1], self.window_sizes[block_name]),
                        strides=(signal.strides[-1] * strides[block_name], *signal.strides),
                    )
                )

                time = t[start:end]
                t_strided = as_strided(
                    x=time,
                    shape=(batch_end, self.window),
                    strides=(int(time.strides[0] * stride), time.strides[0]),
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

        signal_data = dict()
        for block_name, signal in self.signals[record]["data"].items():
            block_index = int(index * self.window_sizes[block_name])
            signal_data[block_name] = torch.FloatTensor(
                signal[..., block_index: block_index + self.window_sizes[block_name]])
        events_data = []

        for event_name, event in self.events[record].items():
            starts, durations = event["data"][0, :], event["data"][1, :]

            index = index * self.window

            # Relative start stop
            starts_relative = (starts - index) / self.window
            durations_relative = durations / self.window
            stops_relative = starts_relative + durations_relative

            valid_indexes = self.get_valid_events_index(
                index, starts, durations, self.window)

            for valid_index in valid_indexes:
                events_data.append((max(0, float(starts_relative[valid_index])),
                                    min(1, float(stops_relative[valid_index])),
                                    event["label"]))

        return signal_data, torch.FloatTensor(events_data)


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

        signals, events = self.extract_balanced_data(
            record=self.index_to_record_event[idx]["record"],
            max_index=self.index_to_record_event[idx]["max_index"],
            events_indexes=self.index_to_record_event[idx]["events_indexes"],
            no_events_indexes=self.index_to_record_event[idx]["no_events_indexes"]
        )

        if self.transformations is not None:
            for signal_type, signal in signals.items():
                signals[signal_type] = self.transformations(signal)

        return signals, events

    def extract_balanced_data(self, record, max_index, events_indexes, no_events_indexes):
        """Extracts an index at random"""

        choice = np.random.choice([0, 1], p=[1 - self.ratio_positive, self.ratio_positive])

        if choice == 0:
            index = no_events_indexes[np.random.randint(len(no_events_indexes))]
            signal_data, events_data = self.get_sample(record, index)
        else:
            index = events_indexes[np.random.randint(len(events_indexes))]
            signal_data, events_data = self.get_sample(record, index)

        return signal_data, events_data
