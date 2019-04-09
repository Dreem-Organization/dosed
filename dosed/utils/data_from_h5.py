"""Transform a folder with h5 files into a dataset for dosed"""

import numpy as np

import h5py

from ..preprocessings import normalizers


def get_h5_data(filename, signals, downsampling_rate):
    with h5py.File(filename, "r") as h5:
        # Check that all signals have the same size and sampling frequency
        signals_size = set([int(h5[signal["h5_path"]].size) for signal in signals])
        assert len(signals_size) == 1, "Different signal sizes found!"
        signal_size = len(range(0, signals_size.pop(), downsampling_rate))

        data = np.zeros((len(signals), signal_size))
        for i, signal in enumerate(signals):
            normalizer = normalizers[signal['processing']["type"]](**signal['processing']['args'])
            data[i, :] = normalizer(h5[signal["h5_path"]][:])[::downsampling_rate]

    return data


def get_h5_events(filename, event):
    with h5py.File(filename, "r") as h5:
        starts = h5[event["h5_path"]]["start"]
        durations = h5[event["h5_path"]]["duration"]
        assert len(starts) == len(durations), "Inconsistents event durations and starts"

        data = np.zeros((2, len(starts)))
        data[0, :] = starts
        data[1, :] = durations
    return data
