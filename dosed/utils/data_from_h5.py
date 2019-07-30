"""Transform a folder with h5 files into a dataset for dosed"""

import numpy as np

import h5py

from ..preprocessing import normalizers
from scipy.interpolate import interp1d


def get_h5_data(filename, signals, fs):
    import time
    time.sleep(5)

    with h5py.File(filename, "r") as h5:

        time_window = min(set([int(h5[signal["h5_path"]].size / signal['fs'])
                               for signal in signals]))

        t_target = np.cumsum([1 / fs] * int(time_window * fs))

        data = np.zeros((len(signals), time_window * fs))
        for i, signal in enumerate(signals):
            t_source = np.cumsum([1 / signal["fs"]] *
                                 int(time_window * signal["fs"]))
            normalizer = normalizers[signal['processing']["type"]](**signal['processing']['args'])
            data[i, :] = interp1d(t_source, normalizer(h5[signal["h5_path"]][:]))(t_target)
    return data


def get_h5_events(filename, event, fs):
    with h5py.File(filename, "r") as h5:
        starts = h5[event["h5_path"]]["start"][:]
        durations = h5[event["h5_path"]]["duration"][:]
        assert len(starts) == len(durations), "Inconsistents event durations and starts"

        data = np.zeros((2, len(starts)))
        data[0, :] = starts * fs
        data[1, :] = durations * fs
    return data
