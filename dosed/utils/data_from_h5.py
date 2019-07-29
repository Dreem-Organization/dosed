"""Transform a folder with h5 files into a dataset for dosed"""

import numpy as np

import h5py

from ..preprocessing import normalizers
from scipy.interpolate import interp1d


def get_h5_data(filename, signals, fs):
    import time
    time.sleep(5)

    signals_fs = [signal['fs'] for signal in signals]
    downsampling_rates = [signal['fs'] / fs for signal in signals]

    with h5py.File(filename, "r") as h5:

        # Check that all signals have the same size after resampling  to target frequency
        signals_size = set([int(h5[signal["h5_path"]].size / f)
                            for signal, f in zip(signals, signals_fs)])
        signal_size = min(signals_size)

        t_source = [np.linspace(0, signal_size, signal_size * f) for f in signals_fs]
        t_target = np.linspace(0, signal_size, signal_size * fs)

        data = np.zeros((len(signals), signal_size * fs))
        for i, signal in enumerate(signals):
            normalizer = normalizers[signal['processing']["type"]](**signal['processing']['args'])
            data[i, :] = interp1d(t_source[i],normalizer(h5[signal["h5_path"]][:]))(t_target)
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

