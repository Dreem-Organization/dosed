"""Transform a folder with h5 files into a dataset for dosed"""

import numpy as np

import h5py

from ..preprocessing import normalizers
from ..preprocessing.resamplers import resample


def get_h5_data(filename, signals, resampling_fs):
    import time
    time.sleep(5)

    signals_fs = [signal['fs'] for signal in signals]
    downsampling_rates = [signal['fs'] / resampling_fs for signal in signals]

    with h5py.File(filename, "r") as h5:

        # Check that all signals have the same size after resampling  to target frequency
        signals_size = set([int(h5[signal["h5_path"]].size / downsampling)
                            for signal, downsampling in zip(signals, downsampling_rates)])
        #assert len(signals_size) == 1, "Different signal sizes found ! {}".format(signals_size)
        signal_size = len(range(0, min(signals_size)))

        data = np.zeros((len(signals), signal_size))
        for i, signal in enumerate(signals):
            normalizer = normalizers[signal['processing']["type"]](**signal['processing']['args'])
            data[i, :] = resample(normalizer(h5[signal["h5_path"]][:]),
                                  signals_fs[i], resampling_fs)[:signal_size]
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

