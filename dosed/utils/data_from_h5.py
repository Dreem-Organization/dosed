"""Transform a folder with h5 files into a dataset for dosed"""

import numpy as np

import h5py
import json
import os

from ..preprocessing import normalizers
from scipy.interpolate import interp1d


def get_h5_data(filename, signals, fs):
    with h5py.File(filename, "r") as h5:

        signal_size = int(fs * min(
            set([h5[signal["h5_path"]].size / signal['fs'] for signal in signals])
        ))

        t_target = np.cumsum([1 / fs] * signal_size)
        data = np.zeros((len(signals), signal_size))
        for i, signal in enumerate(signals):
            t_source = np.cumsum([1 / signal["fs"]] *
                                 h5[signal["h5_path"]].size)
            normalizer = normalizers[signal['processing']["type"]](**signal['processing']['args'])
            data[i, :] = interp1d(t_source, normalizer(h5[signal["h5_path"]][:]),
                                  fill_value="extrapolate")(t_target)
    return data


def get_h5_events(filename, event, fs):
    if "json_path" in event:
        directory, filename = os.path.split(filename)
        filename = os.path.join(directory, event["apnea"], filename[:-3] + ".json")
        with open(filename) as f:
            f = json.load(f)
            starts = []
            durations = []
            for event in f["labels"]:
                starts.append(event["start"])
                durations.append(event["end"] - event["start"])

            assert len(starts) == len(durations), "Inconsistents event durations and starts"

            data = np.zeros((2, len(starts)))
            data[0, :] = np.array(starts) * fs
            data[1, :] = np.array(durations) * fs

    elif "h5_path" in event:
        with h5py.File(filename, "r") as h5:
            starts = h5[event["h5_path"]]["start"][:]
            durations = h5[event["h5_path"]]["duration"][:]
            assert len(starts) == len(durations), "Inconsistents event durations and starts"

            data = np.zeros((2, len(starts)))
            data[0, :] = starts * fs
            data[1, :] = durations * fs

    else:
        raise Exception("No events' path given !")

    return data
