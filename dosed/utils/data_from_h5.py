"""Transform a folder with h5 files into a dataset for dosed"""

import numpy as np

import h5py
import json
import os

from ..preprocessing import dict_filters, get_interpolator


def get_h5_data(filename, signals, window):

    def recursive_processing(block):
        if "h5_paths" in block:
            with h5py.File(filename, "r") as h5:
                signals = [h5[h5_path] for h5_path in block["h5_paths"]]
                signal_size = min([signal.size for signal in signals])
                signals = [signal[:signal_size] for signal in signals]
                fs = block["fs"]
                input_shape = (int(window * fs),)
            return signals, fs, signal_size, input_shape
        else:
            signals = []
            for new_block in block["signals"]:
                signal, old_fs, signal_size, input_shape = recursive_processing(new_block)
                signals.append(signal)

            signals = np.concatenate(signals, axis=0)

            new_fs = block["fs"]

            time_window = signal_size / old_fs

            # Resample the signal to the new frequency
            target_time = np.cumsum([1 / new_fs] * int(time_window * new_fs))
            interpolator = get_interpolator(old_fs, new_fs, signal_size, target_time)
            signals = np.array([interpolator(signal) for signal in signals])
            # Update the signal size
            signal_size = int(time_window * new_fs)
            input_shape = (int(window * new_fs),)

            for filter_params in block["preprocessing"]:
                filter = dict_filters[filter_params["name"]](
                    fs=new_fs,
                    signal_size=signal_size,
                    window=window,
                    input_shape=input_shape,
                    **filter_params["args"])

                signals, new_fs, signal_size, input_shape = filter(signals)

            return signals, new_fs, signal_size, input_shape

    signals_data = dict()
    signals_properties = dict()
    for block in signals:
        signals, fs, signal_size, input_shape = recursive_processing(block)
        input_shape = (len(signals),) + input_shape
        signals_data[block["name"]] = signals
        signals_properties[block["name"]] = {
            "fs": fs, "signal_size": signal_size, "input_shape": input_shape}

    return signals_data, signals_properties


def get_h5_events(filename, event_params):

    if "h5_path" in event_params:
        with h5py.File(filename, "r") as h5:
            starts = h5[event_params["h5_path"]]["start"][:]
            durations = h5[event_params["h5_path"]]["duration"][:]

    elif "json_path" in event_params:
        directory, filename = os.path.split(filename)
        filename = os.path.join(directory, event_params["json_path"], filename[:-3] + ".json")
        with open(filename) as f:
            f = json.load(f)
            starts = []
            durations = []
            for event in f["labels"]:
                if event_params["name"] == "apnea" or event_params["name"] == event["value"]:
                    starts.append(event["start"])
                    durations.append(event["end"] - event["start"])

    else:
        raise Exception("No events'path given")

    assert len(starts) == len(durations), "Inconsistents event durations and starts"

    data = np.zeros((2, len(starts)))
    data[0, :] = starts
    data[1, :] = durations

    return data
