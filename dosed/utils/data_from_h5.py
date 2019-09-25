"""Transform a folder with h5 files into a dataset for dosed"""

import numpy as np

import h5py
import json
import os

from ..preprocessing import normalizers, filters, spectrogram, get_interpolator


def get_h5_data(filename, signals, fs, window):

    signals_raw = []
    signals_spec = []

    fs_raw = fs
    set_tsz = set()
    set_fsz = set()

    for i, signal in enumerate(signals):
        if "spectrogram" in signal.keys():
            signals_spec.append(i)

            nperseg = signal["spectrogram"]["nperseg"]
            nfft = signal["spectrogram"]["nfft"]
            downsampling_t = signal["spectrogram"]["downsampling_t"]
            downsampling_f = signal["spectrogram"]["downsampling_f"]
            padded = signal["spectrogram"]["padded"]

            # frequency size
            fsz = nfft // 2 + 1
            fsz = int(np.ceil(fsz / downsampling_f))
            set_fsz.add(fsz)
            # time size
            tsz = np.ceil((window * fs - int(not padded) * (nperseg - 1) - 1) / (nperseg // 2)) + 1
            tsz = int(np.ceil(tsz / downsampling_t))
            set_tsz.add(tsz)
        else:
            signals_raw.append(i)

    if len(signals_spec) > 0:
        assert len(set_tsz) == 1, set_tsz
        assert len(set_fsz) == 1, set_fsz
        tsz = set_tsz.pop()
        fsz = set_fsz.pop()

    with h5py.File(filename, "r") as h5:

        time_window = min(set([h5[signal["h5_path"]].size / signal['fs'] for signal in signals]))
        signal_size = None

        if len(signals_spec) > 0:
            # /!\ Force resampling frequency to be the same as the spectrogram's one
            nb_windows_spec = int(time_window * signals[signals_spec[0]]["fs"]) // (window * fs)
            signal_size = tsz * nb_windows_spec
            data_spec = np.zeros((len(signals_spec),
                                  fsz,
                                  signal_size))
            fs_raw = tsz / window  # tsz * nb_windows_spec / time_window
            t_target_spec = np.cumsum([1 / fs] * int(time_window * fs))

        if len(signals_raw) > 0:
            if signal_size is None:
                signal_size = int(time_window * fs_raw)
            data_raw = np.zeros((len(signals_raw), signal_size))
            t_target_raw = np.cumsum([1 / fs_raw] * signal_size)

        # Preprocess raw signals
        for i, signal in enumerate([signals[k] for k in signals_raw]):
            interpolator = get_interpolator(
                signal["fs"], fs_raw, h5[signal["h5_path"]].size, t_target_raw)
            normalizer = normalizers[signal['processing']["type"]](**signal['processing']['args'])

            if "filter" in signal.keys():
                filter = filters[signal['filter']['type']](fs=fs_raw, **signal['filter']['args'])

                data_raw[i, :] = interpolator(filter(h5[signal["h5_path"]][:]))
            else:
                data_raw[i, :] = interpolator(h5[signal["h5_path"]][:])

            data_raw[i, :] = normalizer(data_raw[i, :])

        # Preprocess spectrograms
        for i, signal in enumerate([signals[k] for k in signals_spec]):
            interpolator = get_interpolator(
                signal["fs"], fs, h5[signal["h5_path"]].size, t_target_spec)
            normalizer = normalizers[signal['processing']["type"]](**signal['processing']['args'])

            data_spec[i, :] = spectrogram(
                interpolator(h5[signal["h5_path"]][:]),
                fs,
                window, nperseg, nfft,
                downsampling_t, downsampling_f, padded)
            data_spec[i, :] = normalizer(data_spec[i, :])

        window_size = int(window * fs_raw)

        data_dict = dict()
        if len(signals_raw) > 0:
            data_dict["raw"] = data_raw
        if len(signals_spec) > 0:
            data_dict["spec"] = data_spec

    return data_dict, fs_raw, window_size, signal_size


def get_h5_events(filename, event_params, fs):

    if "h5_path" in event_params:
        with h5py.File(filename, "r") as h5:
            starts = h5[event_params["h5_path"]]["start"][:]
            durations = h5[event_params["h5_path"]]["duration"][:]
            assert len(starts) == len(durations), "Inconsistents event durations and starts"

            data = np.zeros((2, len(starts)))
            data[0, :] = starts * fs
            data[1, :] = durations * fs
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

            assert len(starts) == len(durations), "Inconsistents event durations and starts"

            data = np.zeros((2, len(starts)))
            data[0, :] = np.array(starts) * fs
            data[1, :] = np.array(durations) * fs
    else:
        raise Exception("No events'path given")

    return data
