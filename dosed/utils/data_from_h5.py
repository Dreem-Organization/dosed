"""Transform a folder with h5 files into a dataset for dosed"""

import numpy as np

import h5py

from ..preprocessing import normalizers, spectrogram, get_interpolator


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
            signal_size = int(time_window * fs_raw)
            data_raw = np.zeros((len(signals_raw), signal_size))
            t_target_raw = np.cumsum([1 / fs_raw] * signal_size)

        # Preprocess raw signals
        for i, signal in enumerate([signals[k] for k in signals_raw]):
            interpolator = get_interpolator(
                signal["fs"], fs_raw, h5[signal["h5_path"]].size, t_target_raw)
            normalizer = normalizers[signal['processing']["type"]](**signal['processing']['args'])

            data_raw[i, :] = normalizer(interpolator(h5[signal["h5_path"]][:]))

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

        data_dict = {
            "fs": fs_raw,
            "window_size": int(window * fs_raw),
            "signal_size": signal_size,
        }

        if len(signals_raw) > 0:
            data_dict["raw"] = data_raw
        if len(signals_spec) > 0:
            data_dict["spec"] = data_spec

    return data_dict


def get_h5_events(filename, event, fs):
    with h5py.File(filename, "r") as h5:
        starts = h5[event["h5_path"]]["start"][:]
        durations = h5[event["h5_path"]]["duration"][:]
        assert len(starts) == len(durations), "Inconsistents event durations and starts"

        data = np.zeros((2, len(starts)))
        data[0, :] = starts * fs
        data[1, :] = durations * fs
    return data
