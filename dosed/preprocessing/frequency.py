

from scipy import signal
import numpy as np
from scipy.interpolate import interp1d


def spectrogram(data, fs, window, nperseg, nfft, downsampling_t=1, downsampling_f=1, padded=True):
    nb_windows = len(data) // (window * fs)
    fsz = nfft // 2 + 1
    fsz = int(np.ceil(fsz / downsampling_f))
    tsz = np.ceil((window * fs - int(not padded) * (nperseg - 1) - 1) / (nperseg // 2)) + 1
    tsz = int(np.ceil(tsz / downsampling_t))

    spec = np.zeros((nb_windows, fsz, tsz))

    for i in range(nb_windows):
        f, t, Zxx = signal.stft(data[i * window * fs:(i + 1) *
                                     window * fs], fs, nperseg=nperseg, nfft=nfft, padded=padded)
        spec[i, :, :] = np.abs(Zxx[::downsampling_f, ::downsampling_t])

    return spec.transpose((1, 0, 2)).reshape((fsz, nb_windows * tsz))


def get_interpolator(signal_fs, target_fs, original_size, t_target):
    if signal_fs % target_fs == 0:
        def interpolator(x):
            return x[::signal_fs // target_fs][:len(t_target)]
    else:
        def interpolator(x):
            t_source = np.cumsum([1 / signal_fs] * original_size)
            return interp1d(t_source, x, fill_value="extrapolate")(t_target)
    return interpolator
