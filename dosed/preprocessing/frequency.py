

from scipy import signal
import numpy as np
from scipy.interpolate import interp1d


def spectrogram(fs, signal_size, window, input_shape, nperseg, nfft, temporal_downsampling=1,
                frequential_downsampling=1, padded=True):

    def spectrogram_filter(data):
        nb_channels = data.shape[0]
        nb_windows = signal_size // (window * fs)
        fsz = nfft // 2 + 1
        fsz = int(np.ceil(fsz / frequential_downsampling))
        tsz = np.ceil((window * fs - int(not padded) * (nperseg - 1) - 1) / (nperseg // 2)) + 1
        tsz = int(np.ceil(tsz / temporal_downsampling))

        spectrogram = np.zeros((nb_windows, nb_channels, fsz, tsz))

        for i in range(nb_windows):
            f, t, Zxx = signal.stft(data[..., i * window * fs:(i + 1) *
                                         window * fs], fs, nperseg=nperseg, nfft=nfft, padded=padded)
            spectrogram[i, :, :, :] = np.abs(
                Zxx[..., ::frequential_downsampling, ::temporal_downsampling])

        spectrogram = spectrogram.transpose((1, 2, 0, 3))  # nb_channels, fsz, nb_windows, tsz
        spectrogram = spectrogram.reshape((nb_channels, fsz, nb_windows * tsz))
        return spectrogram, tsz / window, nb_windows * tsz, (fsz, tsz)

    return spectrogram_filter


def get_interpolator(signal_fs, target_fs, original_size, t_target):
    if signal_fs % target_fs == 0:
        def interpolator(x):
            return x[::signal_fs // target_fs][:len(t_target)]
    else:
        def interpolator(x):
            t_source = np.cumsum([1 / signal_fs] * original_size)
            return interp1d(t_source, x, fill_value="extrapolate")(t_target)
    return interpolator
