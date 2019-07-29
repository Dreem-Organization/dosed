import numpy as np
from scipy import interpolate


def resample(signal, actual_frequency, target_frequency, interpolation_args={}):

    if actual_frequency == target_frequency:
        return signal
    elif actual_frequency % target_frequency == 0:
        return signal[::round(actual_frequency / target_frequency)]

    resampling_ratio = actual_frequency / target_frequency
    x_base = np.arange(0, len(signal))

    interpolator = interpolate.interp1d(x_base, signal, axis=0, bounds_error=False, fill_value='extrapolate',
                                        **interpolation_args)

    x_interp = np.arange(0, len(signal), resampling_ratio)

    signal_duration = signal.shape[0] / actual_frequency
    resampled_length = round(signal_duration * target_frequency)
    resampled_signal = interpolator(x_interp)
    if len(resampled_signal) < resampled_length:
        padding = np.zeros((resampled_length - len(resampled_signal), signal.shape[-1]))
        resampled_signal = np.concatenate([resampled_signal, padding])

    return resampled_signal
