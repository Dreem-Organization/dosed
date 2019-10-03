from dreempy.filter import (
    iir_highpass_filter,
    iir_bandpass_filter,
    iir_lowpass_filter,
)


def get_bandpass(fs, signal_size, window, input_shape,
                 frequency_band=[0.1, 0.5], order=4, type="butter"):

    def filter(signal, fs=fs, frequency_band=frequency_band, order=order, type=type):

        signal_filtered = iir_bandpass_filter(
            signal=signal,
            fs=fs,
            order=order,
            frequency_band=frequency_band,
            filter_type=type,
            axis=-1,
            forward_backward=True,
        )

        return signal_filtered, fs, signal_size, input_shape

    return filter


def get_lowpass(fs, signal_size, window, input_shape, frequency_cut=0.1, order=4, type="butter"):

    def filter(signal, fs=fs, frequency_cut=frequency_cut, order=order, type=type):

        signal_filtered = iir_lowpass_filter(
            signal=signal,
            fs=fs,
            order=order,
            frequency_cut=frequency_cut,
            filter_type=type,
            axis=-1,
            forward_backward=True,
        )

        return signal_filtered, fs, signal_size, input_shape

    return filter


def get_highpass(fs, signal_size, window, input_shape, frequency_cut=0.5, order=4, type="butter"):

    def filter(signal, fs=fs, frequency_cut=frequency_cut, order=order, type=type):

        signal_filtered = iir_highpass_filter(
            signal=signal,
            fs=fs,
            order=order,
            frequency_cut=frequency_cut,
            filter_type=type,
            axis=-1,
            forward_backward=True,
        )

        return signal_filtered, fs, signal_size, input_shape

    return filter
