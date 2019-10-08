
from scipy.signal import iirfilter, filtfilt, lfilter


def iir_bandpass_filter(signal,
                        fs=250.,
                        order=4,
                        frequency_band=[0.4, 18],
                        filter_type='butter',
                        axis=-1,
                        forward_backward=False):
    """ Perform bandpass filtering using scipy library.

    Parameters
    ----------

    signal : 1D numpy.array
        Array to filter.
    fs : float
        Sampling frequency
    order : int
        Order of the filter
    frequency_band : list
        Specify bandpass eg: [0.5, 20] will keep frequencies between 0.5
        and 20 Hz
    filter_type : str
        Choose type of IIR filter: butter, cheby1, cheby2, ellip, bessel
    axis: int
        Choose axis where to perform filtering.
    forward_backward : boolean
        Set True if you want a null phase shift filtered signal

    Returns
    -------

        1D numpy.array
            The signal filtered
    """
    b, a = iirfilter(order,
                     [ff * 2. / fs for ff in frequency_band],
                     btype='bandpass',
                     ftype=filter_type)
    if forward_backward:
        result = filtfilt(b, a, signal, axis)
    else:
        result = lfilter(b, a, signal, axis)
    return result


def iir_bandstop_filter(signal,
                        fs=250.,
                        order=4,
                        frequency_band=[49.5, 50.5],
                        filter_type='butter',
                        axis=-1,
                        forward_backward=False):
    """ Perform bandstop filtering using scipy library.

    If you want to use it as a notch filter juste put a narrow
    frequency band:
    frequency_band = [49.5, 50.5] will be a good notch 50 Hz notch
    frequency_band = [59.5, 60.5] will be a good 60Hz notch

    Parameters
    ----------
    signal : 1D numpy.array
        Array to filter.
    fs : float
        Sampling frequency
    order : int
        Order of the filter
    frequency_band : list
        Specify bandpass eg: [0.5, 20] will keep frequencies between 0.5
        and 20 Hz
    filter_type : str
        Choose type of IIR filter: butter, cheby1, cheby2, ellip, bessel
    axis: int
        Choose axis where to perform filtering.
    forward_backward : boolean
        Set True if you want a null phase shift filtered signal

    Returns
    -------
        1D numpy.array
            The signal filtered
    """
    b, a = iirfilter(order,
                     [f / (fs / 2.) for f in frequency_band],
                     btype='bandstop',
                     ftype=filter_type)
    if forward_backward:
        result = filtfilt(b, a, signal, axis)
    else:
        result = lfilter(b, a, signal, axis)
    return result


def iir_highpass_filter(signal,
                        fs=250.,
                        order=4,
                        frequency_cut=0.1,
                        filter_type='butter',
                        axis=-1,
                        forward_backward=False):
    """ Perform highpass filtering using scipy library.

    If you want to use it to remove dc
    frequency band:
    frequency_band = [0.1]

    Parameters
    ----------
    signal : 1D numpy.array
        Array to filter.
    fs : float
        Sampling frequency
    order : int
        Order of the filter
    frequency_band : list
        Specify cut frequency eg: [0.1] will keep frequencies higher than 0.1Hz
    filter_type : str
        Choose type of IIR filter: butter, cheby1, cheby2, ellip, bessel
    axis: int
        Choose axis where to perform filtering.
    forward_backward : boolean
        Set True if you want a null phase shift filtered signal

    Returns
    -------
        1D numpy.array
            The signal filtered
    """
    b, a = iirfilter(order,
                     frequency_cut / (fs / 2.),
                     btype='highpass',
                     ftype=filter_type)
    if forward_backward:
        result = filtfilt(b, a, signal, axis)
    else:
        result = lfilter(b, a, signal, axis)
    return result


def iir_lowpass_filter(signal,
                       fs=250.,
                       order=4,
                       frequency_cut=35,
                       filter_type='butter',
                       axis=-1,
                       forward_backward=False):
    """ Perform highpass filtering using scipy library.

    If you want to use it to remove dc
    frequency band:
    frequency_band = [0.1]

    Parameters
    ----------
    signal : 1D numpy.array
        Array to filter.
    fs : float
        Sampling frequency
    order : int
        Order of the filter
    frequency_band : list
        Specify cut frequency eg: [0.1] will keep frequencies higher than 0.1Hz
    filter_type : str
        Choose type of IIR filter: butter, cheby1, cheby2, ellip, bessel
    axis: int
        Choose axis where to perform filtering.
    forward_backward : boolean
        Set True if you want a null phase shift filtered signal

    Returns
    -------
        1D numpy.array
            The signal filtered
    """
    b, a = iirfilter(order,
                     frequency_cut / (fs / 2.),
                     btype='lowpass',
                     ftype=filter_type)
    if forward_backward:
        result = filtfilt(b, a, signal, axis)
    else:
        result = lfilter(b, a, signal, axis)
    return result


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
