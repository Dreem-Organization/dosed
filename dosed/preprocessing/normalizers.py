import numpy as np


def clip(max_value):
    """returns a function to clip data"""

    def clipper(signal_data, max_value=max_value):
        """returns input signal clipped between +/- max_value.
        """
        return np.clip(signal_data, -max_value, max_value)

    return clipper


def clip_and_normalize(min_value, max_value):
    """returns a function to clip and normalize data"""

    def clipper(x, min_value=min_value, max_value=max_value):
        """returns input signal clipped between min_value and max_value
        and then normalized between -0.5 and 0.5.
        """
        x = np.clip(x, min_value, max_value)
        x = ((x - min_value) /
             (max_value - min_value)) - 0.5
        return x

    return clipper


def mask_clip_and_normalize(min_value, max_value, mask_value):
    """returns a function to clip and normalize data"""

    def clipper(x, min_value=min_value, max_value=max_value,
                mask_value=mask_value):
        """returns input signal clipped between min_value and max_value
        and then normalized between -0.5 and 0.5.
        """
        mask = np.ma.masked_equal(x, mask_value)
        x = np.clip(x, min_value, max_value)
        x = ((x - min_value) /
             (max_value - min_value)) - 0.5
        x[mask.mask] = mask_value
        return x

    return clipper
