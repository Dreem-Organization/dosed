import torch


def decode(localization, localizations_default):
    """Opposite of encode"""
    center_encoded, width_encoded = localization[:, 0], localization[:, 1]
    x_plus_y = (center_encoded * localizations_default[:, 1] + localizations_default[:, 0]) * 2
    y_minus_x = torch.exp(width_encoded) * localizations_default[:, 1]
    x = (x_plus_y - y_minus_x) / 2
    y = (x_plus_y + y_minus_x) / 2

    localization_decoded = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], 1)
    return localization_decoded
