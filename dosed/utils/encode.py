import torch


def encode(localization_match, localizations_default):
    """localization_match are converted relatively to their default location

    localization_match has size [batch, number_of_localizations, 2] containing the ground truth
    matched localization (representation x y)
    localization_defaults has size [number_of_localizations, 2]

    returns localization_target [batch, number_of_localizations, 2]
    """
    center = (localization_match[:, 0] + localization_match[:, 1]) / 2 - localizations_default[:, 0]
    center = center / localizations_default[:, 1]
    width = torch.log((localization_match[:, 1] - localization_match[:, 0]) / localizations_default[:, 1])
    localization_target = torch.cat([center.unsqueeze(1), width.unsqueeze(1)], 1)
    return localization_target
