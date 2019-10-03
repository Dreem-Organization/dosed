import tempfile

import torch

from dosed.models import DOSED1, DOSED2, DOSED3, DOSED4


def test_dosed1():
    batch_size = 32
    number_of_channels = 2
    window_duration = 10
    fs = 256
    x = {"raw": torch.rand(batch_size, number_of_channels, window_duration * fs)}

    # number of classes
    number_of_classes = 3

    # default events
    default_event_duration = 1
    overlap_default_event = 2

    net = DOSED1(
        input_shapes={"raw": (number_of_channels, window_duration * fs)},
        number_of_classes=number_of_classes,
        detection_parameters={
            "overlap_non_maximum_suppression": 0.5,
            "classification_threshold": 0.5,
        },
        duration=default_event_duration * fs,
        rho=overlap_default_event
    )
    localizations, classifications, localizations_default = net.forward(x)
    number_of_default_events = int(window_duration / default_event_duration * overlap_default_event)
    assert localizations.shape == (batch_size, number_of_default_events, 2)
    assert classifications.shape == (batch_size, number_of_default_events, number_of_classes + 1)
    assert localizations_default.shape == (number_of_default_events, 2)


def test_dosed2():
    batch_size = 32
    number_of_channels = 2
    window_duration = 10
    fs = 256
    x = {"raw": torch.rand(batch_size, number_of_channels, window_duration * fs)}

    # number of classes
    number_of_classes = 3

    # default events
    default_event_sizes = [1 * fs, 2 * fs]

    net = DOSED2(
        input_shapes={"raw": (number_of_channels, window_duration * fs)},
        number_of_classes=number_of_classes,
        detection_parameters={
            "overlap_non_maximum_suppression": 0.5,
            "classification_threshold": 0.5,
        },
        default_event_sizes=default_event_sizes,
    )
    localizations, classifications, localizations_default = net.forward(x)
    number_of_default_events = sum([
        int(window_duration * fs / default_event_size * 2)
        for default_event_size in default_event_sizes]
    )
    assert localizations.shape == (batch_size, number_of_default_events, 2)
    assert classifications.shape == (batch_size, number_of_default_events, number_of_classes + 1)
    assert localizations_default.shape == (number_of_default_events, 2)


def test_dosed3():
    batch_size = 32
    number_of_channels = 2
    window_duration = 10
    fs = 256
    x = {"raw": torch.rand(batch_size, number_of_channels, window_duration * fs)}

    # number of classes
    number_of_classes = 3

    # default events
    default_event_sizes = [1 * fs, 2 * fs]

    net = DOSED3(
        input_shapes={"raw": (number_of_channels, window_duration * fs)},
        number_of_classes=number_of_classes,
        detection_parameters={
            "overlap_non_maximum_suppression": 0.5,
            "classification_threshold": 0.5,
        },
        default_event_sizes=default_event_sizes,
    )
    localizations, classifications, localizations_default = net.forward(x)
    number_of_default_events = sum([
        int(window_duration * fs / default_event_size * 2)
        for default_event_size in default_event_sizes]
    )
    assert localizations.shape == (batch_size, number_of_default_events, 2)
    assert classifications.shape == (batch_size, number_of_default_events, number_of_classes + 1)
    assert localizations_default.shape == (number_of_default_events, 2)


def test_dosed4():
    batch_size = 32
    number_of_raw_channels = 2
    number_of_spec_channels = 4
    window_duration = 10
    spectrogram_fsize = 15
    fs = 256
    x = {
        "raw": torch.rand(batch_size, number_of_raw_channels, window_duration * fs),
        "spec": torch.rand(batch_size, number_of_spec_channels,
                           spectrogram_fsize, window_duration * fs),
    }

    input_shapes = {
        "raw": (number_of_raw_channels, window_duration * fs),
        "spec": (number_of_spec_channels, spectrogram_fsize, window_duration * fs),
    }

    # number of classes
    number_of_classes = 3

    # default events
    default_event_sizes = [1 * fs, 2 * fs]

    net = DOSED4(
        input_shapes=input_shapes,
        number_of_classes=number_of_classes,
        detection_parameters={
            "overlap_non_maximum_suppression": 0.5,
            "classification_threshold": 0.5,
        },
        default_event_sizes=default_event_sizes,
    )
    localizations, classifications, localizations_default = net.forward(x)
    number_of_default_events = sum([
        int(window_duration * fs / default_event_size * 2)
        for default_event_size in default_event_sizes]
    )
    assert localizations.shape == (batch_size, number_of_default_events, 2)
    assert classifications.shape == (batch_size, number_of_default_events, number_of_classes + 1)
    assert localizations_default.shape == (number_of_default_events, 2)


def test_save_load():
    batch_size = 32
    number_of_raw_channels = 2
    number_of_spec_channels = 4
    window_duration = 10
    spectrogram_fsize = 15
    fs = 256
    x = {
        "raw": torch.rand(batch_size, number_of_raw_channels, window_duration * fs),
        "spec": torch.rand(batch_size, number_of_spec_channels,
                           spectrogram_fsize, window_duration * fs),
    }
    input_shapes = {
        "raw": [number_of_raw_channels, window_duration * fs],
        "spec": [number_of_spec_channels, spectrogram_fsize, window_duration * fs],
    }

    net_parameters = {
        "input_shapes": input_shapes,
        "number_of_classes": 3,
        "detection_parameters": {
            "overlap_non_maximum_suppression": 0.5,
            "classification_threshold": 0.5,
        },
        "default_event_sizes": [64],
    }
    net = DOSED4(
        **net_parameters
    )
    filename = tempfile.mkdtemp() + "/lol.lol"
    net.save(filename, net_parameters)

    net_loaded, net_parameters_loaded = net.load(filename)

    assert net_parameters_loaded == net_parameters

    net.eval()
    localizations, classifications, localizations_default = net.forward(x)
    net_loaded.eval()
    localizations_, classifications_, localizations_default_ = net_loaded.forward(x)

    assert localizations.tolist() == localizations_.tolist()
    assert classifications.tolist() == classifications_.tolist()
    assert localizations_default.tolist() == localizations_default_.tolist()


def test_nelement():
    net_parameters = {
        "input_shapes": {"raw": [1, 20]},
        "number_of_classes": 3,
        "detection_parameters": {
            "overlap_non_maximum_suppression": 0.5,
            "classification_threshold": 0.5,
        },
        "default_event_sizes": [10],
    }
    net = DOSED4(
        **net_parameters
    )
    assert net.nelement == 284
