import shutil
import pytest
import time
import os
from unittest.mock import patch

from dosed.datasets import BalancedEventDataset, EventDataset, get_train_validation_test


@pytest.fixture
def h5_directory():
    return "./tests/test_files/h5/"


@pytest.fixture
def records(h5_directory):
    train, validation, test = get_train_validation_test(h5_directory, 50, 50, seed=2008)
    return train + validation + test


@pytest.fixture
def signals():
    return [
        {'name': 'eeg_raw',
         'signals': [
             {'name': 'eeg_bandpass',
              'signals': [{'h5_paths': ['/eeg_0'],
                           'fs': 64}],
              'fs': 64,
              'preprocessing':[
                  {"name": "bandpass",
                   "args": {
                       "frequency_band": [0.1, 0.5],
                       "order": 4,
                       "type": "butter"
                   }},
                  {"name": "clip_and_normalize",
                   "args": {
                       "min_value": -150,
                       "max_value": 150,
                   }}
              ]
              },
             {'name': 'eeg_highpass',
              'signals': [{'h5_paths': ['/eeg_0'],
                           'fs': 64}],
              'fs': 64,
              'preprocessing':[
                  {"name": "highpass",
                   "args": {
                       "frequency_cut": 0.5,
                       "order": 4,
                       "type": "butter"
                   }},
                  {"name": "clip",
                   "args": {
                           "max_value": 150,
                   }}
              ]
              },
             {'name': 'eeg_lowpass',
              'signals': [{'h5_paths': ['/eeg_0'],
                           'fs': 64}],
              'fs': 64,
              'preprocessing':[
                  {"name": "lowpass",
                   "args": {
                       "frequency_cut": 0.1,
                       "order": 4,
                       "type": "butter"
                   }},
                  {"name": "mask_clip_and_normalize",
                   "args": {
                           "min_value": -150,
                           "max_value": 150,
                           "mask_value": -1,
                   }}
              ]
              },
         ],
         'fs': 32,
         'preprocessing': []
         },
        {'name': 'eeg_spectrogram',
         'signals': [{'h5_paths': ['/eeg_1'],
                      'fs': 64}],
         'fs': 64,
         'preprocessing': [
             {'name': 'spectrogram',
                 'args': {
                     "nperseg": 8,
                     "nfft": 8,
                     "temporal_downsampling": 1,
                     "frequential_downsampling": 1,
                     "padded": True,
                 }},
             {"name": "clip_and_normalize",
                 "args": {
                     "min_value": -150,
                     "max_value": 150,
                 }},
         ]
         }
    ]


@pytest.fixture
def events():
    return [
        {
            "name": "spindle",
            "h5_path": "spindle",
        }
    ]


@pytest.fixture
def cache_directory():
    return "./tests/test_files/h5/.cache"


def test_dataset(signals, events, h5_directory, records):

    window = 2

    dataset = EventDataset(
        h5_directory=h5_directory,
        signals=signals,
        events=events,
        records=sorted(records),
        window=window,
        minimum_overlap=0.5,
        transformations=lambda x: x
    )

    assert len(dataset) == 360

    signals, events = dataset[0]

    for signal_type, signal in signals.items():
        assert signal_type in ["eeg_raw", "eeg_spectrogram"]
        fs = dataset.fs[signal_type]
        if signal_type == "eeg_raw":
            assert tuple(signals[signal_type].shape) == (3, int(window * fs))
        elif signal_type == "eeg_spectrogram":
            assert tuple(signals[signal_type].shape) == (1, 5, int(window * fs))

    if "eeg_spectrogram" not in signals.keys():
        assert signals["eeg_raw"][0][6].tolist() == -0.11056432873010635
    else:
        assert signals["eeg_raw"][0][6].tolist() == -0.008551005274057388
        assert signals["eeg_spectrogram"][0][4][6].tolist() == 0.0006360001862049103


def test_balanced_dataset_ratio_1(h5_directory, signals, events, records):

    dataset = BalancedEventDataset(
        h5_directory=h5_directory,
        signals=signals,
        events=events,
        window=1,
        records=None,
        minimum_overlap=0.5,
        transformations=lambda x: x,
        ratio_positive=1,
    )
    number_of_events = 0
    for i in range(len(dataset)):
        signals, events_data = dataset[i]
        for signal_type, signal in signals.items():
            assert signal_type in ["eeg_raw", "eeg_spectrogram"]
            fs = dataset.fs[signal_type]
            if signal_type == "eeg_raw":
                assert tuple(signals[signal_type].shape) == (3, int(fs))
            elif signal_type == "eeg_spectrogram":
                assert tuple(signals[signal_type].shape) == (1, 5, int(fs))

        if len(events_data) != 0:
            assert events_data.shape[1] == 3
            number_of_events += 1
    assert number_of_events == len(dataset), number_of_events / len(dataset)

    number_of_events = sum(
        [len(dataset.get_record_events(record)[0]) for record in records]
    )
    assert number_of_events == len(dataset) == 103

    assert len(list(dataset.get_record_batch(records[0], 17))) == 22


def test_balanced_dataset_ratio_0(h5_directory, signals, events, records):
    dataset = BalancedEventDataset(
        h5_directory=h5_directory,
        signals=signals,
        events=events,
        window=1,
        records=None,
        minimum_overlap=0.5,
        transformations=lambda x: x,
        ratio_positive=0,
    )

    assert len(dataset) == 103

    nb_without_event = 0
    for i in range(len(dataset)):
        signal, events_data = dataset[i]
        nb_without_event += int(len(events_data) == 0)
    assert nb_without_event == len(dataset), nb_without_event / len(dataset)


def mock_clip_and_normalize(fs, signal_size, window, input_shape, min_value, max_value):
    def clipper(x, min_value=min_value, max_value=max_value):
        time.sleep(1)
        return x, fs, signal_size, input_shape
    return clipper


normalizer = {
    "clip_and_normalize": mock_clip_and_normalize
}


@patch.dict("dosed.utils.data_from_h5.dict_filters", normalizer)
def test_parallel_is_faster(h5_directory, signals, events, records, cache_directory):

    dataset_parameters = {
        "h5_directory": h5_directory,
        "signals": signals,
        "events": events,
        "window": 1,
        "records": None,
        "minimum_overlap": 0.5,
        "ratio_positive": 0.5,
        "cache_data": False,
    }

    shutil.rmtree(cache_directory, ignore_errors=True)
    t1 = time.time()
    BalancedEventDataset(
        n_jobs=-1,
        **dataset_parameters
    )
    t1 = time.time() - t1

    shutil.rmtree(cache_directory, ignore_errors=True)
    t2 = time.time()
    BalancedEventDataset(
        n_jobs=1,
        **dataset_parameters,
    )
    t2 = time.time() - t2

    assert t2 > t1


def test_cache_is_faster(h5_directory, signals, events, records, cache_directory):
    dataset_parameters = {
        "h5_directory": h5_directory,
        "signals": signals,
        "events": events,
        "window": 1,
        "records": None,
        "minimum_overlap": 0.5,
        "ratio_positive": 0.5,
    }

    shutil.rmtree(cache_directory, ignore_errors=True)
    t1 = time.time()
    BalancedEventDataset(
        cache_data=True,
        **dataset_parameters
    )
    t1 = time.time() - t1

    t2 = time.time()
    BalancedEventDataset(
        cache_data=True,
        **dataset_parameters,
    )
    t2 = time.time() - t2

    assert t2 < t1


def test_cache_no_cache(h5_directory, signals, events, records, cache_directory):
    dataset_parameters = {
        "h5_directory": h5_directory,
        "signals": signals,
        "events": events,
        "window": 1,
        "records": None,
        "minimum_overlap": 0.5,
        "ratio_positive": 0.5,
        "n_jobs": -1,
    }

    shutil.rmtree(cache_directory, ignore_errors=True)
    BalancedEventDataset(
        cache_data=False,
        **dataset_parameters
    )
    assert not os.path.isdir(cache_directory)

    BalancedEventDataset(
        cache_data=True,
        **dataset_parameters,
    )
    assert os.path.isdir(cache_directory)


def test_plot(h5_directory, signals, events, records):
    dataset = BalancedEventDataset(
        h5_directory=h5_directory,
        signals=signals,
        events=events,
        window=1,
        records=None,
        minimum_overlap=0.5,
        transformations=lambda x: x,
        ratio_positive=0,
    )

    dataset.plot(5, {"eeg_raw": [1, 2], "eeg_spectrogram": [0]})
