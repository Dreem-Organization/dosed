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
        {
            'h5_path': '/eeg_0',
            'fs': 64,
            'processing': {
                "type": "clip_and_normalize",
                "args": {
                        "min_value": -150,
                    "max_value": 150,
                }
            }
        },
        {
            'h5_path': '/eeg_1',
            'fs': 64,
            'processing': {
                "type": "clip_and_normalize",
                "args": {
                        "min_value": -150,
                    "max_value": 150,
                }
            }
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
        fs=64,
        minimum_overlap=0.5,
        transformations=lambda x: x
    )

    signal, events = dataset[0]

    assert tuple(signal.shape) == (2, int(window * dataset.fs))

    assert len(dataset) == 360

    assert signal[0][6].tolist() == -0.11056432873010635


def test_balanced_dataset_ratio_1(h5_directory, signals, events, records):

    dataset = BalancedEventDataset(
        h5_directory=h5_directory,
        signals=signals,
        events=events,
        window=1,
        fs=64,
        records=None,
        minimum_overlap=0.5,
        transformations=lambda x: x,
        ratio_positive=1,
    )

    signal, events_data = dataset[0]

    assert tuple(signal.shape) == (2, 64)
    assert events_data.shape[1] == 3

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
        fs=64,
        records=None,
        minimum_overlap=0.5,
        transformations=lambda x: x,
        ratio_positive=0,
    )

    signal, events_data = dataset[0]

    assert len(events_data) == 0
    assert len(dataset) == 103


def mock_clip_and_normalize(min_value, max_value):
    def clipper(x, min_value=min_value, max_value=max_value):
        # time.sleep(1)
        return x
    return clipper


normalizer = {
    "clip_and_normalize": mock_clip_and_normalize
}


@patch("dosed.utils.data_from_h5.normalizers", normalizer)
def test_parallel_is_faster(h5_directory, signals, events, records, cache_directory):

    dataset_parameters = {
        "h5_directory": h5_directory,
        "signals": signals,
        "events": events,
        "window": 1,
        "fs": 64,
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
        "fs": 64,
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
        "fs": 64,
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
