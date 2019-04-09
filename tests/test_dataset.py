from dosed.datasets import BalancedEventDataset, EventDataset, get_train_validation_test


def test_dataset():
    h5_directory = "./tests/test_files/h5/"
    train, validation, test = get_train_validation_test(h5_directory, 50, 50)

    window = 1  # in seconds

    signals = [
        {
            'h5_path': '/eeg_0',
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
            'processing': {
                "type": "clip_and_normalize",
                "args": {
                        "min_value": -150,
                    "max_value": 150,
                }
            }
        }
    ]

    events = [
        {
            "name": "spindle",
            "h5_path": "spindle",
        },
    ]

    dataset = EventDataset(
        h5_directory=h5_directory,
        signals=signals,
        events=events,
        window=window,
        downsampling_rate=1,
        records=train,
        minimum_overlap=0.5,
        transformations=lambda x: x
    )

    signal, events = dataset[0]

    assert tuple(signal.shape) == (2, 64)

    assert len(dataset) == 720

    assert signal[0][6].tolist() == -0.11056432873010635


def test_balanced_dataset():
    h5_directory = "./tests/test_files/h5/"
    train, validation, test = get_train_validation_test(h5_directory, 50, 50)

    records = train + test + validation

    window = 1  # in seconds

    signals = [
        {
            'h5_path': '/eeg_0',
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
            'processing': {
                "type": "clip_and_normalize",
                "args": {
                        "min_value": -150,
                    "max_value": 150,
                }
            }
        }
    ]

    events = [
        {
            "name": "spindle",
            "h5_path": "spindle",
        },
    ]

    dataset = BalancedEventDataset(
        h5_directory=h5_directory,
        signals=signals,
        events=events,
        window=window,
        downsampling_rate=1,
        records=None,
        minimum_overlap=0.5,
        transformations=lambda x: x,
        ratio_positive=1,
    )

    signal, events = dataset[0]

    assert tuple(signal.shape) == (2, 64)
    assert events.shape[1] == 3

    number_of_events = sum(
        [len(dataset.get_record_events(record)[0]) for record in records]
    )
    assert number_of_events == 103

    assert len(list(dataset.get_record_batch(records[0], 17))) == 22

    assert len(dataset) == 103

    signals = [
        {
            'h5_path': '/eeg_0',
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
            'processing': {
                "type": "clip_and_normalize",
                "args": {
                        "min_value": -150,
                    "max_value": 150,
                }
            }
        }
    ]

    events = [
        {
            "name": "spindle",
            "h5_path": "spindle",
        },
    ]

    dataset = BalancedEventDataset(
        h5_directory=h5_directory,
        signals=signals,
        events=events,
        window=window,
        downsampling_rate=1,
        records=None,
        minimum_overlap=0.5,
        transformations=lambda x: x,
        ratio_positive=0,
    )

    signal, events = dataset[0]

    assert len(events) == 0
    assert len(dataset) == 103
