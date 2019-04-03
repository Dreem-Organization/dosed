import json

from dosed.datasets import BalancedEventDataset, EventDataset, get_train_validation_test


def test_dataset():
    data_index_filename = "./tests/test_files/memmap/index.json"
    train, validation, test = get_train_validation_test(data_index_filename, 50, 50)

    window = 1  # in seconds

    dataset = EventDataset(
        data_index_filename=data_index_filename,
        records=train + validation + test,
        window=window,
        transformations=lambda x: x
    )

    signal, events = dataset[0]

    assert tuple(signal.shape) == (2, 64)

    assert len(dataset) == 720


def test_balanced_dataset():
    data_index_filename = "./tests/test_files/memmap/index.json"
    index = json.load(open(data_index_filename))
    records = index["records"]
    window = 1  # in seconds

    dataset = BalancedEventDataset(
        data_index_filename=data_index_filename,
        records=records,
        window=window,
        ratio_positive=1,
        transformations=lambda x: x
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

    dataset = BalancedEventDataset(
        data_index_filename=data_index_filename,
        records=records,
        window=window,
        ratio_positive=0,
        transformations=lambda x: x
    )

    signal, events = dataset[0]

    assert len(events) == 0
