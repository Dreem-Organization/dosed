import random
import os

import torch


def collate(batch):
    """collate fn because unconsistent number of events"""
    batch_events = []
    batch_signals = dict()
    for signals, events in batch:
        for signal_name, signal in signals.items():
            batch_signals.setdefault(signal_name, []).append(signal)
        batch_events.append(events)

    for signal_name, signal in batch_signals.items():
        batch_signals[signal_name] = torch.stack(signal, 0)

    return batch_signals, batch_events


def get_train_validation_test(h5_directory,
                              percent_test,
                              percent_validation,
                              black_list=[],
                              seed=None):

    records = [x for x in os.listdir(h5_directory)
               if (x != ".cache" and x[-2:] == "h5") and x not in black_list]

    random.seed(seed)
    index_test = int(len(records) * percent_test / 100)
    random.shuffle(records)
    test = records[:index_test]
    records_train = records[index_test:]

    index_validation = int(len(records_train) * percent_validation / 100)
    random.shuffle(records_train)
    validation = records_train[:index_validation]
    train = records_train[index_validation:]

    return train, validation, test
