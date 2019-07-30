import random
import os

import torch


def collate(batch):
    """collate fn because unconsistent number of events"""
    batch_events = []
    batch_eegs = []
    for eeg, events in batch:
        batch_eegs.append(eeg)
        batch_events.append(events)
    return torch.stack(batch_eegs, 0), batch_events


def get_train_validation_test(h5_directory,
                              percent_test,
                              percent_validation,
                              seed=None):

    records = [x for x in os.listdir(h5_directory) if (x != ".cache" and x[-2:] == "h5")]

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
