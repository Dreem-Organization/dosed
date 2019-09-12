import random
import os

import torch


def collate(batch):
    """collate fn because unconsistent number of events"""
    batch_events = []
    batch_eegs_raw = []
    batch_eegs_spec = []
    for eeg, events in batch:
        if "raw" in eeg:
            batch_eegs_raw.append(eeg["raw"])
        if "spec" in eeg:
            batch_eegs_spec.append(eeg["spec"])
        batch_events.append(events)
    batch_eegs = {}
    if len(batch_eegs_raw) > 0:
        batch_eegs["raw"] = torch.stack(batch_eegs_raw, 0)
    if len(batch_eegs_spec) > 0:
        batch_eegs["spec"] = torch.stack(batch_eegs_spec, 0)

    return batch_eegs, batch_events


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
