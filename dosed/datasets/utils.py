import torch
import random
import json


def collate(batch):
    """collate fn because unconsistent number of events"""
    batch_events = []
    batch_eegs = []
    for eeg, events in batch:
        batch_eegs.append(eeg)
        batch_events.append(events)
    return torch.stack(batch_eegs, 0), batch_events


def get_train_validation_test(data_index_filename,
                              percent_test,
                              percent_validation,
                              seed_test=2018,
                              seed_validation=0):
    data_index = json.load(open(data_index_filename, "r"))

    records = data_index["records"]

    random.seed(seed_test)
    index_test = int(len(records) * percent_test / 100)
    random.shuffle(records)
    test = records[:index_test]
    records_train = records[index_test:]

    random.seed(seed_validation)
    index_validation = int(len(records_train) * percent_validation / 100)
    random.shuffle(records_train)
    validation = records_train[:index_validation]
    train = records_train[index_validation:]

    return train, validation, test
