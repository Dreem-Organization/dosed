"""Transform a folder with h5 files into a dataset for dosed"""

import os
import json
import numpy as np

import tqdm
import h5py
from joblib import Parallel, delayed

from ..preprocessings import normalizers


def h5_to_memmap(h5_directory,
                 memmap_directory,
                 signals,
                 events,
                 parallel=True,
                 ):

    records = [h5_directory + filename for filename in os.listdir(h5_directory)
               if filename[-3:] == ".h5"]

    # Will contain useful informations for training dataset class
    index = {
        "signals": signals,
        "events": events,
        "records": [memmap_directory + record[:-3].split("/")[-1] for record
                    in records]
    }

    # Check sampling frequencies
    sampling_frequencies = set(
        [h5py.File(record)[signal["h5_path"]].attrs["fs"]
         for record in records for signal in signals]
    )
    assert len(sampling_frequencies) == 1
    index["sampling_frequency"] = float(sampling_frequencies.pop())

    # check event names
    assert len(set([event["name"] for event in events])) == 1

    if parallel is True:
        Parallel(n_jobs=-1, verbose=101)(
            delayed(process_record)(record,
                                    signals,
                                    events,
                                    memmap_directory)
            for record in records)
    else:
        for record in tqdm.tqdm(records):
            process_record(record,
                           signals,
                           events,
                           memmap_directory)

    json.dump(index, open(memmap_directory + "index.json", "w"), indent=4)


def process_record(record,
                   signals,
                   events,
                   memmap_directory):
    """processes one record from h5 to memmap"""

    with h5py.File(record, "r") as h5:
        filename_base = memmap_directory + record.split("/")[-1].split(".")[0]

        # Check that all signals have the same size and sampling frequency
        signals_size = set([int(h5[signal["h5_path"]].size) for signal in signals])
        assert len(signals_size) == 1, "Different signal sizes found!"
        signal_size = signals_size.pop()

        # Create data memmap from signals
        filename = "{}_signals.mm".format(filename_base)
        data_mm = np.memmap(filename,
                            dtype='float32',
                            mode='w+',
                            shape=(len(signals), signal_size))

        # Fill the memmaps using the normalized data
        for i, signal in enumerate(signals):
            normalizer = normalizers[signal['processing']["type"]](**signal['processing']['args'])
            data_mm[i, :] = normalizer(h5[signal["h5_path"]][:])

        # For each event create a memmap
        for index_event, event in enumerate(events):

            starts = h5[event["h5_path"]]["start"]
            durations = h5[event["h5_path"]]["duration"]
            number_of_events = len(starts)

            assert len(starts) == len(durations), "Inconsistents event durations and starts"

            filename = "{}_{}.mm".format(filename_base,
                                         event["name"])
            if number_of_events == 0:
                continue

            starts_durations = np.memmap(filename,
                                         dtype='float32',
                                         mode='w+',
                                         shape=(2, number_of_events))

            starts_durations[0, :] = starts
            starts_durations[1, :] = durations
