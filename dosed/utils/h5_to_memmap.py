"""Transform a folder with h5 files into a dataset for dosed"""

import os
import json
import numpy as np
import h5py
from joblib import Parallel, delayed


def clip(max_value):
    """returns a function to clip data"""

    def clipper(signal_data, max_value=max_value):
        """returns input signal clipped between +/- max_value.
        """
        return np.clip(signal_data, -max_value, max_value)

    return clipper


def clip_and_normalize(min_value, max_value):
    """returns a function to clip and normalize data"""

    def clipper(x, min_value=min_value, max_value=max_value):
        """returns input signal clipped between min_value and max_value
        and then normalized between -0.5 and 0.5.
        """
        x = np.clip(x, min_value, max_value)
        x = ((x - min_value) /
             (max_value - min_value)) - 0.5
        return x

    return clipper


def mask_clip_and_normalize(min_value, max_value, mask_value):
    """returns a function to clip and normalize data"""

    def clipper(x, min_value=min_value, max_value=max_value,
                mask_value=mask_value):
        """returns input signal clipped between min_value and max_value
        and then normalized between -0.5 and 0.5.
        """
        mask = np.ma.masked_equal(x, mask_value)
        x = np.clip(x, min_value, max_value)
        x = ((x - min_value) /
             (max_value - min_value)) - 0.5
        x[mask.mask] = mask_value
        return x

    return clipper


def process_record(record,
                   signals,
                   target_folder_name,
                   index):
    """processes one record from h5 to memmap"""

    print(record)
    info = {}

    with h5py.File(record, "r") as h5:
        filename_base = target_folder_name + record[:-3].split("/")[-1]
        size = int(h5[signals["h5_paths"][0]["path"]].size)
        filename = "{}_{}.mm".format(filename_base, signals["name"])
        info["signal_name"] = filename
        info["signal_size"] = size
        signals_mm = np.memmap(filename,
                               dtype='float32',
                               mode='w+',
                               shape=(len(signals["h5_paths"]), size))
        for i, h5_path in enumerate(signals["h5_paths"]):
            normalizer = normalizers[h5_path['processing']["type"]](
                **h5_path['processing']['args'])
            signals_mm[i, 0:len(h5[h5_path["path"]][:])] = \
                normalizer(h5[h5_path["path"]][:])

        for num_event, event in enumerate(index["events"]):
            info[event["name"]] = {}

            starts = h5[event["h5_path"]]["start"]
            durations = h5[event["h5_path"]]["duration"]
            number_of_events = len(starts)

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

            info[event["name"]]["events_{}".format(num_event)] = \
                number_of_events
            info[event["name"]]["name_{}".format(num_event)] = \
                filename

        return info


normalizers = {
    "clip": clip,
    "clip_and_normalize": clip_and_normalize,
    "mask_clip_and_normalize": mask_clip_and_normalize
}


def h5_to_memmap(h5_directory,
                 memmap_directory,
                 signals,
                 events,
                 parallel=True,
                 ):
    if not os.path.isdir(memmap_directory):
        os.mkdir(memmap_directory)

    records = [h5_directory + filename for filename in os.listdir(h5_directory)
               if filename[-3:] == ".h5"]

    for num_event, event in enumerate(events):
        event["label"] = num_event

    for signal_num, signal in enumerate(signals):

        signal["size"] = {}
        for event in events:
            event["size"] = {}

        index = {
            "signals": signal,
            "events": events,
            "records": [memmap_directory + record[:-3].split("/")[-1] for record
                        in
                        records]
        }

        if parallel is True:
            events_info = Parallel(n_jobs=-1, verbose=101)(
                delayed(process_record)(record,
                                        signal,
                                        memmap_directory,
                                        index)
                for record in records)
        else:
            events_info = []
            for record in records:
                info = process_record(record,
                                      signal,
                                      memmap_directory,
                                      index)
                events_info.append(info)

        for record_num, _ in enumerate(records):
            record_info = events_info[record_num]
            signal["size"][record_info["signal_name"]] = record_info["signal_size"]
            for num_event, event in enumerate(index["events"]):
                for consensus, _ in enumerate(event):
                    event["size"][record_info[event["name"]]["name_{}".format(num_event)]] = \
                        record_info[event["name"]]["events_{}".format(num_event)]

        json.dump(index, open(memmap_directory + "index.json", "w"), indent=4)
