import tempfile
import shutil
import os
import json
import numpy as np

from dosed.utils import h5_to_memmap


def remove_root(index):
    index["records"] = sorted([x.split("/")[-1] for x in index["records"]])
    return index


def test_h5_to_memmap():
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
    memmap_directory = tempfile.mkdtemp() + "/"

    h5_to_memmap(h5_directory="./tests/test_files/h5/",
                 memmap_directory=memmap_directory,
                 signals=signals,
                 events=events,
                 parallel=False)
    assert len(os.listdir(memmap_directory)) == 5
    created_index = remove_root(json.load(open(memmap_directory + "/index.json")))
    test_index = remove_root(json.load(open("./tests/test_files/memmap/index.json")))
    assert created_index == test_index
    shutil.rmtree(memmap_directory)


def test_h5_to_memmap_downsamples():
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
    memmap_directory = tempfile.mkdtemp() + "/"

    downsampling_rate = 6

    h5_to_memmap(h5_directory="./tests/test_files/h5/",
                 memmap_directory=memmap_directory,
                 signals=signals,
                 events=events,
                 parallel=False,
                 downsampling_rate=downsampling_rate)

    test_index = json.load(open("./tests/test_files/memmap/index.json"))
    created_index = json.load(open(memmap_directory + "/index.json"))
    assert created_index["sampling_frequency"] == (
        test_index["sampling_frequency"] / downsampling_rate)
    size_memmap = np.memmap(sorted(test_index["records"])[0] + "_signals.mm",
                            dtype='float32').reshape((len(signals), -1)).shape[-1]
    size_downsampled_memmap = np.memmap(sorted(created_index["records"])[0] + "_signals.mm",
                                        dtype='float32').reshape((len(signals), -1)).shape[-1]
    assert (len(range(0, size_memmap, downsampling_rate)) - size_downsampled_memmap) == 0
    shutil.rmtree(memmap_directory)
