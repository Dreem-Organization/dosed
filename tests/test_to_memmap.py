import tempfile
import shutil
import os
import json

from dosed.utils import h5_to_memmap


def remove_root(index):
    index["signals"]["size"] = {
        k.split("/")[-1]: v for k, v in index["signals"]["size"].items()
    }
    index["events"][0]["size"] = {
        k.split("/")[-1]: v for k, v in index["events"][0]["size"].items()
    }
    index["records"] = set([x.split("/")[-1] for x in index["records"]])
    return index


def test_h5_to_memmap():
    signals = [
        {
            "name": "signals",
            "h5_paths": [
                {
                    'path': '/eeg_0',
                    'processing': {
                        "type": "clip_and_normalize",
                        "args": {
                            "min_value": -150,
                            "max_value": 150,
                        }
                    }
                },
                {
                    'path': '/eeg_1',
                    'processing': {
                        "type": "clip_and_normalize",
                        "args": {
                            "min_value": -150,
                            "max_value": 150,
                        }
                    }
                }
            ],
            "fs": 64,
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
