from dosed.utils import h5_to_memmap

from settings import MINIMUM_EXAMPLE_SETTINGS

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

print("\nConverting standard H5 files to memmaps usable for training")

h5_to_memmap(h5_directory=MINIMUM_EXAMPLE_SETTINGS["h5_directory"],
             memmap_directory=MINIMUM_EXAMPLE_SETTINGS["memmap_directory"],
             signals=signals,
             events=events,
             parallel=False)
