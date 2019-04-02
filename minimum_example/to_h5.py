import json
import h5py
import pyedflib
import tqdm
import os

from settings import MINIMUM_EXAMPLE_SETTINGS

print("\n Converting EDF and annotations to standard H5 file")
base_directory = MINIMUM_EXAMPLE_SETTINGS["download_directory"]

records = [
    x.split(".")[0] for x in os.listdir(base_directory) if x[-3:] == "edf"
]

if not os.path.isdir(MINIMUM_EXAMPLE_SETTINGS["h5_directory"]):
    os.mkdir(MINIMUM_EXAMPLE_SETTINGS["h5_directory"])

for record in tqdm.tqdm(records):
    edf_filename = base_directory + record + ".edf"
    spindle_filename = base_directory + record + "_spindle.json"
    h5_filename = '{}/{}.h5'.format(MINIMUM_EXAMPLE_SETTINGS["h5_directory"], record)

    with h5py.File(h5_filename, 'w') as h5:

        # Taking care of spindle annotations
        spindles = [
            (x["start"], x["end"] - x["start"]) for x in json.load(open(spindle_filename))
        ]
        starts, durations = list(zip(*spindles))
        h5.create_group("spindle")
        h5.create_dataset("spindle/start", data=starts)
        h5.create_dataset("spindle/duration", data=durations)

        # Extract signals
        with pyedflib.EdfReader(edf_filename) as f:
            labels = f.getSignalLabels()
            frequencies = f.getSampleFrequencies().astype(int).tolist()

            for i, (label, frequency) in enumerate(zip(labels, frequencies)):

                path = "{}".format(label.lower())
                data = f.readSignal(i)
                h5.create_dataset(path, data=data)
                h5[path].attrs["fs"] = frequency
