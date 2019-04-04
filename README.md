## Dreem One Shot Event Detector (DOSED)

This repository contains a functional implementation of DOSED, a deep learning method proposed first in:

	Stanislas Chambon, Valentin Thorey, Pierrick J. Arnal, Emmanuel Mignot, Alexandre Gramfort
	A deep learning architecture to detect events in EEG signals during sleep
	IEEE 28th International Workshop on Machine Learning for Signal Processing (MLSP), 2018
	https://arxiv.org/abs/1807.05981

and extended in:

	Stanislas Chambon, Valentin Thorey, Pierrick J. Arnal, Emmanuel Mignot, Alexandre Gramfort.
	DOSED: a deep learning approach to detect multiple sleep micro-events in EEG signal
	https://arxiv.org/abs/1812.04079

### Introduction

DOSED in a deep learning approach to jointly predicts locations, durations and types of events in time series.
It was inspired by computer vision object detectors such as YOLO and SSD and relies on a convolutional neural network that builds a feature representation from raw input signals,
 as well as two modules performing localization and classification respectively. DOSED can be easily adapt to detect events of any sort.

 ![dosed_detection_image](https://github.com/Dreem-Organization/dosed/blob/master/dosed_detection.png)

### Citing DOSED

    @inproceedings{chambon2018deep,
      title={A deep learning architecture to detect events in EEG signals during sleep},
      author={Chambon, Stanislas and Thorey, Valentin and Arnal, Pierrick J and Mignot, Emmanuel and Gramfort, Alexandre},
      booktitle={2018 IEEE 28th International Workshop on Machine Learning for Signal Processing (MLSP)},
      pages={1--6},
      year={2018},
      organization={IEEE}
    }

    @article{chambon2018dosed,
      title={DOSED: a deep learning approach to detect multiple sleep micro-events in EEG signal},
      author={Chambon, Stanislas and Thorey, Valentin and Arnal, Pierrick J and Mignot, Emmanuel and Gramfort, Alexandre},
      journal={arXiv preprint arXiv:1812.04079},
      year={2018}
    }

### Minimum example

The folder */minimum_example* contains all necessary code to train a spindle detection model on EEG signals.

We provide a dataset composed of 21 recordings with two EEG central channels downsampled at 64Hz on which spindles have been annotated. The data was collected at [Dreem](http://www.dreem.com) with a Polysomnography device.

The example works out-of-the-box given the following considerations.

#### 1. Package requirements

Packages detailed in *requirements.txt* need to be installed for the example to work.

A settings.py file is used at the root of the project to manage data location. We provide a default
file that should work out of the box and download data in ROOT_PROJECT/data with:

`cp settings-template.py settings.py`

If you want to store data in another location, change the *settings.py* accordingly.

#### 2. Data

Running the script *download\_and\_format\_data.sh* from the */minimum\_example* directory automatically downloads, pre-processes and saves the EEG data in the correct format for training and testing DOSED at
the location provided in your *settings.py*.

Furthemore, the jupyter notebook *download\_and\_data\_format\_explanation.ipynb* provides detailed explanation about the aforementioned steps and about the data formats used to store the information, together with visualizations of the events under consideration.

##### 2.1 To H5

To work with different datasets, and hence data format, we first require you to convert you original
data and annotation into H5 files for each record. An example is provided with the minimum_example 
going from EDF data format + json annotations into H5 format.

Required structure for the .h5 files is the following:

```javascript
/  # root of the h5 file
-> /path/to/signal_1
   + attribute "fs" # sampling frequency

-> /path/to/signal_2
   + attribute "fs"

-> /path/to/signal_3
   + attribute "fs"

-> ... # add as many signals as desired


-> /path/to/event_1/
   -> /start  # array containing start position of each event with respect to the beginning of the recording (in seconds).
   -> /duration  # array containing duration  of each event (in seconds).

-> /path/to/event_2/
   -> /start
   -> /duration

-> ... # add as many events as desired
```

This code is the only dataset-specific code that you will need to write.

##### 2.2 To Memmap

To Train the model, we first convert generic h5 to memmaps files. This allows:
- Selecting which signals we want to train on
- Selecting which events we want to train on
- Allows multi-threading

The *to_memmap.py* script extracts data from the .h5 files, normalizes them and saves them in memmap files. Two variables define this process. The argument *signals* allows to configurate the minimun and maximum clipping boundaries for each signal. Additionally, the argument *events* allows to specify the name of the event types under consideration and the ground truth (note that .h5 files can contain several ground truth versions).

Configuration of variable *signals* . e.g.

```javascript
    signals = {
        [
            "name": "signals",  # memmap save name
                "h5_paths": [
                    {
                        'path': '/h5_path_to_signal_1',
                        'processing': {
                            "type": "clip_and_normalize",
                            "args": {
                                "min_value": -100,
                                "max_value": 100,
                            }
                        }
                    },
                    {
                        'path': '/h5_path_to_signal_2',
                        'processing': {
                            "type": "clip_and_normalize",
                            "args": {
                                "min_value": -100,
                                "max_value": 100,
                            }
                        }
                    },
                    ...  # add as many signals as desired
                ],

            "fs": 256., # (in Hz) all signals in the set have to be sampled at the same frequency
        ],
        ... # add as many sets of signals as desired
    }
```

and of variable *events*. e.g.

```javascript
    events = [
        {
            "name": "my_events",  # name of the events
                    "h5_path": "/event_1",  # annotations path in the .h5 file
        },
    ]
```

#### 3. Training and testing

The jupyter notebook *train\_and\_evaluate\_dosed.ipynb* goes through the training process in detail, describing all important training parameters. It also explains how to generate predictions, and provides a plot of a spindle detection.
