[![CircleCI](https://circleci.com/gh/Dreem-Organization/dosed.svg?style=svg&circle-token=7b6f5fd8d3db49d25417b269c601296b7eebd64f)](https://circleci.com/gh/Dreem-Organization/dosed)

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


#### 2. Minimum example

A minimum example is provided in the folder */minimum\_example* directory.

Running the script ipython notebook *download_and_data_format_explanation.ipynb* or run `make download_example` to download, pre-processes training data.

##### H5 data format

To work with different datasets, and hence data format, we first require you to convert you original
data and annotation into H5 files for each record. *download_and_data_format_explanation.ipynb* and *to_h5.py* provides detailed explanation and an example of that process.

Required structure for the .h5 files is the following:

```
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

#### Training and testing

The jupyter notebook *train\_and\_evaluate\_dosed.ipynb* goes through the training process in detail, describing all important training parameters. It also explains how to generate predictions, and provides a plot of a spindle detection.
