import warnings
from collections import OrderedDict

import torch.nn as nn
import torch

from ..functions import Detection
from .base import BaseNet, get_overlerapping_default_events


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class DOSED4(BaseNet):

    def __init__(self,
                 input_shape,
                 number_of_classes,
                 detection_parameters,
                 default_event_sizes,
                 conv_spec_layers="[[16 , (2,2), (1,1), (2,0), (2,2)]] + "
                                  "[[64, (2,2), (1,1), (2,0), (2,2)]]",
                 conv_raw_layers="[[4 , (5,), (1,), (0,), (2,)]] + "
                                 "[[8 , (5,), (1,), (0,), (2,)]] + "
                                 "[[16, (5,), (1,), (0,), (2,)]] + "
                                 "[[32, (5,), (1,), (0,), (2,)]] + "
                                 "[[64, (5,), (1,), (0,), (2,)]]",
                 pdrop=0.1,
                 fs=256):

        """
             - conv_spec_layers : layers for data as spectrogram
             - conv_raw_layers : layers for raw data
            Both are a list of layers, where each layer is a tuple :
            (channels, kernel, stride,padding, pooling)
        """

        super(DOSED4, self).__init__()
        self.raw_channels = 0
        self.spec_channels = 0
        if "raw" in input_shape.keys():
            self.raw_channels, self.window_size = input_shape["raw"]
        if "spec" in input_shape.keys():
            self.spec_channels, self.fsz, self.window_size = input_shape["spec"]
        self.number_of_classes = number_of_classes + 1  # eventless, real events

        detection_parameters["number_of_classes"] = self.number_of_classes
        self.detector = Detection(**detection_parameters)

        self.conv_spec_layers = eval(conv_spec_layers, {}, {})
        self.conv_raw_layers = eval(conv_raw_layers, {}, {})
        self.pdrop = pdrop

        if max(default_event_sizes) > self.window_size:
            warnings.warn("Detected default_event_sizes larger than"
                          " input_shape! Consider reducing them")

        # Localizations to default tensor
        self.localizations_default = get_overlerapping_default_events(
            window_size=self.window_size,
            default_event_sizes=default_event_sizes
        )

        # model

        if self.spec_channels > 0:
            self.blocks_spectro = nn.ModuleList(
                [
                    nn.Sequential(
                        OrderedDict([
                            ("conv_{}".format(k), nn.Conv2d(
                                in_channels=self.conv_spec_layers[k - 1][0]
                                if k > 0 else self.spec_channels,
                                out_channels=self.conv_spec_layers[k][0],
                                kernel_size=self.conv_spec_layers[k][1],
                                stride=self.conv_spec_layers[k][2],
                                padding=self.conv_spec_layers[k][3],
                                bias=True
                            )),
                            ("batchnorm_{}".format(k), nn.BatchNorm2d(self.conv_spec_layers[k][0])),
                            ("dropout_{}".format(k), nn.Dropout(self.pdrop)),
                            ("relu_{}".format(k), nn.ReLU()),
                            ("max_pooling_{}".format(k), nn.MaxPool2d(
                                kernel_size=self.conv_spec_layers[k][4])),
                        ])
                    ) for k in range(len(self.conv_spec_layers))
                ]
            )
        if self.raw_channels > 0:
            self.blocks_raw = nn.ModuleList(
                [
                    nn.Sequential(
                        OrderedDict([
                            ("conv_{}".format(k), nn.Conv1d(
                                in_channels=self.conv_raw_layers[k - 1][0]
                                if k > 0 else self.raw_channels,
                                out_channels=self.conv_raw_layers[k][0],
                                kernel_size=self.conv_raw_layers[k][1],
                                stride=self.conv_raw_layers[k][2],
                                padding=self.conv_raw_layers[k][3],
                                bias=True
                            )),
                            ("batchnorm_{}".format(k), nn.BatchNorm1d(self.conv_raw_layers[k][0])),
                            ("dropout_{}".format(k), nn.Dropout(self.pdrop)),
                            ("relu_{}".format(k), nn.ReLU()),
                            ("max_pooling_{}".format(k), nn.MaxPool1d(
                                kernel_size=self.conv_raw_layers[k][4])),
                        ])
                    ) for k in range(len(self.conv_raw_layers))
                ]
            )

        def conv(x, conv_layers, dim=0):
            """Compute the final size of an input through the model"""
            for i in range(len(conv_layers)):
                kernel = conv_layers[i][1][dim]
                stride = conv_layers[i][2][dim]
                pad = conv_layers[i][3][dim]
                pool = conv_layers[i][4][dim]
                x = ((x - kernel + 2 * pad) // stride + 1) // pool
            return x

        if self.spec_channels == 0:
            kernel_size = conv(self.window_size, self.conv_raw_layers, dim=0)
            in_channels = self.conv_raw_layers[-1][0]
        else:
            fsz_final = conv(self.fsz, self.conv_spec_layers, dim=0)
            tsz_spec_final = conv(self.window_size, self.conv_spec_layers, dim=1)
            if self.raw_channels == 0:
                in_channels = self.conv_spec_layers[-1][0]  # * fsz_final
                kernel_size = tsz_spec_final * fsz_final
            else:
                tsz_raw_final = conv(self.window_size, self.conv_raw_layers, dim=0)
                in_channels = self.conv_spec_layers[-1][0]  # * (fsz_final + 1)
                kernel_size = tsz_spec_final * fsz_final + tsz_raw_final

        self.localizations = nn.Conv1d(
            in_channels=in_channels,
            out_channels=2 * len(self.localizations_default),
            kernel_size=kernel_size,
            padding=0,
        )

        self.classifications = nn.Conv1d(
            in_channels=in_channels,
            out_channels=self.number_of_classes * len(self.localizations_default),
            kernel_size=kernel_size,
            padding=0,
        )

        self.print_info_architecture(fs)

    def forward(self, x):

        if self.spec_channels > 0:
            if self.raw_channels > 0:
                x_spectro = x["spec"]
                x_raw = x["raw"]
                for block in self.blocks_spectro:
                    x_spectro = block(x_spectro)
                for block in self.blocks_raw:
                    x_raw = block(x_raw)
                bsz, csz, fsz, tsz = x_spectro.shape
                x_spectro = x_spectro.view(bsz, csz, fsz * tsz)
                x = torch.cat((x_spectro, x_raw), 2)
            else:
                x = x["spec"]
                for block in self.blocks_spectro:
                    x = block(x)
                bsz, csz, fsz, tsz = x.shape
                x = x.view(bsz, csz, fsz * tsz)

        else:
            x = x["raw"]
            for block in self.blocks_raw:
                x = block(x)

        batch = x.size(0)

        localizations = self.localizations(x).squeeze().view(batch, -1, 2)
        classifications = self.classifications(x).squeeze().view(batch, -1, self.number_of_classes)

        return localizations, classifications, self.localizations_default

    def print_info_architecture(self, fs):
        pass
