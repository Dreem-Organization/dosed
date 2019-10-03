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


class Embedding1D(nn.Module):
    def __init__(self, input_shape, layers):
        super(Embedding1D, self).__init__()


class DOSED4(BaseNet):

    def __init__(self,
                 input_shapes,
                 window_size,
                 number_of_classes,
                 detection_parameters,
                 default_event_sizes,
                 convolution_layers={
                     "spectrogram": "[[2 , (2,2), (1,1), (2,0), (2,2)]] + "
                                    "[[4 , (2,2), (1,1), (2,0), (2,2)]]",
                     "raw": "[[2 , (5,), (1,), (0,), (2,)]] + "
                            "[[4 , (5,), (1,), (0,), (2,)]]"
                 },
                 pdrop=0.1,
                 fs=256):
        """
             - conv_spec_layers : layers for data as spectrogram
             - conv_raw_layers : layers for raw data
            Both are a list of layers, where each layer is a tuple :
            (channels, kernel, stride,padding, pooling)
        """

        super(DOSED4, self).__init__()
        self.input_shapes = input_shapes
        self.window_size = window_size
        self.number_of_classes = number_of_classes + 1  # eventless, real events

        detection_parameters["number_of_classes"] = self.number_of_classes
        self.detector = Detection(**detection_parameters)

        self.convolution_layers = {block_name: eval(convolution_layer, {}, {})
                                   for block_name, convolution_layer in convolution_layers.items()}
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
        self.blocks_model = nn.ModuleDict()
        for block_name in self.convolution_layers:
            layers = self.convolution_layers[block_name]
            if len(self.input_shapes[block_name]) == 3:
                self.blocks_model[block_name] = nn.ModuleList(
                    [
                        nn.Sequential(
                            OrderedDict([
                                ("conv_{}".format(k), nn.Conv2d(
                                    in_channels=layers[k - 1][0]
                                    if k > 0 else self.input_shapes[block_name][0],
                                    out_channels=layers[k][0],
                                    kernel_size=layers[k][1],
                                    stride=layers[k][2],
                                    padding=layers[k][3],
                                    bias=True
                                )),
                                ("batchnorm_{}".format(k), nn.BatchNorm2d(layers[k][0])),
                                ("dropout_{}".format(k), nn.Dropout(self.pdrop)),
                                ("relu_{}".format(k), nn.ReLU()),
                                ("max_pooling_{}".format(k), nn.MaxPool2d(
                                    kernel_size=layers[k][4])),
                            ])
                        ) for k in range(len(layers))
                    ]
                )
            else:
                self.blocks_model[block_name] = nn.ModuleList(
                    [
                        nn.Sequential(
                            OrderedDict([
                                ("conv_{}".format(k), nn.Conv1d(
                                    in_channels=layers[k - 1][0]
                                    if k > 0 else self.input_shapes[block_name][0],
                                    out_channels=layers[k][0],
                                    kernel_size=layers[k][1],
                                    stride=layers[k][2],
                                    padding=layers[k][3],
                                    bias=True
                                )),
                                ("batchnorm_{}".format(k), nn.BatchNorm1d(layers[k][0])),
                                ("dropout_{}".format(k), nn.Dropout(self.pdrop)),
                                ("relu_{}".format(k), nn.ReLU()),
                                ("max_pooling_{}".format(k), nn.MaxPool1d(
                                    kernel_size=layers[k][4])),
                            ])
                        ) for k in range(len(layers))
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

        in_channels = set([self.convolution_layers[block_name][-1][0]
                           for block_name in self.convolution_layers])
        assert len(in_channels) == 1, in_channels
        in_channels = in_channels.pop()

        kernel_size = 0
        for block_name in self.convolution_layers:
            input_shape = self.input_shapes[block_name]
            if len(input_shape) == 3:
                fsz_final = conv(input_shape[-2], self.convolution_layers[block_name], dim=0)
                tsz_final = conv(input_shape[-1], self.convolution_layers[block_name], dim=1)
                kernel_size += tsz_final * fsz_final
            else:
                tsz_final = conv(input_shape[-1], self.convolution_layers[block_name], dim=0)
                kernel_size += tsz_final

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

    def flatten(self, x):
        if len(x.shape) == 4:
            bsz, csz, fsz, tsz = x.shape
            x = x.view(bsz, csz, fsz * tsz)
        return x

    def reduce(self, x):
        return torch.cat(tuple(x.values()), 2)

    def forward(self, x):
        x = {block_name: x[block_name] for block_name in self.blocks_model}

        for block_name, blocks in self.blocks_model.items():
            for block in blocks:
                x[block_name] = block(x[block_name])
            x[block_name] = self.flatten(x[block_name])

        x = self.reduce(x)

        batch = x.size(0)

        localizations = self.localizations(x).squeeze().view(batch, -1, 2)
        classifications = self.classifications(x).squeeze().view(batch, -1, self.number_of_classes)

        return localizations, classifications, self.localizations_default

    def print_info_architecture(self, fs):
        pass
