from collections import OrderedDict

import torch.nn as nn

from ..functions import Detection
from .base import BaseNet, get_overlerapping_default_events


class DOSED2(BaseNet):

    def __init__(
        self,
        input_shape,
        number_of_classes,
        detection_parameters,
        default_event_sizes,
        k_max=8,
    ):
        super(DOSED2, self).__init__()
        self.number_of_channels, self.window_size = input_shape
        self.number_of_classes = number_of_classes + 1  # eventness, real events

        detection_parameters["number_of_classes"] = self.number_of_classes
        self.detector = Detection(**detection_parameters)

        self.k_max = 8

        # Localizations to default tensor
        self.localizations_default = get_overlerapping_default_events(
            window_size=self.window_size,
            default_event_sizes=default_event_sizes
        )

        # model
        self.spatial_filtering = None
        if self.number_of_channels > 1:
            self.spatial_filtering = nn.Conv2d(
                in_channels=1,
                out_channels=self.number_of_channels,
                kernel_size=(self.number_of_channels, 1),
                padding=0)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict([
                        ("conv_{}".format(k), nn.Conv2d(
                            in_channels=4 * (2 ** (k - 1)) if k > 1 else 1,
                            out_channels=4 * (2 ** k),
                            kernel_size=(1, 3),
                            padding=(0, 1)
                        )),
                        ("batchnorm_{}".format(k), nn.BatchNorm2d(4 * (2 ** k))),
                        ("relu_{}".format(k), nn.ReLU()),
                        ("max_pooling_{}".format(k), nn.MaxPool2d(kernel_size=(1, 2))),
                    ])
                ) for k in range(1, self.k_max + 1)
            ]
        )
        self.localizations = nn.Conv2d(
            in_channels=4 * (2 ** self.k_max),
            out_channels=2 * len(self.localizations_default),
            kernel_size=(self.number_of_channels, int(self.window_size / (2 ** (self.k_max)))),
            padding=(0, 0),
        )

        self.classifications = nn.Conv2d(
            in_channels=4 * (2 ** self.k_max),
            out_channels=self.number_of_classes * len(self.localizations_default),
            kernel_size=(self.number_of_channels, int(self.window_size / (2 ** (self.k_max)))),
            padding=(0, 0),
        )

    def forward(self, x):
        batch = x.size(0)
        x = x.view(batch, 1, self.number_of_channels, -1)

        if self.spatial_filtering:
            x = self.spatial_filtering(x)
            x = x.transpose(2, 1)

        for block in self.blocks:
            x = block(x)

        localizations = self.localizations(x).squeeze().view(batch, -1, 2)
        classifications = self.classifications(x).squeeze().view(batch, -1, self.number_of_classes)

        return localizations, classifications, self.localizations_default
