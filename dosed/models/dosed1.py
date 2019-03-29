"""

"""

import torch
import torch.nn as nn
from dosed.functions import Detection
from collections import OrderedDict
from dosed.models.base import BaseNet


class DOSED1(BaseNet):

    def __init__(
        self,
        input_size,
        number_of_classes,
        detection_parameters,
        duration=256,
        k_max=8,
        rho=2
    ):
        super(DOSED1, self).__init__()
        self.sizes = {}
        self.input_size, self.input_channel_size = input_size
        self.sizes[0] = self.input_size
        self.number_of_classes = number_of_classes + 1  # eventness, real events

        detection_parameters["number_of_classes"] = self.number_of_classes
        self.detector = Detection(**detection_parameters)

        self.localizations_default = []

        self.rho = rho
        self.k_max = k_max
        self.duration = duration
        self.number_of_default_events = self.input_size * self.rho / (2 ** self.k_max)
        assert self.number_of_default_events % 1 == 0

        # model
        self.spatial_filtering = None
        if self.input_channel_size > 1:
            self.spatial_filtering = nn.Conv2d(
                in_channels=1,
                out_channels=self.input_channel_size,
                kernel_size=(self.input_channel_size, 1),
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
            out_channels=2 * self.rho,
            kernel_size=(self.input_channel_size, 3),
            padding=(0, 1),
        )

        self.classifications = nn.Conv2d(
            in_channels=4 * (2 ** self.k_max),
            out_channels=self.number_of_classes * self.rho,
            kernel_size=(self.input_channel_size, 3),
            padding=(0, 1),
        )

        # Localizations to default tensor
        self.localizations_default = torch.Tensor([
            [((2 ** self.k_max)) * (0.5 + i) / (self.rho * self.input_size),
             self.duration / self.input_size] for i in range(int(self.number_of_default_events))
        ]
        )

    def forward(self, x):
        batch = x.size(0)
        x = x.view(batch, 1, self.input_channel_size, -1)

        if self.spatial_filtering:
            x = self.spatial_filtering(x)
            x = x.transpose(2, 1)

        for block in self.blocks:
            x = block(x)
        localizations = self.localizations(x).view(
            batch,
            2 * self.rho,
            int(self.input_size / (2 ** self.k_max))
        ).permute(0, 2, 1).contiguous().view(batch, -1, 2)
        classifications = self.classifications(x).view(
            batch,
            self.number_of_classes * self.rho,
            int(self.input_size / (2 ** self.k_max))
        ).permute(0, 2, 1).contiguous().view(batch, -1, self.number_of_classes)

        return localizations, classifications, self.localizations_default
