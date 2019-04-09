import torch

from dosed.datasets import BalancedEventDataset
from dosed.models import DOSED3
from dosed.trainers import trainers


def test_full_training():
    h5_directory = "./tests/test_files/h5/"

    window = 1  # in seconds

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

    device = torch.device("cuda")

    dataset = BalancedEventDataset(
        h5_directory=h5_directory,
        signals=signals,
        events=events,
        window=window,
        downsampling_rate=1,
        minimum_overlap=0.5,
        transformations=lambda x: x,
        ratio_positive=0.5,
    )

    # default events
    default_event_sizes = [1 * dataset.fs, 2 * dataset.fs]

    net = DOSED3(
        input_size=(dataset.input_size, 2),
        number_of_classes=dataset.number_of_classes,
        detection_parameters={
            "overlap_non_maximum_suppression": 0.5,
            "classification_threshold": 0.5,
        },
        default_event_sizes=default_event_sizes,
    )

    optimizer_parameters = {
        "lr": 5e-3,
        "weight_decay": 1e-8,
    }
    loss_specs = {
        "type": "worst_negative_mining",
        "parameters": {
            "number_of_classes": dataset.number_of_classes,
            "device": device,
        }
    }

    trainer = trainers["adam"](
        net,
        optimizer_parameters=optimizer_parameters,
        loss_specs=loss_specs,
        epochs=2,
    )

    best_net_train, best_metrics_train, best_threshold_train = trainer.train(
        dataset,
        dataset,
        batch_size=12,
    )

    best_net_train.predict_dataset(
        dataset,
        best_threshold_train,
        batch_size=2
    )
