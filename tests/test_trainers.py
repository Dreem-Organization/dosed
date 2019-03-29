from dosed.datasets import BalancedEventDataset
from dosed.models import DOSED3
from dosed.trainers import trainers
import json
import torch


def test_balanced_dataset():
    data_index_filename = "./tests/test_files/memmap/index.json"
    index = json.load(open(data_index_filename))
    records = index["records"]
    window = 1  # in seconds
    device = torch.device("cuda")

    dataset = BalancedEventDataset(
        data_index_filename=data_index_filename,
        records=records,
        window=window,
        ratio_positive=1,
        transformations=lambda x: x
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
