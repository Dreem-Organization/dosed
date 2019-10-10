import random

from dosed.utils import merge_events


def test_merge_events():
    events = [
        (20, 50),  # 0
        (10, 50),  # 1
        (25, 42),  # 2
        (30, 60),  # 3

        (75, 85),  # 4
        (72, 87),  # 5
        (76, 85),  # 6
        (80, 90),  # 7

        (110, 120),  # 8
        (100, 110),  # 9
        (120, 130),  # 10

        (0.1, 0.19),  # 11

        (0.2, 0.4),  # 12
        (0.25, 0.33),  # 13
        (0.35, 0.55),  # 14
    ]

    random.shuffle(events)

    merged_events = merge_events(events)
    true_merged_events = [(10, 60), (72, 90), (100, 130), (0.1, 0.19), (0.2, 0.55)]

    assert len(merged_events) == len(true_merged_events) == 5
    assert set(merged_events) == set(true_merged_events)
