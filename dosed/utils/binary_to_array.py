import numpy as np
from collections import deque


def binary_to_array(x):
    """ Return [start, duration] from binary array

    binary_to_array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1])
    [[4, 8], [11, 13]]
    """
    tmp = np.array([0] + list(x) + [0])
    return np.where((tmp[1:] - tmp[:-1]) != 0)[0].reshape((-1, 2)).tolist()


def merge_events(events):
    if len(events) == 0:
        return []

    events = deque(sorted(events))
    merged_events = []

    event1 = events.popleft()
    while len(events) > 0:
        event2 = events.popleft()

        if event2[0] <= event1[1]:
            event1 = (event1[0], max(event1[1], event2[1]))
        else:
            merged_events.append(event1)
            event1 = event2

    merged_events.append(event1)

    return merged_events
