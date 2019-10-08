from collections import deque


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
