import torch


def non_maximum_suppression(localizations, scores, overlap=0.5):
    """1D nms"""
    x = localizations[:, 0]
    y = localizations[:, 1]

    areas = y - x
    order = scores.sort(0, descending=True)[1]
    keep = []
    while order.numel() > 1:
        i = order[0]
        keep.append([x[i], y[i]])
        order = order[1:]
        xx = torch.clamp(x[order], min=x[i].item())
        yy = torch.clamp(y[order], max=y[i].item())

        intersection = torch.clamp(yy - xx, min=0)

        intersection_over_union = intersection / (areas[i] + areas[order] - intersection)

        order = order[intersection_over_union <= overlap]

    keep.extend([[x[k], y[k]] for k in order])  # remaining element if order has size 1

    return keep
