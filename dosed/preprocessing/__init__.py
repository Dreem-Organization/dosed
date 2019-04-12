from .regularization import GaussianNoise, RescaleNormal, Invert
from .normalizers import clip, clip_and_normalize, mask_clip_and_normalize


normalizers = {
    "clip": clip,
    "clip_and_normalize": clip_and_normalize,
    "mask_clip_and_normalize": mask_clip_and_normalize
}


__all__ = [
    GaussianNoise,
    RescaleNormal,
    Invert,
]
