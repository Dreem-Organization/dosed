import torch

from dosed.utils import encode, decode


def test_encode_decode():
    localizations_default = torch.FloatTensor([1]).repeat([4, 2]).uniform_()

    for line in range(localizations_default.shape[0]):
        if localizations_default[line, 0] > localizations_default[line, 1]:
            aux = localizations_default[line, 0]
            localizations_default[line, 0] = localizations_default[line, 1]
            localizations_default[line, 1] = aux

    localizations = encode(localizations_default, localizations_default)
    is_it_retrieved = decode(localizations, localizations_default)

    assert ((localizations_default - is_it_retrieved).numpy() > 1e-4).sum() == 0
