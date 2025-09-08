import torch
import sd_mecha


def test_crossover_zero_dim():
    a = torch.tensor([])
    b = torch.tensor([])

    actual = sd_mecha.crossover.__wrapped__(a, b)
    assert actual.numel() == 0
