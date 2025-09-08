import torch
import sd_mecha


def test_ties():
    k = 0.5

    models = [
        torch.tensor([
            [-1., 2., 3., 4.],
            [4., -3., 2., 1.],
            [3., 4., 1., -2.],
            [2., 1., -4., 3.],
        ]),
        torch.tensor([
            [3., 4., 1., -2.],
            [2., 1., -4., 3.],
            [-1., 2., 3., 4.],
            [4., -3., 2., 1.],
        ])
    ]
    expected = torch.tensor([
        [3.,  3.,  3., 0.],
        [3., -3., 0., 3.],
        [3.,  3.,  3., 0.],
        [3., -3., 0., 3.]
    ])

    actual = sd_mecha.ties_sum.__wrapped__(*models, k=k)
    assert torch.allclose(actual, expected)

    actual2 = sd_mecha.ties_sum.__wrapped__(
        *models,
        k=k,
        vote_sgn=True,
    )
    assert not torch.allclose(actual, actual2)


def test_ties_zero_dim():
    k = 0.5

    models = [
        torch.tensor([]),
        torch.tensor([])
    ]

    actual = sd_mecha.ties_sum.__wrapped__(*models, k=k)
    assert actual.numel() == 0


def test_ties_no_models():
    k = 0.5

    models = []

    actual = sd_mecha.ties_sum.__wrapped__(*models, k=k)
    assert actual == 0.0
