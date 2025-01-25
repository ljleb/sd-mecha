import torch
import sd_mecha


def test_modelstock():
    cos_eps = 1e-6

    models = [
        torch.tensor([
            [3., 4., 1., -2.],
            [2., 1., -4., 3.],
            [-1., 2., 3., 4.],
            [4., -3., 2., 1.],
        ]),
        torch.tensor([
            [-1., 3., 4., 2.],
            [4., 2., -3., 1.],
            [3., -1., 2., 4.],
            [2., 4., 1., -3.],
        ]),
        torch.tensor([
            [-1., 2., 3., 4.],
            [4., -3., 2., 1.],
            [3., 4., 1., -2.],
            [2., 1., -4., 3.],
        ])
    ]

    expected1 = torch.tensor([
        [0.2727,  2.4545,  2.1818,  1.0909],
        [2.5000,  0.0000, -1.2500,  1.2500],
        [0.8696,  0.8696,  1.0435,  1.0435],
        [-2.0000, -0.5000,  0.2500, -0.2500]
    ])

    stock_only = sd_mecha.model_stock.__wrapped__(*models, cos_eps=cos_eps)
    assert torch.allclose(stock_only, expected1, atol=0.0001)
