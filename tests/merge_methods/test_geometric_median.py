import torch
import sd_mecha


def test_geometric_median():
    eps = 1e-6
    maxiter = 100
    ftol = 1e-20

    models = [
        torch.tensor([
            [ 3.,  4.,  1., -2.],
            [ 2.,  1., -4.,  3.],
            [-1.,  2.,  3.,  4.],
            [ 4., -3.,  2.,  1.],
        ]),
        torch.tensor([
            [-1.,  3.,  4.,  2.],
            [ 4.,  2., -3.,  1.],
            [ 3., -1.,  2.,  4.],
            [ 2.,  4.,  1., -3.],
        ]),
        torch.tensor([
            [-1.,  3.,  2.,  0.],
            [ 3.,  0.,  2.,  1.],
            [-1.,  3.,  1.,  0.],
            [ 3.,  0., -4.,  3.]
        ])
    ]

    models2 = []
    for i in range(100):
        models2.append(torch.rand(1280, 1280))

    expected = torch.tensor([
        [0.4791,  3.3698,  2.2343, -0.1354],
        [2.9323,  0.9739, -1.7289,  1.7395],
        [0.2082,  1.4220,  2.0416,  2.6873],
        [3.0677,  0.0989, -0.2711,  0.4481]
    ])

    median = sd_mecha.geometric_median.__wrapped__(*models, eps=eps, maxiter=maxiter, ftol=ftol)
    assert torch.allclose(median, expected, atol=0.0001)
