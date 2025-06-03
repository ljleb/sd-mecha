import torch
import sd_mecha


def test_della():
    k = 1.0

    probability = 0.40
    della_eps = 0.20
    no_della = 0.0
    seed = 114514
    cos_eps = 1e-6

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

    expected = torch.tensor([
        [ 0.6251,  1.8496,  2.4691,  0.0000],
        [ 3.1303,  0.2084, -0.8335,  1.0780],
        [ 1.0162,  1.7755,  1.0780,  0.0000],
        [ 2.0362,  0.0000,  0.4167,  0.2084]
    ])

    expected2 = torch.tensor([
        [ 0.7469,  1.9883,  2.4127,  0.0000],
        [ 1.7586,  0.9106, -0.9959,  1.1671],
        [ 0.9925,  1.7586,  1.1671,  0.0000],
        [ 1.9223,  0.0000,  0.4979,  0.2490]
    ])

    test_no_della = sd_mecha.ties_sum_with_dropout.__wrapped__(
        *models,
        probability=probability,
        della_eps=no_della,
        rescale=False,
        k=k,
        vote_sgn=True,
        seed=seed,
        apply_stock=False,
        apply_median=True,
        cos_eps=cos_eps,
        eps=eps,
        maxiter=maxiter,
        ftol=ftol,
    )
    assert torch.allclose(test_no_della, expected, atol=0.0001)

    actual_della = sd_mecha.ties_sum_with_dropout.__wrapped__(
        *models,
        probability=probability,
        della_eps=della_eps,
        rescale=False,
        k=k,
        vote_sgn=True,
        seed=seed,
        apply_stock=False,
        apply_median=True,
        cos_eps=cos_eps,
        eps=eps,
        maxiter=maxiter,
        ftol=ftol,
    )
    assert torch.allclose(actual_della, expected, atol=0.0001)

    actual_della_flipped = sd_mecha.ties_sum_with_dropout.__wrapped__(
        *models,
        probability=probability,
        della_eps=-della_eps,
        rescale=False,
        k=k,
        vote_sgn=True,
        seed=seed,
        apply_stock=False,
        apply_median=True,
        cos_eps=cos_eps,
        eps=eps,
        maxiter=maxiter,
        ftol=ftol,
    )
    assert torch.allclose(actual_della_flipped, expected2, atol=0.0001)
