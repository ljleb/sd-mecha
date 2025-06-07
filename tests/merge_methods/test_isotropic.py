import torch
import sd_mecha

def test_isotropic():
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
    #for i in range(100):
    #    models2.append(torch.rand(1280, 1280))
    for i in range(20):
        models2.append(torch.rand(640, 320, 3, 3))

    expected = torch.tensor([
        [ 0.6583,  4.5956,  2.5195, -1.1094],
        [ 3.1552,  1.5443, -2.4330,  2.2136],
        [ 0.0614,  0.7001,  2.6028,  4.5609],
        [ 4.1000,  0.1333,  0.2084, -0.3341]
    ])

    Iso_c = sd_mecha.isotropic.__wrapped__(*models)
    assert torch.allclose(Iso_c, expected, atol=0.0001)

    #s_bar = sd_mecha.isotropic.__wrapped__(*models2, return_sbar=True)
    #_ = sd_mecha.isotropic_overrided.__wrapped__(models2[0], s_bar)


if __name__ == "__main__":
    test_isotropic()