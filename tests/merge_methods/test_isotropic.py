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
        [ 0.4487,  8.7236,  3.9089, -2.5803],
        [ 4.0244,  4.1229, -6.7844,  4.3609],
        [-0.0294,  0.0581,  5.3719,  8.3293],
        [ 9.0466, -2.2665,  2.8416, -1.7849]
    ])

    apply_exp_1 = False
    apply_high_dim_1 = False

    Iso_c = sd_mecha.isotropic.__wrapped__(*models, apply_exp=apply_exp_1, apply_high_dim=apply_high_dim_1)
    #print(Iso_c)
    assert torch.allclose(Iso_c, expected, atol=0.0001)

    #s_bar = sd_mecha.isotropic.__wrapped__(*models2, return_sbar=True)
    #_ = sd_mecha.isotropic_overrided.__wrapped__(models2[0], s_bar)


if __name__ == "__main__":
    test_isotropic()