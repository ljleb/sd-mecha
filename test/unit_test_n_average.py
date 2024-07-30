import torch
import sd_mecha

_models = [
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
        [-1.,  3.,  2.,  0.], 
        [ 3.,  0.,  2.,  1.], 
        [-1.,  3.,  1.,  0.],
        [ 3.,  0., -4.,  3.]
    ])
]

_expected = torch.tensor([
    [ 0.3333,  3.3333,  2.3333,  0.0000],
    [ 3.0000,  1.0000, -1.6667,  1.6667],
    [ 0.3333,  1.3333,  2.0000,  2.6667],
    [ 3.0000,  0.3333, -0.3333,  0.3333]
])

avg = sd_mecha.merge_methods.n_average.__wrapped__(*_models)

assert torch.allclose(avg, _expected, atol=0.0001)
