import torch
import sd_mecha

_k = 1.0
_use_delta = 0.0
_use_signs = 1.0

_probability = 0.25
_use_rescale = 0.0
_no_rescale = 1.0
_seed = 114514

_alpha = 0.0 #Not used

# Sudoku of 4x4
_models = [
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

_expected1 = torch.tensor([
    [-1.3333,  4.0000,  2.6667,  0.0000], 
    [ 4.0000, -4.0000,  2.6667,  1.3333], 
    [-1.3333,  4.0000,  1.3333,  0.0000], 
    [ 4.0000,  0.0000, -5.3333,  4.0000]
])

_expected2 = torch.tensor([
    [-1.,  3.,  2.,  0.], 
    [ 3.,  0.,  2.,  1.], 
    [-1.,  3.,  1.,  0.], 
    [ 3.,  0., -4.,  3.]
])

#Visual inspect if dropout really happens

_dare1 = sd_mecha.ties_sum_with_dropout.__wrapped__(*_models, probability=_probability, k=_k, seed=_seed)
#print(_dare1)

assert torch.allclose(_dare1, _expected1, atol = 0.0001)

_dare2 = sd_mecha.ties_sum_with_dropout.__wrapped__(*_models, probability=_probability, no_rescale=_no_rescale, k=_k, vote_sgn=_use_signs, seed=_seed)

#print(_dare2)
assert torch.allclose(_dare2, _expected2, atol = 0.0001)

#_ties1 = sd_mecha.ties_sum.__wrapped__(*_models, k=_k)
#print(_ties1)

#_ties2 = sd_mecha.ties_sum.__wrapped__(*_models, k=_k, vote_sgn=_use_signs)
#print(_ties2)
