import torch
import sd_mecha

_alpha = 0.33
_k = 0.9
_use_delta = 0.0
_use_signs = 1.0
_probability = 0.1
_seed = 114514

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

_expected = torch.tensor([
    [ 3.3333,  3.3333,  2.2222,  4.4444],
    [ 3.3333, -3.3333, -4.4444,  2.2222],
    [ 3.3333,  3.3333,  2.2222,  4.4444],
    [ 3.3333,  1.1111, -4.4444,  2.2222]
])

_dare1 = sd_mecha.ties_sum_with_dropout.__wrapped__(*_models, probability=_probability, k=_k, seed=_seed)
print(_dare1)

assert torch.allclose(_dare1, _expected, atol = 0.0001)

_dare2 = sd_mecha.ties_sum_with_dropout.__wrapped__(*_models, probability=_probability, k=_k, vote_sgn=_use_signs, seed=_seed)

print(_dare2)
assert not torch.allclose(_dare1, _dare2, atol = 0.0001)

#Visual inspect if dropout really happens

_ties1 = sd_mecha.ties_sum.__wrapped__(*_models, k=_k)
print(_ties1)

_ties2 = sd_mecha.ties_sum.__wrapped__(*_models, k=_k, vote_sgn=_use_signs)
print(_ties2)
