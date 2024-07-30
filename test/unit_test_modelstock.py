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
_cos_eps = 1e-6
_apply_stock = 1.0
_no_stock = 0.0

# This time more 4x4 sudoku.

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
        [-1., 2., 3., 4.],
        [4., -3., 2., 1.],
        [3., 4., 1., -2.],
        [2., 1., -4., 3.],
    ])
]

_expected1 = torch.tensor([
    [ 0.2727,  2.4545,  2.1818,  1.0909],
    [ 2.5000,  0.0000, -1.2500,  1.2500],
    [ 0.8696,  0.8696,  1.0435,  1.0435],
    [-2.0000, -0.5000,  0.2500, -0.2500]
])

# Notice that it can be brutal.
_expected2 = torch.tensor([
    [ 0.0000,  2.6592,  2.3638,  3.5456],
    [ 2.8031,  1.2614, -3.3638,  1.6819],
    [ 0.0000,  0.0000,  0.0000,  0.0000],
    [ 2.6077,  0.0000,  1.9557,  0.9779]
])

_expected3 = torch.tensor([
    [ 0.0000,  3.0000,  2.6667,  4.0000],
    [ 3.3333,  1.5000, -4.0000,  2.0000],
    [ 3.0000,  3.0000,  2.0000,  0.0000],
    [ 2.6667,  0.0000,  2.0000,  1.0000]
])

#Visual inspect if dropout really happens

_stock_only = sd_mecha.model_stock_for_tensor.__wrapped__(*_models, cos_eps=_cos_eps)
#print(_stock_only)
assert torch.allclose(_stock_only, _expected1, atol=0.0001)

_with_dare = sd_mecha.ties_sum_with_dropout.__wrapped__(*_models, probability=_probability, no_rescale=_no_rescale, k=_k, vote_sgn=_use_signs, seed=_seed, apply_stock = _apply_stock, cos_eps = _cos_eps)
#print(_with_dare)
assert torch.allclose(_with_dare, _expected2, atol=0.0001)

_dare_only = sd_mecha.ties_sum_with_dropout.__wrapped__(*_models, probability=_probability, no_rescale=_no_rescale, k=_k, vote_sgn=_use_signs, seed=_seed, apply_stock = _no_stock, cos_eps = _cos_eps)
#print(_dare_only)
assert torch.allclose(_dare_only, _expected3, atol=0.0001)
