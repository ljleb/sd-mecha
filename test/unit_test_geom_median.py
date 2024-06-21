import torch
import sd_mecha
import time


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

_apply_median = 1.0
_no_median = 0.0

_eps = 1e-6
_maxiter = 100 #1 iter = 10 sec, avg 5-10 iter
_ftol = 1e-20

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

_models2 = []
for i in range(100):
    _models2.append(torch.rand(1280, 1280))

# Not used
_weights = torch.ones(len(_models), device=_models[0].device)

_expected = torch.tensor([
    [ 0.4791,  3.3698,  2.2343, -0.1354],
    [ 2.9323,  0.9739, -1.7289,  1.7395],
    [ 0.2082,  1.4220,  2.0416,  2.6873],
    [ 3.0677,  0.0989, -0.2711,  0.4481]
])

_expected2 = torch.tensor([
    [ 0.0000,  3.1750,  2.2089,  0.0000],
    [ 3.0170,  0.5588, -0.6999,  1.1580],
    [ 0.5758,  2.2492,  1.1580,  0.0000],
    [ 2.9830,  0.0000,  0.3500,  0.1750]
])

median = sd_mecha.geometric_median.__wrapped__(*_models, eps=_eps, maxiter=_maxiter, ftol=_ftol)
#print(median2)
assert torch.allclose(median, _expected, atol = 0.0001)

_with_dare = sd_mecha.ties_sum_with_dropout.__wrapped__(*_models, probability=_probability, no_rescale=_no_rescale, k=_k, vote_sgn=_use_signs, seed=_seed, apply_stock = _no_stock, apply_median = _apply_median, cos_eps = _cos_eps, eps=_eps, maxiter=_maxiter, ftol=_ftol)
#print(_with_dare)
assert torch.allclose(_with_dare, _expected2, atol = 0.0001)

ts = time.time()
median2 = sd_mecha.geometric_median.__wrapped__(*_models2, eps=_eps, maxiter=_maxiter, ftol=_ftol)
#print(median2) #Around 0.5 but will flutter
te = time.time()
#print(te - ts) #WS = 0.9, Notebook = 1.76
assert (te - ts) < 10.0 