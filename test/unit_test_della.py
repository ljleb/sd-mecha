import torch
import sd_mecha
import time

_k = 1.0
_use_delta = 0.0
_use_signs = 1.0

_probability = 0.40
_della_eps = 0.20
_no_della = 0.0
_use_rescale = 1.0 #Since 0.0.26
_no_rescale = 0.0 #Since 0.0.26
_rescale = 0.0 #Since 0.0.26
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
    [ 0.6251,  1.8496,  2.4691,  0.0000],
    [ 3.1303,  0.2084, -0.8335,  1.0780],
    [ 1.0162,  1.7755,  1.0780,  0.0000],
    [ 2.0362,  0.0000,  0.4167,  0.2084]
])

# Hard to tell which is better: see poc_della.py
# However it is a lot closer to DARE result.
_expected2 = torch.tensor([
    [ 0.7469,  1.9883,  2.4127,  0.0000],
    [ 1.7586,  0.9106, -0.9959,  1.1671],
    [ 0.9925,  1.7586,  1.1671,  0.0000],
    [ 1.9223,  0.0000,  0.4979,  0.2490]
])

# This is mode "nax" instead of "ordinal". The "rank" distribution will be more extreme.
_expected_scipy_max = torch.tensor([
    [-0.7853,  2.3560,  2.1193,  0.5487],
    [ 0.4294,  0.7633, -0.8230,  0.9184],
    [-0.7257,  0.0000,  1.0597,  0.8587],
    [ 2.9403,  0.0000,  0.2743,  1.7477]
]).type(torch.DoubleTensor)

test_no_della = sd_mecha.ties_sum_with_dropout.__wrapped__(*_models, probability=_probability, della_eps=_no_della, rescale=_no_rescale, k=_k, vote_sgn=_use_signs, seed=_seed, apply_stock = _no_stock, apply_median = _apply_median, cos_eps = _cos_eps, eps=_eps, maxiter=_maxiter, ftol=_ftol)
#print(test_no_della)
assert torch.allclose(test_no_della, _expected, atol = 0.0001)

# Guess what? Mathmatically it is identical! Wow!
test_have_della = sd_mecha.ties_sum_with_dropout.__wrapped__(*_models, probability=_probability, della_eps=_della_eps, rescale=_no_rescale, k=_k, vote_sgn=_use_signs, seed=_seed, apply_stock = _no_stock, apply_median = _apply_median, cos_eps = _cos_eps, eps=_eps, maxiter=_maxiter, ftol=_ftol)
#print(test_have_della)
assert torch.allclose(test_have_della, _expected, atol = 0.0001)

# So that we should expect della_eps should be negative!
test_have_della_flipeps = sd_mecha.ties_sum_with_dropout.__wrapped__(*_models, probability=_probability, della_eps=_della_eps * -1.0, rescale=_no_rescale, k=_k, vote_sgn=_use_signs, seed=_seed, apply_stock = _no_stock, apply_median = _apply_median, cos_eps = _cos_eps, eps=_eps, maxiter=_maxiter, ftol=_ftol)
#print(test_have_della_flipeps)
assert torch.allclose(test_have_della_flipeps, _expected2, atol = 0.0001)

# Integration test: TGMD2
# WS = 61.9
ts = time.time()
stress_test = sd_mecha.ties_sum_with_dropout.__wrapped__(*_models2, probability=_probability, della_eps=_della_eps, rescale=_no_rescale, k=_k, vote_sgn=_use_signs, seed=_seed, apply_stock = _no_stock, apply_median = _apply_median, cos_eps = _cos_eps, eps=_eps, maxiter=_maxiter, ftol=_ftol)
te = time.time()
print(te - ts) 
assert (te - ts) < 120.0