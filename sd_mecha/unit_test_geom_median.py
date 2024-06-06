import torch
import sd_mecha
import time

_eps = 1e-6
_maxiter = 100 #1 iter = 10 sec, avg 5-10 iter
_ftol = 1e-20

_models2 = [
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

_models = []
for i in range(100):
    _models.append(torch.rand(1280, 1280))

# Not used
_weights = torch.ones(len(_models), device=_models[0].device)

_expected = torch.tensor([
    [ 0.4791,  3.3698,  2.2343, -0.1354],
    [ 2.9323,  0.9739, -1.7289,  1.7395],
    [ 0.2082,  1.4220,  2.0416,  2.6873],
    [ 3.0677,  0.0989, -0.2711,  0.4481]
])

ts = time.time()
median = sd_mecha.geometric_median_list_of_array.__wrapped__(*_models, eps=_eps, maxiter=_maxiter, ftol=_ftol)
#print(median) #Around 0.5 but will flutter
te = time.time()
#print(te - ts) #WS = 0.9, Notebook = 1.76
assert (te - ts) < 10.0 

median2 = sd_mecha.geometric_median_list_of_array.__wrapped__(*_models2, eps=_eps, maxiter=_maxiter, ftol=_ftol)
#print(median2)
assert torch.allclose(median2, _expected, atol = 0.0001)