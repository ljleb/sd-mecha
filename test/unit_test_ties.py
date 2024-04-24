"""
## Comment / LaTeX translation ##
- View in https://upmath.me/
### add_difference_ties ###
- `base`: $$ \theta_{init} $$
- `*models`: $$ \{\theta_{init}\}_{t=1}^n $$
- `models` after `subtract`: $$ \tau_t $$
- `alpha`: $$ \lambda $$
- `k`: $$ k $$ ( From $$ \% $$ to $$ 1 $$ )
- `res`: $$ \lambda * \tau_m $$
- `return`: $$ \theta_m $$
### ties_sum ###
- `delta`: $$ \hat{\tau}_t $$
- `signs`: $$ \gamma_t $$ 
- `final_sign`: $$ \gamma_m^p = sgn(\sum_{t=1}^n \hat{\tau}_t^p) $$ 
- `delta_filters`: $$ \{ \gamma_t^p = \gamma_m^p \} $$
- `param_counts`: $$ |A^p| $$
- `filtered_delta`: $$ \sum_{t\in{A^p}} \hat{\tau}_t^p $$
- `return`: $$ \lambda * \tau_m $$
"""

import torch
import sd_mecha

_alpha = 0.33
_k = 0.5
# Sudoku of 4x4, "top k" should be 2.
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
    [3.,  3.,  3., 4.],
    [3., -3., -4., 3.],
    [3.,  3.,  3., 4.],
    [3., -3., -4., 3.]
])

_actual = sd_mecha.ties_sum.__wrapped__(*_models, k=_k)
assert torch.allclose(_actual, _expected)
