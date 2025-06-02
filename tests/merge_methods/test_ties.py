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


def test_ties():
    k = 0.5

    models = [
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
    expected = torch.tensor([
        [3.,  3.,  3., 0.],
        [3., -3., 0., 3.],
        [3.,  3.,  3., 0.],
        [3., -3., 0., 3.]
    ])

    actual = sd_mecha.ties_sum.__wrapped__(*models, k=k)
    assert torch.allclose(actual, expected)

    actual2 = sd_mecha.ties_sum.__wrapped__(*models, k=k, vote_sgn=True)
    assert not torch.allclose(actual, actual2)
