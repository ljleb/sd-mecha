"""
## Commnet / LaTeX translation by DammK ##
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

_alpha = 0.33
_k = 0.5
# Sudoku of 4x4, "top k" should be 2.
_models = [
    torch.tensor([    
        [-1.,2.,3.,4.],
        [4.,-3.,2.,1.],
        [3.,4.,1.,-2.],
        [2.,1.,-4.,3.],
    ]),
    torch.tensor([        
        [3.,4.,1.,-2.],
        [2.,1.,-4.,3.],
        [-1.,2.,3.,4.],
        [4.,-3.,2.,1.],
    ])
]

def ties_sum(  
    *models: torch.tensor,
    alpha: float,
    k: float = 0.2,
) -> torch.tensor:

    # Step 1: Trim redundant parameters

    # $$ \hat{\tau}_t $$ O(N) in space
    deltas = [filter_top_k(m, k) for m in models]
    deltas = torch.stack(deltas, dim=0)

    # $$ \gamma_t $$ 
    signs = torch.sign(deltas)

    # $$ \gamma_m^p = sgn(\sum_{t=1}^n \hat{\tau}_t^p) $$
    final_sign = torch.sign(torch.sum(deltas,dim=0)) 

    # Step 3: Disjoint merge.

    # $$ \{ \gamma_t^p = \gamma_m^p \} $$
    delta_filters = (signs == final_sign).float()

    # $$ |A^p| $$
    param_counts = torch.sum(delta_filters, dim=0)

    # $$ \sum_{t\in{A^P}} \hat{\tau}_t^p $$
    filtered_delta = (deltas * delta_filters).sum(dim=0)

    # $$ \lambda * \tau_m $$
    tau_m = alpha * torch.nan_to_num(filtered_delta / param_counts)

    res = {
        'deltas': deltas,
        'signs': signs,
        'final_sign': final_sign,
        'delta_filters': delta_filters,
        'param_counts': param_counts,
        'filtered_delta': filtered_delta,
        'tau_m': tau_m
    }
    for rk, rv in res.items():
        print(rk)
        print(rv)

    return tau_m

def filter_top_k(a: torch.Tensor, k: float):
    k = max(int((1 - k) * torch.numel(a)), 1)
    k_value, _ = torch.kthvalue(torch.abs(a.flatten()).float(), k)
    top_k_filter = (torch.abs(a) >= k_value).float()
    return a * top_k_filter


ties_sum(*_models, alpha = _alpha, k =_k)