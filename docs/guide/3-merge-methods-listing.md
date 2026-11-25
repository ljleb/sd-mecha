# Merge Methods Listing

**NOTE: this listing is a work in progress and may not be exhaustive.**

Here follows a list of built-in merge methods from the `sd_mecha` module.

### `weighted_sum(a, b, alpha: float | Tensor = 0.5)`

Linear interpolation between `a` and `b`. If `alpha` is a `Tensor`, then it is broadcasted over the input models.
This is useful when a separate weight is needed for each corresponding parameter in `a` and `b`.

Input models: `a`, `b`  
Input params: `alpha`

$$
\theta_{out} = a(1 - \alpha) + b\alpha
$$

### `add_difference(a, b, *, alpha: float | Tensor = 1.0, fallback_to_a: bool = True)`

Add the delta `b` to `a` at a rate of `alpha`. If `alpha` is a `Tensor`, then it is broadcasted over the input models.
This is useful when a separate weight is needed for each corresponding parameter in `a` and `b`.

sd-mecha can convert a compatible delta to the key format of `a`. By default, keys without a delta keep their values from `a`. Set `fallback_to_a=False` to omit those keys.

Input models: `a`, `b`  
Input params: `alpha`

$$
\theta_{out} = a + b\alpha
$$

### `add_difference(a, b, c, *, alpha: float | Tensor = 1.0, clamp_to_ab: bool = False, fallback_to_a: bool = True)`

Add the delta `b - c` to `a` at a rate of `alpha`. If `alpha` is a `Tensor`, then it is broadcasted over the input models.
This is useful when a separate weight is needed for each corresponding parameter in `a`, `b` and `c`.

The same `fallback_to_a` option applies to keys without a delta from `b - c`.

If `clamp_to_ab` is `True`, then the result of add difference is clipped to `a` and `b`.
The bounds must use the same merge space. See `sd_mecha.clamp(...)` for more details.

Input models: `a`, `b`, `c`  
Input params: `alpha`

$$
f_{clip}(m) =
\begin{cases}
clip(m, a, b) & \text{if clamping is enabled}, \\
m & \text{otherwise}.
\end{cases}
$$

$$
m_{out} = f_{clip}(a + \alpha(b - c))
$$

### `slerp(a, b, alpha: float | Tensor = 0.5)`

Spherical interpolation of `a` and `b`.

Input models: `a`, `b`  
Input params: `alpha`

$$
\Omega = \arccos\left(\frac{a \cdot b}{\lVert a\rVert\,\lVert b\rVert}\right)
$$

$$
m' = \frac{a}{|a|} \frac{\sin((1 - \alpha)\Omega)}{\sin \Omega} + \frac{b}{|b|} \frac{\sin(\alpha\Omega)}{\sin \Omega}
$$

$$
m_{out} = (|a|(1-\alpha) + |b|\alpha) m'
$$

If the spherical interpolation produces NaN, sd-mecha uses linear interpolation for that tensor.

For more information, see the [corresponding wikipedia article](https://en.wikipedia.org/wiki/Slerp).

### `add_perpendicular(a, b, c, *, alpha: float | Tensor = 1.0)`

Add the orthogonalized delta `b - c` to `a` at a rate of `alpha`. If `alpha` is a `Tensor`, then it is broadcasted over the input models.
This is useful when a separate weight is needed for each corresponding parameter in `a`, `b` and `c`.

Input models: `a`, `b`, `c`  
Input params: `alpha`

$$
\Delta_a = a - c
$$

$$
\Delta_b = b - c
$$

$$
m_{out} = a + \alpha(\Delta_b - \Delta_a \frac{\Delta_a \cdot \Delta_b}{\Delta_a \cdot \Delta_a})
$$

### `geometric_sum(a, b, alpha: float | Tensor = 0.5)`

Geometric interpolation between `a` and `b`.
For positive inputs it is equivalent to interpolating their logarithms and taking the exponential.
For negative inputs, sd-mecha computes complex powers and returns the real part. The result can be nonzero when the signs differ.

For nonnegative inputs and $0 < \alpha < 1$:

- If $a = b$, the output is `a`.
- If $a = 0$ or $b = 0$, the output is zero.
- Otherwise, the result tends toward the smaller value compared to a linear blend.

Input models: `a`, `b`  
Input params: `alpha`

$$
m_{out} = \Re(a^{1-\alpha} \cdot b^\alpha)
$$

### `add_cosine_a(a, b, alpha: float | Tensor = 1.0)`

"Cosine A" method from supermerger. It computes cosine similarity along the first axis after normalization.

Input models: `a`, `b`  
Input params: `alpha`


### `add_cosine_b(a, b, alpha: float | Tensor = 1.0)`

"Cosine B" method from supermerger. It averages cosine similarity along the first axis with cosine similarity of the whole tensor.

Input models: `a`, `b`  
Input params: `alpha`

### `add_difference_ties(base, *models, alpha: float | Tensor = 1.0, k: float = 1.0)`

For checkpoint inputs, sd-mecha subtracts `base` from each input. Use `k` to keep the largest-magnitude fraction of each difference. Let $d_i$ be the trimmed difference for input $i$. Each entry gets a sign by vote:

$$
s = \operatorname{sign}\left[\sum_{i=1}^{n}\operatorname{sign}(d_i)\right]
$$

sd-mecha averages the entries with that sign. `alpha` scales the resulting delta before sd-mecha adds it to `base`.

If `alpha` is a `Tensor`, then it is broadcasted over the input models.
This is useful when a separate weight is needed for each corresponding parameter in `base` and `*m`.

Input models: `base`, `models`

Input params: `alpha`

For more information, see the [paper](https://arxiv.org/abs/2306.01708).

### `rotate(a, b, c=None, *, alpha=0.0)`

Use an orthogonal transform to align `a` with `b`. Pass `c` to rotate the differences from that checkpoint.

## Compress differences

| Method | Result |
| --- | --- |
| `get_rank_from_ratio(a, ratio=0.5)` | Compute an integer rank for each tensor from its dimensions and the ratio. |
| `truncate_rank(a, rank=8, ...)` | Approximate a delta using at most the specified rank. A rank recipe can choose a different rank per key. |
| `truncate_kronecker(a, dims=None, ...)` | Approximate a delta with a Kronecker product. |

Both `sd_mecha.lokr_dims(a, ratio=0.5)` and `sd_mecha.lokr_dims_lycoris(a, factor=-1, unbalanced_factorization=False)` produce per-key factor dimensions for `sd_mecha.truncate_kronecker()`.

`truncate_rank()` can reuse cached singular factors for a lower rank. Exact factors can serve any basis request. Approximate factors require matching iteration and seed settings. Changing `use_approximate_basis` alone keeps eligible factors in the cache. The cache uses `device="cpu"` and `dtype=torch.float16` unless you override either setting when [creating the cache](2-user-defined-merge-methods.md#cache-work-across-merges).

## Apply and extract adapters

`apply_lycoris(base, lyco)` supports LoRA, LoKr, LoHa, Norm, full-delta, and BOFT adapters for registered base key formats. The adapter config uses the suffix `_lycoris_lora`, including for adapter types other than LoRA.

Use `sd_mecha.lycoris.resolve(name)` to get an extractor.

| Name | Arguments | Result |
| --- | --- | --- |
| `"lora"` | `(base, rank=8, use_approximate_basis=True, approximate_basis_iters=2, approximate_basis_seed=None)` | Fit low-rank factors to a delta. |
| `"lokr"` | `(base, dims=None)` | Fit Kronecker factors to a delta. |
| `"norm"` | `(diff)` | Store one-dimensional weight and bias differences. |
| `"boft"` | `(base, target, block_size=10, stages=5, solver="svd_sweeps")` | Fit rotations and per-output-channel gains to the base and target weights. Use `solver="lbfgs"` to refine the four-sweep fit. |

`to_base(recipe)` converts additive adapter updates to a delta in the base model's key format. It warns if it skips BOFT updates, which need the base weights. Use `apply_lycoris()` to apply those updates.

## Inspect the registry

Use `sd_mecha.extensions.merge_methods.get_all()` to list registered methods. Their `get_signature()` method exposes the parameters, defaults, merge spaces, and model configs. The registry also includes generated conversions and implementations selected by interfaces.

## Saved recipe compatibility

Loading an old recipe can emit a warning. Save it again to write the current method and config names.

- Loading saved `rotate()` calls with `alignment`, `stiefel_eps`, or `stiefel_max_iters` ignores those arguments and warns.
- Loading old LoRA and LoKr extraction names or adapter config names that use `_kohya_lora` emits a warning.
- Use `anima-diffusion_pipe` in place of `anima-diffuser`, and `moss_image-openmoss` in place of `moss_image-diffusers`.
