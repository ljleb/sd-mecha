# Merge Methods Listing

**NOTE: this listing is a work in progress and may not be exhaustive.**

Here follows a list of built-in merge methods from the `sd_mecha` module.

### `weighted_sum(a, b, alpha: float | Tensor = 0.5)`

Linear interpolation between `a` and `b`. If `alpha` is a `Tensor`, then it is broadcasted over the input models.
This is useful when a separate weight is needed for each corresponding parameter in `a` and `b`.

Input models: `a`, `b`  
Input params: `alpha`

```math
\theta_{out} = a(1 - \alpha) + b\alpha
```

### `add_difference(a, b, *, alpha: float | Tensor = 1.0)`

Add the delta `b` to `a` at a rate of `alpha`. If `alpha` is a `Tensor`, then it is broadcasted over the input models.
This is useful when a separate weight is needed for each corresponding parameter in `a` and `b`.

Input models: `a`, `b`  
Input params: `alpha`

```math
\theta_{out} = a + b\alpha
```

### `add_difference(a, b, c, *, alpha: float | Tensor = 1.0, clip_to_ab: bool = False)`

Add the delta `b - c` to `a` at a rate of `alpha`. If `alpha` is a `Tensor`, then it is broadcasted over the input models.
This is useful when a separate weight is needed for each corresponding parameter in `a`, `b` and `c`.

If `clip_to_ab` is `True`, then the result of add difference is clipped to `a` and `b`.
See `sd_mecha.clamp(...)` for more details.

Input models: `a`, `b`, `c`  
Input params: `alpha`

```math
f_{clip}(m) = 
\begin{cases}
clip(m, a, b) & \text{if clip_to_a_b}, \\
m & \text{otherwise}.
\end{cases}
```

```math
m_{out} = f_{clip}(a + \alpha(b - c))
```

### `slerp(a, b, alpha: float = 0.5)`

Spherical interpolation of `a` and `b`.

Input models: `a`, `b`  
Input params: `alpha`

```math
\Omega = arccos(\frac{a}{|a|} \frac{b}{|b|})
```
```math
m' = \frac{a}{|a|} \frac{sin((1 - \alpha) \Omega)}{sin \Omega} + \frac{b}{|b|} \frac{sin \alpha \Omega}{sin \Omega}
```
```math
m_{out} = (|a|(1-\alpha) + |b|\alpha) m'
```

For more information, see the [corresponding wikipedia article](https://en.wikipedia.org/wiki/Slerp).

### `add_perpendicular(a, b, c, *, alpha: float | Tensor = 1.0)`

Add the orthogonalized delta `b - c` to `a` at a rate of `alpha`. If `alpha` is a `Tensor`, then it is broadcasted over the input models.
This is useful when a separate weight is needed for each corresponding parameter in `a`, `b` and `c`.

Input models: `a`, `b`, `c`  
Input params: `alpha`

```math
\Delta_a = a - c
```
```math
\Delta_b = b - c
```
```math
m_{out} = a + \alpha(\Delta_b - \Delta_a \frac{\Delta_a \cdot \Delta_b}{\Delta_a \cdot \Delta_a})
```

### `geometric_sum(a, b, alpha: float = 0.5)`

Geometric interpolation between `a` and `b`.
The result is zero for any position in the tensor where the sign of `a` and `b` is different.

It is the exact same method as `weighted_sum` but in log space.

A compelling way of thinking about this method is that it acts as a sort of smooth "AND gate" between `a` and `b`:

- if $`a = b`$, then the output is `a` (or `b` since they are equal)
- if $`a = 0`$ or $`b = 0`$, then the output is `0`
- otherwise, the output is some sort of interpolation between the above cases.

Another way to think of this method is that a geometric interpolation tends to be skewed towards smaller values.

Input models: `a`, `b`  
Input params: `alpha`

```math
m_{out} = \Re(a^{1-\alpha} \cdot b^\alpha)
```

### `add_cosine_a(a, b, alpha: float)`

"Cosine A" method from supermerger. I have not looked deeply into how this method works, so I don't have any useful insight.
Feel free to contribute to this section by opening a PR or a discussions thread.

Input models: `a`, `b`  
Input params: `alpha`


### `add_cosine_b(a, b, alpha: float)`

"Cosine B" method from supermerger. I have not looked deeply into how this method works, so I don't have any useful insight.
Feel free to contribute to this section by opening a PR or a discussions thread.

Input models: `a`, `b`  
Input params: `alpha`

### `add_difference_ties(base, *m, alpha: float | Tensor = 1.0)`

Discard parameters of $m_i$ when their sign differs from the majority sign.
The majority sign is calculated parameter-wise as  
  
$`s = \text{sign} \Biggl [ \sum_{i = 1}^{n}{(m_i - \text{base})} \Biggr ]`$  
  
`alpha` controls the rate at which the resolved delta is added to `base`.  
  
If `alpha` is a `Tensor`, then it is broadcasted over the input models.
This is useful when a separate weight is needed for each corresponding parameter in `base` and `*m`.

Input models: `base`, `m[i]` for $`i \in 1..n`$  
Input params: `alpha`

For more information, see the [paper](https://arxiv.org/abs/2306.01708).
