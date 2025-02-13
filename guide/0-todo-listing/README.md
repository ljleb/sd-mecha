**NOTE: put this information in docstrings, and then use a tool to generate docs from these**

# Recipes listing

Here is a comprehensive list of built-in merge methods and a description of what they do.

### `sd_mecha.weighted_sum(a, b, *, alpha: float = 0.5)`

Linear interpolation between $a$ and $b$.

Input models: $a$, $b$  
Input hypers: $\alpha$

```math
m_{out} = a(1 - \alpha) + b\alpha
```

### `sd_mecha.add_difference(a, b, *, alpha: float = 1.0)`

Add the delta $b$ to $a$ at a rate of $\alpha$.

Input models: $a$, $b$  
Input hypers: $\alpha$

```math
m_{out} = a + b\alpha
```

### `sd_mecha.add_difference(a, b, c, *, alpha: float = 1.0, clip_to_ab: bool = True)`

Add the delta $b - c$ to $a$ at a rate of $\alpha$.

If `clip_to_ab` is `True`, then the result of add difference is clipped to $a$ and $b$.
See `clip(a, *bounds, ...)` below for more info.

Input models: $a$, $b$, $c$  
Input hypers: $\alpha$

```math
f_{clip}(m) = 
\begin{cases}
clip(m, a, b) & \text{if clip to $a$ and $b$}, \\
m & \text{otherwise}.
\end{cases}
```

```math
m_{out} = f_{clip}(a + \alpha(b - c))
```

### `sd_mecha.slerp(a, b, *, alpha: float = 0.5)`

Circular interpolation between $a$ and $b$.

Input models: $a$, $b$  
Input hypers: $\alpha$

```math
\Omega = arccos(\frac{a}{|a|} \frac{b}{|b|})
```
```math
m' = \frac{a}{|a|} \frac{sin((1 - \alpha) \Omega)}{sin \Omega} + \frac{b}{|b|} \frac{sin \alpha \Omega}{sin \Omega}
```
```math
m_{out} = (|a|(1-\alpha) + |b|\alpha) m'
```

For more information: https://en.wikipedia.org/wiki/Slerp

### `sd_mecha.add_perpendicular(a, b, c, *, alpha: float = 1.0)`

Add orthogonalized delta $b - c$ to $a$ at a rate of $\alpha$.

Input models: $a$, $b$, $c$  
Input hypers: $\alpha$

```math
\Delta_a = a - c
```
```math
\Delta_b = b - c
```
```math
m_{out} = a + \alpha(\Delta_b - \Delta_a \frac{\Delta_a \cdot \Delta_b}{\Delta_a \cdot \Delta_a})
```

### `sd_mecha.geometric_sum(a, b, *, alpha: float = 0.5)`

Geometric sum of each parameter of $a$ with the corresponding parameter in $b$˙.
The sum is computed on the complex plane in case any parameter of $a$ and $b$ is negative to avoid NaNs, then projected back onto the real line.

It is the exact same method as `weighted_sum` but in log space.

A compelling way of thinking about this method is that it acts as a sort of smooth "AND gate" between $a$ and $b$:
- if $a = b$, then the output is $a$ (or $b$ since they are equal)
- if $a = 0$ and $b \neq 0$, or $b = 0$ and $a \neq 0$, then the output is $0$
- otherwise, the output is some sort of interpolation between the above cases.

Another way to think of this method is that a geometric sum tends to be skewed towards smaller values.

Input models: $a$, $b$  
Input hypers: $\alpha$

```math
m_{out} = \Re(a^{1-\alpha} \cdot b^\alpha)
```

### `sd_mecha.add_cosine_a(a, b, *, alpha: float)`

"Cosine A" method from supermerger. I have not looked deeply into how this method works, so I don't have any useful insight.
Feel free to contribute to this section by opening a PR or a discussions thread.

Input models: $a$, $b$  
Input hypers: $\alpha$


### `sd_mecha.add_cosine_b(a, b, *, alpha: float)`

"Cosine B" method from supermerger. I have not looked deeply into how this method works, so I don't have any useful insight.
Feel free to contribute to this section by opening a PR or a discussions thread.

Input models: $a$, $b$  
Input hypers: $\alpha$

### `sd_mecha.add_difference_ties(base, *m, alpha: float = 1.0)`

Discard parameters of $m_i$ when their sign differs from the majority sign.
The majority sign is calculated parameter-wise as
$$\text{sign} \Biggl [ \sum_{i = 1}^{n}{(m_i - \text{base})} \Biggr ]$$
$\alpha$ controls the rate at which the resolved delta is added to $\text{base}$.

Input models: $\text{base}$, $m_i$ for $i \in 1..n$  
Input hypers: $\alpha$

For more information, see the [paper](https://arxiv.org/abs/2306.01708).
