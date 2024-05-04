# Recipes listing

Here is a comprehensive list of built-in recipes and a description of what they do.

## `weighted_sum(a, b, *, alpha: Hyper)`

Linear interpolation between $a$ and $b$.

Input models: $a$, $b$  
Input hypers: $\alpha$

```math
m = a(1 - \alpha) + b\alpha
```

## `add_difference(a, b, c, *, alpha: Hyper)`

Add the delta $b - c$ to $a$ at a rate of $\alpha$.

Input models: $a$, $b$, $c$  
Input hypers: $\alpha$

```math
m = a + \alpha(b - c)
```

## `slerp(a, b, *, alpha: Hyper)`

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
m = (|a|(1-\alpha) + |b|\alpha) m'
```

For more information: https://en.wikipedia.org/wiki/Slerp

## `add_perpendicular(a, b, c, *, alpha: Hyper)`

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
m = c + \alpha(\Delta_a - \Delta_b \frac{\Delta_b \cdot \Delta_a}{\Delta_b \cdot \Delta_b})
```
