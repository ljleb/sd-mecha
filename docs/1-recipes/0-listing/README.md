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
m = a + \alpha(\Delta_b - \Delta_a \frac{\Delta_a \cdot \Delta_b}{\Delta_a \cdot \Delta_a})
```

## `geometric_sum(a, b, *, alpha: Hyper)`

Geometric sum of each parameter of $a$ with the corresponding parameter in $b$˙.
The sum happens on the complex plane in case any of $a$ and $b$ is negative to avoid NaNs, then projected back onto the real axis.

A compelling way of thinking about this method is that it acts as a sort of smooth "AND gate" between $a$ and $b$:
- if $a = b$, then the output is $a$ (or $b$ since they are equal)
- if $a = 0$ and $b \neq 0$, or $b = 0$ and $a \neq 0$, then the output is $0$
- otherwise, the output is some sort of interpolation between the above cases.

Another way to think of this method is that a geometric sum tends to be skewed towards smaller values.


Input models: $a$, $b$  
Input hypers: $\alpha$

```math
m = \Re(a^{1-\alpha} * b^\alpha)
```
