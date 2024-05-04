# Recipes listing

Here is a comprehensive list of built-in recipes and a description of what they do.

## `weighted_sum(a, b, *, alpha: Hyper)`

Linear interpolation between $a$ and $b$.

Input models: $a$, $b$  
Input hypers: $\alpha$

$m = a(1 - \alpha) + b\alpha$

## `add_difference(a, b, c, *, alpha: Hyper)`

Add the task vector $b - c$ to $a$ at a rate of $\alpha$.

Input models: $a$, $b$, $c$  
Input hypers: $\alpha$

$m = a + \alpha(b - c)$

## `slerp(a, b, *, alpha: Hyper)`

Circular interpolation between $a$ and $b$.

Input models: $a$, $b$  
Input hypers: $\alpha$

$\Omega = arccos(\frac{a}{|a|} \frac{b}{|b|})$  
$m' = \frac{a}{|a|} \frac{sin((1 - \alpha) \Omega)}{sin \Omega} + \frac{b}{|b|} \frac{sin \alpha \Omega}{sin \Omega}$  
$m = (|a|(1-\alpha) + |b|\alpha) m'$

For more information: https://en.wikipedia.org/wiki/Slerp

