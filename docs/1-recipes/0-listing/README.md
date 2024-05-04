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

$\omega = arccos(\frac{a}{|a|} \frac{b}{|b|})$  
$n = \frac{a}{|a|} \frac{sin((1 - \alpha) \omega)}{sin \omega} + \frac{b}{|b|} \frac{sin \alpha \omega}{sin \omega}$  
$m = (|a|(1-\alpha) + |b|\alpha) n$
