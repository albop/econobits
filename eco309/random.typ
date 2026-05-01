#import "@preview/touying:0.7.1": *
#import themes.simple: *

#show: simple-theme.with(aspect-ratio: "16-9")

= Title

== First Slide

Hello, Touying!

#pause

Hello, Typst!

== STeady state:

$ k = (1-delta) k + k^alpha s $

$ delta k^(1-alpha) = s $

$ k = (s/delta)^(1/(1-alpha)) $


For a recursvie sequence $x_n = f(x_(n-1))$, close to $overline(x)$, we have:

$ (x_(n+1) - overline(x))/(x_(n) - overline(x)) approx (x_(n) - overline(x))/(x_(n-1) - overline(x)) $

we get a guess:

$ overline(x) =  x_(t-1) - frac((x_t-x_(t-1))^2, x_(t+1)-2 x_t + x_(t-1)) $

== Markov Chain

Stochastic process:

- $(X_t)_t$: random variables indexed by $t$

Markov chain: stochastic process whose ewvolution after $t$, only depends on $X_t$ 
- memory-less property

Random walk ( in discrete time)

$ x_t = x_(t-1) + epsilon_t$

-> Brownian motion in continuous time

Conitinous time/ states +  brownian => stochastic calculus

$x' = alpha x + d omega$


== Iterates of P

Show that if $P$ is a stochastic matrix then $P^k$ is a stochastic matrix


Assume $P^(k-1)$ is  a stochastic matrix,

$ P P^(k-1) = mat(mu_1; mu_2; ... ; mu_n) P^(k-1) = mat(mu_1 P^(k-1); mu_2 P^(k-1); ... ; mu_n P^(k-1))$

where $mu_1, ... mu_n$ are probability vectors.
We see that all rows are probability vectors: $P^k$ is a stochastic matrix.

== Question

Given $mu_0$ and stochastic matrix $P$.

- Does a steady-state $overline(mu)$ exist s.t. $overline(mu)' P = overline(mu)'$ ?
- Does $(mu_0)' P^k$ converge to anything ?
