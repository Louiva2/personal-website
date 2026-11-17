---
title: Why the harmonic series diverges
date: 2026-11-14
summary: Grouping and a comparison with an integral show why the harmonic numbers have no upper bound, and how slowly they grow. This tidies up textbook material; it is not a new result.
---

The harmonic series

$$
\sum_{n=1}^{\infty}\frac{1}{n}=1+\frac{1}{2}+\frac{1}{3}+\frac{1}{4}+\cdots
$$

is an example one meets early in analysis: the terms tend to zero, yet the sum is not a finite number. This is not a new result. The grouping below is usually credited to Nicole Oresme, in the fourteenth century. All this note does is write out two common ways of seeing it, so that they are easy to come back to.

Write the sum of the first $n$ terms, the $n$-th harmonic number, as

$$
H_n=\sum_{k=1}^{n}\frac{1}{k}.
$$

The question is whether $H_n$ tends to a finite limit as $n\to\infty$.

## Grouping

Take a positive integer $n$ and cut $H_{2^n}$ at the powers of 2:

$$
\begin{aligned}
H_{2^n}
&=1+\frac{1}{2}
+\left(\frac{1}{3}+\frac{1}{4}\right)
+\left(\frac{1}{5}+\cdots+\frac{1}{8}\right)\\
&\quad+\cdots
+\left(\frac{1}{2^{n-1}+1}+\cdots+\frac{1}{2^n}\right).
\end{aligned}
$$

Apart from the leading $1$, every piece that follows is at least $\frac{1}{2}$.

The lone $\frac{1}{2}$ is $\frac{1}{2}$ itself. For $k\ge 2$, the piece

$$
\frac{1}{2^{k-1}+1}+\cdots+\frac{1}{2^k}
$$

has $2^{k-1}$ terms, each at least $1/2^k$, so

$$
\frac{1}{2^{k-1}+1}+\cdots+\frac{1}{2^k}
\ge 2^{k-1}\cdot\frac{1}{2^k}
=\frac{1}{2}.
$$

From $k=2$ to $k=n$ there are $n-1$ pieces; with the lone $\frac{1}{2}$ that makes $n$ copies of $\frac{1}{2}$. When $n=1$ there are no brackets after it and only $1+\frac{1}{2}$ is left, so the claim holds there too. Hence

$$
H_{2^n}\ge 1+\frac{n}{2}.
$$

The right-hand side grows without bound as $n$ does, so $(H_m)$ has no upper bound and therefore does not converge (a convergent sequence is bounded). It is also increasing, so it tends to plus infinity. The harmonic series diverges.

Terms tending to zero is only a necessary condition for convergence. Here $\frac{1}{n}\to 0$, the condition holds, and the series still diverges.

## Against the logarithm

$\frac{1}{x}$ is decreasing on $[1,\infty)$, so for every integer $k\ge 1$,

$$
\int_{k}^{k+1}\frac{1}{x}\,dx\le\frac{1}{k}.
$$

For $k\ge 2$ there is the other side as well:

$$
\frac{1}{k}\le\int_{k-1}^{k}\frac{1}{x}\,dx.
$$

Summing the first from $k=1$ to $n$ and the second from $k=2$ to $n$ gives

$$
\ln(n+1)\le H_n\le 1+\ln n.
$$

The lower bound already tends to infinity, which is another proof of divergence. The two bounds also say how fast: $H_n$ and $\ln n$ differ by a bounded amount, that is, $H_n=\ln n+O(1)$. It diverges, but slowly. Multiply $n$ by a hundred and the logarithm goes up by about $\ln 100$, less than five units, and the harmonic number gains only about that much.

Tightening the $O(1)$, one can show that the limit

$$
\gamma=\lim_{n\to\infty}\left(H_n-\ln n\right)
$$

exists. Write $a_n=H_n-\ln n$. Then

$$
a_n-a_{n+1}
=\ln\left(1+\frac{1}{n}\right)-\frac{1}{n+1}
=\int_{n}^{n+1}\left(\frac{1}{x}-\frac{1}{n+1}\right)dx.
$$

On the interval $[n,n+1]$ we have $\frac{1}{x}\ge\frac{1}{n+1}$; the integrand is non-negative and not identically zero, so $a_n>a_{n+1}$ and the sequence is decreasing. On the other hand, by the lower bound above,

$$
a_n=H_n-\ln n\ge\ln(n+1)-\ln n=\ln\left(1+\frac{1}{n}\right)>0.
$$

So $(a_n)$ is decreasing and bounded below, the limit exists, and $\gamma\ge 0$. This number is called the Euler–Mascheroni constant. Its value is approximately $0.577$; the decimals are not computed here. A slightly finer statement is therefore

$$
H_n=\ln n+\gamma+o(1).
$$

## To keep

The partial sums have no upper bound, yet they grow only as slowly as a logarithm. Both can be true at once. Grouping shows that the series diverges, the comparison with an integral shows how slowly, and Euler’s constant marks the remainder after $\ln n$ that slowly settles down.
