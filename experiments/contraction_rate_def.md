# Empirical Contraction Rate $\hat\beta$

## Definition

The linear convergence guarantee for all methods takes the form

$$
\mathrm{RSE}_k^X \;=\; \frac{\|\mathcal{X}^{(k)} - \mathcal{X}_*\|_F^2}{\|\mathcal{X}_*\|_F^2} \;\leq\; \beta^k \cdot \mathrm{RSE}_0^X, \qquad 0 < \beta < 1,
$$

where $\beta$ is the theoretical per-iteration contraction factor.

Given an observed history $\{r_k\}_{k=0}^{K}$ with $r_k = \mathrm{RSE}_k^X$, we estimate $\beta$ by fitting the log-linear model

$$
\log r_k \;\approx\; a + k\,\log\hat\beta,
$$

via ordinary least squares:

$$
\boxed{
\log\hat\beta \;=\; \frac{\displaystyle\sum_{k=0}^{K} k\,\log r_k \;-\; \frac{1}{K+1}\!\left(\sum_k k\right)\!\left(\sum_k \log r_k\right)}
{\displaystyle\sum_{k=0}^{K} k^2 \;-\; \frac{1}{K+1}\!\left(\sum_k k\right)^{\!2}}
}
$$

so that $\hat\beta = \exp(\log\hat\beta) \in (0, 1)$.

## Interpretation

| Value of $\hat\beta$ | Meaning |
|---|---|
| Close to $0$ | RSE halves in very few iterations — very fast convergence |
| Close to $1$ | RSE barely decreases per step — slow convergence |

A smaller $\hat\beta$ means fewer iterations are needed to reach any fixed tolerance.

## Relation to iteration count

For a method with contraction rate $\hat\beta$ starting from $r_0 = 1$, the number of iterations to reach tolerance $\varepsilon$ is approximately

$$
K \;\approx\; \frac{\log\varepsilon}{\log\hat\beta}.
$$

So the ratio of iteration counts between two methods with rates $\hat\beta_1 < \hat\beta_2$ is

$$
\frac{K_2}{K_1} \;=\; \frac{\log\hat\beta_1}{\log\hat\beta_2} \;>\; 1.
$$

## Why log-linear fit rather than the two-point formula?

The two-point estimator $\hat\beta = (r_K/r_0)^{1/K}$ is sensitive to the endpoint $r_K$: if the method just converged, $r_K$ is at the tolerance floor and $\hat\beta$ is underestimated; if it hit the iteration budget without converging, $r_K$ is large and $\hat\beta$ is overestimated. The OLS fit uses the full history and is therefore more stable across both converging and non-converging runs.

## Z-iterate analogue

For TGDBEK the same estimator is applied to the Z-RSE history $\{r_k^Z\}$ to obtain $\hat\beta_Z$:

$$
r_k^Z \;=\; \frac{\|\mathcal{Z}^{(k)} - \mathcal{B}^\perp\|_F^2}{\|\mathcal{B} - \mathcal{B}^\perp\|_F^2},
$$

where $\mathcal{B}^\perp = \mathcal{B} - \mathcal{A}*\mathcal{A}^\dagger*\mathcal{B}$ is the component of $\mathcal{B}$ orthogonal to the range of $\mathcal{A}$. Empirically $\hat\beta_Z < \hat\beta_X$, confirming that the Z-iterate contracts faster than the X-iterate.
