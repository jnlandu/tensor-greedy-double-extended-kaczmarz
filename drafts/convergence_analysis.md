# Convergence of TGDBEK as $k \to \infty$

## The Bound

The corrected theorem gives, for all $k \geq 0$:

$$\|\mathcal{X}^{(k+1)} - \mathcal{X}_*\|_F^2
\;\leq\;
\underbrace{\alpha^{k+1}\|\mathcal{X}^{(0)} - \mathcal{X}_*\|_F^2}_{\text{Term 1}}
\;+\;
\underbrace{\frac{\alpha^{k+1}-\beta^{k+1}}{\alpha - \beta}
\cdot\frac{\beta}{\alpha}
\,\|\mathcal{B} - \mathcal{B}^\perp\|_F^2}_{\text{Term 2}}$$

For convergence $\mathcal{X}^{(k)} \to \mathcal{X}_*$, both terms must vanish as
$k \to \infty$. This requires $\alpha \in (0,1)$ and $\beta \in (0,1)$.

---

## Term 1 — Vanishes if $\alpha < 1$

$$\alpha^{k+1} \;\xrightarrow{k\to\infty}\; 0
\quad\Longleftrightarrow\quad
\alpha < 1,$$

where

$$\alpha = 1 - \theta\,
\frac{\sigma^2_{\min}\!\left(\mathrm{bcirc}(\mathcal{A}_{J,:,:})\right)}
     {\sigma^2_{\max}\!\left(\mathcal{A}_{J,:,:}\right)}.$$

**When is $\alpha < 1$?**  Whenever:

- $\theta > 0$: the error $\mathcal{X}^{(k)} - \mathcal{X}_*$ has a non-zero
  component in $\mathrm{range}\!\left((\mathcal{A}_{J_k,:,:})^\top\right)$, and
- $\sigma_{\min}(\mathrm{bcirc}(\mathcal{A}_{J,:,:})) > 0$: the accumulated
  row-block $\mathcal{A}_{J,:,:}$ has full column rank.

**When is $\alpha \geq 0$?**  Since $\theta \leq 1$ and
$\sigma^2_{\min} \leq \sigma^2_{\max}$, the subtracted term is at most $1$,
so $\alpha \geq 0$ always.

Therefore, under the two conditions above, $\alpha \in (0,1)$ and Term 1 $\to 0$. $\checkmark$

---

## Term 2 — Vanishes if $\beta < 1$ as Well

Apply the geometric series identity:

$$\frac{\alpha^{k+1}-\beta^{k+1}}{\alpha - \beta}
= \sum_{l=0}^{k} \alpha^{\,l}\,\beta^{\,k-l}.$$

Each summand $\alpha^l \beta^{k-l} \leq \max(\alpha,\beta)^k \to 0$.
The full sum satisfies

$$\sum_{l=0}^{k} \alpha^{\,l}\,\beta^{\,k-l}
\;\leq\; (k+1)\max(\alpha,\beta)^k \;\to\; 0,$$

since $r^k / k \to 0$ for any $r \in (0,1)$.
Hence Term 2 $\to 0$ as well, and the entire right-hand side collapses to $0$. $\checkmark$

**When is $\beta < 1$?**

$$\beta = 1 - \eta\,
\frac{\sigma^2_{\min}\!\left(\mathrm{bcirc}(\mathcal{A}^\top)\right)}
     {\|\mathcal{A}\|_F^2 - \|\mathcal{A}_{:,\overline{U},:}\|_F^2}.$$

$\beta < 1$ holds whenever:

| Condition | Role |
|-----------|------|
| $\eta > 0$ | Always satisfied by assumption ($\eta \in (0,1]$). |
| $\sigma_{\min}(\mathrm{bcirc}(\mathcal{A}^\top)) > 0$ | $\mathcal{A}$ has full column rank (in the t-product sense). |
| $\|\mathcal{A}_{:,\overline{U},:}\|_F^2 < \|\mathcal{A}\|_F^2$ | The active column-block never exhausts the full Frobenius norm of $\mathcal{A}$. |

The third condition is the subtle one. It fails only if $U_k = [N_2]$ (all column
slices are simultaneously selected), which would make the denominator zero and
$\beta$ undefined. In that edge case the z-step projection is a full orthogonal
projection onto $\mathrm{null}(\mathcal{A}^\top)$, so $\mathcal{Z}^{(k)}$ has
already reached $\mathcal{B}^\perp$ exactly and convergence holds trivially.
The current proof does not address this case explicitly — this is a gap.

---

## Summary

| Term | Vanishes when | Sufficient condition |
|------|--------------|----------------------|
| $\alpha^{k+1}\|\mathcal{X}^{(0)}-\mathcal{X}_*\|_F^2$ | $\alpha \in (0,1)$ | $\theta > 0$ and $\mathcal{A}_{J,:,:}$ has full column rank |
| $\frac{\alpha^{k+1}-\beta^{k+1}}{\alpha-\beta}\cdot\frac{\beta}{\alpha}\|\mathcal{B}-\mathcal{B}^\perp\|_F^2$ | $\beta \in (0,1)$ | $\mathcal{A}$ has full column rank and $\|\mathcal{A}_{:,\overline{U},:}\|_F^2 < \|\mathcal{A}\|_F^2$ |

**Answer:** Yes, the bound implies $\|\mathcal{X}^{(k)} - \mathcal{X}_*\|_F^2 \to 0$
as $k \to \infty$, provided $\alpha, \beta \in (0,1)$.
These conditions are not currently stated explicitly in the theorem — the paper
should add a remark or corollary confirming they hold under standard
assumptions (full column rank of $\mathcal{A}$ and $\eta > 0$).
