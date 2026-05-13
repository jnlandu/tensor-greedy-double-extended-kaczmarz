# Are $\alpha$ and $\beta$ Always in $(0,1)$?

## Context

The convergence bound of the TGDBEK theorem depends on two contraction
factors:

$$\alpha = 1 - \theta\,
\frac{\sigma^2_{\min}(\mathrm{bcirc}(\mathcal{A}_{J,:,:}))}
     {\sigma^2_{\max}(\mathcal{A}_{J,:,:})},
\qquad
\beta = 1 - \eta\,
\frac{\sigma^2_{\min}(\mathrm{bcirc}(\mathcal{A}^\top))}
     {\|\mathcal{A}\|_F^2 - \|\mathcal{A}_{:,\overline{U},:}\|_F^2}.$$

For the bound to imply convergence, both must lie in $[0,1)$.
The upper bound ($< 1$) is straightforward; the lower bound ($\geq 0$)
is non-trivial, especially for $\beta$.

---

## $\alpha \in [0, 1)$

### $\alpha \geq 0$ — always guaranteed

Since $\theta \leq 1$ and $\sigma^2_{\min} \leq \sigma^2_{\max}$, the
subtracted term is at most $1$, so $\alpha \geq 0$ always holds.

Equality $\alpha = 0$ occurs only when $\theta = 1$ **and** the accumulated
row-block $\mathcal{A}_{J,:,:}$ is a scaled isometry (all singular values
equal). This means the x-step converges in one shot — a best case, not a
problem.

### $\alpha < 1$ — two mild conditions

| Condition | Role |
|-----------|------|
| $\theta > 0$ | The error $\mathcal{X}^{(k)} - \mathcal{X}_*$ has a non-zero component in $\mathrm{range}((\mathcal{A}_{J_k,:,:})^\top)$. Fails only if the error already lies in $\mathrm{null}(\mathcal{A}_{J_k,:,:})$, i.e., the x-step is already at the solution. |
| $\sigma_{\min}(\mathrm{bcirc}(\mathcal{A}_{J,:,:})) > 0$ | The accumulated row-block has full column rank. Standard for overdetermined systems. |

### Conclusion for $\alpha$

$$\alpha \in [0, 1) \quad \text{always;}$$

strict positivity $\alpha > 0$ requires $\theta < 1$ or $\sigma_{\min} <
\sigma_{\max}$, but $\alpha = 0$ only accelerates convergence.

---

## $\beta \in [0, 1)$

### $\beta < 1$ — three conditions

| Condition | Role |
|-----------|------|
| $\eta > 0$ | Always satisfied by assumption ($\eta \in (0,1]$). |
| $\sigma_{\min}(\mathrm{bcirc}(\mathcal{A}^\top)) > 0$ | $\mathcal{A}$ has full column rank in the t-product sense. |
| $\|\mathcal{A}_{:,\overline{U},:}\|_F^2 < \|\mathcal{A}\|_F^2$ | The active column-block never covers all of $\mathcal{A}$; the denominator stays positive. |

### $\beta \geq 0$ — not obvious from the formula, but guaranteed by geometry

The formula can *look* negative. When
$\eta\,\sigma^2_{\min} > \|\mathcal{A}\|_F^2 - \|\mathcal{A}_{:,\overline{U},:}\|_F^2$
(e.g., large $\eta$, well-conditioned $\mathcal{A}$, large active block),
the expression $1 - \eta\,\sigma^2_{\min}/\mathrm{denom}$ is negative.
Yet $\beta < 0$ is **impossible**. Here is why.

#### The projection argument

The z-step update is a **projection**:

$$\mathcal{Z}^{(k+1)}
= \mathcal{Z}^{(k)}
- (\mathcal{A}_{:,U_k,:}^\top)^\dagger
  * \mathcal{A}_{:,U_k,:}^\top
  * (\mathcal{Z}^{(k)} - \mathcal{B}^\perp).$$

Let $D = \|\text{projected term}\|_F^2$. Since projections are contractions:

$$0 \;\leq\; D \;\leq\; \|\mathcal{Z}^{(k)} - \mathcal{B}^\perp\|_F^2.$$

Define the true contraction coefficient
$C_k = D\,/\,\|\mathcal{Z}^{(k)} - \mathcal{B}^\perp\|_F^2 \in [0,1]$.

The proof lower-bounds $D$ and shows:

$$C_k \;\geq\; \frac{\eta\,\sigma^2_{\min}(\mathrm{bcirc}(\mathcal{A}^\top))}{\|\mathcal{A}\|_F^2 - \|\mathcal{A}_{:,\overline{U},:}\|_F^2},$$

so $\beta = 1 - \eta\,\sigma^2_{\min}/\mathrm{denom} \leq 1 - C_k$.

#### Why $\beta \geq 0$ follows

Since $C_k \in [0,1]$, we have $1 - C_k \geq 0$. If $\beta < 0$, then

$$1 - C_k \;\leq\; \beta \;<\; 0,$$

which contradicts $1 - C_k \geq 0$.

Therefore $\beta \geq 0$ **must hold** — the geometry of the projection
forces it, regardless of how the formula looks.

> **Gap in the paper.** This implicit guarantee is never stated. The proof
> defines $\beta$ by a formula that can appear negative and uses it directly
> without verifying $\beta \geq 0$. A one-line remark after the definition
> of $\beta$ — noting that $C_k \leq 1$ forces $\beta \geq 0$ — would close
> this gap.

---

## Summary Table

| Factor | $\geq 0$? | $< 1$? | Gap in current proof |
|--------|-----------|--------|----------------------|
| $\alpha$ | Always (ratio $\leq 1$, $\theta \leq 1$) | Under $\theta > 0$ and $\sigma_{\min} > 0$ | Conditions not stated explicitly in the theorem |
| $\beta$ | Yes — projection contraction forces it implicitly | Under $\eta > 0$, $\sigma_{\min} > 0$, $\|\mathcal{A}_{:,\overline{U},:}\|_F^2 < \|\mathcal{A}\|_F^2$ | The geometric argument for $\beta \geq 0$ is nowhere written down |

Both factors lie in $[0,1)$ under standard assumptions, but the proof leaves
the non-negativity of $\beta$ entirely implicit. Making this explicit would
strengthen the theorem and pre-empt a natural referee question.




"$\Vert \fm A^\dagger * \fm X \Vert_F \leq \Vert \fm A^\dagger \Vert_2^2 \Vert \fm X \Vert_F^2$" has both the exponent and the norm type wrong. The standard sub-multiplicativity bound is
$$\Vert \fm A^\dagger * \fm X \Vert_F \leq \Vert \fm A^\dagger \Vert_2 \Vert \fm X \Vert_F.$$
