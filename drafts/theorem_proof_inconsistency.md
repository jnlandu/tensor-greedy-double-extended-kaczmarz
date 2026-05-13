# Inconsistency Between Theorem Statement and Proof

## Overview

The theorem promises a linear convergence bound in terms of three scalars
α, β, γ. However, each of those scalars is defined differently in the theorem
statement versus the proof, and the final inequality produced by the proof
differs from the one stated in the theorem. There are three concrete
discrepancies.

---

## Discrepancy 1 — γ is defined as its own reciprocal, then unused

### In the theorem statement

$$\gamma = \sigma^2_{\min}(\mathcal{A}^J)$$

A squared minimum singular value; a positive number.

### In the proof

$$\gamma = \frac{1}{\sigma^2_{\min}(\mathcal{A}^{J_k})}$$

The **reciprocal** of the above.

### Why this matters

The proof's induction step writes the coupling term as `(1/γ)β`. With the
proof's own definition, `1/γ = σ²_min`, so the coupling term is
`σ²_min · β` — a small, well-behaved factor that shrinks as the minimum
singular value grows. With the theorem's definition instead, `1/γ = 1/σ²_min`,
making the coupling term *larger* and the bound *worse*. The two definitions
give opposite roles to γ.

### The deeper problem

γ appears in the theorem's "where" clause but is **completely absent from the
theorem's bound formula**. The inequality the theorem actually states (see
below) contains no γ whatsoever. Defining a quantity and never using it in
the statement is both confusing and a sign that the statement was not
synchronized with the proof.

---

## Discrepancy 2 — α and β are built from different spectral quantities

|          | Theorem statement | Proof |
|----------|------------------|-------|
| **α**    | $1 - \theta\,\dfrac{\sigma^2_{\min}(\mathrm{bcirc}(\mathcal{A}^\top))}{\sigma^2_{\max}(\mathrm{bcirc}(\mathcal{A}^\top))}$ | $1 - \theta\,\dfrac{\sigma^2_{\min}(\mathrm{bcirc}(\mathcal{A}_{:,J,:}))}{\sigma^2_{\max}(\mathcal{A}_{:,J,:})}$ |
| **β**    | $1 - \dfrac{\sigma^2_{\max}(\mathcal{A})}{\|\mathcal{A}\|_F^2 - \|\mathcal{A}(\overline{U})\|_F^2}$ | $1 - \eta\,\dfrac{\sigma^2_{\min}(\mathrm{bcirc}(\mathcal{A}^\top))}{\|\mathcal{A}\|_F^2 - \|\mathcal{A}_{:,\overline{U},:}\|_F^2}$ |

### For α

- The **theorem** uses the full tensor transpose $\mathcal{A}^\top$ in both
  the numerator and denominator.
- The **proof** uses the accumulated greedy row-block
  $\mathcal{A}_{:,J,:}$, the union of all active sets seen so far.

These are different operators. The proof's version is what actually falls out
of the x-step analysis; the theorem's version looks like it was copied from
the z-step formula.

### For β

- The **theorem** uses $\sigma^2_{\max}(\mathcal{A})$ — the largest singular
  value — and has **no** η factor.
- The **proof** uses $\eta \cdot \sigma^2_{\min}(\mathrm{bcirc}(\mathcal{A}^\top))$ —
  the *smallest* singular value of the transpose, scaled by the greedy
  threshold η.

The two quantities are not just scaled versions of each other: max vs. min
singular value are conceptually opposite. Moreover, η is the single most
important parameter of the TGDBEK algorithm, and its absence from the
theorem's β means the theorem does not capture how the threshold controls
the convergence rate.

---

## Discrepancy 3 — The final bound formula itself is different

### Theorem states

$$\|\mathcal{X}^{(k+1)} - \mathcal{X}_\ast\|_F^2
\;\leq\;
\alpha^{k+1}\|\mathcal{X}^{(0)} - \mathcal{X}_\ast\|_F^2
+
\frac{\alpha^{k+1}-\beta^{k+1}}{\alpha - \beta}
\,\|\mathcal{B}^\perp\|_F^2$$

### Proof produces

$$\|\mathcal{X}^{(k+1)} - \mathcal{X}_\ast\|_F^2
\;\leq\;
\alpha^{k+1}\|\mathcal{X}^{(0)} - \mathcal{X}_\ast\|_F^2
+
\frac{\alpha^{k+1}-\beta^{k+1}}{\alpha - \beta}
\cdot\frac{\beta}{\alpha}
\,\|\mathcal{B}_{R(\mathcal{A})}\|_F^2$$

### Two differences

**Different norm.** The theorem has $\|\mathcal{B}^\perp\|_F^2$, where
$\mathcal{B}^\perp = \mathcal{B} - \mathcal{A}\mathcal{A}^\dagger\mathcal{B}$
is the component of $\mathcal{B}$ **orthogonal** to the range of
$\mathcal{A}$. The proof has $\|\mathcal{B}_{R(\mathcal{A})}\|_F^2$, which is
the component of $\mathcal{B}$ **in** the range of $\mathcal{A}$, i.e.,
$\mathcal{A}\mathcal{A}^\dagger\mathcal{B}$. These two are the complementary
parts of the orthogonal decomposition $\mathcal{B} = \mathcal{B}_{R(\mathcal{A})} + \mathcal{B}^\perp$;
one is not a scalar multiple of the other.

**Different coefficient.** The theorem has coefficient **1** in front of the
norm. The proof has the extra factor **β/α**, which is less than 1 when
β < α (which holds generically). Omitting β/α makes the theorem's bound
looser, but the mismatch still means the two inequalities are not the same
statement.

---

## Root Cause

The theorem and the proof were not kept in sync. Most likely the theorem
statement was written first, based on an intended result, and then the proof
was carried out separately — the proof's algebra revealed different constants
(the row-block A_{:,J,:} for α, the η-weighted min singular value for β, the
range component B_{R(A)} for the residual norm), but the theorem was never
updated to match.

---

## What Needs to Change

The proof's derivation is the ground truth. The fix is to **rewrite the
theorem statement** to match what the proof actually shows:

1. **Replace** the theorem's α with the proof's version:
   $\alpha = 1 - \theta\,\sigma^2_{\min}(\mathrm{bcirc}(\mathcal{A}_{:,J,:}))\,/\,\sigma^2_{\max}(\mathcal{A}_{:,J,:})$.

2. **Replace** the theorem's β with the proof's version, restoring the η factor:
   $\beta = 1 - \eta\,\sigma^2_{\min}(\mathrm{bcirc}(\mathcal{A}^\top))\,/\,(\|\mathcal{A}\|_F^2 - \|\mathcal{A}_{:,\overline{U},:}\|_F^2)$.

3. **Either** remove γ from the where-clause entirely, or define it as
   $\gamma = 1/\sigma^2_{\min}(\mathcal{A}^J)$ (the proof's version) and use
   it explicitly inside the bound.

4. **Replace** $\|\mathcal{B}^\perp\|_F^2$ with
   $(\beta/\alpha)\|\mathcal{B}_{R(\mathcal{A})}\|_F^2$ in the bound, or
   re-examine whether a tighter z-step bound can recover the cleaner
   $\|\mathcal{B}^\perp\|_F^2$ form (which would also require justifying why
   $\mathcal{B}^\perp$ — not $\mathcal{B}_{R(\mathcal{A})}$ — is the right
   quantity to bound the residual coupling term).








Does the bound imply convergence to $\mathcal{X}_*$ as $k \to \infty$?
The updated bound is:

$$|\mathcal{X}^{(k+1)} - \mathcal{X}_|F^2 ;\leq; \underbrace{\alpha^{k+1}|\mathcal{X}^{(0)} - \mathcal{X}|F^2}{\text{Term 1}} + \underbrace{\frac{\alpha^{k+1}-\beta^{k+1}}{\alpha - \beta}\cdot\frac{\beta}{\alpha},|\mathcal{B} - \mathcal{B}^\perp|F^2}{\text{Term 2}}$$

Term 1 — vanishes if $\alpha < 1$
$$\alpha^{k+1} \xrightarrow{k\to\infty} 0 \quad \Longleftrightarrow \quad \alpha < 1$$

$$\alpha = 1 - \theta,\frac{\sigma^2_{\min}(\mathrm{bcirc}(\mathcal{A}{J,:,:}))}{\sigma^2{\max}(\mathcal{A}_{J,:,:})} < 1$$

as long as $\theta > 0$ (the error has a non-zero component in $\mathrm{range}((\mathcal{A}{J,:,:})^\top)$) and $\sigma{\min}(\mathrm{bcirc}(\mathcal{A}_{J,:,:})) > 0$ (the accumulated row-block has full column rank). Since $\theta \leq 1$ and the singular value ratio $\leq 1$, we also have $\alpha \geq 0$, so $\alpha \in (0,1)$ under these conditions. ✓

Term 2 — vanishes if $\beta < 1$ as well
Use the geometric series identity:

$$\frac{\alpha^{k+1}-\beta^{k+1}}{\alpha - \beta} = \sum_{l=0}^{k} \alpha^l \beta^{k-l}$$

Each of the $k+1$ terms goes to 0, and the sum is bounded by $(k+1)\max(\alpha,\beta)^k \to 0$ since $\max(\alpha,\beta) < 1$. Therefore Term 2 $\to 0$ as well, and the full right-hand side collapses to 0. ✓

$$\beta = 1 - \eta,\frac{\sigma^2_{\min}(\mathrm{bcirc}(\mathcal{A}^\top))}{|\mathcal{A}|F^2 - |\mathcal{A}{:,\overline{U},:}|_F^2} < 1$$

provided $\eta > 0$, $\sigma_{\min}(\mathrm{bcirc}(\mathcal{A}^\top)) > 0$ (i.e. $\mathcal{A}$ has full column rank), and critically:

$$|\mathcal{A}_{:,\overline{U},:}|_F^2 < |\mathcal{A}|_F^2$$

This means the active column-block never exhausts the entire Frobenius norm of $\mathcal{A}$. If at some step $U_k = [N_2]$ (all columns selected), the denominator hits zero and $\beta$ is undefined. In practice this edge case means the z-step has already driven $\mathcal{Z}^{(k)}$ to $\mathcal{B}^\perp$ exactly, so convergence holds trivially — but the proof does not address this case explicitly.

Answer
Yes, the bound implies $|\mathcal{X}^{(k)} - \mathcal{X}_*|_F^2 \to 0$ as $k \to \infty$, provided:

Condition	Guarantees
$\sigma_{\min}(\mathrm{bcirc}(\mathcal{A}_{J,:,:})) > 0$ and $\theta > 0$	$\alpha \in (0,1)$ → Term 1 $\to 0$
$\sigma_{\min}(\mathrm{bcirc}(\mathcal{A}^\top)) > 0$, $\eta > 0$, and $|\mathcal{A}_{:,\overline{U},:}|_F^2 < |\mathcal{A}|_F^2$	$\beta \in (0,1)$ → Term 2 $\to 0$
Neither condition is currently stated explicitly in the theorem — this is a gap the paper should close, ideally by adding a corollary or a remark right after the theorem that states these sufficient conditions and confirms $\alpha, \beta \in (0,1)$ under them.





## Convergence analysis
Does the bound imply convergence to $\mathcal{X}_*$ as $k \to \infty$?
The updated bound is:

$$|\mathcal{X}^{(k+1)} - \mathcal{X}_|F^2 ;\leq; \underbrace{\alpha^{k+1}|\mathcal{X}^{(0)} - \mathcal{X}|F^2}{\text{Term 1}} + \underbrace{\frac{\alpha^{k+1}-\beta^{k+1}}{\alpha - \beta}\cdot\frac{\beta}{\alpha},|\mathcal{B} - \mathcal{B}^\perp|F^2}{\text{Term 2}}$$

Term 1 — vanishes if $\alpha < 1$
$$\alpha^{k+1} \xrightarrow{k\to\infty} 0 \quad \Longleftrightarrow \quad \alpha < 1$$

$$\alpha = 1 - \theta,\frac{\sigma^2_{\min}(\mathrm{bcirc}(\mathcal{A}{J,:,:}))}{\sigma^2{\max}(\mathcal{A}_{J,:,:})} < 1$$

as long as $\theta > 0$ (the error has a non-zero component in $\mathrm{range}((\mathcal{A}{J,:,:})^\top)$) and $\sigma{\min}(\mathrm{bcirc}(\mathcal{A}_{J,:,:})) > 0$ (the accumulated row-block has full column rank). Since $\theta \leq 1$ and the singular value ratio $\leq 1$, we also have $\alpha \geq 0$, so $\alpha \in (0,1)$ under these conditions. ✓

Term 2 — vanishes if $\beta < 1$ as well
Use the geometric series identity:

$$\frac{\alpha^{k+1}-\beta^{k+1}}{\alpha - \beta} = \sum_{l=0}^{k} \alpha^l \beta^{k-l}$$

Each of the $k+1$ terms goes to 0, and the sum is bounded by $(k+1)\max(\alpha,\beta)^k \to 0$ since $\max(\alpha,\beta) < 1$. Therefore Term 2 $\to 0$ as well, and the full right-hand side collapses to 0. ✓

$$\beta = 1 - \eta,\frac{\sigma^2_{\min}(\mathrm{bcirc}(\mathcal{A}^\top))}{|\mathcal{A}|F^2 - |\mathcal{A}{:,\overline{U},:}|_F^2} < 1$$

provided $\eta > 0$, $\sigma_{\min}(\mathrm{bcirc}(\mathcal{A}^\top)) > 0$ (i.e. $\mathcal{A}$ has full column rank), and critically:

$$|\mathcal{A}_{:,\overline{U},:}|_F^2 < |\mathcal{A}|_F^2$$

This means the active column-block never exhausts the entire Frobenius norm of $\mathcal{A}$. If at some step $U_k = [N_2]$ (all columns selected), the denominator hits zero and $\beta$ is undefined. In practice this edge case means the z-step has already driven $\mathcal{Z}^{(k)}$ to $\mathcal{B}^\perp$ exactly, so convergence holds trivially — but the proof does not address this case explicitly.

Answer
Yes, the bound implies $|\mathcal{X}^{(k)} - \mathcal{X}_*|_F^2 \to 0$ as $k \to \infty$, provided:

Condition	Guarantees
$\sigma_{\min}(\mathrm{bcirc}(\mathcal{A}_{J,:,:})) > 0$ and $\theta > 0$	$\alpha \in (0,1)$ → Term 1 $\to 0$
$\sigma_{\min}(\mathrm{bcirc}(\mathcal{A}^\top)) > 0$, $\eta > 0$, and $|\mathcal{A}_{:,\overline{U},:}|_F^2 < |\mathcal{A}|_F^2$	$\beta \in (0,1)$ → Term 2 $\to 0$
Neither condition is currently stated explicitly in the theorem — this is a gap the paper should close, ideally by adding a corollary or a remark right after the theorem that states these sufficient conditions and confirms $\alpha, \beta \in (0,1)$ under them.




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
