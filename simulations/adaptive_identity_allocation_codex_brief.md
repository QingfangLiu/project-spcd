# Adaptive Identity Allocation: Simulation Project Brief

## Project goal

Build a small simulation project that formalizes and visualizes a **normative model of adaptive outcome-identity representation**.

The model is intended to support the following theoretical idea:

> Detailed outcome-identity representations should be allocated where they provide the greatest marginal improvement in decision quality. In a task containing both directly learned and inferred cues, explicit identity representation can be more computationally valuable for inferred outcomes because directly learned cues may already have a reliable alternative associative route to outcome identity.

This project is **not** intended to fit behavioral or fMRI data. The goal is to produce a transparent proof-of-principle simulation and a publication-quality conceptual figure that makes the optimization logic intuitive.

The model should **not** hard-code that preconditioned cues receive more representation. Preferential representation of inferred outcomes should emerge only under parameter regimes where explicit identity representation has greater computational utility for those cues.

---

## Core variables

Consider two cue classes:

- `P`: preconditioned / inferred cue
- `C`: directly conditioned cue

For cue \(i \in \{P,C\}\):

- \(r_i \ge 0\): representational resource / strength / fidelity allocated to the cue's outcome identity
- \(a_i > 0\): computational value of explicitly representing the cue's outcome identity
- \(L_i\): decision loss associated with insufficient identity representation

Use the single-cue loss function:

\[
L_i = a_i e^{-r_i}
\]

Interpretation:

- Larger \(r_i\) reduces decision loss.
- Larger \(a_i\) means failure to explicitly represent identity is more costly.
- The exponential form gives diminishing returns: the first units of representational resource help more than later units.
- The exponential form is chosen for simplicity and tractability, not because it has a special biological interpretation.

The total loss is:

\[
L = a_P e^{-r_P} + a_C e^{-r_C}
\]

subject to a finite representational budget:

\[
r_P + r_C \le R
\]

where \(R>0\) is total available representational resource.

For the main derivation, assume the full budget is used:

\[
r_P+r_C=R.
\]

---

## Optional interpretation of \(a_i\)

A useful decomposition is:

\[
a_i = D_i(1-q_i)
\]

where:

- \(D_i\): behavioral cost of using the wrong outcome identity for cue \(i\)
- \(q_i\): reliability of an alternative route that can provide outcome identity without requiring this explicit representation

This gives the model a concrete interpretation.

For a directly conditioned cue:

\[
C \rightarrow O
\]

the system may already have a relatively reliable direct associative route, so \(q_C\) can be high.

For a preconditioned cue:

\[
P \rightarrow C \rightarrow O
\]

there is no directly experienced \(P \rightarrow O\) association, so the alternative route may be less reliable and \(q_P\) may be lower.

If \(D_P=D_C\) and \(q_C>q_P\), then:

\[
a_P>a_C.
\]

Important conceptual point:

> The theory is **not** that preconditioned cues intrinsically deserve more OFC representation. The theory is that explicit identity representation should be stronger wherever its **marginal computational utility** is greater.

---

## Optimization derivation

With:

\[
L = a_P e^{-r_P} + a_C e^{-r_C}
\]

and:

\[
r_P+r_C=R,
\]

write:

\[
r_C=R-r_P.
\]

Then:

\[
L(r_P)=a_Pe^{-r_P}+a_Ce^{-(R-r_P)}.
\]

Differentiate:

\[
\frac{dL}{dr_P}
=
-a_Pe^{-r_P}+a_Ce^{-r_C}.
\]

At an interior optimum:

\[
\frac{dL}{dr_P}=0,
\]

so:

\[
a_Pe^{-r_P}=a_Ce^{-r_C}.
\]

Interpret these terms as marginal values:

- \(a_Pe^{-r_P}\): benefit of allocating a little more resource to \(P\)
- \(a_Ce^{-r_C}\): cost of removing that same amount from \(C\)

At the optimum, these are equal.

Rearranging:

\[
\frac{a_P}{a_C}=e^{r_P-r_C}
\]

and therefore:

\[
\boxed{
r_P-r_C=\log\frac{a_P}{a_C}
}
\]

Together with:

\[
r_P+r_C=R,
\]

the interior solution is:

\[
r_P^*=
\frac{R+\log(a_P/a_C)}{2}
\]

and:

\[
r_C^*=
\frac{R-\log(a_P/a_C)}{2}.
\]

The solution must satisfy:

\[
r_P\ge0,\qquad r_C\ge0.
\]

Therefore, when:

\[
\log\frac{a_P}{a_C}\ge R,
\]

the constrained optimum reaches the boundary:

\[
r_P^*=R,\qquad r_C^*=0.
\]

Similarly, if:

\[
\log\frac{a_C}{a_P}\ge R,
\]

then:

\[
r_C^*=R,\qquad r_P^*=0.
\]

This boundary behavior is important because it gives a qualitative regime resembling:

- strong inferred-outcome identity representation
- weak / absent directly learned identity representation

without hard-coding that pattern.

---

## Key predictions

### Equal computational value

If:

\[
a_P=a_C,
\]

then:

\[
r_P=r_C=R/2.
\]

### Greater value for inferred identity

If:

\[
a_P>a_C,
\]

then:

\[
r_P>r_C.
\]

### Strong asymmetry under limited resources

If \(a_P/a_C\) is sufficiently large relative to \(R\), then:

\[
r_P=R,\qquad r_C=0.
\]

### Direct-only environment

If only conditioned cues are present, the optimization reduces to:

\[
\min_{r_C} a_Ce^{-r_C}
\]

subject to:

\[
r_C\le R.
\]

Thus:

\[
r_C^*=R.
\]

This provides a simple conceptual explanation for why directly learned reward identity can be strongly represented in experiments containing only directly learned cues, while the same class of cues may receive less explicit representation when inferred cues are also present and have greater marginal computational value.

The model should make clear that this is a **normative explanation**, not a causal claim already established by the current experiment.

---

# Figure plan

Create one publication-quality figure with **five labeled panels and six plots** arranged in two rows of three. Panels A, B, and C occupy the top row. Panel D shows the simulated prediction at bottom left; panel E spans the remaining two bottom positions and tests the prediction in lateral and medial OFC.

Match the typography and spacing of Figures 2 and 3. Keep long parameter explanations in the caption, and use gray for simulation, teal for lateral OFC, and purple for medial OFC. Simulated coding has arbitrary units, while empirical coding uses RSA coefficients.

## Panel A — Benefits of identity coding

Plot:

\[
L_i=a_i e^{-r_i}
\]

with:

- x-axis: representational resource \(r_i\)
- y-axis: decision loss \(L_i\)

Show several values of \(a_i\), e.g. \(0.5,1,2,4\).

Purpose:

1. Show that more resource reduces loss.
2. Show diminishing returns.
3. Show that larger \(a_i\) means identity representation is more computationally valuable.

Suggested annotation:

> Higher \(a_i\): greater cost of failing to explicitly represent outcome identity.

Do not over-interpret \(r_i\) biologically. Label it as **representational resource** or **identity representational strength**.

---

## Panel B — Optimal allocation

Use an illustrative parameter set such as:

\[
R=4,\qquad a_P=4,\qquad a_C=1.
\]

Since:

\[
r_C=R-r_P,
\]

plot:

\[
L(r_P)=a_Pe^{-r_P}+a_Ce^{-(R-r_P)}
\]

for \(0\le r_P\le R\).

Axes:

- x-axis: \(r_P\)
- y-axis: total decision loss

Optionally add a secondary/top label showing \(r_C=R-r_P\).

Mark the minimum:

\[
r_P^*,r_C^*.
\]

For the example above:

\[
r_P^*\approx2.69,\qquad r_C^*\approx1.31.
\]

Purpose:

> Show visually that allocating too little resource to either cue increases total loss, and that the optimum is a balance determined by relative computational value.

Suggested annotations:

- left side: **Too little resource for P**
- right side: **Too little resource for C**
- minimum: **Optimal allocation**

---

## Panel C — Allocation follows utility

Plot:

\[
r_P-r_C
\]

as a function of:

\[
\frac{a_P}{a_C}.
\]

Use a log-scaled x-axis if that makes the relationship clearer.

For interior solutions:

\[
r_P-r_C=\log(a_P/a_C).
\]

Enforce the budget and non-negativity constraints in the actual simulation, so the curve saturates at \(+R\) and \(-R\).

Mark:

\[
a_P/a_C=1
\]

where:

\[
r_P-r_C=0.
\]

Indicate:

- \(a_P/a_C>1\): more resource to inferred outcome identity
- \(a_P/a_C<1\): more resource to directly learned outcome identity

Purpose:

> Show that preferential allocation is a consequence of relative computational utility, not cue category itself.

---

## Panel D — Model prediction

Simulate 100 participants with a common budget \(R=4\). Draw each
participant's \(\log(a_P/a_C)\) from a normal distribution with mean
\(\log 4\) and standard deviation 1.2, set \(a_C=1\), and calculate
\(r_P^*,r_C^*\) using the same constrained optimum as Panel B.

Plot direct identity coding on the x-axis and inferred identity coding on
the y-axis, using noisy measurements of those latent allocations:

\[
x_j=r_{C,j}^*+\epsilon_{C,j},\qquad
y_j=r_{P,j}^*+\epsilon_{P,j}.
\]

The measurement errors are independent, zero-mean Gaussian draws with
standard deviation 0.75 in arbitrary resource units. Use random seed
20261023 for reproducibility. Latent allocations remain nonnegative and sum
to \(R\); noisy readouts are not clipped. These readouts are illustrative
coding proxies, not simulated RSA coefficients, and need not lie in
\([-1,1]\).

Show the simulated Spearman correlation, a descriptive linear fit, and its
pointwise 95% confidence interval for the mean response. The shading is not
a confidence interval for Spearman rho. Do not assign empirical ROI labels
or significance stars to the simulated sample.

At a fixed budget without measurement noise, \(r_P^*=R-r_C^*\), so the
rank correlation is exactly \(-1\) when allocations vary across participants.
Independent readout noise attenuates this relationship. Across-participant
budget variation, if added in a future model, can further weaken or obscure
the negative association. A negative empirical correlation alone does not
establish resource competition.

All parameters are illustrative and are not optimized to reproduce the fMRI
correlation. This panel demonstrates a qualitative prediction of the
shared-budget assumption.

---

## Panel E — Test in fMRI data

Use the existing first-cue, premeal odor-identity RSA coefficients for conditioned and preconditioned odor-predictive cues. Retain the same subject exclusions as Figure 3 (`sub-19`, `sub-31`, and `sub-33`), leaving 31 participants. Place lateral OFC and medial OFC in matched plots under one shared E heading.

Plot conditioned-cue identity coding on x and preconditioned-cue identity coding on y, with RSA Spearman coefficients as the units. Use common empirical axis limits, zero-reference lines, and the original one-sided, uncorrected Spearman tests of negative associations:

- Lateral OFC: rho = -0.316564, p = 0.041365.
- Medial OFC: rho = -0.029640, p = 0.437117.

Draw descriptive linear least-squares fits with pointwise 95% confidence intervals for the mean response, using the same convention as panel D. The empirical correlation supports a qualitative prediction; it does not by itself establish resource competition or provide a numerical fit of the simulation.

## Optional future panel — Phase diagram

Create a heat map or contour plot with:

- x-axis: \(\log(a_P/a_C)\) or \(a_P/a_C\)
- y-axis: total budget \(R\)
- color: \(r_P-r_C\)

Alternatively, show three regimes:

- P-dominated allocation
- shared allocation
- C-dominated allocation

The boundary condition:

\[
\log(a_P/a_C)=R
\]

marks the transition to:

\[
r_C=0.
\]

The opposite boundary is:

\[
\log(a_C/a_P)=R.
\]

This panel is optional but would strengthen the point that the empirical pattern is one parameter regime of a more general model.

---

# Optional link to neural decoding

Do **not** make this the core model, but optionally include a simple mapping from latent representational resource to expected decoding performance.

For a two-class identity decoder, use:

\[
A_i
=
0.5
+
0.5(1-e^{-\kappa r_i}),
\]

where:

- \(A_i\): expected decoding accuracy
- \(0.5\): chance
- \(\kappa>0\): scaling parameter

Thus:

\[
r_i=0 \Rightarrow A_i=0.5
\]

and:

\[
r_i\uparrow \Rightarrow A_i\uparrow.
\]

This mapping is **illustrative only** and should not be described as a fitted measurement model.

---

# Conceptual guardrails

1. **Do not hard-code cue type.** Allocation should depend on \(a_i\), \(R\), and the loss function—not on an if-statement assigning more resource to preconditioned cues.

2. **Do not claim that inferred cues are intrinsically more important.** The model predicts stronger inferred-cue representation only when \(a_P>a_C\).

3. **Do not equate absence of OFC decoding with absence of identity information everywhere.** A low \(r_C\) only means little resource is allocated to the explicit identity representation under study.

4. **Treat the model as normative.** It asks how representation *should* be allocated under a finite budget to minimize decision loss. It does not claim that the brain literally solves this exact optimization problem.

5. **The exponential form is illustrative.** If easy, include a robustness check with another monotonically decreasing, diminishing-returns function such as:

\[
L_i=\frac{a_i}{1+r_i}.
\]

The qualitative allocation principle should remain.

---

# Suggested repository structure

```text
adaptive-identity-allocation/
├── README.md
├── requirements.txt
├── src/
│   ├── model.py
│   ├── simulate.py
│   └── plotting.py
├── scripts/
│   └── make_figure.py
├── figures/
│   ├── fig_model.pdf
│   └── fig_model.png
└── results/
    └── simulation_summary.csv
```

Suggested functions:

```python
single_cue_loss(a, r)
total_loss(a_p, a_c, r_p, R)
optimal_allocation(a_p, a_c, R)
marginal_value(a, r)
decoding_accuracy(r, kappa=1.0)
```

`optimal_allocation` should implement the constrained analytic solution, including boundary cases.

Also verify the analytic solution numerically with a grid search or optimizer.

---

# Numerical checks

### Example 1: asymmetric interior solution

\[
R=4,\quad a_P=4,\quad a_C=1
\]

Expected:

\[
r_P\approx2.693,\qquad r_C\approx1.307.
\]

### Example 2: symmetric solution

\[
R=4,\quad a_P=a_C=1
\]

Expected:

\[
r_P=r_C=2.
\]

### Example 3: boundary solution

\[
R=2,\quad a_P=10,\quad a_C=1.
\]

Since:

\[
\log(10)\approx2.303>2,
\]

expected:

\[
r_P=2,\qquad r_C=0.
\]

---

# Figure aesthetics

Target a neuroscience / cognitive science journal figure.

Use:

- vector PDF output
- high-resolution PNG preview
- sans-serif fonts
- labels large enough for two-column journal reproduction
- minimal legends where direct labeling is possible
- panel letters `A`, `B`, `C`, `D`, and `E`, with E spanning both empirical ROI plots
- consistent notation across panels
- clean white background
- no unnecessary 3D effects or decoration

The figure should be understandable before reading the detailed mathematical derivation.

---

# Draft figure caption

**Figure 4. A resource-limited model of outcome-identity representation.**

**A. Benefits of identity coding.** Decision loss for a cue decreases with the amount of representational resource allocated to its outcome identity, \(L_i=a_i e^{-r_i}\). The parameter \(a_i\) determines the computational value of explicitly representing that outcome identity, whereas \(r_i\) denotes representational resource. The exponential form produces diminishing returns.

**B. Optimal allocation.** When inferred and directly learned cues share a finite representational budget, \(r_P+r_C=R\), total loss is minimized by balancing the costs associated with insufficient representation of each cue. An example with \(a_P>a_C\) yields greater optimal allocation to the inferred cue.

**C. Allocation follows utility.** The optimal allocation difference follows \(r_P-r_C=\log(a_P/a_C)\) for interior solutions, with boundary solutions imposed by the finite resource constraint. Thus, preferential representation of inferred outcomes emerges when their explicit identity representation has greater computational value; it is not assumed from cue type itself.

**D. Model prediction.** Simulated participants differ in relative computational value while sharing a fixed budget. Direct and inferred identity coding are represented by optimal allocations plus independent Gaussian measurement noise, yielding a negative between-participant association. Points show 100 simulated participants, the line is a descriptive linear fit, and shading is its pointwise 95% confidence interval for the mean response. Coding units and parameters are illustrative; the model is not fitted to fMRI RSA values.

**E. Test in fMRI data.** Correlations between conditioned- and preconditioned-cue odor-identity RSA coefficients test the prediction in lateral and medial OFC (31 participants per region). The lateral OFC association is negative (Spearman rho = -0.317, p = 0.041); the medial OFC association is not significant (rho = -0.030, p = 0.437). Tests assess negative associations and are one-sided. Lines show descriptive linear least-squares fits; shading indicates pointwise 95% confidence intervals for the fitted mean response.

---

# Main takeaway

The simulation should make this logic visually obvious:

\[
\text{identity representation reduces decision loss}
\]

plus

\[
\text{representational resources are finite}
\]

plus

\[
\text{explicit identity can have different computational value across cues}
\]

therefore

\[
\boxed{
\text{optimal representation is preferentially allocated where its marginal utility is greatest}
}
\]

In the parameter regime relevant to the current hypothesis:

\[
a_P>a_C
\]

therefore:

\[
r_P>r_C.
\]

The model should present this as a **general adaptive-allocation principle**, not as a special-purpose rule for sensory preconditioning.
