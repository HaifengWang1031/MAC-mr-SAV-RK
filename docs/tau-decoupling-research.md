# Decoupling the step size from the shifted MAC Stokes solve

Let

\[
A_\tau=\begin{bmatrix}H_\tau&G\\-D&0\end{bmatrix},\qquad
H_\tau=I+\sigma K,\qquad \sigma=\tau\nu a,
\]

with `G = -D.T` and pressure understood on the mean-zero subspace. The main conclusion is:

> Exact removal of \(\tau\) from the pressure solve is available for a commuting periodic MAC discretization (and for a free-slip closure only when the discrete commutator is verified). It is not exact for this repository's no-slip closure. On the present uniform rectangle, the best reusable route is therefore tensor-product Helmholtz and pressure-Poisson solves inside an outer exact saddle-point iteration. The outer iteration corrects a boundary-supported commutator defect.

## What the current code actually solves

The MAC operator is especially favorable for reuse. [`operators.py`](../solver/mac/operators.py#L6-L29) constructs \(K=\operatorname{diag}(K_u,K_v)\) from Kronecker sums of one-dimensional second differences. Eliminated normal wall velocities give the usual diagonal `2`; odd extension of tangential velocity across a half-cell no-slip wall changes the endpoint diagonal to `3`. The same file constructs the cell divergence and sets \(G=-D^T\).

[`stokes.py`](../solver/mac/stokes.py#L33-L65) currently assembles the gauge-reduced coupled matrix, factors it with SuperLU, and caches the factorization by `(mass, viscosity)`. The cache therefore reuses a fixed step size but must refactor for a new \(\tau\). It already solves multiple right-hand sides together. This matters because every mrSAV stage supplies two columns, while every stage of a given SDIRK2/3 step has the same diagonal coefficient and hence the same \(H_\tau\) ([SDIRK2](../solver/schemes/sdirk2_mrsav.py#L13-L25), [SDIRK3](../solver/schemes/sdirk3_mrsav.py#L18-L32)). Adaptive stepping is the adverse case: many distinct accepted and rejected \(\tau\)'s can churn the bounded LU cache.

## Exact algebra and the commutator test

For every boundary condition, exact block elimination gives

\[
A_\tau=
\begin{bmatrix}I&0\\-DH_\tau^{-1}&I\end{bmatrix}
\begin{bmatrix}H_\tau&0\\0&S_\tau\end{bmatrix}
\begin{bmatrix}I&H_\tau^{-1}G\\0&I\end{bmatrix},
\qquad S_\tau=DH_\tau^{-1}G.
\]

Thus a projection is an exact solver only if it retains the true Schur complement (or is embedded as a preconditioner in a converged outer iteration). Perot derived fractional-step methods as block LU factorizations; Chang, Giraldo, and Perot exhibited an exact staggered-mesh projection with no splitting error and machine-precision incompressibility ([Perot 1993](https://doi.org/10.1006/jcph.1993.1162); [Chang--Giraldo--Perot 2002](https://doi.org/10.1006/jcph.2002.7087)).

Define the cell-centered positive pressure Laplacian \(L_p=-DG=DD^T\). If

\[
DK=L_pD \quad\text{(equivalently, }KG=GL_p\text{)},
\]

then

\[
-S_\tau=L_p(I+\sigma L_p)^{-1},\qquad
(-S_\tau)^{-1}=L_p^\dagger+\sigma I.
\]

For a divergence-free right-hand side constraint, pressure can therefore be obtained from the single \(\tau\)-independent Poisson equation \(L_pp=-Df\), followed by the shifted velocity solve \(H_\tau u=f-Gp\). Equivalently, the exact inverse pressure Schur action is one fixed Poisson inverse plus the scalar update \(\sigma I\). Cai et al. identify the corresponding gradient--Laplacian identity as exact for a constant-coefficient periodic staggered grid; with exact subsolvers their projection preconditioner becomes an exact Stokes solver ([Cai et al. 2014](https://doi.org/10.4208/cicp.070114.170614a)). The formula is also the finite-difference form of the generalized-Stokes pressure preconditioner introduced by Cahouet and Chabard ([1988](https://doi.org/10.1002/fld.1650080802)).

### Boundary conditions are decisive

| Boundary closure | Helmholtz separability | Exact pressure commutation |
|---|---:|---:|
| Fully periodic, constant coefficients | yes (FFT) | yes |
| Rectangular free slip | yes (sine/cosine transforms) | closure-dependent; verify \(DK-L_pD=0\) |
| This repository's homogeneous no slip | yes (mixed sine transforms) | no |

The no-slip failure is not a small roundoff detail. Directly constructing

\[
C=DK-L_pD
\]

from the repository matrices shows that it is supported only on boundary pressure cells and boundary-tangential velocities. With equal mesh spacings, an \(8\times6\) grid gives 48 nonzeros, 24 supported pressure rows, and rank 23; a \(16\times12\) grid gives 104 nonzeros, 52 supported rows, and rank 51. These observations match `boundary cells - 1`, namely \(2(n_x+n_y)-5\), on all checked grids from \(3^2\) through \(16\times12\). Replacing only the tangential endpoint diagonal `3` by the even-extension/free-slip value `1` makes the same commutator zero to roundoff. This is code-specific experimental evidence, not a general proof for every free-slip discretization.

The remaining dependence can be isolated exactly. Set \(H_p=I+\sigma L_p\). Since

\[
DH_\tau=H_pD+\sigma C,
\]

the positive pressure Schur operator obeys

\[
H_p(-S_\tau)=L_p-\sigma C H_\tau^{-1}D^T,
\]

and the exact pressure equation is

\[
\bigl(L_p-\sigma C H_\tau^{-1}D^T\bigr)p
=-Df+\sigma C H_\tau^{-1}f.
\]

Thus the periodic/free-slip cancellation is the special case \(C=0\). For this no-slip stencil, all nontrivial \(\tau\)-dependence left after the fixed pressure Poisson operator is a correction of rank at most the number of boundary pressure cells. This makes a Woodbury/capacitance formulation algebraically possible, but not automatically economical: each new \(\tau\) changes the boundary correction through \(H_\tau^{-1}\).

Published staggered-grid analysis likewise finds that nonperiodic boundary treatment creates a boundary-sized commutator discrepancy and bounds the number of nonunit eigenvalues accordingly ([Cai 2015](https://doi.org/10.1016/j.apnum.2014.12.003), [author preprint](https://arxiv.org/abs/1310.1059)). At the continuous level, failure of the Laplacian and Leray projection to commute is also the source of no-slip pressure/boundary-layer complications ([Liu--Liu--Pego 2010](https://doi.org/10.1016/j.jcp.2010.01.010)). Consequently, using \(L_p^\dagger+\sigma I\) as though it were exact would change the present coupled stage equations. It is appropriate as a Schur preconditioner whose remaining wall error is removed by an outer Krylov solve.

## Local prototype evidence

The following exploratory measurements use the production geometry and coefficients:
MAC \(128^2\) on \((0,2\pi)^2\), \(\nu=0.02\), the SDIRK2 diagonal
\(a=1-1/\sqrt2\), and ten distinct steps between `0.00135697` and
`0.00362322` with mean `0.0025`. CPU timings use one BLAS/OpenMP thread. They
are implementation evidence, not a replacement for a committed backend and its
regression tests.

- The velocity Laplacian has extremal eigenvalues approximately `0.499975` and
  `3319.843`; hence \(I+\sigma K\) has condition number only `1.026`--`1.070`
  over this step range. The varying systems are not becoming difficult because
  of velocity-block conditioning.
- Precomputing the four one-dimensional eigendecompositions takes about
  `0.0069 s`. Applying the resulting tensor-product inverse to all ten shifted
  velocity systems takes about `0.0021 s`; comparison with sparse LU gives a
  relative difference `5.5e-14` and scaled residual `4.4e-14`.
- For the full coupled problem, use the exact matrix-free Schur action
  \(D H_\tau^{-1}D^T\) and precondition CG by
  \(L_p^\dagger+\sigma I\). A single tau-independent pressure-Poisson LU plus
  the four one-dimensional eigenspaces costs about `0.036 s`. The ten random
  Stokes systems then take `0.085 s` after setup (`0.121 s` including setup),
  with 5--6 pressure iterations each. Maximum full divergence is `2.22e-10`.
- At the minimum, mean, and maximum tested steps, velocity and pressure agree
  with the current coupled SuperLU solve to relative errors below `8.2e-14`.
  By comparison, refactoring the coupled saddle matrix at all ten steps takes
  about `4.82 s` on the same run, so this prototype is about 40 times faster
  including its reusable setup.

The prototype used the existing pinned-pressure reduction for the Poisson
preconditioner. A production tensor solver should instead operate on the full
mean-zero pressure space, set the constant cosine mode to zero explicitly, and
retain the repository's full momentum and divergence acceptance checks.

## Implemented backend and measured result

The branch `feature/tau-decoupled-stokes` now contains that production-oriented
path in [`tensor_stokes.py`](../solver/mac/tensor_stokes.py). `TensorStokes` uses
DST-I for eliminated-normal Dirichlet directions, DST-II for tangential
half-cell no-slip directions, and DCT-II for the mean-zero pressure Poisson
inverse. It applies the true Schur complement in every CG iteration, so the
Cahouet--Chabard expression is only a preconditioner and the no-slip stage
equations are unchanged. The existing full momentum-residual and full-divergence
checks remain acceptance criteria; `DirectStokes` remains the default fallback.

The reproducible benchmark entry point is
[`benchmark_stokes_backends.py`](../experiments/kolmogorov_adaptive/benchmark_stokes_backends.py).
On the same 128² Kolmogorov case, ten steps, three repeats, and one numerical
thread, the median CPU times including backend setup were:

| Scheme | fixed Direct → Tensor | speedup | random Direct → Tensor | speedup |
|---|---:|---:|---:|---:|
| SDIRK2 | 0.647889 → 0.134525 s | 4.82× | 4.891160 → 0.124791 s | 39.19× |
| SDIRK2-mrSAV | 0.736653 → 0.260393 s | 2.83× | 4.927767 → 0.261176 s | 18.87× |
| SDIRK3 | 0.758802 → 0.256676 s | 2.96× | 5.128740 → 0.260301 s | 19.70× |
| SDIRK3-mrSAV | 1.012436 → 0.524722 s | 1.93× | 5.210182 → 0.522797 s | 9.97× |

For `TensorStokes`, random/fixed CPU ratios are 0.928, 1.003, 1.014, and
0.996 respectively: within this short benchmark the varying-step penalty has
disappeared into timing noise. The Schur solves use 4.475--5.0 iterations per
right-hand side on average. Across all cases, the largest final-velocity
difference from `DirectStokes` is `5.99e-13` relative and the largest recorded
divergence is `7.68e-13`. The complete machine-readable result is
[`timings.json`](../reports/timings/20260927T110237-380d2a7a-tau-decoupled-10-step/timings.json).

These numbers validate this grid and coefficient range, not arbitrary geometry,
variable viscosity, or nonuniform meshes. The transform backend currently
supports only the repository's uniform rectangular homogeneous no-slip MAC
operator.

The adaptive Kolmogorov configuration now selects this backend through
`stokes_backend="tensor"`; configurations without that key retain the historical
direct default. A 128², `T=0.01`, three-repeat comparison exercised all four
schemes with both I and PI controllers. Direct-to-tensor median CPU speedups were
24.08×/28.47× for SDIRK2 I/PI, 15.69×/17.10× for SDIRK2-mrSAV, 15.10×/16.38×
for SDIRK3, and 9.30×/9.61× for SDIRK3-mrSAV. Acceptance decisions matched;
the maximum relative attempted-step difference was `8.29e-7`, the maximum final
velocity difference was `7.66e-13`, and the largest tensor-backend divergence
was `1.29e-12`. The short timing window had no rejected trials, so a separate
tight-tolerance regression forces one rejection followed by two accepted steps
and compares both backends. Full records are in
[`timings.json`](../reports/timings/20260927T111819-9752d294-adaptive-stokes-backends/timings.json).

## Reusable computational options

### 1. Fast diagonalization is the strongest fit for this grid

Each velocity block is a Kronecker sum. Diagonalize its one-dimensional factors once:

\[
K_q=(Q_y\otimes Q_x)\,\Lambda_q\,(Q_y\otimes Q_x)^T,\qquad
H_{\tau,q}^{-1}=Q\,(I+\sigma\Lambda_q)^{-1}Q^T.
\]

Only the diagonal reciprocals change with \(\tau\); the bases do not. The normal direction uses the standard eliminated-Dirichlet sine basis, and the tangential half-cell odd extension uses the corresponding shifted sine basis. Likewise, \(L_p\) is a cell-centered Neumann Kronecker sum, with its constant cosine mode set to zero to impose mean-zero pressure. The tensor-product direct method originates with Lynch, Rice, and Thomas ([1964](https://doi.org/10.1007/BF01386067)); FFT-based rectangular Helmholtz solvers cover Dirichlet, Neumann, and periodic cases ([Boisvert 1987](https://doi.org/10.1145/29380.214342)). On these uniform operators, sine/cosine transforms give \(O(N\log N)\) applications and \(O(N)\) storage.

No-slip walls therefore do **not** obstruct fast, \(\tau\)-reusable solves with \(H_\tau\). They obstruct replacing the coupled system by one uncorrected Helmholtz solve plus one fixed Poisson solve.

### 2. Use projection/block factorization as a preconditioner

A practical exact solver is FGMRES on the original gauge-fixed saddle system with a triangular projection preconditioner:

1. apply the exact tensor-product \(H_\tau^{-1}\);
2. approximate \((-S_\tau)^{-1}\) by \(L_p^\dagger+\sigma I\);
3. apply the block correction and iterate until the existing full momentum and divergence checks pass.

This preserves the unsplit equations and their no-slip boundary treatment. Griffith used precisely the projection-as-preconditioner principle to retain general physical boundary conditions in an unsplit staggered-grid method ([2009](https://doi.org/10.1016/j.jcp.2009.07.001)). Cai et al. built uniform-MAC Stokes preconditioners from independent generalized Helmholtz and Poisson solves and reported mesh-robust variants even with inexact multigrid subsolves ([2014](https://doi.org/10.4208/cicp.070114.170614a)). Uniform-in-step-size generalized-Stokes preconditioning also has operator-theoretic support ([Mardal--Winther 2004](https://doi.org/10.1007/s00211-004-0529-6)).

The boundary support of \(C\) suggests a second, more experimental route: represent the difference between the commuting Schur approximation and the true \(S_\tau\) as a boundary-rank correction and use a Woodbury/capacitance solve. It could produce an exact direct method with transform-based bulk solves and an \(O(n_x+n_y)\) dense boundary system, but this needs derivation and conditioning tests before it should replace the simpler outer Krylov correction.

### 3. Shifted and recycling Krylov methods have narrower roles

The velocity family can be written \(H_\tau=\sigma(K+\sigma^{-1}I)\), so genuine batches with the **same** right-hand side are shifted systems. Shifted matrices share a Krylov subspace, and restarted shifted GMRES can solve several shifts with one matrix-vector product per iteration ([Frommer--Glässner 1998](https://doi.org/10.1137/S1064827596304563)). This does not automatically apply to the full saddle matrices, which are not scalar shifts of one another, or to normal time stepping, where both \(\tau\) and the stage right-hand side change. Classical simultaneous-shift methods also require common/collinear residuals; unrelated right-hand sides require block/Sylvester variants ([Soodhalter 2016](https://doi.org/10.1137/140998214)).

Recycling is more relevant to adaptive runs: matrices and right-hand sides change slowly, so selected subspaces from one solve can accelerate the next ([Parks et al. 2006](https://doi.org/10.1137/040607277)). Simultaneously combining shift invariance, one fixed recycled space, and fixed storage has intrinsic restrictions ([Soodhalter--Szyld--Xue 2014](https://doi.org/10.1016/j.apnum.2014.02.006)). In this repository, transform diagonalization removes the main need for multi-shift Krylov on \(H_\tau\); recycling is a possible second-stage optimization of the outer wall-correction iteration.

## PETSc mapping

If a distributed backend is restored, PETSc already exposes the necessary composition:

- [`PCFIELDSPLIT`](https://petsc.org/release/manualpages/PC/PCFIELDSPLIT/) implements diagonal, lower, upper, and full Schur factorizations. For this matrix \(A_{00}=H_\tau\), \(A_{01}=G\), \(A_{10}=-D\), and PETSc's Schur complement is exactly \(S_\tau=DH_\tau^{-1}G\). Since \(A_{11}=0\), provide the pressure approximation explicitly rather than using the default `a11`; PETSc documents the user-Schur path in [`PCFieldSplitSetSchurPre`](https://petsc.org/release/manualpages/PC/PCFieldSplitSetSchurPre/). [`PCLSC`](https://petsc.org/release/manualpages/PC/PCLSC/) is the general algebraic commutator alternative, but the explicit \(L_p^\dagger+\sigma I\) approximation is more specific to this constant-coefficient MAC system.
- A virtual [`MatSchurComplement`](https://petsc.org/release/manualpages/KSP/MatCreateSchurComplement/) retains the true Schur action; its blocks can be changed for a new \(\tau\) with [`MatSchurComplementUpdateSubMatrices`](https://petsc.org/release/manualpages/KSP/MatSchurComplementUpdateSubMatrices/).
- PETSc automatically reuses setup when the operator is unchanged. [`KSPSetReusePreconditioner`](https://petsc.org/release/manualpages/KSP/KSPSetReusePreconditioner/) can freeze a preconditioner across changed values, but the official documentation warns that iteration counts may increase substantially. Reusing ordering/symbolic structure while recomputing numeric factors is safer for direct methods.
- [`KSPEKSM`](https://petsc.org/release/manualpages/KSP/KSPEKSM/) is PETSc's common-right-hand-side solver for \((K+\sigma_iM)x_i=b\); its external preconditioner must be `PCNONE`, reinforcing that it is not a drop-in solver for the full preconditioned saddle sequence.
- PETSc's HPDDM interface exposes GCRO-DR/BGCRO-DR and preserves the deflation space across solves unless reset ([`KSPHPDDMSetType`](https://petsc.org/release/manualpages/KSP/KSPHPDDMSetType/)). Attach the constant pressure nullspace or retain the repository's current gauge removal.

## Recommended order of work

1. Prototype exact DST/DCT applications of \(H_\tau^{-1}\) and \(L_p^\dagger\), with operator-action tests against sparse solves for square and rectangular grids.
2. Wrap them in a triangular projection/Cahouet--Chabard preconditioner for the **unchanged** coupled matrix; use flexible GMRES and retain the current full residual and full-divergence acceptance checks.
3. Benchmark fixed and adaptive step sequences separately. Compare setup time, solve time, iterations, and memory against cached SuperLU, including the existing two-column mrSAV solve.
4. Only if outer iterations are still material, test recycled GCRO-DR or the boundary-rank exact correction. Use multi-shift Krylov only for an identified same-right-hand-side batch.

This route decouples reusable spatial work from \(\tau\) without silently changing the no-slip stage equations or pressure semantics.
