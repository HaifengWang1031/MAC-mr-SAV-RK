# Spectral discretisation (single-element Dirichlet-combined Legendre)

Slices S1, S2 and S3 of the plan agreed on 2026-09-21: the basis and the assembly for a
third discretisation, its `solver/model.py` implementation, and the moving-lid lifting with
the sharp and regularised lid-driven cavity compared against the validated MAC solutions.
No new dependency is involved; only numpy, scipy and sympy.

## What is implemented

`solver/spectral/basis.py`

- The Dirichlet combination `phi_k = L_k - L_{k+2}` on the reference interval, evaluated
  and differentiated by recurrences that never divide by `1 - xi**2`.
- The exact 1D matrices: mass, stiffness, divergence and the pressure-velocity projection.
  They are built from `int L_k L_l = 2/(2k+1) delta_kl` and from
  `L_m' = sum_{j in J_m} (2j+1) L_j`, so they carry no quadrature error at all. The
  Dirichlet stiffness turns out to be *diagonal* (entries `2(2k+3)`) and the divergence
  matrix bidiagonal, which is what keeps the 2D operators sparse.

`solver/spectral/assembly.py`

- The tensor-product space with per-direction quadrature.
- Kronecker assembly of the 2D blocks: velocity mass, velocity stiffness and the two
  divergence blocks.
- Galerkin loads by Gauss-Legendre quadrature, evaluation and L2 norms.

`solver/spectral/stokes.py`

- The shifted Stokes saddle point
  `[[mass*Mv + viscosity*Lv, 0, -Dx^T], [0, ., -Dy^T], [Dx, Dy, 0]]` with the pressure
  constant mode removed, which is exactly the nullspace of that block, factorizations
  cached on `(mass, viscosity)`, and both an algebraic residual and a pointwise divergence
  check after every solve.

`solver/spectral/model.py`

- The discretisation presented through the seam in `solver/model.py`, so `SDIRK2` and
  `SDIRK2-mr-ccSAV` drive it unmodified. A modal basis forces two decisions that the MAC
  model never has to make:
  - **One vector space.** The schemes add `vector(state)`, `force(t)` and `nonlinear(state)`
    together and hand the sum to `solve`, so all three have to live in the same space. The
    MAC model satisfies this silently, because its basis functions are nodal indicators and
    its mass matrix is diagonal, so load and coefficient spaces coincide. Here the packed
    vector holds *coefficients* and the weak operators are folded into field operators:
    `apply_K = M^-1 S`, `apply_G = M^-1 G`, and `solve` turns its right-hand side into a
    load. `Space.load` is only the weak right-hand side, so anything built from an analytic
    expression goes through `model.project`, which solves the mass matrix.
  - `max_abs` returns the largest coefficient, not a sup-norm of the field. It only has to
    be a consistent scale for the relative stage checks; the physical divergence is
    reported separately by `diagnostics`, evaluated in physical space.
- `nonlinear` assembles the convective term on a wider quadrature than the projection uses,
  which makes the projection of the polynomial product exact and lets the term be verified
  against an analytic value instead of only against itself.

`solver/spectral/lifting.py`

- The inhomogeneous Dirichlet values, following shenfun's construction: the wall profile is
  carried by two extra basis functions, `(1+y/ly)/2` and `(1-y/ly)/2`, whose coefficients
  are *known*, so their contribution moves to the right-hand side as an explicit field g
  and the saddle-point matrix is untouched. `Space.stiffness_load` and
  `Space.divergence_load` assemble those loads, and each is pinned by an identity: for a
  field that lies in the velocity space they must reproduce `velocity_stiffness @
  coefficients` and the assembled divergence blocks exactly.
- The lid profile is projected onto the 1D Dirichlet basis first, exactly as shenfun's
  `set_boundary_dofs` does. `LidLifting` takes the profile in the reference coordinate, so
  the constant sharp lid and the regularised `(1-xi^2)^2` differ only in one callable.

## What is verified

| check | result |
|---|---|
| 1D matrices against sympy symbolic integration | exact, deviation 0 |
| 2D blocks against quadrature of the same weak forms, basis by basis, square and non-square | below 1e-12 relative |
| Dirichlet basis vanishes at both ends; tensor product on all four walls | 1e-14 / 1e-12 |
| load of the constant and of one basis function | `lx*ly` exactly, and the mass column |
| quadrature nodes per direction (regression for the lx/ly bug below) | passes |

Two real bugs were found by these references and are fixed: the y quadrature nodes were
scaled by `lx`, which pushed the reference coordinate outside `[-1, 1]`; and the y factor
of `Dx` used the velocity mass where the pressure test function requires the
pressure-times-velocity projection, which silently enforced a different constraint.

## What was wrong, and how it was found

The first version of this slice looked like a solver bug: the manufactured Stokes velocity
came out three times too small, the errors did not move with resolution, and the pointwise
divergence sat at O(1). It was not a solver bug — the algebraic residuals were machine zero
throughout. Three real defects, all of them in the *projection and comparison* code:

1. `Space.load` contracted the `meshgrid` output in the wrong order, so on a rectangle the
   load was the transpose of the right one. Every square-domain test passed either way,
   because the manufactured fields and the basis are then symmetric under `x <-> y`; the
   defect appeared only against symbolic integration on a rectangle, and only in the (0,2)
   and (2,0) entries.
2. `Space.evaluate` had the same transposition, so it returned the field of the transposed
   coefficients. That is invisible for symmetric coefficients — a square domain, or a field
   built from `phi_0 psi_0` — which is why the exact-polynomial diffusion test passed while
   the manufactured comparison did not.
3. The pressure was evaluated with the *velocity* basis. The pressure space uses plain
   Legendre, so the manufactured pressure looked non-convergent while the velocity was
   already at machine precision. `Space.evaluate_pressure` now exists for it.

Three process lessons, recorded because they cost hours:

- An index convention cannot be settled by reasoning; it has to be compared against a value
  produced somewhere else. Symbolic integration is that somewhere else.
- A check that passes on a square domain proves nothing about mode ordering. Ordering
  defects need a rectangle, or a mode pattern that is not symmetric.
- When a residual is machine zero and the answer looks wrong, suspect the comparison before
  the solver. Every intermediate here was telling the truth except the diagnostics.

## What is verified now

| check | result |
|---|---|
| 1D matrices against sympy symbolic integration | exact, deviation 0 |
| 2D blocks against quadrature of the same weak forms, basis by basis, square and rectangle | below 1e-12 relative |
| load of a separable function on a rectangle against the outer product of two 1D integrals | below 1e-12; the transposed order fails this |
| diffusion with a polynomial exact solution, square and rectangle, `nu = 0.1` and `2.5` | exact to 1e-16 |
| manufactured steady Stokes, velocity | 1.2e-05 at 8 modes, 3.2e-09 at 12, 2.7e-13 at 16, 4e-16 at 20 |
| manufactured steady Stokes, pressure | 1.0e-06 at 8 modes, 1.8e-10 at 12, 1.2e-14 at 16, 1.3e-16 at 20 |
| modal continuity residual | machine zero, including the row whose pressure mode is the gauge |

Both fields converge spectrally on the square and on the rectangle. Beyond 20 modes the
error sits at roundoff and drifts slowly upward: that is the accuracy limit of a single
Legendre element in double precision, and it is the reason a multi-element basis exists.

## What the seam verified

- **The nonlinear term** against sympy: for a polynomial velocity that is in the space
  (both components vanish on all four walls), `N(v) = (u.grad)u` matches the analytic value
  pointwise to 1.9e-18, a relative 4.2e-15. The quadrature is wide enough that the
  projection of the product is exact, so this compares against an outside reference and is
  not a self-check.
- **The operators**: `D` and `G` are adjoints under the mass inner product
  (`<Gp, v> = -<p, Dv>`, residual 1e-11 on random vectors) and the stiffness form is
  symmetric there. Both statements go through `inner` rather than a plain dot product,
  because the operators exposed by the seam are `M^-1 S` and `M^-1 G`.
- **Time convergence** on the transient manufactured solution (`experiments/problems.py`,
  amplitude 0.1, nu 0.1, T 0.2, 16 modes, reference the same scheme at dt = T/256):

  | scheme | dt=T/4 | dt=T/8 | dt=T/16 | dt=T/32 | observed order |
  |---|---|---|---|---|---|
  | SDIRK2 | 2.386e-03 | 5.921e-04 | 1.457e-04 | 3.568e-05 | 2.01, 2.02, 2.03 |
  | SDIRK2-mr-ccSAV | 1.953e-03 | 4.989e-04 | 1.400e-04 | 4.626e-05 | 1.97, 1.83, 1.60 |

  Against a dt = T/1024 reference the SAV errors are unchanged (5.005e-04, 1.426e-04,
  4.963e-05, 2.084e-05; orders 1.81, 1.52, 1.25) and are identical at 16 and at 24 modes,
  so the degradation is temporal rather than spatial, and not an artefact of the reference.

### The SAV order at small steps is not a defect of this discretisation

The same measurement on the validated MAC model, same manufactured solution, T/256
reference, and resolution-independent in the same way:

| model | dt=T/8 | dt=T/16 | dt=T/32 | observed order |
|---|---|---|---|---|
| MAC 32x32, SDIRK2 | 2.651e-05 | 6.606e-06 | 1.633e-06 | 2.00, 2.02 |
| MAC 32x32, SAV | 2.769e-05 | 7.495e-06 | 2.266e-06 | 1.89, 1.73 |
| MAC 48x48, SAV | 2.776e-05 | 7.473e-06 | 2.237e-06 | 1.89, 1.74 |

Both discretisations show the same signature, so the two implementations agree with each
other rather than one being wrong. The second order recorded for both schemes in
`docs/validation.md` is the leading order, and the SAV scheme falls below it at small
steps. Whether that is intended by the scheme or a defect in it is a question about the
scheme, not about this discretisation, and it is not addressed here.

## S3: the lid-driven cavity against the validated MAC solutions

### Verifying the lifting itself

An inhomogeneous manufactured solution, built from a stream function so that the exact
velocity is divergence-free and its wall trace *is* the lifting's trace (`A(xi) = xi
(1-xi^2)^2` on the lid, chosen odd so the net flux is zero). Solving the steady Stokes
problem with the analytic forcing:

| pressure | size 8 | size 12 | size 16 |
|---|---|---|---|
| polynomial, `(2x-1)(2y-1)` | 2.11e-16 | 3.65e-16 | 1.63e-16 |
| smooth, `cos(pi x) cos(pi y)` | 1.29e-08 | 1.39e-13 | 2.36e-16 |

With a polynomial pressure the exact velocity minus the lifting lies in the velocity space,
so the discrete solve reproduces it at roundoff and the sequence is not monotone; with a
smooth pressure only the pressure projection is left, and it falls through eight decades to
roundoff. A wrong sign, a missing factor or the wrong mass convention in either load makes
this fail at every size rather than converge slowly.

### Two discretisations, one scheme, one benchmark

> **Corrected 2026-09-21, twice over.** Nothing in this subsection is a current measurement.
> The numbers predate (a) the SAV pairing with a fixed lifting, which changed the auxiliary
> variable's start and added the lifting work scalar to the stage scalar equation, and (b) the
> repair of the comparison metric itself. The yardstick was measured by striding face values,
> which samples half a cell away, because the face offsets of two grids differ by
> `(factor-1)/2`; that mixed interpolation error into the yardstick and made the sharp-lid
> ratio look far better than it is. Under the cell-centred metric the sharp lid *fails* the
> "closer than the coarse grid" criterion (ratio 3.4-3.9) while the regularised lid satisfies
> it comfortably (0.674). `docs/validation.md`, section "尖锐顶盖判据的度量缺陷", carries the
> derivation and the current measurements; the assertions in `tests/test_spectral.py` now use
> the narrowed criteria recorded there.

The lid is imposed completely differently on the two sides of the comparison: the MAC
solver adds a ghost-point viscous load `2 nu U / hy^2` to the top row of u faces (a pure
right-hand side term), while the spectral solve splits the wall value off as a lifting whose
trace is the boundary condition. Both runs use SDIRK2-mr-ccSAV from rest, Re=100, unit
cavity. The yardstick is the MAC solver's *own* grid sensitivity `|32^2 - 64^2|`, measured
in the same way rather than assumed:

| lid | MAC 32^2 vs 64^2 | spectral 20 modes vs MAC 64^2 | spectral 28 modes vs MAC 64^2 |
|---|---|---|---|
| sharp | 0.009001 | 0.006722 | 0.003775 |
| regularised | 0.006192 | 0.000838 | 0.000838 |

Kinetic energy at the same steady states: sharp 0.034008366 (MAC 64^2) against 0.034469608
(20 modes) and 0.034456119 (28 modes); regularised 0.018663548 against 0.018824777 at both
sizes. The table is a T=40 steady state from an ad-hoc script (unit cavity, Re=100, SDIRK2-mr-ccSAV
from rest, dt=0.02, MAC 32^2/64^2 and spectral 20/28 modes); the test suite reproduces the
same comparison at T=10, where the numbers are already those of the steady state (0.006805
against 0.006722 for the sharp lid). The spectral solution is closer to the fine MAC
solution than the coarse MAC grid is,
and for the regularised lid it stops changing between 20 and 28 modes, so the residual
disagreement is the MAC's own discretisation error rather than the spectral one. The sharp
lid keeps improving with resolution (`0.0067 -> 0.0038`) but does not collapse, which is the
corner singularity: its constant profile is not representable, so it arrives as a projection
with Gibbs oscillations, and the pointwise divergence diagnostic sits at 52 (20 modes) and
70 (28 modes) in the corner layer against 5e-13 for the MAC. That diagnostic is a corner
layer, not a failed constraint: the constraint residual the schemes gate on is the weak one,
which is machine zero, and the constant pressure row that would hold the cavity's net flux
is the removed gauge.

### Three places the seam had to be made consistent with a lifting

All three showed up as the SAV stage gate failing rather than as a wrong answer, and each is
recorded because the same trap applies to any future model whose state is not the physical
field:

- `apply_K` has to act on the *total* field, `M^-1 S (w + g)`. The schemes build their stage
defect from it, and `solve` subtracts the same lifting load, so the two have to agree: with
`apply_K` on the interior only, the defect kept a leftover of 0.40 instead of machine zero.
- `solve_columns` gives the boundary datum to the *first* column only. The schemes add the
later columns to the first as corrections, so solving every column with the same datum
multiplies it by the sum of the coefficients; because the model is linear, solving the
corrections with a homogeneous constraint is exactly equivalent to solving the combined
right-hand side.
- `apply_D` reports the physical weak divergence over the *constrained* rows. The removed
gauge row is where the cavity's net flux would live, and it is deliberately not imposed.

### A limit on transient comparisons

Short transients cannot be compared between the two codes: the MAC starts from a completely
stationary state, lid included, while a lifting imposes the lid from t=0, so the two initial
conditions differ by an O(1) boundary layer. The difference decays (0.081 at T=1, 0.049 at
T=2, 0.0068 at T=10, 0.0067 at T=40 in the L2 measure above) and is not a discretisation
error: it is unchanged between 20 and 24 modes. The comparison is therefore made at a time
by which the flow is steady, and dt=0.04 and dt=0.02 give identical numbers because both
codes run the same scheme and the temporal errors cancel.

## Next step

- The cavity is compared against the MAC solver, not against Ghia's tabulated centreline
data, so the absolute accuracy of both is bounded by the MAC's own error; adding that
comparison is the remaining independent check.
- Open, and separate from S3: whether the SAV scheme's sub-second order at small steps is
  intended. The decisive experiment is a dt -> 0 study of the scalar `r` and of the stage
  residual on one fixed discretisation, which this seam now makes cheap to run.


## Fixed lifting: SAV pairing with total velocity (2026-09-21)

The stored state is the homogeneous part $w_h$, while the physical velocity is $u_h=w_h+g$ for a fixed lifting. Both homogeneous and lifted convection now use the antisymmetric quadrature form
\[
b_h(a,b,c)=\tfrac12[(a\cdot\nabla b,c)_Q-(a\cdot\nabla c,b)_Q].
\]
The model returns both the homogeneous-test load $N_h(w_h)$ and the scalar $\ell_h(w_h)=b_h(u_h,u_h,g)$ through `nonlinear_with_lifting`. The latter is integrated directly against the lifting on the same nodes, not against a projected lifting. Thus $(N_h(w_h),w_h)+\ell_h(w_h)=0$ to roundoff. MAC models return zero lifting work.

For the incremental stages, combine the scalar using the same weights as the convective vector:
\[
\ell_{n,1}=\eta\ell_h(w^n),\qquad
\ell_{n,2}=-\ell_h(w^n)+(1-\delta)\ell_h(w_{n,1}).
\]
The scalar stage equation uses $(B_{n,i},w_{n,i})+\ell_{n,i}$, representing the weak pairing with total stage velocity. In the existing cubic only $\alpha$ changes to $(B_{n,i},V)+\ell_{n,i}$; $\beta=(B_{n,i},W)$ and the two Stokes solves are unchanged. The scalar residual check includes the same term. Cross-stage pairing is not forced to zero.

This restores the zero-$r$ invariant of the fixed-space semidiscrete extension. It does not prove a general time-uniform $O(\tau)$ estimate or a moving-wall energy theorem: the boundary-driven energy balance needs its own lifting terms. Time-dependent lifting is not supported by this derivation. Underintegration remains a separate accuracy issue even though algebraic skew cancellation is exact. Earlier cavity error numbers in this document predate this correction and are historical, not current measurements.


## Polynomial de-aliasing quadrature (2026-09-21)

With N=space.size modes, phi_k=P_k-P_{k+2}, k=0,...,N-1, has maximum degree p=N+1. Each convective weak integrand is a product of three fields with one derivative; in the direction not differentiated its degree can reach 3p. A safe tensor Gauss rule therefore satisfies 2q-1 >= 3(N+1) in each direction. The effective node count is

`q = max(N + dealias, (3*(N+1)+2)//2)`.

`dealias` retains its meaning as the user-requested extra-node lower bound. `quadrature_extra` exposes the effective value, recalculated if that request changes. Convection loads, the direct lifting pairing, and spectral-model physical diagnostics use this rule consistently. The built-in LidLifting is a projected polynomial in x and linear in y, so the same degree bound applies, including the sharp-lid polynomial projection. This does not remove sharp-corner spatial truncation/Gibbs error, nor prove exact integration for arbitrary non-polynomial custom liftings or forcings. Force projection remains independently controlled by its existing extra argument.

The earlier warning that the default convection rule underintegrates high modes is resolved by this change. The skew identity alone was insufficient; mass-norm agreement with a substantially denser quadrature is now tested at 20, 28 and 40 modes, both with and without a lifting.
