# Spectral discretisation (single-element Dirichlet-combined Legendre)

Slice S1 of the plan agreed on 2026-09-21: build the basis and the assembly for a third
discretisation that will implement `solver/model.py`, so the existing schemes, drivers and
records are reused. The lid-driven cavity comparison, in both the sharp and the
regularised form, is S3. No new dependency is involved; only numpy, scipy and sympy.

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

## Next step

- S2: implement `solver/model.py` for this discretisation, so `SDIRK2` and `SDIRK2-mr-ccSAV`
  drive it and the unsteady manufactured test shows the expected second order in time.
- S3: add the moving-lid lifting for the inhomogeneous boundary, run the sharp and the
  regularised lid-driven cavity, and compare against the validated MAC solutions.
