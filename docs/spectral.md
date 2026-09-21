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

## What is not verified yet: the manufactured steady Stokes solution

`tests/test_spectral.py::test_manufactured_steady_stokes_converges_spectrally` is marked
xfail. Measured behaviour with the manufactured fields of `experiments/problems.py`
(`psi = A sin^2(pi x) sin^2(pi y)`, `nu = 0.1`, `A = 0.2`):

- the algebraic residuals are machine zero — momentum 1e-16 and the continuity rows 1e-16,
  including the row whose pressure mode was removed;
- but the discrete velocity is about three times smaller than the exact one
  (max 0.19 against 0.62), and the L2 errors stay at 3.4e-1 and 4.5e-1 for every
  resolution from 8 to 32 modes: they do not converge;
- the pointwise divergence sits at about 3.1 and drifts slightly upward with resolution.

A non-zero load, operators verified against quadrature of the same weak forms, and a
machine-zero residual together mean the discrete problem being solved is not the intended
one. The error is therefore in the setup rather than in the assembly, and every
self-consistent check — assembly against quadrature of the same weak form — is blind to it
by construction. Earlier attempts to explain it as a leak, a quadrature error or a
missing factor were each refuted by measurement and are recorded here so they are not
retried.

## Next step

Decouple the pieces instead of guessing:

1. solve a pure diffusion (Helmholtz) problem with the same machinery and a polynomial
   exact solution that the Dirichlet basis reproduces exactly — no pressure coupling, so
   any remaining error is in the operator or the load;
2. add the pressure coupling and repeat the manufactured test, expecting spectral
   convergence;
3. add the moving-lid lifting for the inhomogeneous boundary;
4. only then compare against the validated MAC cavity solutions and produce the figures.

## Progress after the diffusion decoupling (2026-09-21)

The decoupled diffusion problem — no pressure coupling, and a polynomial exact solution
that lies in the trial space — is **exact on the square domain**: `max|u_h - u|` between
1e-17 and 3e-17 for `nu = 0.1` and for `nu = 2.5`, at every resolution tried. Operator,
load and solve are therefore all correct there.

On the rectangular domain the error stays at 3.5e-3 regardless of resolution, and it is now
localized to exactly two load entries:

| entry | sympy | assembled load | assembled `nu*S*c_exact` |
|---|---|---|---|
| (0,2) | -0.00410256 | **-0.01083333** | -0.00410256 |
| (2,0) | -0.01083333 | **-0.00410256** | -0.01083333 |

The stiffness applied to the exact solution matches symbolic integration exactly, so the
stiffness is right and the load is transposed in that pair. The continuous identity
`int nu grad(u):grad(v) = int f v` holds symbolically to 1e-14 for those same modes, which
is what makes the table decisive rather than suggestive.

Two traps cost real time here and are recorded so they are not repeated:

- On a square domain the transposition is *invisible*, because the manufactured fields and
  the basis are then symmetric under `x <-> y`. Every square-domain test passes either way,
  so only a rectangular domain can expose it.
- The one-line discriminator tried earlier — the non-zero pattern of `int x phi_k psi_l` —
  is ambiguous, because that pattern is itself symmetric: a load with non-zeros only in the
  first row and one with non-zeros only in the first column produce the same set of row and
  column maxima. Only the numeric value of a known entry settles the convention.

## Next step

Assemble the load component by component, one basis pair at a time, with no tensor-product
shortcut and compare against the symbolic table above. That removes the ambiguity by
construction instead of by argument. Once the rectangular diffusion problem is exact too,
the pressure coupling can be revisited with the same per-mode references.
