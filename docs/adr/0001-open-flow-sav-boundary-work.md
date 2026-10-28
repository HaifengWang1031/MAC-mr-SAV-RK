# Fixed inlet, natural outlet and SAV boundary work

The DFG2D-2 cylinder benchmark uses steady nonzero inlet and natural gradient-form traction at the outlet, unlike the existing closed homogeneous FEM problem. Retain the shared time-stage formulas, represent known Dirichlet coefficients separately from physical free coefficients, and account explicitly for convective boundary energy work in the SAV scalar pairing; forcing closed-wall zero-work or mean-zero-pressure assumptions onto this problem would solve a different system.

## Consequences

Let g contain the known fixed Dirichlet coefficients and zero free coefficients, U the full velocity, M the free consistent mass, A the full gradient stiffness and B the full weak divergence. Free momentum and continuity are

M u_t + nu(A_II u + A_Ib g) + N_I(U) - B_I^T p = F_I,
B_I u + B_b g = 0.

Here N_I is a weak load; the Model returns its Riesz coefficients M^(-1)N_I, not that load itself.

The natural outlet makes pressure unique; no mean gauge or pressure penalty is added. First Stokes RHS imposes the known boundary/continuity load; every SAV correction RHS is homogeneous. Exposed K/D and residual checks use exactly those same affine boundary contributions.

Use N(U;v)=b_skew(U,U,v)+1/2 integral_boundary (U.n)(U.v). For incompressible fields this is the advective weak form with the benchmark's natural outlet, rather than a modified nonlinear traction. On any discrete field, N(U;U)=P_boundary=1/2 integral_boundary (U.n)|U|^2 because the volume form is antisymmetric. Supply scalar boundary work ell(U)=N(U;g)-P_boundary, so <M^(-1)N_I(U),u>_M+ell(U)=0 at equal states while cross-stage pairings retain the existing SAV formulas. Compute both boundary terms on the same actual facets/quadrature, not by inventing a zero scalar or cancelling a coefficient dot product.

This restores the zero-r continuous-time invariant of the fixed-lifting semidiscrete extension; it does not assert a new unconditional open-boundary energy theorem or support time-dependent inlet traces. Fixed boundary values also mean their time derivative is zero. Independent Poiseuille/pressure-force tests and developed cylinder/reference comparisons are required.

Stage pressures are still increments and must be labelled natural-outflow (not mean-zero). For physical forces/probes solve the instantaneous homogeneous-boundary velocity-rate constraint with mass-only Stokes; this provides a separate semidiscrete physical pressure, not a claimed high-order endpoint approximation from a stage increment. The DFG stress is nu*grad(U)-pI, not the symmetric-gradient traction; the body normal is opposite the fluid normal on the obstacle.

Existing closed/no-slip FEM keeps its old pressure gauge, boundary work, layout and record semantics. This decision is specific to fixed-inlet open flow and does not silently change old models or time formats.
