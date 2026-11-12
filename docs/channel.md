# Channel MAC model

`solver.channel.ChannelModel` solves two-dimensional incompressible velocity-pressure
Navier–Stokes on `(0,Lx) × (0,H)`, periodic in x and stationary homogeneous no-slip
at y=0,H. Both spatial directions use second-order MAC differences. The default
coupled sparse `DirectStokes` and optional FFT backend solve the same MAC equations.
All six shared schemes operate
through the Model seam: `imex_euler/imex_euler_mrsav`, `sdirk2/sdirk2_mrsav`, and
`sdirk3/sdirk3_mrsav`. All support fixed/prescribed steps; the four SDIRK schemes
also support embedded I/PI adaptivity. Euler has no adaptive estimator. See the
[registry](../solver/schemes/registry.py) and [adaptive guide](../solver/adaptivity/README.md).

## Faces and independent degrees of freedom

`u` has shape `(ny,nx+1)` at `(i*hx,(j+.5)*hy)`, with `u[:,-1]=u[:,0]`.
`v` has shape `(ny+1,nx)` at `((i+.5)*hx,j*hy)`, with zero bottom/top rows.
Pressure has shape `(ny,nx)` at cell centres. Packing concatenates `u[:,:nx]`
and `v[1:-1,:]`; there are `nx*(2*ny-1)` real velocity unknowns. The repeated
periodic endpoint is never counted in inner products or diagnostics.

`pack` checks finite real data, periodic endpoint agreement to roundoff, and exact
zero normal wall velocities. `unpack` restores canonical boundary values without
mutating input. `pack_load` samples only independent faces and does not require a
physical force to vanish on the wall. `curl` accepts a periodic vertex streamfunction
with each wall constant in x; distinct bottom/top constants can represent net flux.

Tangential u values are half a cell from the walls; they are not forced to zero.
Odd ghost values give the u negative-Laplacian endpoint diagonal `3/hy²`.
The v block eliminates zero normal wall faces and uses the ordinary tridiagonal
Dirichlet stencil. Periodic x neighbours connect across the seam.

## Coupled solve and mean flow

`D` contains every cell's face-flux difference, `G=-D.T`, and `K` is symmetric
positive definite because of the y walls. The solve is

```text
(mass I + viscosity K) w + G p = rhs,  D w = 0.
```

`viscosity` here is the shifted stage coefficient, e.g. `nu*dt*Dii`, not always
physical nu. `DirectStokes` accepts common sparse operators via a structural protocol;
its original solve algorithm is unchanged. It removes one dependent continuity row
and one pressure gauge, then checks ALL momentum and continuity equations. Factors
use the exact `(mass,viscosity)` key with bounded LRU reuse (default four entries).
SAV right-hand sides share a matrix and use one batched solve.

The only pressure ambiguity is a global constant. No streamwise mean-velocity
constraint is imposed. A steady solve with mass=0 accepts nonzero mean forcing.
Uniform acceleration `f=(F,0)` represents `-d(p/rho)/dx=F`; the computed pressure
is the periodic reduced pressure, excluding the background `-F*x`.

## FFT backend

`ChannelModel(grid,nu,backend='fft')` selects `FFTStokes`; `backend='direct'` or
the default selects the sparse reference backend. A custom `StokesBackend` can
still be injected. The FFT solver uses orthonormal rFFT in x only, retaining ALL
MAC modes, including the even-grid Nyquist mode. The x divergence symbol is
`d=(exp(i*theta)-1)/hx`, gradient is `-conj(d)`, and negative-Laplacian eigenvalue
is `4*sin(theta/2)^2/hx²`. Using `i*k` or dropping a 2/3 band would change the
discretisation and is deliberately avoided.

Each nonzero mode solves the full one-dimensional velocity-pressure system.
Reordering by y gives bandwidth three; pivoted complex LAPACK band LU supports
multiple right-hand sides and bounded exact-shift caching. Mode zero has v=0,
a tridiagonal u solve, and pressure reconstructed from the vertical load with
zero y mean. The y mean of nonzero pressure modes remains unrestricted.
Factors/solves have O(nx*ny) work/storage per cached shift, plus x FFT cost
O(nx*ny*log(nx)); physical momentum and complete divergence are checked against
the SAME assembled MAC operators after inverse FFT.

## Nonlinearity and diagnostics

The channel Numba kernel evaluates centred dual-cell fluxes with periodic x indices
and odd tangential y ghosts, retaining the conservative-minus-half-dual-divergence
skew form. Its energy pairing vanishes to roundoff even away from ker(D).
Forces and nonlinear loads remain unprojected so stage pressure information is retained.
There is no spectral filter or dealiasing operation.

Inner products are `hx*hy` times the dot product of independent velocities. Diagnostics
include kinetic/modified energy, full physical divergence, H1 seminorm, force power,
viscous dissipation, bulk velocity, mean flow rate, and streamwise flow-rate spread.
`Q_i=hy*sum_j u[j,i]` is independent of i for a solenoidal impermeable field.
`wall_shear_bottom/top` are kinematic signed y-gradient estimates `nu*(2*u_bottom/hy)`
and `nu*(-2*u_top/hy)` averaged in x. They are not signed outward wall tractions,
and a general second-order wall-shear accuracy claim has not been established.

## References and validation boundaries

The steady continuous Poiseuille profile is `F*y*(H-y)/(2*nu)`. With this half-cell
closure, the exact discrete equilibrium adds `F*hy²/(8*nu)` to its u samples.
Consequently continuum L2 error is second order, while discrete-equilibrium residuals
can be at roundoff. The poiseuille case starts from the discrete equilibrium;
startup starts from rest; perturbed adds a solenoidal vertex-curl perturbation.

Continuous MMS uses analytic derivatives of
`psi=a(t)*(sin(k*x)+.2*sin(2*k*x))*sin²(pi*y/H)` and a nonzero analytic pressure.
It verifies spatial accuracy. The fixed-grid discrete MMS builds loads using
independent NumPy stencils/fluxes, without invoking the solver's operators or kernel,
and verifies temporal order without a spatial error floor. Both initial fields use
the vertex curl. The startup sine-series reference is evaluated at the actual saved
time; initial wall regularity makes it unsuitable as the sole temporal-order test.

Stage pressure is the existing incremental multiplier, not a demonstrated high-order
physical endpoint pressure. Scalar local-defect control is not a global bound on |r|.
Short MMS and perturbed-channel checks do not validate long-time high-Re turbulence.
Moving walls, prescribed flow-rate feedback, stretched grids, MPI backends and
checkpoint continuation are not implemented in the current model.

See experiment entry points（本地材料：`../experiments/channel_ns/README.md`） and
[recorded verification](validation.md).
