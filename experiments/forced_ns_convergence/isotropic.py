"""Isotropic no-slip initial stream function built from clamped-clamped beam modes.

`psi = sum A_pq phi_p(x) phi_q(y)` with `phi_p'''' = beta_p^4 phi_p` and
`phi_p = phi_p' = 0` at both ends, so that `u = d_y psi` and `v = -d_x psi` are divergence
free in the continuum and both vanish on all four walls. `k_lo`, `k_hi` and `alpha` are the
knobs; wavenumbers are wavelengths per unit length, the same convention as the
`sin(2*pi*m*x)` forcing, so a band and `m` are directly comparable.

The naive `cosh`/`sinh` form of `phi_p` loses every significant digit once `beta_p` exceeds
roughly 30 (measured against a 60-digit reference: 2e-2 absolute error at p=12 and 8e-1 at
p=24, against mode values of order one), so the `delta` form below is used and is checked
against that reference in `tests/test_isotropic_ic.py`.

`docs/validation.md` records the measured anisotropy of one realisation, the band/lifetime
trade-off (`tau = 1/(nu (2 pi K)^2)`) and why the field is only statistically isotropic:
the clamped basis and the four walls break rotation invariance away from the domain centre.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from solver.mac.grid import MACGrid, Array

TARGETS=('max','rms','energy')

def _require_index(p: int) -> None:
    if type(p) is not int or p<1: raise ValueError('Beam index must be a positive integer')

@lru_cache(maxsize=256)
def beam_eigenvalue(p: int) -> float:
    """`beta_p`, the p-th root of `cos(b) cosh(b) = 1`; `beta_p ~ (p+1/2) pi`."""
    _require_index(p)
    # Dividing cos(b) cosh(b) - 1 by cosh(b) > 0 keeps the sign and cannot overflow.
    shape=lambda b: np.cos(b)-2.*np.exp(-b)/(1.+np.exp(-2.*b))
    centre=(p+0.5)*np.pi
    for half in (0.3,0.5,0.8,1.2,1.5):
        if shape(centre-half)*shape(centre+half)<0.: break
    else: raise ValueError(f'No bracket for beam eigenvalue {p}')
    return float(brentq(shape,centre-half,centre+half,xtol=1e-15,rtol=1e-15))

def _delta(b: float) -> float:
    """`1 - sigma` with `sigma = (cosh b - cos b)/(sinh b - sin b)`, free of cancellation."""
    return float((-np.exp(-b)+(np.cos(b)-np.sin(b)))/(np.sinh(b)-np.sin(b)))

def _beam(b: float, x: Array) -> Array:
    d=_delta(b)
    return 0.5*(d*np.exp(b*x)+(2.-d)*np.exp(-b*x))-np.cos(b*x)+(1.-d)*np.sin(b*x)

def beam(p: int, x: Array) -> Array:
    """Clamped beam mode `phi_p`; `phi_p(0) = phi_p(1) = 0` and `phi_p(1-x) = (-1)^(p+1) phi_p(x)`."""
    return _beam(beam_eigenvalue(p),x)

def beam_derivative(p: int, x: Array) -> Array:
    """`phi_p'`, zero at both ends, so the tangential velocity vanishes on the walls."""
    b=beam_eigenvalue(p); d=_delta(b)
    return b*(0.5*(d*np.exp(b*x)-(2.-d)*np.exp(-b*x))+np.sin(b*x)+(1.-d)*np.cos(b*x))

@lru_cache(maxsize=256)
def beam_norm(p: int) -> float:
    """`||phi_p||_2` on [0,1] by 200-node Gauss-Legendre quadrature (the integrand is stable)."""
    _require_index(p)
    nodes,weights=np.polynomial.legendre.leggauss(200)
    return float(np.sqrt(0.5*np.sum(weights*_beam(beam_eigenvalue(p),0.5*(nodes+1.))**2)))

def mode_wavenumber(p: int, q: int) -> float:
    """Wavelengths per unit length of mode `(p,q)` on the unit domain; the forcing carries `m`.

    On a domain of length `lx` the physical wavenumber of the same mode is `mode_wavenumber/lx`.
    """
    return float(np.hypot(beam_eigenvalue(p),beam_eigenvalue(q))/(2.*np.pi))

def velocity_at(grid: MACGrid, modes: list[tuple[int,int]], coefficients: Array,
                x: Array, y: Array) -> tuple[Array,Array]:
    """Continuum `(u, v) = (d_y psi, -d_x psi)` of the mode sum at arbitrary coordinates.

    `psi(x,y) = sum a_pq phi_p(x/lx) phi_q(y/ly)`, so the chain rule puts `1/ly` on `u` and
    `1/lx` on `v`; the test suite uses this to evaluate the field on the wall itself, where a
    face-centred sample can only ever be `O(h)` rather than zero.
    """
    u=np.zeros_like(x); v=np.zeros_like(y)
    for (p,q),a in zip(modes,coefficients):
        u+=a*beam(p,x/grid.lx)*beam_derivative(q,y/grid.ly)/grid.ly
        v-=a*beam_derivative(p,x/grid.lx)*beam(q,y/grid.ly)/grid.lx
    return u,v

def coefficient_table(grid: MACGrid, *, k_lo: float, k_hi: float, alpha: float, seed: int,
                      symmetric: bool=False) -> tuple[list[tuple[int,int]], Array]:
    """Modes inside the band and their coefficients `~ K^(-1-alpha/2)`, divided by the L2 norms.

    `symmetric` reuses one coefficient per unordered pair, which makes the field invariant
    under `x <-> y` and so makes the two component energies exactly equal.
    """
    modes=[(p,q) for p in range(1,grid.nx) for q in range(1,grid.ny) if k_lo<=mode_wavenumber(p,q)<=k_hi]
    if not modes: raise ValueError('Empty wavenumber band')
    wavenumbers=np.array([mode_wavenumber(p,q) for p,q in modes])
    raw=np.random.default_rng(seed).standard_normal(len(modes))*wavenumbers**(-(1.+alpha/2.))
    if symmetric:
        drawn: dict[tuple[int,int],float]={}
        for index,(p,q) in enumerate(modes):
            key=(min(p,q),max(p,q))
            raw[index]=drawn.setdefault(key,float(raw[index]))
    return modes, raw/np.array([beam_norm(p)*beam_norm(q) for p,q in modes])

def build_isotropic(grid: MACGrid, *, k_lo: float=4., k_hi: float=12., alpha: float=5./3., seed: int=0,
                    target: str='max', value: float=1., symmetric: bool=False) -> Array:
    """Packed MAC velocity from an isotropic band of clamped-beam stream modes.

    The returned field is divergence free to the O(h^2) sampling error of the discrete
    divergence and satisfies no slip on all four walls; `experiments.workflow` optionally
    applies one discrete Leray projection (`mass=1, viscosity=0`) before the first step.
    """
    if target not in TARGETS: raise ValueError(f'Unknown target {target}')
    if type(seed) is not int or seed<0: raise ValueError('Invalid seed')
    if not np.isfinite([k_lo,k_hi,alpha,value]).all() or not 0.<k_lo<k_hi or value<=0:
        raise ValueError('Invalid isotropic controls')
    if k_hi>min(grid.nx,grid.ny)/8.:
        raise ValueError('Band exceeds the wavenumbers this grid can resolve')
    modes,coefficients=coefficient_table(grid,k_lo=k_lo,k_hi=k_hi,alpha=alpha,seed=seed,symmetric=symmetric)
    xu,yu=grid.coordinates('u'); xv,yv=grid.coordinates('v')
    u,_=velocity_at(grid,modes,coefficients,xu,yu)
    _,v=velocity_at(grid,modes,coefficients,xv,yv)
    # The analytic field is already zero on the walls; this also keeps `pack`'s exact-zero
    # boundary contract, which a rounded evaluation could otherwise break.
    u[:,[0,-1]]=0.; v[[0,-1],:]=0.
    squares=grid.area*(float(np.dot(u[:,1:-1].ravel(),u[:,1:-1].ravel()))
                       +float(np.dot(v[1:-1,:].ravel(),v[1:-1,:].ravel())))/(grid.lx*grid.ly)
    if target=='max': scale=value/max(float(np.abs(u).max()),float(np.abs(v).max()))
    elif target=='rms': scale=value/float(np.sqrt(0.5*squares))
    else: scale=float(np.sqrt(value/(0.5*squares)))
    return grid.pack(scale*u,scale*v)
