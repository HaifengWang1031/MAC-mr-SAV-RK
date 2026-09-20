"""Enumerate finite numerical real roots and select minimum |r| (signed tie break)."""
from dataclasses import dataclass
import numpy as np
from ..mac.grid import Array

@dataclass
class RootChoice:
    candidates: list[float]
    residuals: list[float]
    selected: float

def real_roots(coefficients: Array, tolerance: float = 1e-10) -> RootChoice:
    c=np.asarray(coefficients,dtype=float)
    if c.ndim!=1 or not np.isfinite(c).all() or c.size<2:
        raise ValueError('Invalid polynomial')
    # Strip exact zeros only: a small cubic term can still create genuine far roots.
    c=np.trim_zeros(c,'f')
    if c.size<2: raise RuntimeError('Polynomial has no isolated root')
    c=c/np.max(np.abs(c))
    accepted=[]
    # Multiple real roots split into tiny complex clusters under coefficient
    # roundoff. Degree-dependent O(eps**(1/degree)) neighborhoods must reach
    # the real-axis residual check; the stricter residual still decides validity.
    imaginary_tolerance=max(tolerance,8*np.finfo(float).eps**(1/(c.size-1)))
    for root in np.roots(c):
        if abs(root.imag)>imaginary_tolerance*(1+abs(root.real)): continue
        r=float(root.real)
        # Two Newton refinements, accepted only when their scaled residual improves.
        for _ in range(2):
            derivative=float(np.polyval(np.polyder(c),r))
            if derivative==0: break
            new=r-float(np.polyval(c,r))/derivative
            if not np.isfinite(new): break
            if abs(np.polyval(c,new))<=abs(np.polyval(c,r)): r=new
        denominator=float(np.polyval(np.abs(c),abs(r)))
        residual=abs(float(np.polyval(c,r)))/max(denominator,np.finfo(float).tiny)
        if np.isfinite(r) and np.isfinite(residual) and residual<=tolerance:
            accepted.append((r,residual))
    if not accepted: raise RuntimeError('No residual-validated numerical real root')
    accepted.sort()
    # Preserve multiplicities returned numerically; root_count counts these candidates.
    minimum=min(abs(r) for r,_ in accepted)
    tied=[r for r,_ in accepted if abs(r)<=minimum+32*np.finfo(float).eps*(1+minimum)]
    return RootChoice([r for r,_ in accepted],[e for _,e in accepted],min(tied))
