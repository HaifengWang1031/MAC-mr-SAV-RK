"""Fourth-order/negative-fourth-order dissipative MHD; shared SAV stages unchanged."""
from collections.abc import Callable

import numpy as np

from .mhd import FourierMHD, MHDState
from .space import ComplexArray, FourierSpace, RealArray


class ForcedFourierMHD(FourierMHD):
    """Equal u/B damping nu_plus*k^4 + nu_minus*k^-4 (zero mode untouched).

    nu remains nu_plus at the shared stage seam; apply_K carries both multipliers.
    Forcing is an additive physical load, not multiplied by the SAV coupling.
    """

    def __init__(self, space: FourierSpace, nu_plus: float, nu_minus: float, *,
                 force: Callable[[float], ComplexArray], tolerance: float = 1e-10) -> None:
        super().__init__(space, nu_plus, nu_plus, tolerance=tolerance)
        if not np.isfinite(nu_minus) or nu_minus < 0:
            raise ValueError('nu_minus must be finite and nonnegative')
        self.nu_minus = nu_minus
        self.small = nu_plus*space.k2**2
        self.large = nu_minus*space.inverse_k2**2
        self._k = (self.small+self.large)[None, None]/nu_plus
        self._external_force = force

    def _evaluate_force(self, t: float) -> ComplexArray:
        value = np.asarray(self._external_force(t))
        if value.shape != self._zero.shape:
            raise ValueError('Expected paired MHD forcing')
        for block in value:
            self.space.check_velocity(block)
        if self.physical_divergence_inf(value) > self.tolerance*(1+self.max_abs(value)):
            raise ValueError('MHD forcing must be divergence free')
        return value

    def diagnostics(self, state: MHDState) -> dict[str, float]:
        s = self.space
        z = self.vector(state)
        forcing = self.force(state.t)
        a = flux_potential(s, z[1])
        ga = flux_potential(s, forcing[1])
        result = super().diagnostics(state)
        result.update(power_u=s.inner(z[0], forcing[0])/s.area,
                      power_B=s.inner(z[1], forcing[1])/s.area,
                      potential_squared=.5*s.inner(a, a)/s.area,
                      power_Q=s.inner(a, ga)/s.area,
                      mean_u=float(z[0, 0, 0, 0].real), mean_v=float(z[0, 1, 0, 0].real),
                      mean_Bx=float(z[1, 0, 0, 0].real), mean_By=float(z[1, 1, 0, 0].real))
        for name, multiplier in (('small', self.small), ('large', self.large)):
            result['dissipation_E_'+name] = s.inner(z, multiplier*z)/s.area
            result['dissipation_Q_'+name] = s.inner(a, multiplier*a)/s.area
        result['dissipation'] = result['dissipation_E_small']+result['dissipation_E_large']
        result['power'] = result['power_u']+result['power_B']
        return result


def flux_potential(space: FourierSpace, magnetic: ComplexArray) -> ComplexArray:
    """Zero-mean a for B=(a_y,-a_x), also valid for an induction load."""
    return 1j*(space.kx*magnetic[1]-space.ky*magnetic[0])*space.inverse_k2


SAMPLED_SPECTRA = ('Pi_E', 'Pi_Q', 'E_u', 'E_B')
LOWPASS_SPECTRA = ('E_low', 'Q_low', 'power_E_low', 'power_Q_low',
                   'diss_E_plus_low', 'diss_E_minus_low', 'diss_Q_plus_low', 'diss_Q_minus_low')


def transfer_spectra(model: FourierMHD, z: ComplexArray, t: float | None = None) -> dict[str, RealArray]:
    """Physical unmodified nonlinear flux through |k|<=K, K=0,1,... .

    Shell j is (j-1,j], including zero in shell 0. rFFT weights give spatial
    averages, not integrals. Flux is positive out of the low-pass ball. These
    are PDE fluxes evaluated on the state, not a reconstruction of RK stage work.

    With ``t`` supplied and an additive force available, the same call also returns
    cumulative low-pass energy, injection and both dissipation channels, so that
    ``dE_<K/dt = P_<K - D+_<K - D-_<K - Pi_E(K)`` can be checked per shell. The
    identity uses exact shell-summed mode contributions, never a shell-centre
    wavenumber approximation.
    """
    s = model.space
    nonlinear = model.nonlinear(z)
    a, na = flux_potential(s, z[1]), flux_potential(s, nonlinear[1])
    shells = np.ceil(np.sqrt(s.k2)).astype(int)
    size = int(shells[s.mask].max())+1

    def shell_sum(values: RealArray) -> RealArray:
        return np.bincount(shells[s.mask], weights=(s.weights*values)[s.mask], minlength=size)

    energy_transfer = np.real(np.sum(np.conj(z)*nonlinear, axis=(0, 1)))
    energy_density = .5*np.sum(np.abs(z)**2, axis=(0, 1))
    q_density = .5*np.abs(a)**2
    result = {'K': np.arange(size, dtype=float),
              'Pi_E': np.cumsum(shell_sum(energy_transfer)),
              'Pi_Q': np.cumsum(shell_sum(np.real(np.conj(a)*na))),
              'E_u': shell_sum(.5*np.sum(np.abs(z[0])**2, axis=0)),
              'E_B': shell_sum(.5*np.sum(np.abs(z[1])**2, axis=0)),
              'E_low': np.cumsum(shell_sum(energy_density)),
              'Q_low': np.cumsum(shell_sum(q_density))}
    small = getattr(model, 'small', None)
    large = getattr(model, 'large', None)
    force = getattr(model, 'force', None)
    if small is not None and large is not None:
        damping = np.sum(np.abs(z)**2, axis=(0, 1))
        result['diss_E_plus_low'] = np.cumsum(shell_sum(small*damping))
        result['diss_E_minus_low'] = np.cumsum(shell_sum(large*damping))
        result['diss_Q_plus_low'] = np.cumsum(shell_sum(small*np.abs(a)**2))
        result['diss_Q_minus_low'] = np.cumsum(shell_sum(large*np.abs(a)**2))
    if t is not None and force is not None:
        forcing = force(t)
        ga = flux_potential(s, forcing[1])
        result['power_E_low'] = np.cumsum(shell_sum(np.real(np.sum(np.conj(z)*forcing, axis=(0, 1)))))
        result['power_Q_low'] = np.cumsum(shell_sum(np.real(np.conj(a)*ga)))
    return result
