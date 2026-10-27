"""Explicit identities and storage/estimator metadata for implemented time schemes."""
from dataclasses import dataclass
from collections.abc import Callable

from ..core import Scheme
from .imex_euler import IMEXEuler
from .imex_euler_mrsav import IMEXEulerMRSAV
from .sdirk2 import SDIRK2
from .sdirk2_mrsav import SDIRK2MRSAV
from .sdirk3 import SDIRK3
from .sdirk3_mrsav import SDIRK3MRSAV


@dataclass(frozen=True)
class SchemeInfo:
    order: int
    n_stages: int
    n_roots: int
    estimator_order: int | None
    pressure_semantics: str = 'incremental_stage_multiplier_mean_zero'


DEFAULT_SCHEMES = ('sdirk2', 'sdirk2_mrsav', 'sdirk3', 'sdirk3_mrsav')
_INFO = {
    'sdirk2': SchemeInfo(2, 2, 3, 2),
    'sdirk2_mrsav': SchemeInfo(2, 2, 3, 2),
    'sdirk3': SchemeInfo(3, 4, 5, 3),
    'sdirk3_mrsav': SchemeInfo(3, 4, 5, 3),
    'imex_euler': SchemeInfo(1, 1, 1, None, 'euler_step_multiplier_mean_zero'),
    'imex_euler_mrsav': SchemeInfo(1, 1, 1, None, 'euler_step_multiplier_mean_zero'),
}
SCHEMES = tuple(_INFO)
_FACTORIES: dict[str, Callable[[float], Scheme]] = {
    'sdirk2': lambda gamma: SDIRK2(),
    'sdirk2_mrsav': lambda gamma: SDIRK2MRSAV(gamma),
    'sdirk3': lambda gamma: SDIRK3(),
    'sdirk3_mrsav': lambda gamma: SDIRK3MRSAV(gamma),
    'imex_euler': lambda gamma: IMEXEuler(),
    'imex_euler_mrsav': lambda gamma: IMEXEulerMRSAV(gamma),
}


def scheme_info(name: str) -> SchemeInfo:
    try:
        return _INFO[name]
    except KeyError as error:
        raise ValueError(f'Unknown scheme: {name}') from error


def make_scheme(name: str, gamma: float = 1.) -> Scheme:
    scheme_info(name)
    return _FACTORIES[name](gamma)
