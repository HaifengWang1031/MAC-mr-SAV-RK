"""Explicit optional FEM solver inputs and computation-only library provenance.

Validation and record loading do not import PETSc or construct a solver.
"""
from importlib.metadata import version
from typing import Any

import numpy as np

FEM_BACKEND_KEYS = ('mass_backend', 'stokes_amg_type', 'fem_load_mode')
STOKES_BACKENDS = ('direct', 'recycled', 'petsc')


def validate_solver_options(config: dict[str, Any]) -> None:
    backend = config.get('stokes_backend', 'direct')
    if backend not in STOKES_BACKENDS:
        raise ValueError('Unknown FEM Stokes backend')
    if config.get('fem_load_mode', 'riesz') not in ('riesz', 'weak'):
        raise ValueError('Unknown FEM fem_load_mode')
    if config.get('mass_backend', 'direct') not in ('direct', 'cg', 'metis'):
        raise ValueError('Unknown FEM mass_backend')
    if config.get('stokes_amg_type', 'gamg') not in ('gamg', 'hypre'):
        raise ValueError('Unknown FEM stokes_amg_type')
    if 'stokes_amg_type' in config and backend != 'petsc':
        raise ValueError('stokes_amg_type requires the petsc backend')


def model_solver_options(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in FEM_BACKEND_KEYS if key in config}


def solver_provenance(source: dict[str, Any], config: dict[str, Any]) -> None:
    """Enrich only an explicitly selected PETSc computation's run identity."""
    if config.get('stokes_backend') != 'petsc':
        return
    from petsc4py import PETSc
    source['packages']['petsc4py'] = version('petsc4py')
    source['petsc'] = dict(version=list(PETSc.Sys.getVersion()),
        scalar_dtype=str(np.dtype(PETSc.ScalarType)), index_dtype=str(np.dtype(PETSc.IntType)),
        communicator='COMM_SELF', amg_type=config.get('stokes_amg_type', 'gamg'),
        has_hypre=bool(PETSc.Sys.hasExternalPackage('hypre')))
