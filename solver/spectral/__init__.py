"""Spectral (single-element Dirichlet-combined Legendre) discretisation.

Seam 1 of the plan in `docs/model-seam-spec.md`'s sibling: a third discretisation that
implements `solver/model.py` so the existing schemes, drivers and records are reused.
Slice S1 builds and validates the basis, the assembly and the shifted Stokes solve;
nothing here depends on the MAC discretisation.

This package must be importable without matplotlib, h5py or numba; only numpy, scipy and
sympy (tests) are used.
"""
