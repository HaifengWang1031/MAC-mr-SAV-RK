"""Consistent-mass Taylor–Hood models for closed and fixed-inlet open flow."""
from .cylinder import CylinderSpace
from .model import FEMModel
from .open_flow import OpenFlowModel
from .space import MeshSpace, Space

__all__ = ['FEMModel', 'OpenFlowModel', 'MeshSpace', 'CylinderSpace', 'Space']
