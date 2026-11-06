"""Independent doubly periodic Fourier–Leray discretization."""
from .space import FourierSpace
from .model import FourierModel
from .mhd import FourierMHD, MHDState

__all__ = ['FourierSpace', 'FourierModel', 'FourierMHD', 'MHDState']
