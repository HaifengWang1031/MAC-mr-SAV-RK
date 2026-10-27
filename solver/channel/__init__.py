"""Channel MAC discretisation: periodic x, stationary no-slip y."""
from .grid import ChannelGrid
from .operators import ChannelOperators
from .model import ChannelModel
from .fft_stokes import FFTStokes

__all__ = ['ChannelGrid', 'ChannelOperators', 'ChannelModel', 'FFTStokes']
