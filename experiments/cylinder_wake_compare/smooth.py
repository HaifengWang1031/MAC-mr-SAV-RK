"""Continuous vorticity recovery and masked smoothing for visualization only."""
from typing import Any

import matplotlib.tri as mtri
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from skfem import Basis, ElementTriP1

from experiments.cylinder_ns.reconstruct import decode_velocity


class SmoothVorticityGrid:
    """Recover saved P2 curl in continuous P1, then sample/filter a fixed raster.

    This is a display transform, not a velocity update or a flow solve. The
    cylinder is masked before filtering; normalized weights avoid zero dilution
    near its edge. No temporal interpolation or source-field mutation occurs.
    """

    def __init__(self, basis: Any, *, bounds: tuple[float, float, float, float],
                 width: int = 1600, sigma: float = .008) -> None:
        if type(width) is not int or width < 32 or not np.isfinite(sigma) or sigma < 0.:
            raise ValueError('Expected raster width >=32 and finite nonnegative smoothing sigma')
        xmin, xmax, ymin, ymax = bounds
        if not np.isfinite(bounds).all() or xmax <= xmin or ymax <= ymin:
            raise ValueError('Expected ordered finite physical display bounds')
        self.basis = basis
        self.extent = bounds
        self.sigma = float(sigma)
        self.width = width
        self.height = max(2, round(width*(ymax-ymin)/(xmax-xmin)))
        dx, dy = (xmax-xmin)/width, (ymax-ymin)/self.height
        self.x = xmin+(np.arange(width)+.5)*dx
        self.y = ymin+(np.arange(self.height)+.5)*dy
        x, y = np.meshgrid(self.x, self.y)

        p1 = Basis(basis.mesh, ElementTriP1(), quadrature=basis.quadrature)
        self._phi = np.stack([item[0] for item in p1.basis])
        self._dx = p1.dx
        self._dofs = p1.element_dofs
        self._size = p1.N
        mass = np.einsum('ieq,jeq,eq->ije', self._phi, self._phi, self._dx)
        shape = mass.shape
        rows = np.broadcast_to(self._dofs[:, None, :], shape).ravel()
        cols = np.broadcast_to(self._dofs[None, :, :], shape).ravel()
        self._mass = splu(coo_matrix((mass.ravel(), (rows, cols)), shape=(p1.N, p1.N)).tocsc())

        mesh = basis.mesh
        triangle = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
        cells = np.asarray(triangle.get_trifinder()(x, y), dtype=np.int64)
        self.mask = cells < 0
        self._valid = ~self.mask
        valid_cells = cells[self._valid]
        vertices = mesh.p[:, mesh.t[:, valid_cells]]
        a, b, c = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        xx, yy = x[self._valid]-a[0], y[self._valid]-a[1]
        determinant = (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        lb = (xx*(c[1]-a[1])-yy*(c[0]-a[0]))/determinant
        lc = ((b[0]-a[0])*yy-(b[1]-a[1])*xx)/determinant
        self._weights = np.array([1.-lb-lc, lb, lc])
        self._pixels = p1.element_dofs[:, valid_cells]
        self._sigma_pixels = (sigma/dy, sigma/dx)
        self._normalization = gaussian_filter(self._valid.astype(float), self._sigma_pixels,
                                               mode='nearest') if sigma else np.ones(x.shape)

    def frame(self, u: np.ndarray, v: np.ndarray) -> np.ma.MaskedArray:
        """Return filtered physical vorticity on the fluid raster, with the body masked."""
        gradient = self.basis.interpolate(decode_velocity(self.basis, u, v)).grad
        curl = gradient[1, 0]-gradient[0, 1]
        local = np.einsum('eq,ieq,eq->ie', curl, self._phi, self._dx)
        rhs = np.bincount(self._dofs.ravel(), weights=local.ravel(), minlength=self._size)
        recovered = self._mass.solve(rhs)
        values = np.zeros(self.mask.shape)
        values[self._valid] = np.sum(recovered[self._pixels]*self._weights, axis=0)
        if self.sigma:
            filtered = gaussian_filter(values, self._sigma_pixels, mode='nearest')
            values = np.divide(filtered, self._normalization, out=np.zeros_like(values),
                               where=self._normalization > 0.)
        return np.ma.array(values, mask=self.mask)

    def metadata(self) -> dict[str, Any]:
        """Describe every display transform independently of the numerical run config."""
        return {'recovery': 'continuous P1 L2 projection of the saved affine-P2 curl',
                'filter': 'mask-normalized spatial Gaussian; nearest extension at rectangle boundary',
                'sigma_physical': self.sigma, 'raster_shape': [self.height, self.width],
                'raster_extent': list(self.extent), 'cylinder_masked': True,
                'image_interpolation': 'bilinear', 'temporal_interpolation': False,
                'qualification': 'display only; does not establish accuracy or remove raw-field oscillations'}
