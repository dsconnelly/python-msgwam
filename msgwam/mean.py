from __future__ import annotations
from copy import copy
from typing import TYPE_CHECKING, Optional

import numpy as np

from . import config

if TYPE_CHECKING:
    from .rays import RayCollection

class MeanFlow:
    def __init__(self) -> None:
        """
        Initialize a MeanFlow using the configuration settings. Note that
        config.load_config must have been called for initialization to succeed.
        """

        self.r_faces = np.linspace(*config.grid_bounds, config.n_grid)
        self.r_centers = (self.r_faces[:-1] + self.r_faces[1:]) / 2
        self.dr: float = self.r_faces[1] - self.r_faces[0]

        self.rho = self.init_rho()
        self.u, self.v = self.init_uv()
        self.dp_dx, self.dp_dy = self.init_grad_p()

        nu_centers = config.mu / self.rho
        self.nu = np.interp(self.r_faces, self.r_centers, nu_centers)
        self.pmf_bar = np.zeros((2, *self.r_faces.shape))

    def __add__(self, other: np.ndarray) -> MeanFlow:
        """
        Return a new MeanFlow object sharing the same background profiles
        (density and pressure gradients) but with velocities equal to those of
        this object added to the data in other. Defined to make writing
        time-stepping routines easier.

        Parameters
        ----------
        other
            Array of data to be added to the wind profiles (for example, the
            calculated mean flow tendency multiplied by the time step).

        Returns
        -------
        MeanFlow
            MeanFlow with updated wind profiles.

        Raises
        ------
        ValueError
            Indicates that other does not have the correct shape.

        """

        if other.shape != (2, len(self.r_centers)):
            raise ValueError('other does not have correct shape')

        output = copy(self)
        output.u = self.u + other[0]
        output.v = self.v + other[1]

        return output

    def init_rho(self) -> np.ndarray:
        """
        Initialize the mean density profile depending on whether or not the
        Boussinesq approximation is made.

        Returns
        -------
        np.ndarray
            Mean density profile at cell centers.

        """

        if config.boussinesq:
            return config.rhobar0 * np.ones(self.r_centers.shape)

        return config.rhobar0 * np.exp(-self.r_centers / config.H_scale)

    def init_uv(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialize the mean wind profiles using the method identified in config.

        Returns
        -------
        np.ndarray, np.ndarray
            Mean u and v at cell centers.

        Raises
        ------
        ValueError
            Indicates that an unknown method was specified for initialization.
            Currently, the only methods are 'sine_homogeneous' and 'gaussian'.

        """

        method = config.uv_init_method
        if method == 'sine_homogeneous':
            tanh = np.tanh((self.r_centers - config.r0) / config.sig_r) + 1
            sine = np.sin(2 * np.pi * self.r_centers / config.sig_r)

            u = 0.5 * config.u0 * tanh * sine
            v = np.zeros(u.shape)

            return u, v

        if method == 'gaussian':
            arg = -(((self.r_centers - config.r0) / config.sig_r) ** 2)
            u = config.u0 * np.exp(arg)
            v = np.zeros(u.shape)

            return u, v

        message = f'Unknown method for initializing mean flow: {method}'
        raise ValueError(message)

    def init_grad_p(self) -> np.ndarray:
        """
        Initialize the horizontal pressure gradients according to the
        geostrophic approximation.

        Returns
        -------
        np.ndarray
            Array whose first and second rows correspond to the zonal and
            meridional pressure gradients, respectively, at cell centers.

        """

        return np.vstack((
            self.rho * config.f0 * self.v,
            -self.rho * config.f0 * self.u
        ))

    def project(
        self,
        rays: RayCollection,
        data: np.ndarray,
        onto: str
    ) -> np.ndarray:
        """
        Project ray volume data onto a vertical grid.

        Parameters
        ----------
        rays
            Current ray volume properties.
        data
            The data to project (for example, pseudo-momentum fluxes). Should
            have the same shape as the properties stored in rays.
        onto
            Whether to project onto cell 'faces' or cell 'centers'.

        Returns
        -------
        np.ndarray
            Projected values at each vertical grid point. Has the same length
            as either `self.r_faces` or `self.r_centers`, depending on `onto`.

        """

        if config.proj_method == 'discrete':
            grid = {'faces' : self.r_centers, 'centers' : self.r_faces}[onto]
            output = np.nansum(self.get_fracs(rays, grid) * data, axis=1)

            if onto == 'faces':
                output = np.pad(output, 1, 'edge')

            return output
        
        elif config.proj_method == 'gaussian':
            sigma = rays.dr * config.smoothing
            grid = {'faces' : self.r_faces, 'centers' : self.r_centers}[onto]
            env = np.exp(-0.5 * ((rays.r - grid[:, None]) / sigma) ** 2)
            amplitude = data * rays.dr / np.sqrt(2 * np.pi) / sigma

            return np.nansum(amplitude * env, axis=1)

    def pmf(self, rays: RayCollection, onto: str='faces') -> np.ndarray:
        """
        Calculate the zonal and meridional pseudomomentum fluxes.

        Parameters
        ----------
        rays
            RayCollection with current ray properties.
        onto
            Passed to `self.project` to indicate where to calculate the fluxes.
            Can be either 'faces' (useful during integration) or 'centers' (for
            saving offline diagnostics).

        Returns
        -------
        np.ndarray
            Array of shape (2, config.n_grid) whose rows correspond to the zonal
            and meridional pseudomomentum fluxes, respectively, at cell faces.

        """

        action_flux = rays.cg_r() * rays.action / config.epsilon
        pmf_x = self.project(rays, action_flux * rays.k, onto=onto)
        pmf_y = self.project(rays, action_flux * rays.l, onto=onto)

        if config.shapiro_filter:
            pmf_x[1:-1] = self._shapiro_filter(pmf_x)
            pmf_y[1:-1] = self._shapiro_filter(pmf_y)

        pmf = np.vstack((pmf_x, pmf_y))
        if config.tau == 0:
            return pmf

        pmf_bar = self.pmf_bar.copy()
        if onto == 'centers':
            pmf_bar = np.vstack((
                np.interp(self.r_centers, self.r_faces, pmf_bar[0]),
                np.interp(self.r_centers, self.r_faces, pmf_bar[1])
            ))

        ratio = config.dt / config.tau
        return np.exp(-ratio) * (pmf_bar + ratio * pmf)

    def dmean_dt(self, rays: RayCollection) -> np.ndarray:
        """
        Calculate the time tendency of the mean wind, including Coriolis terms
        and pseudomomentum flux divergences from the ray volumes.

        Parameters
        ----------
        rays
            RayCollection with current ray properties.

        Returns
        -------
        np.ndarray
            Array whose first and second rows are the time tendencies of the
            zonal and meridional wind, respectively, at cell centers.
            
        """

        self.pmf_bar = self.pmf(rays)
        dpmf_dr = np.diff(self.pmf_bar, axis=1) / self.dr
        du_dt = config.f0 * self.v - (self.dp_dx + dpmf_dr[0]) / self.rho
        dv_dt = -config.f0 * self.u - (self.dp_dy + dpmf_dr[1]) / self.rho

        return np.vstack((du_dt, dv_dt))
    
    @staticmethod
    def get_fracs(
        rays: RayCollection,
        grid: np.ndarray
    ) -> np.ndarray:
        """
        Find the fraction of each grid cell that intersects each ray volume.

        Parameters
        ----------
        rays
            Current ray volume properties.
        grid
            The edges of the vertical regions to project onto.

        Returns
        -------
        np.ndarray
            Fraction of each grid cell intersected by each ray. Has shape
            (len(grid - 1), config.n_ray_max).

        """

        r_lo = rays.r - 0.5 * rays.dr
        r_hi = rays.r + 0.5 * rays.dr

        r_mins = np.maximum(r_lo, grid[:-1, None])
        r_maxs = np.minimum(r_hi, grid[1:, None])

        return np.maximum(r_maxs - r_mins, 0) / (grid[1] - grid[0])
    
    @staticmethod
    def _shapiro_filter(data: np.ndarray) -> np.ndarray:
        """
        Apply a zeroth-order Shapiro filter.

        Parameters
        ----------
        data
            Array to filter.

        Returns
        -------
        np.ndarray
            Filtered array. Has two fewer elements than the data passed in.
        
        """

        return (data[:-2] + 2 * data[1:-1] + data[2:]) / 4
