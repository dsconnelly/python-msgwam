from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from . import config
from .utils import interp, get_fracs, pad_ends, shapiro_filter

if TYPE_CHECKING:
    from .rays import RayCollection

class MeanState:
    def __init__(self) -> None:
        """Initialize the mean state of the model."""

        self.z_faces = torch.linspace(*config.grid_bounds, config.n_grid)
        self.z_centers = (self.z_faces[:-1] + self.z_faces[1:]) / 2
        self.dz = self.z_faces[1] - self.z_faces[0]

        self.rho = self._init_rho()
        self.wind = self._init_wind()
        self.grad_p = self._init_grad_p()

        self.nu = interp(
            self.z_faces,
            self.z_centers,
            config.mu / self.rho
        )

    @property
    def u(self) -> torch.Tensor:
        """
        Return the zonal component of the mean wind.

        Returns
        -------
        torch.Tensor
            Tensor of zonal wind velocities at cell centers.

        """

        return self.wind[0]
    
    @property
    def v(self) -> torch.Tensor:
        """
        Return the meridional component of the mean wind.

        Returns
        -------
        torch.Tensor
            Tensor of meridional wind velocities at cell centers.

        """

        return self.wind[1]
    
    def project(
        self,
        rays: RayCollection,
        data: torch.Tensor,
        onto: str
    ) -> torch.Tensor:
        """
        Project data corresponding to each ray volume onto the vertical grid of
        the mean state.

        Parameters
        ----------
        rays
            Collection of current ray volume properties.
        data
            Data to project (for example, pseudomomentum fluxes). Should have
            the same shape as the properties stored in `rays`.
        onto
            Whether to project onto cell `'faces'` or cell `'centers'`.

        Returns
        -------
        torch.Tensor
            Projected values at each vertical grid point. Has the same length as
            either `self.z_faces` or `self.z_centers`, depending on `onto`.

        Raises
        ------
        ValueError
            Indicates that an unsupported projection method was specified.

        """

        if config.proj_method == 'discrete':
            edges = {'faces' : self.z_centers, 'centers' : self.z_faces}[onto]
            output = torch.nansum(get_fracs(rays, edges) * data, dim=1)

            if onto == 'faces':
                output = pad_ends(output)

            return output
        
        message = f'Unknown projection method: {config.proj_method}'
        raise ValueError(message)
    
    def pmf(self, rays: RayCollection, onto: str='faces') -> torch.Tensor:
        """
        Calculate the zonal and meridional pseudomomentum fluxes.

        Parameters
        ----------
        rays
            Collection of current ray volume properties.
        onto, optional
            Passed to `self.project` to indicate where to calculate the fluxes.
            Can be either `'faces'` (default, useful during integration) or
            `'centers'` (for saving offline diagnostics).

        Returns
        -------
        torch.Tensor
            Tensor of shape (2, config.n_grid) or (2, config.n_grid - 1),
            depending on the value of `onto`, whose rows contain the zonal and
            meridional pseudomomentum fluxes, respectively.

        """

        action_flux = rays.cg_r() * rays.action / config.epsilon
        pmf_x = self.project(rays, action_flux * rays.k, onto=onto)
        pmf_y = self.project(rays, action_flux * rays.l, onto=onto)

        if config.shapiro_filter:
            pmf_x[1:-1] = shapiro_filter(pmf_x)
            pmf_y[1:-1] = shapiro_filter(pmf_y)

        return torch.vstack((pmf_x, pmf_y))

    def dmean_dt(self, rays: RayCollection) -> torch.Tensor:
        """
        Calculate the time tendency of the mean wind, including Coriolis terms
        as well as the pseudomomentm flux divergences. Note that the diffusion
        of the mean flow is not included here, as it is handled implicitly by
        the integrator.

        Parameters
        ----------
        rays
            Collection of current ray volume properties.

        Returns
        -------
        torch.Tensor
            Tensor whose first and second rows contain the time tendencies of
            the zonal and meridional wind, respectively, at cell centers.

        """

        dpmf_dz = torch.diff(self.pmf(rays), dim=1) / self.dz
        coriolis = config.f0 * torch.vstack((self.v, -self.u))

        return coriolis - (self.grad_p + dpmf_dz) / self.rho

    def _init_grad_p(self) -> torch.Tensor:
        """
        Initialize the geostrophic horizontal pressure gradients.

        Returns
        -------
        torch.Tensor
            Tensor whose first and second rows contain the zonal and meridional
            pressure gradients, respectively, at cell centers.

        """

        return torch.vstack((
            self.rho * config.f0 * self.v,
            -self.rho * config.f0 * self.u
        ))

    def _init_rho(self) -> torch.Tensor:
        """
        Initialize the mean density profile, depending on whether the Boussinesq
        approximation should be made.

        Returns
        -------
        torch.Tensor
            Mean density profile at cell centers.

        """

        if config.boussinesq:
            return config.rho0 * torch.ones_like(self.z_centers)
        
        return config.rho0 * torch.exp(-self.z_centers / config.H_rho)
    
    def _init_wind(self) -> torch.Tensor:
        """
        Initialize the mean wind profiles with the method specified in config.

        Returns
        -------
        torch.Tensor
            Tensor whose first and second rows contain the zonal and meridional
            velocities, respectively, at cell centers.

        Raises
        ------
        ValueError
            Indicates that an unknown method was specified for initialization.
            Currently, the only supported method is 'gaussian'.

        """

        if not config.interactive_mean:
            return torch.zeros((2, config.n_grid - 1))

        method = config.wind_init_method
        if method == 'gaussian':
            scale = (self.z_centers - config.z0) / config.sigma_uv
            envelope = torch.exp(-0.5 * scale ** 2)

            return torch.vstack((
                config.u0 * envelope,
                config.v0 * envelope
            ))
        
        message = f'Unknown method for initializing mean wind: {method}'
        raise ValueError(message)
