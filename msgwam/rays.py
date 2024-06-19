from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast

import torch

from . import config, sources
from .dispersion import cg_r, omega_hat
from .utils import interp, get_fracs, put

if TYPE_CHECKING:
    from .mean import MeanState

class RayCollection:
    props = ['r', 'dr', 'k', 'l', 'm', 'dk', 'dl', 'dm', 'dens', 'age', 'meta']
    indices = {prop : i for i, prop in enumerate(props)}

    r: torch.Tensor; dr: torch.Tensor
    k: torch.Tensor; l: torch.Tensor; m: torch.Tensor
    dk: torch.Tensor; dl: torch.Tensor; dm: torch.Tensor
    dens: torch.Tensor; age: torch.Tensor; meta: torch.Tensor

    def __init__(self) -> None:
        """Initialize the collection of ray volumes."""

        shape = (len(self.props), config.n_ray_max)
        self.data = torch.nan * torch.zeros(shape)
        self.next_meta = -1

        cls_name = config.source_type.capitalize() + 'Source'
        self.source: sources.Source = getattr(sources, cls_name)()

        self.ghosts = torch.zeros(self.source.n_slots).int()
        for slot, data in enumerate(self.source.data.T):
            self.ghosts[slot] = self.add_ray(data)

        if config.purge_mode == 'network':
            self.model = torch.jit.load(config.purge_network_path)

    def __getattr__(self, prop: str) -> Any:
        """
        Return the row of self.data corresponding to the named ray property.
        This function is added so that ray properties can be accessed as rays.k,
        rays.dr, and so on. Because __getattr__ is only called if an error is
        thrown during __getattribute__ (i.e. if a ray property is requested), we
        only have to handles those cases and can raise an error otherwise.
        
        Parameters
        ----------
        prop
            Name of the ray property to return. Should be in self.props.

        Returns
        -------
        torch.Tensor
            Corresponding row of self.data.

        Raises
        ------
        AttributeError
            Indicates that no ray property with the given name exists.

        """

        if prop in self.indices:
            return self.data[self.indices[prop]]
        
        message = f'{type(self).__name__} object has no attribute {prop}'
        raise AttributeError(message)
    
    @property
    def valid(self) -> torch.Tensor:
        """
        Determine which columns of self.data correspond to active ray volumes.

        Returns
        -------
        torch.Tensor
            Boolean tensor indicating whether each column of the data tensor is
            tracking an active ray volume or is free to be written over.

        """

        return ~torch.isnan(self.meta)
    
    @property
    def count(self) -> int:
        """
        Count the number of active ray volumes in the collection. The cast is
        for the benefit of the type checker; `self.valid` is a Boolean tensor
        and so its sum is guaranteed to be an integer.

        Returns
        -------
        int
            Number of active ray volumes.

        """

        return cast(int, self.valid.sum().item())
    
    @property
    def action(self) -> torch.Tensor:
        """
        Calculate the wave action density of each ray volume.

        Returns
        -------
        torch.Tensor
            Tensor of wave action densities, calculated as the spectral wave
            action density multiplied by the spectral volume of each ray.

        """

        return self.dens * abs(self.dk * self.dl * self.dm)
    
    def add_ray(self, data: torch.Tensor) -> int:
        """
        Add a ray to the collection, storing its data in the first available
        column. Raises an error if the collection is already at the maximum
        allowable number of active ray volumes.

        Parameters
        ----------
        data
            Vector of ray properties (r, dr, k, l, m, dk, dl, dm, dens).

        Returns
        -------
        int
            Index of the column where the new ray volume was added.

        Raises
        ------
        RuntimeError
            Indicates that the RayCollection already has config.n_ray_max rays.

        """

        if self.count == config.n_ray_max:
            raise RuntimeError('RayCollection has too many rays')

        self.next_meta = self.next_meta + 1
        j = int(torch.argmin(self.valid.int()))

        self.data[:-2, j] = data
        self.data[-2:, j] = torch.tensor([0, self.next_meta])
        
        return j
    
    def delete_rays(self, j: int | torch.Tensor) -> None:
        """
        Delete one or more ray volumes by filling the corresponding columns of
        self.data with torch.nan.

        Parameters
        ----------
        j
            Index or tensor of indices of ray volumes to delete.

        """

        self.data[8, j] = 0
        self.data[9:, j] = torch.nan

    def purge(self, excess: int, mean: MeanState) -> None:
        """
        Delete enough rays to enforce the bottom boundary condition. The rays to
        be purged will be selected according to `config.purge_mode`.

        Parameters
        ----------
        excess
            How many rays need to be deleted.
        u
            Current zonal wind profile.

        """

        if config.purge_mode == 'none':
            return

        elif config.purge_mode == 'action':
            criterion = self.action

        elif config.purge_mode == 'cg_r':
            criterion = abs(self.cg_r())

        elif config.purge_mode == 'energy':
            criterion = self.action * self.omega_hat()

        elif config.purge_mode == 'pmf':
            criterion = abs(self.k * self.cg_r() * self.action)

        elif config.purge_mode == 'random':
            criterion = torch.rand(config.n_ray_max)

        else:
            message = f'Unknown purge mode: {config.purge_mode}'
            raise ValueError(message)

        idx = torch.argsort(criterion)
        idx = idx[(~torch.isin(idx, self.ghosts)) & self.valid[idx]]
        self.delete_rays(idx[:excess])

    def check_boundaries(self, mean: MeanState) -> None:
        """
        Delete rays that have propagated outside of the physical domain.

        Parameters
        ----------
        mean
            MeanState used to determine the vertical extent of the system.

        """

        below = self.r - 0.5 * self.dr < mean.z_faces[0]
        above = self.r + 0.5 * self.dr > mean.z_faces[-1]
        self.delete_rays(below | above)

        pmf = self.k * self.cg_r() * self.action
        self.delete_rays(abs(pmf) < config.min_pmf)

    def check_source(self, mean: MeanState) -> None:
        """
        Enforce the bottom boundary condition by adding ray volumes as necessary
        to replace those that have cleared the ghost layer.

        """

        r_lo = self.r[self.ghosts] - 0.5 * self.dr[self.ghosts]
        crossed = torch.where(r_lo > config.r_ghost)[0]

        if len(crossed) == 0:
            return
        
        datas, replaced = self.source.launch(crossed, mean.u)
        excess = self.count + datas.shape[1] - config.n_ray_max

        if excess > 0:
            self.purge(excess, mean)

        for j, data in enumerate(datas.T):
            self.ghosts[replaced[j]] = self.add_ray(data)

    def cg_r(self) -> torch.Tensor:
        """
        Calculate the vertical group velocity of each ray in the collection.
        This function is just a wrapper around cg_r called with the appropriate
        wave properties stored in this object.

        Returns
        -------
        torch.Tensor
            Tensor of vertical group velocities, with torch.nan at indices where
            no ray volume is propagating.

        """

        return cg_r(self.k, self.l, self.m)
    
    def omega_hat(self) -> torch.Tensor:
        """
        Calculate the intrinsic frequency of each ray in the collection. This
        function is just a wrapper around cg_r called with the appropriate wave
        properties stored in this object.

        Returns
        -------
        torch.Tensor
            Tensor of intrinsic frequencies, with torch.nan at indices where no
            ray volume is propagating.

        """

        return omega_hat(self.k, self.l, self.m)
    
    def drays_dt(self, mean: MeanState) -> torch.Tensor:
        """
        Calculate the time tendency of each ray property. Note that no tendency
        is returned for dens, age, or meta, as we have exact update equations
        for these properties and so the integrator handles them separately.

        Parameters
        ----------
        mean
            Current mean state of the system.

        Returns
        -------
        torch.Tensor
            Tensor of time tendencies, each row of which corresponds to a ray
            property named in self.props.

        """

        dr_dt = self.cg_r()
        du_dr = interp(self.r, mean.z_faces[1:-1], torch.diff(mean.u) / mean.dz)
        dv_dr = interp(self.r, mean.z_faces[1:-1], torch.diff(mean.v) / mean.dz)
        dm_dt = -(self.k * du_dr + self.l * dv_dr)

        ddr_dt, ddm_dt = torch.zeros((2, config.n_ray_max))
        dk_dt, dl_dt, ddk_dt, ddl_dt = torch.zeros((4, config.n_ray_max))

        idx = self.r < config.r_launch
        dm_dt[idx] = ddr_dt[idx] = ddm_dt[idx] = 0

        return torch.vstack((
            dr_dt, ddr_dt,
            dk_dt, dl_dt, dm_dt,
            ddk_dt, ddl_dt, ddm_dt
        ))
    
    def dissipate_and_break(self, mean: MeanState) -> None:
        """
        First, dissipate waves according to viscosity. Then, determine where
        convective instability-induced wave breaking should occur, and adjust
        the spectral wave action densities accordingly. This function is meant
        to be called separately be the integrator to advance the spectral wave
        action densities, as they are the sole ray volume property that has a
        term that needs to be handled implicitly.

        Parameters
        ----------
        mean
            Current mean state of the system.

        """

        omega_hat = self.omega_hat()
        wvn_sq = self.k ** 2 + self.l ** 2 + self.m ** 2

        nu = config.dissipation * interp(self.r, mean.z_faces, mean.nu)
        damping = nu * wvn_sq * (1 + config.f0 ** 2 / omega_hat ** 2)
        damped = self.dens * torch.exp(-config.dt * damping)
        self.data = put(self.data, 8, damped)

        if config.n_chromatic == 0:
            return

        S = self.m ** 2 * omega_hat * self.action
        threshold = mean.rho * config.N0 ** 2 / 2
        intersects = get_fracs(self, mean.z_faces)
        intersects[intersects > 0] = 1

        if config.n_chromatic != -1:
            S = S.reshape(-1, config.n_chromatic)
            wvn_sq = wvn_sq.reshape(-1, config.n_chromatic)
            intersects = intersects.reshape(intersects.shape[0], *S.shape)
            threshold = threshold[:, None]

        P = mean.project(self, S, 'centers') - threshold
        Q = mean.project(self, S * wvn_sq, 'centers')

        idx = Q != 0
        kappa = torch.zeros(P.shape)
        kappa[idx] = torch.clamp(P[idx], min=0) / Q[idx]

        factor = 1 - wvn_sq * (intersects * kappa[..., None]).max(dim=0)[0]
        self.data = put(self.data, 8, self.dens * factor.reshape(-1))
