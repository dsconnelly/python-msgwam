from __future__ import annotations
from typing import TYPE_CHECKING, Any

import numpy as np

from . import config, sources
from .utils import _cg_r, _omega_hat

if TYPE_CHECKING:
    from .mean import MeanFlow

class RayCollection:
    props = ['r', 'dr', 'k', 'l', 'm', 'dk', 'dl', 'dm', 'dens', 'age', 'meta']
    indices = {prop: i for i, prop in enumerate(props)}

    r: np.ndarray; dr: np.ndarray
    k: np.ndarray; l: np.ndarray; m: np.ndarray
    dk: np.ndarray; dl: np.ndarray; dm: np.ndarray
    dens: np.ndarray; age: np.ndarray; meta: np.ndarray

    def __init__(self, mean: MeanFlow) -> None:
        """
        Initialize a RayCollection based on the loaded config settings and an
        array of source information.

        Parameters
        ----------
        mean
            Initialized MeanFlow to be passed to the source function.

        """

        shape = (len(self.props), config.n_ray_max)
        self.data = np.nan * np.zeros(shape)
        self.next_meta = -1

        source_func = getattr(sources, config.source_type)
        self.sources: np.ndarray = source_func(mean)
        
        self.ghosts = {}
        for slot, data in enumerate(self.sources.T):
            self.ghosts[slot] = self.add_ray(data)

        if config.epsilon < 1:
            self.cg_source = self.cg_r()[:self.sources.shape[1]]
            times = np.ceil(config.dr_init / (self.cg_source * config.dt))
            self.launch_rate = config.epsilon * (1 / times).sum()

    def __getattr__(self, prop: str) -> Any:
        """
        Return the row of self.data corresponding to the named ray property.
        Note that since __getattr__ is only called if an error is thrown during
        __getattribute__ (i.e. if a ray property is requested) we only have to
        handle those cases and can raise an error otherwise. This function is
        added so that ray properties can be accessed as `rays.dr`, etc.

        Parameters
        ----------
        prop
            Name of the ray property to return. Should be in self.props.

        Returns
        -------
        np.ndarray
            Corresponding row of self.data.

        Raises
        ------
        AttributeError
            Indicates that no ray property of the given name exists.

        """

        if prop in self.indices:
            return self.data[self.indices[prop]]

        message = f'{type(self).__name__} object has no attribute {prop}'
        raise AttributeError(message)

    @property
    def valid(self) -> np.ndarray:
        """
        Determine which columns of self.data correspond to active ray volumes.

        Returns
        -------
        np.ndarray
            Boolean array indicating whether each column of the array data is
            tracking an active ray volume or is free to be written over.

        """

        return np.isnan(self.data).sum(axis=0) == 0

    @property
    def count(self) -> int:
        """
        Count the number of active ray volumes in the collection.

        Returns
        -------
        int
            Number of active ray volumes.

        """

        return self.valid.sum()
    
    @property
    def action(self) -> np.ndarray:
        """
        Calculate the wave action density.

        Returns
        -------
        np.ndarray
            Array of wave action densities, calculated as the spectral wave
            action multiplied by the spectral volume of each ray.

        """

        return self.dens * abs(self.dk * self.dl * self.dm)
    
    def add_ray(self, data: np.ndarray) -> int:
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
        
        j = int(np.argmin(self.valid))
        self.next_meta = self.next_meta + 1
        self.data[:, j] = [*data, 0, self.next_meta]

        return j
    
    def delete_rays(self, j: int | np.ndarray) -> None:
        """
        Delete a ray volume by filling the corresponding column with np.nan.

        Parameters
        ----------
        j
            Index or array of indices of ray volumes to delete.

        """

        self.data[:, j] = np.nan

    def omega_hat(self) -> np.ndarray:
        """
        Calculate the intrinsic frequencies of each ray in the collection. This
        function is just a shorthand for calling _omega_hat with the appropriate
        wave properties stored in this object.

        Returns
        -------
        np.ndarray
            Array of intrinsic frequencies. Will be np.nan at indices where no
            ray volume is propagating.

        """

        return _omega_hat(self.k, self.l, self.m)
    
    def cg_r(self) -> np.ndarray:
        """
        Calculate the vertical group velocities for each ray in the collection.
        This function is just a shorthand for calling _omega_hat with the
        appropriate wave properties stored in this object.

        Returns
        -------
        np.ndarray
            Array of vertical group velocities. Will be np.nan at indices where
            no ray volume is propagating.

        """

        return _cg_r(self.k, self.l, self.m)
    
    def check_boundaries(self, mean: MeanFlow) -> None:
        """
        Delete rays that have strayed outside of the physical domain.
        
        Parameters
        ----------
        mean
            MeanFlow providing the vertical extent of the system.

        """

        below = self.r - 0.5 * self.dr < mean.r_faces[0]
        above = self.r + 0.5 * self.dr > mean.r_faces[-1]
        self.delete_rays(below | above)

    def check_source(self) -> None:
        """
        Enforce the bottom boundary condition by adding ray volumes as necessary
        to replace those that have cleared the ghost layer.
        """

        crossed: list | np.ndarray = []
        for slot in range(self.sources.shape[1]):
            i = self.ghosts[slot]
            if self.r[i] - 0.5 * self.dr[i] > config.r_ghost:
                crossed.append(slot)

        if not crossed:
            return

        if config.epsilon < 1:
            weights = self.cg_source[crossed]
            weights = weights / weights.sum()

            size = int(np.floor(self.launch_rate))
            if np.random.rand() < self.launch_rate - size:
                size = size + 1

            size = min(len(crossed), size)
            crossed = np.random.choice(crossed, size, False, weights)

        excess = self.count + len(crossed) - config.n_ray_max
        if config.purge and excess > 0:
            idx = np.argsort(self.action)
            exclude = list(self.ghosts.values())
            idx = idx[~np.isin(idx, exclude)]

            self.delete_rays(idx[:excess])

        for slot in crossed:
            data = self.sources[:, slot].copy()
            self.ghosts[slot] = self.add_ray(data)
    
    def drays_dt(self, mean: MeanFlow) -> np.ndarray:
        """
        Calculate the time tendency of each ray property.

        Parameters
        ----------
        mean
            Current mean state of the system.

        Returns
        -------
        np.ndarray
            Array of time tendencies, each row of which corresponds to the ray
            property named in self.props.

        """

        dr_dt = self.cg_r()
        du_dr = np.interp(self.r, mean.r_faces[1:-1], np.diff(mean.u) / mean.dr)
        dv_dr = np.interp(self.r, mean.r_faces[1:-1], np.diff(mean.v) / mean.dr)
        dm_dt = -(self.k * du_dr + self.l * dv_dr)
        
        ddr_dt, ddm_dt = np.zeros((2, config.n_ray_max))
        dk_dt, dl_dt, ddk_dt, ddl_dt = np.zeros((4, config.n_ray_max))

        ddens_dt = np.zeros(config.n_ray_max)
        dage_dt = np.ones(config.n_ray_max)
        dmeta_dt = np.zeros(config.n_ray_max)

        idx = self.r < config.r_launch
        dm_dt[idx] = ddr_dt[idx] = ddm_dt[idx] = 0

        return np.vstack((
            dr_dt, ddr_dt,
            dk_dt, dl_dt, dm_dt,
            ddk_dt, ddl_dt, ddm_dt,
            ddens_dt, dage_dt, dmeta_dt
        ))
    
    def dissipate_and_break(self, mean: MeanFlow) -> None:
        """
        First, dissipate waves according to viscosity. (For now, dissipation is
        handled here rather than in the time tendencies, as dissipation means
        the ray tracing equations have a term that should be handled implicitly,
        but only for wave action density.) Then, determine where convective
        instability-induced wave breaking should occur, and adjust the spectral
        wave action densities accordingly.

        Parameters
        ----------
        MeanFlow
            Current mean state of the system.

        """

        omega_hat = self.omega_hat()
        wvn_sq = self.k ** 2 + self.l ** 2 + self.m ** 2
        
        nu = config.dissipation * np.interp(self.r, mean.r_faces, mean.nu)
        damping = nu * wvn_sq * (1 + config.f0 ** 2 / omega_hat ** 2)
        self.dens[:] = self.dens * np.exp(-config.dt * damping)

        S = self.m ** 2 * omega_hat * self.action
        P = mean.project(self, S, 'centers') - mean.rho * config.N0 ** 2 / 2
        Q = mean.project(self, S * wvn_sq, 'centers')

        idx = Q != 0
        kappa = np.zeros(mean.rho.shape)
        kappa[idx] = np.maximum(P[idx], 0) / Q[idx]

        intersects = mean.get_fracs(self, mean.r_faces)
        intersects[intersects > 0] = 1

        factor = 1 - wvn_sq * np.max(intersects * kappa[:, None], axis=0)
        self.dens[:] = self.dens * factor
        