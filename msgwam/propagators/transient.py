from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, Self, cast

import numpy as np

from msgwam.means import MeanState

from .. import config
from ..constants import PROP_NAMES
from ..dispersion import get_cg_r, get_cp_x, get_omega_hat
from ..utils import shapiro_filter

from .base import Propagator

if TYPE_CHECKING:
    from ..means import MeanState

class TooManyRaysError(Exception):
    pass

class TransientPropagator(Propagator):
    """
    Class implementing the ray tracing scheme MS-GWaM, whose study is the main
    focus of this package. See Bölöni et al. (2021) for details.
    """

    r: np.ndarray; dr: np.ndarray
    k: np.ndarray; l: np.ndarray; m: np.ndarray
    dk: np.ndarray; dl: np.ndarray; dm: np.ndarray
    dens: np.ndarray; age: np.ndarray; meta: np.ndarray

    def __init__(self, mean: MeanState) -> None:
        """
        Initialize the ray tracer by creating an array to hold as many ray
        volumes are allowed. Then launch a ray volume for each spectral element
        of the source.
        """

        super().__init__(mean)
        self._indices = {name : i for i, name in enumerate(PROP_NAMES)}

        shape = (len(PROP_NAMES), config.n_max)
        self._data = np.nan * np.zeros(shape)
        self._next_meta = 0

        self._r_init = config.z_min - 0.5 * config.dr_init
        self._ghosts = np.zeros(config.n_source).astype(int)
        datas, cdx = self._source.launch(mean, 0)

        for k, data in zip(cdx, datas.T):
            self._ghosts[k] = self._add_ray(data, mean)

    def __getattr__(self, name: str) -> Any:
        """
        Return the row of `self._data` corresponding to the named ray property.
        This function allows ray properties to be accessed directly as fields of
        this object. Because `__getattr__` is called only if an error is thrown
        during `__getattribute__` (i.e. if a ray property is requested), we have
        to handle only those cases, and can raise an error otherwise.

        Parameters
        ----------
        name
            Name of the ray property to return. Should be in `PROP_NAMES`.

        Returns
        -------
        np.ndarray
            Corresponding row of `self.data`.

        Raises
        ------
        AttributeError
            Indicates that no ray property with the given name exists.

        """

        try:
            return self._data[self._indices[name]]
        
        except KeyError:
            message = f'{type(self).__name__} object has no attribute {name}'
            raise AttributeError(message)
        
    @property
    def action(self) -> np.ndarray:
        """
        Calculate the wave action density of each ray volume.

        Returns
        -------
        np.ndarray
            Array of wave action densities, calculated as the spectral wave
            action density multiplied by the spectral volume of each volume.

        """

        return self.dens * self.dk * self.dl * self.dm

    def get_fluxes(self, mean: MeanState, net: bool = True) -> np.ndarray:
        """
        For the transient propagator, getting the fluxes involves projecting the
        flux associated with each ray volume onto the vertical grid.
        """

        if net:
            wvns = [self.k, self.l]

        else:
            wvns = [
                np.maximum(self.k, 0), np.minimum(self.k, 0),
                np.maximum(self.l, 0), np.minimum(self.l, 0)
            ]

        action_flux = self.action * self._get_cg_r(mean)
        func = lambda wvn: self._project(wvn * action_flux, mean.z_padded)
        fluxes = np.vstack([func(wvn) for wvn in wvns])

        if config.shapiro_filter:
            fluxes[:, 1:-1] = shapiro_filter(fluxes.T).T

        return fluxes

    @property
    def n_active(self) -> int:
        """
        Counts the number of ray volumes currently propagating.

        Returns
        -------
        int
            Number of active ray volumes.

        """

        return self._valid.sum()
    
    def step(self, mean: MeanState, n_step: int) -> Self:
        """
        First, the transient propagator advances the ODEs with an RK3 step. Next
        wave dissipation and breaking is applied, after which waves that have
        exited the domain or are otherwise invalid are removed. Finally, the
        bottom boundary condition is enforced and new rays are instantiated.
        """
        
        self._take_RK3_step(mean)
        self._apply_sinks(mean)
        self._check_boundaries(mean)
        self._check_source(mean, n_step)

        return self

    def _add_ray(self, data: np.ndarray, mean: MeanState) -> int:
        """
        Add a ray volume to the propagator, storing its data in the first open
        column. Raises an error if the propagator is already at the maximum
        number of active ray volumes and `config.n_increment == 0`.

        Parameters
        ----------
        data
            Vector of wave properties (k, l, m, dk, dl, dm, dens).
        mean
            Current mean state of the system.

        Returns
        -------
        int
            Index of the column where the new ray volume was added. The cast is
            for the benefit of the type checker.

        Raises
        ------
        TooManyRaysError
            Indicates that the propagator already has `config.n_max` rays.

        """

        if self.n_active >= self._n_max:
            if config.n_increment > 0:
                shape = (len(PROP_NAMES), config.n_increment)
                self._data = np.hstack((self._data, np.nan * np.zeros(shape)))

            else:
                raise TooManyRaysError

        k, l, m, *_ = data
        u = np.interp(self._r_init, mean.z_centers, mean.u)
        sign = np.sign(get_cp_x(k, l, m, mean.N[0]) - u)
        
        self._next_meta = self._next_meta + 1
        j = np.argmin(self._valid)

        self._data[2:-2, j] = data
        self._data[:2, j] = [self._r_init, config.dr_init]
        self._data[-2:, j] = [0, sign * self._next_meta]

        return cast(int, j)

    def _apply_sinks(self, mean: MeanState) -> None:
        """
        First dissipate waves according to viscosity. Next, remove any rays that
        have artificially passed through critical layers. Finally, determine if
        convective instability-induced wave breaking should occur, and adjust
        the spectral wave action densities accordingly.

        Parameters
        ----------
        mean
            Current mean state of the system.

        """

        omega_hat = self._get_omega_hat(mean)
        wvn_sq = self.k ** 2 + self.l ** 2 + self.m ** 2

        nu = config.dissipation * np.interp(self.r, mean.z_faces, mean.nu)
        damping = nu * wvn_sq * (1 + config.f ** 2 / (omega_hat ** 2))
        self._data[8] = self.dens * np.exp(-config.dt * damping)

        if config.check_sign_changes:
            cp_x = self._get_cp_x(mean)
            u = np.interp(self.r, mean.z_centers, mean.u)
            self._delete_rays(np.sign(cp_x - u) != np.sign(self.meta))

        if config.n_chromatic == 0:
            return
        
        threshold = mean.rho * mean.N ** 2 / 2
        S = self.m ** 2 * omega_hat * self.action
        intersects = self._get_fracs(mean.z_faces)
        intersects[intersects > 0] = 1

        if config.n_chromatic != -1:
            S = S.reshape(-1, config.n_chromatic)
            wvn_sq = wvn_sq.reshape(-1, config.n_chromatic)
            intersects = intersects.reshape(intersects.shape[0], *S.shape)
            threshold = threshold[:, None]

        P = self._project(S, mean.z_faces) - threshold
        Q = self._project(S * wvn_sq, mean.z_faces)

        idx = Q != 0
        kappa = np.zeros(P.shape)
        kappa[idx] = np.maximum(P[idx], 0) / Q[idx]

        factor = 1 - wvn_sq * (intersects * kappa[:, None]).max(axis=0)
        self._data[8] = self.dens * factor.flatten()

    def _check_boundaries(self, mean: MeanState) -> None:
        """
        Delete rays that have propagated outside of the physical domain and rays
        that no longer have more than `config.min_flux` momentum flux.

        Parameters
        ----------
        mean
            Current mean state of the system.

        """

        below = self.r < self._r_init
        above = self.r - 0.5 * self.dr > config.z_max
        self._delete_rays(below | above)

        flux = self.k * self.action * self._get_cg_r(mean)
        self._delete_rays(abs(flux) < config.min_flux)

    def _check_source(self, mean: MeanState, n_step: int) -> None:
        """
        Enforce the bottom boundary condition by adding ray volumes as necessary
        to replace those that have cleared the ghost layer.

        Parameters
        ----------
        mean
            Current mean state of the system.
        n_step
            Index of the current time step.

        """

        crossed = self.r[self._ghosts] > config.z_min
        cdx, *_ = np.where(crossed)

        if crossed.sum() == 0:
            return
        
        datas, cdx = self._source.launch(mean, n_step, cdx)
        excess = self.n_active + datas.shape[1] - self._n_max

        jdx = self._ghosts[cdx]
        r_hi = self.r[jdx] + 0.5 * self.dr[jdx]
        self._data[0, jdx] = (config.z_min + r_hi) / 2
        self._data[1, jdx] = r_hi - config.z_min

        self._prune(excess, mean)
        for k, data in zip(cdx, datas.T):
            self._ghosts[k] = self._add_ray(data, mean)

    def _delete_rays(self, j: int | np.ndarray) -> None:
        """
        Delete one or more ray volumes by filling the corresponding columns of
        `self._data` with `np.nan`.

        Parameters
        ----------
        j
            Index or array of indices of ray volumes to delete.

        """

        self._data[:, j] = np.nan

    def _get_cg_r(
        self,
        mean: MeanState,
        r: Optional[np.ndarray]=None
    ) -> np.ndarray:
        """
        Return the vertical group velocity of each propagating ray volume. This
        function is a wrapper around `get_cg_r` called with the appropriate wave
        properties and using the buoyancy frequency at each ray's position.

        Parameters
        ----------
        mean
            Current mean state of the system.
        r
            Where in the vertical coordinate to compute group velocities. If not
            provided, the ray volume centers will be used, but other values can
            be passed, e.g. to compute velocities at ray tops and bottoms.

        Returns
        -------
        np.ndarray
            Array of vertical group velocities.

        """

        if r is None:
            r = self.r

        N = np.interp(r, mean.z_centers, mean.N)
        return get_cg_r(self.k, self.l, self.m, N)
    
    def _get_cp_x(self, mean: MeanState) -> np.ndarray:
        """
        Return the zonal phase velocity of each propagating ray volume. This
        function is a wrapper around `get_cg_r` called with the appropriate wave
        properties and using the buoyancy frequency at each ray's position.

        Parameters
        ----------
        mean
            Current mean state of the system.

        Returns
        -------
        np.ndarray
            Array of zonal phase velocities.

        """

        N = np.interp(self.r, mean.z_centers, mean.N)
        return get_cp_x(self.k, self.l, self.m, N)

    def _get_fracs(self, edges: np.ndarray) -> np.ndarray:
        """
        Given the edges of arbitrary regions in the vertical grid, find the
        fraction of each region that is intersected by each ray volume.

        Parameters
        ----------
        edges
            Edges of regions of the vertical grid. Likely either the cell faces
            (for projection onto cell centers) or the padded set of cell centers
            (for projection onto cell faces).

        Returns
        -------
        np.ndarray
            Array of shape (len(faces) - 1, self._n_max), where the value at
            index `[i, j]` corresponds to the fraction of region `i` that is
            intersected by ray volume `j`.

        """

        r_lo = self.r - 0.5 * self.dr
        r_hi = self.r + 0.5 * self.dr
        
        r_mins = np.maximum(r_lo, edges[:-1, None])
        r_maxs = np.minimum(r_hi, edges[1:, None])
        dz = edges[1:, None] - edges[:-1, None]

        return np.maximum(r_maxs - r_mins, 0) / dz

    def _get_drays_dt(self, mean: MeanState) -> np.ndarray:
        """
        Calculate the time tendecy of each ray property. Note that no tendencies
        are returned for spectral density, age, or meta, as we have exact update
        equations for those properties and so they are handled separately.

        Parameters
        ----------
        mean
            Current mean state of the system.

        Returns
        -------
        np.ndarray
            Array of time tendencies for each of the first eight properties.

        """

        cg_lo = self._get_cg_r(mean, self.r - 0.5 * self.dr)
        cg_hi = self._get_cg_r(mean, self.r + 0.5 * self.dr)
        dr_dt = 0.5 * (cg_lo + cg_hi)
        ddr_dt = cg_hi - cg_lo

        N = np.interp(self.r, mean.z_centers, mean.N)
        du_dr = np.interp(self.r, mean.z_faces[1:-1], np.diff(mean.u) / mean.dz)
        dv_dr = np.interp(self.r, mean.z_faces[1:-1], np.diff(mean.v) / mean.dz)
        dN_dr = np.interp(self.r, mean.z_faces[1:-1], np.diff(mean.N) / mean.dz)

        omega_hat = self._get_omega_hat(mean)
        wvn_hor_sq = self.k ** 2 + self.l ** 2
        coeff = N * wvn_hor_sq / omega_hat / (wvn_hor_sq + self.m ** 2)

        dk_dt, dl_dt, ddk_dt, ddl_dt = np.zeros((4, self._n_max))
        dm_dt = -(self.k * du_dr + self.l * dv_dr + coeff * dN_dr)
        ddm_dt = -self.dm * ddr_dt / self.dr

        idx = self.r < config.z_min
        dm_dt[idx] = ddr_dt[idx] = ddm_dt[idx] = 0

        return np.vstack((
            dr_dt, ddr_dt,
            dk_dt, dl_dt, dm_dt,
            ddk_dt, ddl_dt, ddm_dt
        ))

    def _get_omega_hat(self, mean: MeanState) -> np.ndarray:
        """
        Return the intrinsic frequency of each propagating ray volume. This
        function is a wrapper around `get_omega_hat` called with the appropriate
        wave properties and using the buoyancy frequency at each ray's position.

        Parameters
        ----------
        mean
            Current mean state of the system.

        Returns
        -------
        np.ndarray
            Array of intrinsic frequencies.

        """

        N = np.interp(self.r, mean.z_centers, mean.N)
        return get_omega_hat(self.k, self.l, self.m, N)

    @property
    def _n_max(self) -> int:
        """
        Return the current number of ray volumes allowed by the propagator.

        Returns
        -------
        int
            Current maximum allowable number of rays. In most cases, this will
            be identical to `config.n_max`, but if `config.n_increment > 0` then
            the size of the underlying data may have been increased.

        """

        return self._data.shape[1]
    
    def _project(self, data: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Project data corresponding to each ray volume onto the vertical grid.

        Parameters
        ----------
        data
            Data to project (e.g. momentum fluxes or buoyancy perturbations).
            Should have `self._n_max` elements. If `data` has more than one
            dimension, then the projection is done in batches and returns a
            profile for each batch.
        edges
            Edges of the regions to project onto, to be passed to `_get_fracs`.

        Returns
        -------
        np.ndarray
            Projected values at each vertical grid point. If `data` had more
            than one dimension, the first dimension of this array will range
            over vertical profiles corresponding to each batch.

        """

        fracs = self._get_fracs(edges).reshape(-1, *data.shape)
        return np.nansum(fracs * data, axis=-1)

    def _prune(self, excess: int, mean: MeanState) -> None:
        """
        Delete enough rays to add `excess` more, presumably at the bottom
        boundary of the domain. The rays to be pruned will be selected according
        to `config.prune_by`. If `excess` is non-positive, no rays are pruned.

        Parameters
        ----------
        excess
            How many ray volumes must be deleted.
        mean
            Current mean state of the system.

        """

        if excess <= 0 or config.prune_by == 'none':
            return
        
        if config.prune_by == 'energy':
            criterion = self.action * self._get_omega_hat(mean)

        elif config.prune_by == 'random':
            criterion = np.random.rand(self._n_max)

        idx = np.argsort(criterion)
        keep = (~np.isin(idx, self._ghosts)) & self._valid[idx]
        self._delete_rays(idx[keep][:excess])

    def _take_RK3_step(self, mean: MeanState) -> None:
        """
        Take a step using the memory-efficient formulation of the RK3 method.
        Note that this method changes `self._data` in place.

        Parameters
        ----------
        mean
            Current mean state of the system. We make the approximation that the
            mean state is constant across stages of the Runge-Kutta scheme.

        """

        As = [0, -5 / 9, -153 / 128]
        Bs = [1 / 3, 15 / 16, 8 / 15]
        increment: float | np.ndarray = 0

        for A, B in zip(As, Bs):
            increment = self._get_drays_dt(mean) * config.dt + A * increment
            self._data[:8] = self._data[:8] + B * increment

        self._data[9] = self._data[9] + config.dt

    @property
    def _valid(self) -> np.ndarray:
        """
        Indicates the columns of `self._data` tracking active ray volumes.

        Returns
        -------
        np.ndarray
            Boolean array indicating whether each column of the underlying data
            array is tracking an active ray volume or can be written over.

        """

        return ~np.isnan(self.meta)