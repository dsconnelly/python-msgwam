from __future__ import annotations
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional, Self, cast
from warnings import warn

import pandas as pd
import numpy as np

from .. import config
from ..constants import PROP_NAMES
from ..dispersion import get_cg_r, get_omega_hat
from ..utils import shapiro_filter

from .base import Propagator
from .jitted import get_importances, get_max_intersects, interp, project

if TYPE_CHECKING:
    from ..means import MeanState

class CFLWarning(Warning):
    pass

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
    attrition: np.ndarray

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
        self._next_meta = -1

        padding = (mean.z_centers[0] - mean.dz, mean.z_centers[-1] + mean.dz)
        self._z_padded = np.pad(mean.z_centers, 1, constant_values=padding)
        k = np.argmax(self._z_padded > config.r_source) - 1

        if config.logging:
            self._statistics = defaultdict(lambda: [np.inf, -np.inf, 0, 0])

        self._r_ghost = config.r_source - config.dr_ghost
        self._r_ghost = np.minimum(self._r_ghost, self._z_padded[k])
        self._ghosts = np.zeros(config.n_source).astype(int)

        self._check_source(mean, 0)
        self._apply_sinks(mean, damp=False)

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
    
    def export_log(self) -> pd.DataFrame:
        """Export the logged statistics as a `DataFrame`."""

        names = list(self._statistics.keys())
        data = np.zeros((len(names), 3))
        names = sorted(names)

        for i, name in enumerate(names):
            vmin, vmax, vtot, count = self._statistics[name]
            data[i] = [vmin, vtot / count, vmax]

        return pd.DataFrame(data, index=names, columns=['min', 'mean', 'max'])

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
        drop = (self.m < 0) & (self.r - 0.5 * self.dr < self._r_ghost)
        action_flux[drop] = 0

        data = np.vstack([wvn * action_flux for wvn in wvns])
        fluxes = self._project(data, self._z_padded) / config.epsilon

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

        dt = self._get_dt(mean)
        for _ in range(config.dt // dt):    
            self._take_RK4_step(mean, dt)

        self._check_source(mean, n_step)
        self._apply_sinks(mean)
        self._check_boundaries(mean)

        if config.logging:
            if self.n_active > 0:
                frac = (self.k != 0)[self._valid].sum() / self.n_active
                self._log_value('zonal fraction (%)', 100 * frac)

            self._log_value('active rays', self.n_active)
            self._log_value('age', abs(self.age)[self._valid])

        return self

    def _add_ray(
        self,
        data: np.ndarray,
        r: float
    ) -> int:
        """
        Add a ray volume to the propagator, storing its data in the first open
        column. Raises an error if the propagator is already at the maximum
        number of active ray volumes and `config.n_increment == 0`.

        Parameters
        ----------
        data
            Vector of wave properties (k, l, m, dk, dl, dm, dens).
        r
            Vertical position of the array to add.

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

        self._next_meta = self._next_meta + 1
        j = np.argmin(self._valid)

        self._data[0, j] = r
        self._data[1:-3, j] = data
        self._data[-3:, j] = [0, self._next_meta, 0]

        return cast(int, j)

    def _apply_sinks(self, mean: MeanState, damp: bool=True) -> None:
        """
        First dissipate waves according to viscosity. Next, remove any rays that
        have artificially passed through critical layers. Finally, determine if
        convective instability-induced wave breaking should occur, and adjust
        the spectral wave action densities accordingly.

        Parameters
        ----------
        mean
            Current mean state of the system.
        damp
            Whether to calculate damping or just wave breaking. Should always be
            `False` except for at initialization.

        """

        wvn = abs(self.k + self.l)
        omega_hat = self._get_omega_hat(mean)
        wvn_hor_sq = self.k ** 2 + self.l ** 2
        wvn_sq = wvn_hor_sq + self.m ** 2

        G2 = interp(self.r, mean.z_centers, mean.G2)
        nu = config.dissipation * interp(self.r, mean.z_faces, mean.nu)
        damping = nu * (wvn_sq + G2) * (1 + config.f ** 2 / (omega_hat ** 2))
        damping = np.exp(-config.dt * damping)

        if config.n_sponge > 0:
            gap = mean.z_faces[-1] - self.r
            idx = self.r > mean.z_faces[-(config.n_sponge + 1)]
            sponge = 1 - (self._get_cg_r(mean) * config.dt / gap)
            damping[idx] = np.minimum(np.maximum(sponge[idx], 0), 1)

        if config.max_age_warning > 0:
            time_left = config.max_age - self.age
            idx = time_left < config.max_age_warning
            frac = np.minimum(np.maximum(1 - config.dt / time_left, 0), 1)
            damping[idx] = np.minimum(damping[idx], frac[idx])

        if damp:
            damping[self._notouch] = 1
            self._data[11] = (1 - damping) * wvn * self.action
            self._data[8] = self.dens * damping

        if config.n_chromatic == 0:
            return

        threshold = mean.rho / 2
        S = self.action * wvn_hor_sq * self.m ** 2 / (omega_hat * wvn_sq)
        
        if config.n_chromatic == -1:
            pdx = np.zeros(self._n_max).astype(int)
        else:
            _, pdx = self._get_packet_info()

        data = np.vstack((S, S * wvn_sq)) / config.epsilon
        P, Q = self._project(data, mean.z_faces, pdx)
        P = P - threshold

        idx = Q != 0
        kappa = np.zeros(P.shape)
        kappa[idx] = P[idx] / Q[idx]

        maxes = get_max_intersects(self.r, self.dr, mean.z_faces, kappa, pdx)
        factor = np.maximum(0, 1 - config.epsilon * wvn_sq * maxes)
        factor[self._notouch & (self.age > 0)] = 1

        self._data[11] += (1 - factor) * wvn * self.action
        self._data[8] = self.dens * factor

    def _check_boundaries(self, mean: MeanState) -> None:
        """
        Delete rays that have propagated outside of the physical domain and rays
        that no longer have more than `config.min_flux` momentum flux.

        Parameters
        ----------
        mean
            Current mean state of the system.

        """

        r_lo = self.r - 0.5 * self.dr
        r_hi = self.r + 0.5 * self.dr

        cg_r = self._get_cg_r(mean)
        wvn = np.sqrt(self.k ** 2 + self.l ** 2)
        flux = wvn * self.action * cg_r

        drop = r_lo > config.z_max
        drop = drop | ((r_hi < config.z_min) & (self.m > 0))
        drop = drop | (abs(flux) < config.min_flux)
        drop = drop | (cg_r < config.min_cg)

        if config.max_age > 0:
            drop = drop | (self.age > config.max_age)

        drop[self._notouch] = False
        self._data[11] += drop * wvn * self.action
        self._data[8, drop] = 0
        
        if config.oob_action == 'delete':
            self._delete_rays(drop)

        else:
            self._data[9, drop] = -self.age[drop]

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

        if n_step == 0 and config.source_type == 'constant':
            datas, cdx = self._source.launch(mean, 0)
            r_init = self._r_ghost - 0.5 * datas[0]

            for k, data in zip(cdx, datas.T):
                self._ghosts[k] = self._add_ray(data, r_init[k])

            if config.jitter > 0:
                noise = np.random.rand(self._n_max) - 0.5
                self._data[0] += config.jitter * noise

            return

        cdx: Optional[np.ndarray] = None
        if config.source_type == 'constant':
            r_lo = (self.r - 0.5 * self.dr)[self._ghosts]
            crossed = (r_lo > self._r_ghost)

            if config.max_age_ghost > 0:
                age = self.age[self._ghosts]
                crossed = crossed | (age > config.max_age_ghost)

            cdx, *_ = np.where(crossed)
            if crossed.sum() == 0:
                return

            if not config.strict_source:
                r_lo = np.minimum(r_lo, config.r_source)

        datas, cdx = self._source.launch(mean, n_step, cdx)
        to_add: list[tuple[int, np.ndarray, float]] = []

        if config.source_type == 'constant':
            for k, data in zip(cdx, datas.T):
                while True:
                    r = r_lo[k] - 0.5 * data[0]
                    r_lo[k] = r_lo[k] - data[0]
                    to_add.append((k, data, r))

                    if r_lo[k] <= self._r_ghost:
                        break

        else:
            repeats = {}
            for k, data in zip(cdx, datas.T):
                n_shift = repeats.setdefault(k, 0)
                r = self._r_ghost - (n_shift + 0.5) * data[0]
                repeats[k] = repeats[k] + 1
                to_add.append((k, data, r))

        excess = self.n_active + len(to_add) - self._n_max
        self._prune(excess, mean)

        for k, data, r in to_add:
            self._ghosts[k] = self._add_ray(data, r)

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
            r = self.r.copy()
            r[self._notouch] = config.r_source

        N = interp(r, mean.z_centers, mean.N)
        G2 = interp(r, mean.z_centers, mean.G2)

        return get_cg_r(self.k, self.l, self.m, N, G2)

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

        N = interp(self.r, mean.z_centers, mean.N)
        G2 = interp(self.r, mean.z_centers, mean.G2)

        du_dr = interp(self.r, mean.z_faces[1:-1], np.diff(mean.u) / mean.dz)
        dv_dr = interp(self.r, mean.z_faces[1:-1], np.diff(mean.v) / mean.dz)
        dN_dr = interp(self.r, mean.z_faces[1:-1], np.diff(mean.N) / mean.dz)
        dG2_dr = interp(self.r, mean.z_faces[1:-1], np.diff(mean.G2) / mean.dz)

        dN2_dr = 2 * N * dN_dr
        omega_hat = self._get_omega_hat(mean)
        wvn_hor_sq = self.k ** 2 + self.l ** 2
        wvn_sq = wvn_hor_sq + self.m ** 2 + G2

        dm_dt = -(
            self.k * du_dr + self.l * dv_dr + (
                wvn_hor_sq * dN2_dr +
                (config.f ** 2 - omega_hat ** 2) * dG2_dr
            ) / (2 * wvn_sq * omega_hat)
        )

        cg_mid = self._get_cg_r(mean)
        dk_dt, dl_dt, ddk_dt, ddl_dt, ddm_dt = np.zeros((5, self._n_max))
        dm_dt[self._notouch] = ddr_dt[self._notouch] = 0
        dr_dt[self._notouch] = cg_mid[self._notouch]

        return np.vstack((
            dr_dt, ddr_dt,
            dk_dt, dl_dt, dm_dt,
            ddk_dt, ddl_dt, ddm_dt
        ))
    
    def _get_dt(self, mean: MeanState) -> int:
        """
        Choose an adaptive time step to prevent critical level jumping.
        
        Parameters
        ----------
        mean
            Current mean state of the system.

        Returns
        -------
        int
            Time step small enough so that individual stages of the RK3 time
            stepper don't skip over entire cells of the vertical grid.

        """

        cg_max = np.nanmax(abs(self._get_cg_r(mean)))
        for n in range(1, config.max_dt_multiplier + 1):
            dt, remainder = divmod(config.dt, n)
            if remainder != 0:
                continue

            if cg_max * dt / mean.dz / 4 < 1:
                return dt

        message = f'could not find sufficiently small time step'
        warn(message, CFLWarning)

        return dt

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

        r = self.r.copy()
        r[self._notouch] = config.r_source

        N = interp(r, mean.z_centers, mean.N)
        G2 = interp(r, mean.z_centers, mean.G2)
        
        return get_omega_hat(self.k, self.l, self.m, N, G2)

    def _get_packet_info(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Sort the ray volumes into packets according to the `meta` attribute and
        the packet size specified by `config.n_chromatic`.

        Returns
        -------
        np.ndarray
            Labels of the packets that are currently active. Packet labels will
            not be reused throught the entire integration.
        np.ndarray
            Array indicating which of the currently active packets, numbered
            starting at zero, each ray volume belongs to.

        """

        floors = np.floor(self.meta / config.n_chromatic)
        labels, pdx = np.unique(floors, return_inverse=True)
        pdx[~self._valid] = -1

        return labels[~np.isnan(labels)].astype(int), pdx.astype(int)

    def _log_value(self, name: str, v: float | np.ndarray) -> None:
        """
        Log a diagnostic value. Keeps track of minima, maxima, and means.

        Parameters
        ----------
        name
            Name of the statistic to update.
        v
            Value of the statistic to log. If an array, the elements of the
            flattened array will be treated as individual observations.

        """

        v = np.asarray(v).flatten()
        if len(v) == 0:
            return

        self._statistics[name][0] = min(self._statistics[name][0], v.min())
        self._statistics[name][1] = max(self._statistics[name][1], v.max())
        self._statistics[name][2] = self._statistics[name][2] + v.sum()
        self._statistics[name][3] = self._statistics[name][3] + len(v)

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
    
    @property
    def _notouch(self) -> np.ndarray:
        """
        Indicates ray volumes that have yet to enter the domain interior and so
        should be protected from pruning, refraction, dissipation, and breaking.

        Returns
        -------
        np.ndarray
            Boolean array indicating waves that should not be touched.

        """

        notouch = (self.r < config.r_source) & (self.m < 0)

        if config.max_age_ghost > 0:
            notouch = notouch & (self.age < config.max_age_ghost)

        return notouch

    def _project(
        self,
        data: np.ndarray,
        edges: np.ndarray,
        pdx: Optional[np.ndarray]=None
    ) -> np.ndarray:
        """
        Project data corresponding to each ray volume onto the vertical grid.
        This function is mainly a wrapper around `project`, which cannot be an
        instance method as it is JIT-compiled.

        Parameters
        ----------
        data
            Variables to project (e.g. momentum fluxes). To save computation,
            multiple variables are projected at once, so that `data` should have
            two dimensions, the first of which ranges over variables to project
            and the second of which ranges over ray volumes.
        edges
            Edges of regions of the vertical grid. Likely either the cell faces
            (for projection onto cell centers) or the padded set of cell centers
            (for projection onto cell faces).
        pdx
            Precomputed packet index for each ray volume. If `None`, ray volumes
            will be projected all together.

        Returns
        -------
        np.ndarray
            Projected values at each vertical grid point for each variable
            passed in as a row of `data`. If `pdx` was not passed as `None`, the
            returned array will have three dimensions, the second of which
            ranges over non-negative indices in `pdx`.

        """

        jdx = 0 if pdx is None else slice(None, None)
        pdx = np.zeros(self._n_max, dtype=int) if pdx is None else pdx
        out = project(self.r, self.dr, edges, data, pdx)

        return out[:, jdx]

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

        if config.prune_by in ['cg_r', 'flux', 'importance'] or config.logging:
            mom = (self.k + self.l) * self.action
            cg = self._get_cg_r(mean)
            flux = abs(mom * cg)

        match config.prune_by:
            case 'cg_r':
                criterion = abs(cg)

            case 'energy':
                criterion = self.action * self._get_omega_hat(mean) * self.dr

            case 'flux':
                criterion = flux * self.dr

            case 'importance':
                criterion = np.zeros(self._n_max)
                for idx in [self.k > 0, self.k < 0, self.l > 0, self.l < 0]:
                    args = (self.r[idx], self._z_padded, flux[idx] * self.dr[idx])
                    criterion[idx] = get_importances(*args)

            case 'random':
                criterion = np.random.rand(self._n_max)

        ubound = np.nanmax(criterion)
        criterion[~self._valid] = ubound + 3
        criterion[self._notouch] = ubound + 2
        drop = np.argsort(criterion)[:excess]

        if config.logging:
            self._log_value('pruned age (h)', self.age[drop] / 3600)
            self._log_value('pruned flux (mPa)', flux[drop] * 1000)
            self._log_value('pruned r (km)', self.r[drop] / 1000)
            self._log_value('pruned cg (m / s)', cg[drop])

        self._delete_rays(drop)

    def _take_RK4_step(self, mean: MeanState, dt: int) -> None:
        """
        Take a step using a reasonably memory-efficient formulation of RK4. Note
        that this method changes `self._data` in place.

        Parameters
        ----------
        mean
            Current mean state of the system. We make the approximation that the
            mean state is constant across stages of the Runge-Kutta scheme.
        dt
            Time step to use. Passed as an argument so that it can be adaptive.

        """

        area = self.dr * self.dm
        Cs = [1 / 2, 1 / 2, 1, 1 / 6]
        incs = np.zeros((5, *self._data[:8].shape))
        incs[0] = self._data[:8]

        for stage, C in enumerate(Cs, 1):
            incs[stage] = dt * self._get_drays_dt(mean)
            self._data[:8] = incs[0] + C * incs[stage]

        self._data[:8] = self._data[:8] + incs[1] / 6 + (incs[2] + incs[3]) / 3
        self._data[9] = self._data[9] + dt
        
        self._data[1] = abs(self.dr)
        self._data[1, self.dr < config.dr_min] = config.dr_min
        self._data[1, self.dr > config.dr_max] = config.dr_max
        self._data[7] = area / self.dr

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