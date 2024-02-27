from __future__ import annotations
from abc import ABC, abstractmethod
from copy import copy
from time import time as now
from typing import Any, Optional

import numpy as np
import scipy as sp
import tqdm
import xarray as xr

from scipy.linalg import lu_factor, lu_solve

from . import config
from .mean import MeanFlow
from .rays import RayCollection

class Integrator(ABC):
    def __init__(self) -> None:
        """
        Initialize the Integrator and the arrays that will hold snapshots of the
        mean flow and the waves when the system is integrated.
        """

        mean = MeanFlow()
        rays = RayCollection(mean)

        self.time = config.dt * np.arange(config.n_t_max)
        if not config.interactive_mean:
            with xr.open_dataset(config.mean_file) as ds:
                ds = ds.interp(time=self.time)
                self.prescribed_u = ds['u'].values
                self.prescribed_v = ds['v'].values

            mean.u = self.prescribed_u[0]
            mean.v = self.prescribed_v[0]
        
        self.int_mean = [mean]
        self.int_rays = [rays]

        pmf = mean.pmf(rays, onto='centers')
        self.int_pmf = [pmf]
        
    @abstractmethod
    def step(
        self,
        mean: MeanFlow,
        rays: RayCollection
    ) -> tuple[MeanFlow, RayCollection]:
        """
        Advance the state of the system by one time step. Should be implemented
        by every Integrator subclass.

        Parameters
        ----------
        mean
            Current MeanFlow.
        rays
            Current RayCollection.

        Returns
        -------
        MeanFlow
            Updated MeanFlow.
        RayCollection
            Updated RayCollection.

        """

        pass

    def integrate(self) -> Integrator:
        """
        Integrate the system over the time interval specified in config.

        Returns
        -------
        Integrator
            The integrated system.

        """

        mean = self.int_mean[0]
        rays = self.int_rays[0]

        format = (
            '{desc}: {percentage:3.0f}%|' +
            '{bar}' +
            '| {n:.2f}/{total_fmt} [{rate_fmt}{postfix}]'
        )

        iterator = tqdm.trange(
            1, config.n_t_max,
            bar_format=format,
            disable=(not config.show_progress),
            unit_scale=(config.dt / 86400),
            unit='day'
        )

        start = now()
        for i in iterator:
            mean, rays = self.step(mean, rays)
            rays.check_boundaries(mean)
            rays.break_waves(mean)

            if not config.interactive_mean:
                mean.u = self.prescribed_u[i]
                mean.v = self.prescribed_v[i]

            if i % config.n_skip == 0:
                self.int_mean.append(mean)
                self.int_rays.append(rays)
                
                pmf = mean.pmf(rays, onto='centers')
                self.int_pmf.append(pmf)

        self.runtime = now() - start
        
        return self

    def to_dataset(self) -> xr.Dataset:
        """
        Return a Dataset holding the data from the integrated system.
        """

        data: dict[str, Any] = {
            'time' : self.time[::config.n_skip],
            'nray' : np.arange(config.n_ray_max),
            'grid' : self.int_mean[0].r_centers
        }
        
        for name in ['u', 'v']:
            stacked = np.vstack([getattr(mean, name) for mean in self.int_mean])
            data[name] = (('time', 'grid'), stacked)

        for name in RayCollection.props:
            stacked = np.vstack([getattr(rays, name) for rays in self.int_rays])
            data[name] = (('time', 'nray'), stacked)

        stacked = np.stack(self.int_pmf).transpose(1, 0, 2)
        data['pmf_u'] = (('time', 'grid'), stacked[0])
        data['pmf_v'] = (('time', 'grid'), stacked[1])

        return xr.Dataset(data, attrs={'runtime' : self.runtime})
    
class RK3Integrator(Integrator):
    aa = [0, -5 / 9, -153 / 128]
    bb = [1 / 3, 15 / 16, 8 / 15]

    def step(
        self,
        mean: MeanFlow,
        rays: RayCollection
    ) -> tuple[MeanFlow, RayCollection]:
        """Take an RK3 step."""

        p: float | np.ndarray = 0
        q: float | np.ndarray = 0

        for a, b in zip(self.aa, self.bb):
            dmean_dt = mean.dmean_dt(rays)
            drays_dt = rays.drays_dt(mean)

            p = config.dt * dmean_dt + a * p
            q = config.dt * drays_dt + a * q

            mean = mean + b * p
            rays = rays + b * q

        return mean, rays
    
class SBDF2Integrator(Integrator):
    def __init__(self) -> None:
        """Initialize an SBDF2Integrator. See Wang and Ruuth (2008)."""

        super().__init__()

        nu = self.int_mean[0].nu
        diag = -nu[:-1] - nu[1:]
        off_diag = nu[1:-1]

        D = (
            np.diag(diag) +
            np.diag(off_diag, k=1) +
            np.diag(off_diag, k=-1)
        )

        D[0, 0] = D[0, 0] - nu[0]
        D[-1, -1] = D[-1, -1] - nu[-1]
        D = D / self.int_mean[0].dr ** 2

        m, _ = D.shape
        self.A = lu_factor(np.eye(m) - config.dt * D)
        self.B = lu_factor(3 * np.eye(m) / 2 - config.dt * D)

        self.last: Optional[list[np.ndarray]] = None
        self.dlast_dt: Optional[list[np.ndarray]] = None

    def step(
        self,
        mean: MeanFlow,
        rays: RayCollection
    ) -> tuple[MeanFlow, RayCollection]:
        """Take an SBDF2 step, using semi-implicit Euler to initialize."""
        
        du_dt, dv_dt = mean.dmean_dt(rays)
        drays_dt = rays.drays_dt(mean)
        mean, rays = copy(mean), copy(rays)

        if self.last is None:
            mean.u = lu_solve(self.A, mean.u + config.dt * du_dt)
            mean.v = lu_solve(self.A, mean.v + config.dt * dv_dt)
            rays.data = rays.data + config.dt * drays_dt

        else:
            last_u, last_v, last_rays = self.last
            last_du_dt, last_dv_dt, last_drays_dt = self.dlast_dt

            idx = np.isnan(last_drays_dt[0])
            last_rays[:, idx] = rays.data[:, idx]
            last_drays_dt[:, idx] = 0

            mean.u = self.lhs(mean.u, last_u, du_dt, last_du_dt, stiff=True)
            mean.v = self.lhs(mean.v, last_v, dv_dt, last_dv_dt, stiff=True)
            rays.data = self.lhs(rays.data, last_rays, drays_dt, last_drays_dt)

        self.last = [mean.u, mean.v, rays.data]
        self.dlast_dt = [du_dt, dv_dt, drays_dt]
        
        return mean, rays

    def lhs(
        self,
        curr: np.ndarray,
        last: np.ndarray,
        dcurr_dt: np.ndarray,
        dlast_dt: np.ndarray,
        stiff: bool=False
    ) -> np.ndarray:
        """
        Compute the new function value using the SBDF2 algorithm.

        Parameters
        ----------
        curr
            Current values.
        last
            Values at previous time step.
        dcurr_dt
            Current time tendency.
        dlast_dt
            Time tendency at previous time step.
        stiff
            Whether the matrix on the left-hand side of the SBDF2 discretization
            should include diffusive effects. Should be true for the mean flow
            and false for the rays.
        
        """

        rhs = 2 * curr - 0.5 * last + config.dt* (2 * dcurr_dt - dlast_dt)
        return lu_solve(self.B, rhs) if stiff else 2 * rhs / 3
