from __future__ import annotations
from abc import ABC, abstractmethod
from copy import copy
from time import time as now
from typing import Any

import cftime
import torch
import xarray as xr

from torch.linalg import lu_factor, lu_solve as _lu_solve
lu_solve = lambda A, b: _lu_solve(*A, b.T).T

from . import config
from .constants import EPOCH
from .mean import MeanState
from .rays import RayCollection
from .utils import get_iterator, open_dataset

class Integrator(ABC):
    def __init__(self) -> None:
        """
        Initialize the integrator by creating the lists that will hold snapshots
        of the mean state and the ray volumes when the system is integrated.
        Also load in the prescribed wind file, if there is one.
        """

        mean = MeanState()
        rays = RayCollection()

        self.time = cftime.num2date(
            config.dt * torch.arange(config.n_t_max),
            units=f'seconds since {EPOCH}'
        )

        if not config.interactive_mean:
            if isinstance(config.prescribed_wind, str):
                with open_dataset(config.prescribed_wind) as ds:
                    ds = ds.interp(time=self.time)
                    u = torch.tensor(ds['u'].values)
                    v = torch.tensor(ds['v'].values)

                self.prescribed_wind = torch.stack((u, v), dim=1).float()

            else:
                shape = (len(self.time), -1, -1)
                self.prescribed_wind = config.prescribed_wind.expand(shape)

            mean.wind = self.prescribed_wind[0]

        self.snapshots_mean = [mean]
        self.snapshots_rays = [rays]

        pmf = mean.pmf(rays, onto='centers')
        self.snapshots_pmf = [pmf]

    @abstractmethod
    def step(
        self,
        mean: MeanState,
        rays: RayCollection
    ) -> tuple[MeanState, RayCollection]:
        """
        Advance the state of the system by one time step. Should be implemented
        by every Integrator subclass.

        Parameters
        ----------
        mean
            Current mean state of the system.
        rays
            Collection of current ray volume properties.

        Returns
        -------
        mean
            Updated mean state of the system.
        rays
            Collection of updated ray volume properties.
        
        """
        ...

    def integrate(self) -> Integrator:
        """
        Integrate the system over the time interval specified in config.

        Returns
        -------
        Integrator
            The integrated system.

        """

        mean = self.snapshots_mean[0]
        rays = self.snapshots_rays[0]

        start = now()
        for i in get_iterator():
            if i * config.dt % config.dt_launch == 0:
                rays.check_source(mean)

            mean, rays = self.step(mean, rays)
            rays.check_boundaries(mean)
            rays.dissipate_and_break(mean)

            if not config.interactive_mean:
                mean.wind = self.prescribed_wind[i]

            if i % config.n_skip == 0:
                self.snapshots_mean.append(mean)
                self.snapshots_rays.append(rays)

                pmf = mean.pmf(rays, onto='centers')
                self.snapshots_pmf.append(pmf)

        self.runtime = now() - start

        return self
    
    def to_dataset(self) -> xr.Dataset:
        """Return a dataset containing the integration results."""

        data: dict[str, Any] = {
            'time' : self.time[::config.n_skip],
            'nray' : torch.arange(config.n_ray_max),
            'z' : self.snapshots_mean[0].z_centers
        }

        for name in ['u', 'v']:
            fields = [getattr(mean, name) for mean in self.snapshots_mean]
            data[name] = (('time', 'z'), torch.vstack(fields))

        for name in RayCollection.props:
            fields = [getattr(rays, name) for rays in self.snapshots_rays]
            data[name] = (('time', 'nray'), torch.vstack(fields))

        stacked = torch.stack(self.snapshots_pmf)
        data['pmf_u'] = (('time', 'z'), stacked[:, 0])
        data['pmf_v'] = (('time', 'z'), stacked[:, 1])

        return xr.Dataset(data, attrs={'runtime' : self.runtime})
    
class SBDF2Integrator(Integrator):
    def __init__(self) -> None:
        """Initialize an SBDF2Integrator. See Wang and Ruuth (2008)."""

        super().__init__()

        nu = self.snapshots_mean[0].nu
        diag = -nu[:-1] - nu[1:]
        off_diag = nu[1:-1]

        D = (
            torch.diag(diag) +
            torch.diag(off_diag, diagonal=1) +
            torch.diag(off_diag, diagonal=-1)
        )

        D[0, 0] = D[0, 0] - nu[0]
        D[-1, -1] = D[-1, -1] - nu[-1]
        D = D / self.snapshots_mean[0].dz ** 2

        m, _ = D.shape
        self.A = lu_factor(torch.eye(m) - config.dt * D)
        self.B = lu_factor(3 * torch.eye(m) / 2 - config.dt * D)

        self.last: list[torch.Tensor] = []
        self.dlast_dt: list[torch.Tensor] = []

    def step(
        self,
        mean: MeanState,
        rays: RayCollection
    ) -> tuple[MeanState, RayCollection]:
        """
        Take an SBDF2 step. We use semi-implict Euler to initialize. The scheme
        is slightly complicated because ray volumes are created every time step,
        and so we must always distinguish between trajectories that need to use
        an Euler step and ones that can use the multi-step scheme.

        Note also that wave action spectral density, age, and metadata are
        handled separately, since we have exact update equations for them.
        """

        dmean_dt = mean.dmean_dt(rays)
        drays_dt = rays.drays_dt(mean)
        first_step = len(self.last) == 0

        if not first_step:
            last_mean, last_rays = self.last
            last_dmean_dt, last_drays_dt = self.dlast_dt

        self.last = [mean.wind, rays.data]
        self.dlast_dt = [dmean_dt, drays_dt]
        mean, rays = copy(mean), copy(rays)

        if first_step:
            mean.wind = lu_solve(self.A, mean.wind + config.dt * dmean_dt)
            new_rays = rays.data[:8] + config.dt * drays_dt

        else:
            mean.wind = self.lhs(
                mean.wind, last_mean,
                dmean_dt, last_dmean_dt,
                stiff=True
            )

            euler_jdx = rays.age == 0
            euler_data = rays.data[:8, euler_jdx]

            new_rays = self.lhs(
                rays.data[:8], last_rays[:8],
                drays_dt, last_drays_dt
            )

            euler_delta = config.dt * drays_dt[:, euler_jdx]
            new_rays[:8, euler_jdx] = euler_data + euler_delta

        rays.data = torch.vstack((new_rays, rays.data[8:]))
        rays.age[:] = rays.age[:] + config.dt

        return mean, rays

    def lhs(
        self,
        curr: torch.Tensor,
        last: torch.Tensor,
        dcurr_dt: torch.Tensor,
        dlast_dt: torch.Tensor,
        stiff: bool=False
    ) -> torch.Tensor:
        """
        Compute an updated vector of function values using the SBDF2 algorithm.

        Parameters
        ----------
        curr
            Current vector of function values.
        last
            Vector of function values at previous time step.
        dcurr_dt
            Current vector of time tendencies.
        dlast_dt
            Vector of time tendencies at previous time step.
        stiff
            Whether the matrix on the left-hand side of the SBDF2 discretization
            should include diffusive effects. Should be `True` for the mean
            state and `False` for the rays.

        Returns
        -------
        torch.Tensor
            Updated vector of function values.

        """

        rhs = 2 * curr - 0.5 * last + config.dt * (2 * dcurr_dt - dlast_dt)
        return lu_solve(self.B, rhs) if stiff else 2 * rhs / 3
                