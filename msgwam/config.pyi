from typing import Any, Optional

import torch

def load(path: str) -> None:
    ...

def refresh(config: Optional[dict[str, Any]]=None) -> None:
    ...

def reset() -> None:
    ...

# ==============================================================================
# global flags
# ==============================================================================
show_progress: bool # Whether to show a progress bar during integration.
interactive_mean: bool # Whether the ray volumes affect the mean state.
prescribed_wind: str | torch.Tensor # Prescribed mean wind field to use if
    # interactive_mean is False. If specified in the namelist, should be a path
    # to a netCDF file containing a (time-varying) mean wind field. If specified
    # dynamically, should be a tensor with two rows containing the constant u
    # and v profiles, respectively.

# ==============================================================================
# time stepping
# ==============================================================================
dt: int # Time step (s).
dt_output: int # Output time step (s)
n_day: int # Number of days to integrate for.

# ==============================================================================
# vertical grid
# ==============================================================================
grid_bounds: tuple[float, float] # Top and bottom of vertical grid (m).
n_grid: int # Number of vertical grid cell boundaries.

# ==============================================================================
# mean state profiles
# ==============================================================================
N0: float # Buoyancy frequency (1 / s).
rho0: float # Density at zero altitude (kg / m^3).
boussinesq: bool # Whether density is constant with height.
H_rho: float # Density scale height (m).
latitude: float # Column latitude (degrees N).

# ==============================================================================
# mean wind initialization
# ==============================================================================
wind_init_method: str # How to initialize the mean wind profiles.
u0: float # Amplitude of the u jet if the 'gaussian' method is used (m / s).
v0: float # Amplitude of the v jet if the 'gaussian' method is used (m / s).
z0: float # Altitude of the jet if the 'gaussian' method is used (m).
sigma_uv: float # Width of the jet if the 'gaussian' method is used (m).

# ==============================================================================
# pseudomomentum flux calculation
# ==============================================================================
shapiro_filter: bool # Whether to apply a Shapiro filter to flux profiles.
proj_method: str # How to project wave quantities onto the vertical grid. Must
    # be 'discrete' or 'gaussian'.
smoothing: float # If Gaussian projection is used, higher smoothing values widen
    # the Gaussian momentum flux curve associated with each wave packet.

# ==============================================================================
# viscosity and dissipation
# ==============================================================================
mu: float # Dynamic viscosity (m^2 / s).
dissipation: float # Ratio of the viscosity used in wave dissipation to that
    # used for the mean flow. If zero, waves do not dissipate.

# ==============================================================================
# ray volumes and propagation
# ==============================================================================
n_ray_max: int # Maximum number of ray volumes that can exist at once.
n_chromatic: int # Number of waves considered at once when determining which ray
    # volumes should break. If 1, breaking is monochromatic, and if -1, breaking
    # is polychromatic over the whole collection of rays. If 0, no ray volume
    # breaking occurs. Values are than these are most useful during batch
    # integrations, so that subsets of ray volumes can break individually.
epsilon: float # Intermittency parameter defining the percentage of time that a
    # new ray volume will be launched. Must be in (0, 1].

# ==============================================================================
# source and spectrum (parameters in this section are likely to be used by more
# than one source or spectrum type)
# ==============================================================================
source_type: str # How ray volumes should be sampled from the source spectrum.
    # Must be the name of a subclass of Source defined in sources.py.
spectrum_type: str # Parameterization of the source spectrum to use. Must be the
    # name of a function defined in spectra.py. For dynamically created spectra,
    # a function that returns the desired ray volume data should be monkey
    # patched into that module.
purge: bool # Whether to purge existing rays to enforce the bottom boundary
    # condition. If False, an error will be raised if the boundary condition
    # cannot be enforced because there are two many rays.

relaunch: bool # Whether to launch waves beyond the first time step.
bc_mom_flux: float # Momentum flux across lower boundary (Pa).

r_launch: float # Launch height of ray volumes (m).
dr_init: float # Initial vertical extent of ray volumes (m).

dk_init: float # Initial extent of ray volumes in k space. 
dl_init: float # Initial extent of ray volumes in l space.

wvl_hor_char: float # Characteristic horizontal wavelength (m).
direction: float # Direction of horizontal propagation (deg relative to east).

# ==============================================================================
# 'gaussians' spectrum
# ==============================================================================
c_center: float # Phase speed with peak amplitude (m / s).
c_width: float # Width of Gaussian in phase speed space (m / s).
n_source: int # How many ray volumes to discretize source into. Must be even.

# ==============================================================================
# derived parameters (set by the code, not the namelist)
# ==============================================================================
name: str # Name of the configuration, derived from the file path.
n_t_max: int # Number of time steps to take.
n_skip: int # Number of time steps to skip between outputs.
f0: float # Coriolis parameter (1 / s).
r_ghost: float # Vertical extent of the ghost layer (m).
