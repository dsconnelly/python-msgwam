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
average_output: bool # Whether or not to average mean state variables (wind and
    # momentum flux profiles) in the output file. Has no effect unless dt_output
    # is larger than dt.

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
min_pmf: float # Absolute momentum flux below which rays will be considered too
    # weak and removed from the system.
n_chromatic: int # Number of waves considered at once when determining which ray
    # volumes should break. If 1, breaking is monochromatic, and if -1, breaking
    # is polychromatic over the whole collection of rays. If 0, no ray volume
    # breaking occurs. Values are than these are most useful during batch
    # integrations, so that subsets of ray volumes can break individually.

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
purge_mode: str # Criterion to use when purging rays to enforce the bottom
    # boundary condition. Must be one of 'action', 'cg_r', 'energy', or 'pmf'.
    # If 'none', rays will not be purged when checking the source, but an error
    # will be raised if the boundary condition cannot be enforced.
n_increment: int # If nonzero, then n_ray_max will be increased by this amount
    # whenever necessary to allow the boundary condition to be enforced.

dt_launch: int # Time step between calls to the source. If inf, new ray volumes
    # are not launched beyond the initial packet.

r_launch: float # Launch height of ray volumes (m).
dr_init: float # Initial vertical extent of ray volumes (m).

dk_init: float # Initial extent of ray volumes in k space. 
dl_init: float # Initial extent of ray volumes in l space.

wvl_hor_char: float # Characteristic horizontal wavelength (m).
direction: float # Direction of horizontal propagation (deg relative to east).

n_source: int # How many ray volumes to discretize source into. Must be even.
c_max: float # Maximum absolute phase speed in source spectrum (m / s).

# ==============================================================================
# 'gaussians' spectrum
# ==============================================================================
c_center: float # Phase speed with peak amplitude (m / s).
c_width: float # Width of Gaussian in phase speed space (m / s).
bc_mom_flux: float # Total momentum flux across lower boundary (Pa).

# ==============================================================================
# 'from_file' spectrum
# ==============================================================================
spectrum_file: str # Path to saved spectrum.

# ==============================================================================
# 'network' source
# ==============================================================================
model_path: str # Path where JIT-compiled neural network is saved.

# ==============================================================================
# derived parameters (set by the code, not the namelist)
# ==============================================================================
name: str # Name of the configuration, derived from the file path.
n_t_max: int # Number of time steps to take.
n_skip: int # Number of time steps to skip between outputs.
f0: float # Coriolis parameter (1 / s).
r_ghost: float # Vertical extent of the ghost layer (m).
rays_per_packet: int # How many ray volumes the neural network accepts at once.
    # Calculated as the product of coarse_height and coarse_width.
