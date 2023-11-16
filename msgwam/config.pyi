def load_config(path: str) -> None:
    ...

def refresh(config: dict[str]=None) -> None:
    ...

################################################################################
# global flags
################################################################################
boussinesq: bool # Whether density is constant with height.
hprop: bool # Whether waves propagate horizontally (currently unsupported).

################################################################################
# wave-mean interactions
################################################################################
interactive_mean: bool # Whether waves affect the mean flow.
saturate_online: bool # Whether wave breaking is included in tendency terms.
filter_pmf: bool # Whether to filter momentum flux before differentiating.
mean_file: str # Path to netCDF file containing prescribed wind time series
    # (only used if not interactive_mean).

################################################################################
# time stepping
################################################################################
dt: int # Time step (s).
n_day: int # Number of days to integrate for.
dt_output: int # Output time step (s).

################################################################################
# vertical grid
################################################################################
n_grid: int # Number of vertical grid cell boundaries.
grid_bounds: tuple[float, float] # Extent of vertical grid (m).

################################################################################
# mean state parameters
################################################################################
phi0: float # Latitude (deg).
rhobar0: float # Density at zero height (kg / m^3).
hh: float # Density scale height (m).
N0: float # Buoyancy frequency (1 / s).

################################################################################
# mean flow initialization
################################################################################
uv_init_method: str # How to initialize interactive u and v profiles. Must be
    # one of 'sine_homogeneous' or 'gaussian'.
u0: float # Reference velocity for initialization scheme.
r0: float # Reference height for initialization scheme.
sig_r: float # Reference standard deviation for initialization scheme.

################################################################################
# flow parameters
################################################################################
nu: float # Mean velocity diffusivity (m^2 / s)
alpha: float # Fraction of Lindzen criterion at which waves break (unitless).

################################################################################
# source and spectrum options
################################################################################
spectrum_type: str # How to calculate the source spectrum. Must be the name of a
    # function defined in sources.py
source_type: str # How to launch ray volumes from the source. Must be the name
    # of a subclass of Source defined in sources.py, lowercase and with the
    # 'Source' suffix removed.

n_ray_max: int # Maximum number of rays that can exist at once.
bc_mom_flux: float # Momentum flux across lower boundary (Pa).

wvl_hor_char: float # Characteristic horizontal wavelength (m).
wvl_ver_char: float # Characteristic vertical wavelength (m).
direction: float # Horizontal direction (deg relative to due east).

dk_init: float # Initial width of volumes in k space.
dl_init: float # Initial width of volumes in l space.
dm_init: float # Initial width of volumes in m space.

r_launch: float # Launch height of ray volumes (m).
dr_init: float # Initial height of ray volumes (m).

################################################################################
# 'stochastic' source parameters
################################################################################
source_fraction: float # Fraction of source ray volumes to keep active at once.

################################################################################
# 'legacy' spectrum parameters
################################################################################
r_m_area: float # Initial area of volumes in r-m space (unitless).
r_init_bounds: tuple[float, float] # Lower and upper extent of wave packet.

################################################################################
# 'desaubies' spectrum parameters
################################################################################
n_c_tilde: int # Number of c_tilde grid points.
n_omega_tilde: int # Number of omega_tilde grid points.
c_tilde_bounds: tuple[float, float] # Bounds on c_tilde (m / s).
omega_tilde_bounds: tuple[float, float] # Bounds on omega_tilde (1 / s).

################################################################################
# 'bimodal' spectrum parameters
################################################################################
n_per_mode: int # Number of ray volumes to launch for each of the positive and
    # negative k parts of the source.

################################################################################
# 'gaussians' spectrum parameters
################################################################################
c_bounds: tuple[float, float] # Mininum and maximum zonal phase speeds (m / s).
c_center: float # Peak of Gaussians in phase speed space (m / s).
c_width: float # Width of the Gaussian in phase speed space (m / s).
n_c: int # How many phase speeds to discretize the interval into.

################################################################################
# internal variables (set by the code, not the namelist)
################################################################################
n_t_max: int # Number of time steps to take.
n_skip: int # Number of time steps to skip between outputs.
f0: float # Coriolis parameter (1 / s).
r_ghost: float # Height of the ghost layer (m).
