################################################################################
# gated-oscillation scenario
################################################################################

gate_closed: float
gate_open: list[float]
gate_period : float
gate_width: float

noise_amp: float
noise_cutoffs: list[float]
noise_decays: list[float]

shear_width: float

time_scales: list[float]

wave_amp: float
wvl: float

z_gate_bounds: list[float]
z_tide_bounds: list[float]

################################################################################
# other parameters
################################################################################

components: str
ICON_region: str
n_columns: int
spectrum_type: str