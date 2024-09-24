import sys

sys.path.insert(0, '.')
from msgwam import config
from msgwam.integration import SBDF2Integrator

N_MAX = 2502
SPEEDUP = 9

def integrate(resources: str, resolution: str, mode: str) -> None:
    """
    Integrate the model with the loaded configuration file and given resource
    and resolution specifications.

    Parameters
    ----------
    resources
        Affects how many rays can exist at once and controls intermittency. Must
        be one of `{'inf', 'many', 'few', 'intermittent'}`.
    resolution
        Affects the spatial and temporal resolution of the ray volumes. Must be
        one of `{'hyperfine', 'fine', 'coarse'}`.
    mode
        Whether to run with a `'prescribed'` or `'interactive'` mean flow. Or,
        if set to `'convective'`, runs with prescribed mean winds but a source
        spectrum that varies in time according to ICON convection data.

    """

    _set_resources(resources)
    _set_resolution(resolution)

    if mode == 'interactive':
        config.interactive_mean = True

    elif mode == 'convective':
        config.spectrum_type = 'from_file'

    config.refresh()
    ds = SBDF2Integrator().integrate()

    name = '-'.join([resources, resolution, mode])
    ds.to_netcdf(f'data/{config.name}/{name}.nc')
    config.reset()

def _set_resolution(resolution: str) -> None:
    """
    Set the spatial and temporal resolution of the integration.

    Parameters
    ----------
    resolution
        Which resolution specification to use. `'hyperfine'` indicates ray
        volumes more finely resolved in space and time than by the loaded config
        file, useful for reference integrations. `'fine'` uses the settings set
        by the config file, and `'coarse'` reduces the resolution by the square
        root of `SPEEDUP` in both physical and spectral space.  `'intermittent'`
        is `'fine'` but with ray volumes launched only very three hours.

    """

    if resolution == 'hyperfine':
        config.dt = 40
        config.dt_launch = 40

        config.rescale_fluxes = False
        config.n_source *= 5
        config.dr_init /= 5

    elif resolution == 'fine':
        pass

    elif resolution in ['coarse', 'network']:
        root = int(SPEEDUP ** 0.5)
        config.n_source //= root
        config.dr_init *= root

        if resolution == 'network':
            config.source_type = 'network'

    elif resolution == 'intermittent':
        config.dt_launch = 3 * 60 * 60

    else:
        raise ValueError(f'Unknown resolution specification {resolution}')
    
def _set_resources(resources: str) -> None:
    """
    Set the resource allotment of the integration.

    Parameters
    ----------
    resources
        Which resource specification to use. `'inf'` allows the model to have
        arbitrarily many ray volumes. `'many'` uses `N_MAX` as the maximum, and
        `'few'` uses `N_MAX // SPEEDUP`.

    """

    if resources == 'inf':
        config.n_ray_max = 80000
        config.n_increment = 10000

    elif resources == 'many':
        config.n_ray_max = N_MAX
        config.purge_mode = 'energy'

    elif resources == 'few':
        config.n_ray_max = N_MAX // SPEEDUP
        config.purge_mode = 'energy'

    else:
        raise ValueError(f'Unknown resource specification {resources}')
