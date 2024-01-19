# python-msgwam

Python ray tracer for internal gravity waves following the technique of Muraschko et al. (2015), Bölöni et al. (2016, 2021) and Kim et al. (2021). The theory is based on Achatz et al. (2017) and Achatz (2022).

## Getting started
To run the ray tracer you will need

- a Python 3.x interpreter
- matplotlib
- numpy

The code can be invoked by calling the `msgwam` directory as a module. You must pass a path to a configuration TOML file and a path where the integration data can be saved in netCDF form. For example
```
python -m msgwam config/bichromatic.toml data/bichromatic.nc
```

Several configuration files are provided in the `config` directory. Note that `desaubies.toml` probably behaves strangely.

The function `make_plots` in `plotting.py` can plot integrator output as below.
![oscillation](oscillation.png)
There is also `animate.py`, a utility for animating ray volume and mean flow behavior.

## License
The code is licensed under the Creative Commons Attribution 4.0 license. For more info see
https://creativecommons.org/licenses/by/4.0/
