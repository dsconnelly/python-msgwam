#!/bin/bash
set -e

config="$1"
python -m msgwam config/${config}.toml data/${config}.nc
python plot.py data/${config}.nc plots/${config}.png
python animate.py data/${config}.nc plots/${config}.mp4
