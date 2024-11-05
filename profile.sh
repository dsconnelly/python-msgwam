#!/bin/bash

python -m cProfile -o profile.pstats -m msgwam config/benchmark.toml
python -m gprof2dot -f pstats -n 5 profile.pstats | dot -Tpng -o profile.png