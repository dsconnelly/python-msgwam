#!/bin/bash

python ()
{
    cmd="python -u $@";
    singularity exec \
        --overlay ${overlay}:ro ${image} \
        /bin/bash -c "source /ext3/env.sh; $cmd"
}

python -m acceleration config/ml-accel.toml $@