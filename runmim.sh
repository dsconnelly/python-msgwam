#!/bin/bash

python ()
{
    cmd="python -u $@";
    singularity exec \
        --overlay ${overlay}:ro ${image} \
        /bin/bash -c "source /ext3/env.sh; $cmd"
}

name=$1
shift

python -m acceleration config/mima-$name.toml $@