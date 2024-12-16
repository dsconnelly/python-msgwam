#!/bin/bash -e

#SBATCH --mail-user=dsconnelly@nyu.edu
#SBATCH --mail-type=END,FAIL

python ()
{
    cmd="python -u $@";
    singularity exec \
        --overlay ${overlay}:ro ${image} \
        /bin/bash -c "source /ext3/env.sh; $cmd"
}

python -m experiments.ml-accel $@