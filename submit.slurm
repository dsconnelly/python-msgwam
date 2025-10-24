#!/bin/bash -e

#SBATCH --mail-user=dsconnelly@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

python ()
{
    cmd="/ext3/miniforge3/envs/msgwam/bin/python -u $@";
    singularity exec --nv \
        --overlay ~/singularity/cuda-overlay.ext3:ro ${image} \
        /bin/bash -c "source /ext3/env.sh; $cmd"
}

python -m acceleration $@
