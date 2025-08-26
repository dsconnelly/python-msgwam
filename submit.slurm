#!/bin/bash -e

#SBATCH --mail-user=dsconnelly@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

python ()
{
    cmd="python -u $@";
    singularity exec \
        --overlay ${overlay}:ro ${image} \
        /bin/bash -c "source /ext3/env.sh; $cmd"
}

python -m acceleration $@
