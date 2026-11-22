#!/bin/bash -e

#SBATCH --mail-user=dsconnelly@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

mython() {
    singularity exec --fakeroot \
        --bind $HOME \
        --overlay "${OVERLAY}:ro" \
        $IMAGE /bin/bash -c 'source /ext3/env.sh; python $@' -- "$@"
}

mython -m acceleration $@
