#!/bin/bash -e

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=64G

#SBATCH --mail-user=dsconnelly@nyu.edu
#SBATCH --mail-type=END,FAIL

python ()
{ 
    cmd="python -u $@";
    singularity exec \
        --overlay ${overlay}:ro ${image} \
        /bin/bash -c "source /ext3/activate.sh; $cmd"
}

config="config/ml-accel.toml"
params="hgrids/surrogate.toml"

for arg in "$@"; do

    if [ "$arg" = "preprocess" ]; then
        python experiments/ml-accel $config $params \
            save-training-data save-indices
    
    elif [ "$arg" = "train-surrogate" ]; then
        python experiments/ml-accel $config $params \
            train-network:surrogate

    elif [ "$arg" = "solve-inversion" ]; then
        python experiments/ml-accel $config $params \
            solve-inversion save-supervolumes
    
    fi

done
