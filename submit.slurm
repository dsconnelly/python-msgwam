#!/bin/bash -e

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --mem=128G

#SBATCH --mail-user=dsconnelly@nyu.edu
#SBATCH --mail-type=END,FAIL

python ()
{ 
    cmd="python $@";
    singularity exec \
        --overlay ${overlay}:ro ${image} \
        /bin/bash -c "source /ext3/activate.sh; $cmd"
}

case=$1
shift

if [ "$case" = "1" ]; then
    config="config/centered.toml"
    mode="prescribed"
elif [ "$case" = "2" ]; then
    config="config/centered.toml"
    mode="interactive"
elif [ "$case" = "3" ]; then
    config="config/centered-tropics.toml"
    mode="convective"
fi

for arg in "$@"; do

    if [ "$arg" = "preprocess" ]; then

        if [ "$case" = "1" ]; then
            python experiments/testing $config \
                save-mean-state:descending-jets
        elif [ "$case" = "3" ]; then
            python experiments/testing $config \
                save-mean-state:ICON
        fi

        if [ "$mode" = "convective" ]; then
            python experiments/testing $config \
                save-ICON-spectra
        fi

    elif [ "$arg" = "integrate" ]; then

        python experiments/testing $config \
            integrate:reference:$mode \
            integrate:many-fine:$mode \
            integrate:few-fine:$mode \
            integrate:many-coarse:$mode \
            integrate:few-coarse:$mode \
            integrate:intermittent:$mode

    elif [ "$arg" = "analyze" ]; then
    
        python experiments/testing $config \
            plot-fluxes:$mode \
            plot-lifetimes:$mode \
            plot-scores:$mode:pmf_u

        if [ "$mode" = "interactive" ]; then
            python experiments/testing $config \
                plot-scores:$mode:u
        fi

    fi

done



