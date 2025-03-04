#!/bin/bash

name=$1
dep_arg=$2

if [[ $name == icon* ]]; then
    n="59"
else
    n="99"
fi

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=1:00:00 \
    -a 0-$n \
    -J coarsening \
    -o logs/$name/coarsening-%a.out \
    $dep_arg \
    submit.slurm config/$name.toml save-coarsenings
)

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=1:00:00 \
    -J update \
    -o logs/$name/coarsening-update.out \
    --dependency=afterok:$job_id \
    submit.slurm config/$name.toml \
        plot-coarse-errors \
        update-config
)

echo $job_id
