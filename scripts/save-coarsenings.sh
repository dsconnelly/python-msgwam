#!/bin/bash

name=$1
dep_arg=$2

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=1:00:00 \
    -a 0-24 \
    -J "${name}-coarsenings" \
    -o logs/$name/coarsening-%a.out \
    $dep_arg \
    submit.slurm config/$name.toml save-coarsenings
)

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=1:30:00 \
    -J "${name}-update" \
    -o logs/$name/coarsening-update.out \
    --dependency=afterok:$job_id \
    submit.slurm config/$name.toml \
        save-coarse-errors \
        plot-coarse-errors \
        update-config
)

echo $job_id
