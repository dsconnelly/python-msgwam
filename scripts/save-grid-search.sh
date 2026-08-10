#!/bin/bash

name=$1
dep_arg=$2

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=1:00:00 \
    -a 0-24 \
    -J "${name}-grid-search" \
    -o logs/$name/grid-search-%a.out \
    $dep_arg \
    submit.slurm config/$name.toml save-grid-search
)

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=1:30:00 \
    -J "${name}-update" \
    -o logs/$name/grid-search-update.out \
    --dependency=afterok:$job_id \
    submit.slurm config/$name.toml \
        save-grid-search-errors:all \
        plot-grid-search-errors:all \
        update-config:flux
)

echo $job_id
