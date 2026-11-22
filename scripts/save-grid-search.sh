#!/bin/bash

name=$1
dep_arg=$2

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=1:00:00 \
    --account=torch_pr_433_general \
    -a 0-24 \
    -J "${name}-grid-search" \
    -o logs/$name/grid-search-%a.out \
    $dep_arg \
    submit.slurm config/$name.toml save-grid-search:mima
)

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=1:30:00 \
    --account=torch_pr_433_general \
    -J "${name}-update" \
    -o logs/$name/grid-search-update.out \
    --dependency=afterok:$job_id \
    submit.slurm config/$name.toml \
        save-grid-search-errors:all:mima \
        plot-grid-search-errors:all::mima \
        update-config:flux
)

echo $job_id
