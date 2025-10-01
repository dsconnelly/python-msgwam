#!/bin/bash

name=$1
dep_arg=$2

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=6:00:00 \
    -J $name-save-training-data \
    -o logs/$name/save-training-data.out \
    $dep_arg \
    submit.slurm config/$name.toml \
        save-training-data \
        plot-training-series
)

echo $job_id