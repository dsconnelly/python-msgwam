#!/bin/bash

name=$1
dep_arg=$2

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=16G \
    --time=3:00:00 \
    -a 0-11 \
    -J "ml-accel-context" \
    -o logs/ml-accel/save-training-context-%a.out \
    $dep_arg \
    submit.slurm config/$name.toml save-training-context
)

echo $job_id