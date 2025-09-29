#!/bin/bash

name=$1
dep_arg=$2

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=6:00:00 \
    -a 0-143 \
    -J "ml-accel-search" \
    -o logs/ml-accel/train-network-%a.out \
    $dep_arg \
    submit.slurm config/$name.toml train-network:va
)

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=64G \
    --time=6:00:00 \
    -J "ml-accel-best" \
    -o logs/ml-accel/train-network-best.out \
    --dependency=afterok:$job_id \
    submit.slurm config/$name.toml train-network:te
)

echo $job_id
