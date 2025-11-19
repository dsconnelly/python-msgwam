#!/bin/bash

name=$1
dep_arg=$2

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=128G \
    --time=3:00:00 \
    -J "cache-arrays" \
    -o logs/ml-accel/cache-arrays.out \
    $dep_arg \
    submit.slurm config/$name.toml \
        cache-arrays:2 \
        cache-arrays:6
)

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=128G \
    --time=10:30:00 \
    --gres=gpu:1 \
    -J "train-network" \
    -o logs/ml-accel/train-network-mse.out \
    --dependency=afterok:$job_id \
    submit.slurm config/$name.toml \
        search-hyperparameters:8 \
        train-network
)

echo $job_id
