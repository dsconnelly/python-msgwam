#!/bin/bash

name=$1
dep_arg=$2

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=128G \
    --time=4:00:00 \
    -J "cache-arrays" \
    -o logs/ml-accel/cache-arrays.out \
    $dep_arg \
    submit.slurm config/$name.toml \
        cache-arrays:2 \
        cache-arrays:3 \
        cache-arrays:4 \
        cache-arrays:5
)

dep_arg="--dependency=afterok:$job_id"

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=128G \
    --time=10:30:00 \
    --gres=gpu:v100:1 \
    -J "train-network" \
    -o logs/ml-accel/train-network-providence.out \
    $dep_arg \
    submit.slurm config/$name.toml \
        search-hyperparameters:0:5 \
        train-network
)

echo $job_id
