#!/bin/bash

name=$1
dep_arg=$2

# job_id=$(sbatch \
#     --parsable \
#     --ntasks=1 \
#     --mem=128G \
#     --time=12:00:00 \
#     --gres=gpu:v100:1 \
#     -J "train-network" \
#     -o logs/ml-accel/train-network.out \
#     $dep_arg \
#     submit.slurm config/$name.toml \
#         search-hyperparameters \
#         train-network
# )

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=128G \
    --time=2:00:00 \
    --gres=gpu:v100:1 \
    -J "train-network" \
    -o logs/ml-accel/train-network-final.out \
    $dep_arg \
    submit.slurm config/$name.toml \
        train-network
)

echo $job_id
