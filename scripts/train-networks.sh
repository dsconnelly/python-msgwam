#!/bin/bash

name=$1
dep_arg=$2

rm data/ml-accel/records/*.txt
n=$(python get-gridsize.py hyperparameters/$name.toml)

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=3:30:00 \
    -a 0-$n \
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
