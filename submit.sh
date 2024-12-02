#!/bin/bash

name="idealized"
config="config/$name.toml"

mkdir -p "data/$name/coarsenings"
mkdir -p "data/$name/models"
mkdir "plots/$name"

reference_id=$(sbatch \
    --parsable \
    --n-tasks=1 \
    --cpus-per-task=8 \
    --time=18:00:00 \
    --mem=64G \
    -J reference \
    -o logs/ml-accel/reference.out \
    ml-accel.slurm $config save-spectrum save-descending-jets integrate:reference
)

coarsenings_id=$(sbatch \
    --parsable \
    --n-tasks=1 \
    --cpus-per-task=8 \
    --time=12:00:00 \
    --mem=16G \
    -J coarsenings \
    -o logs/ml-accel/coarsenings.out \
    --dependency=afterok:${reference_id} \
    ml-accel.slurm $config save-coarsenings update-config
)

baselines_id=$(sbatch \
    --parsable \
    --n-tasks=1 \
    --cpus-per-task=8 \
    --time=1:00:00 \
    --mem=16G \
    -J baselines \
    -o logs/ml-accel/baselines.out \
    --dependency=afterok:${coarsenings_id} \
    ml-accel.slurm $config integrate:coarse integrate:stochastic integrate:instantaneous
)

generate_id=$(sbatch \
    --parsable \
    --n-tasks=1 \
    --cpus-per-task=8 \
    --time=6:00:00 \
    --mem=16G \
    -J generate \
    -o logs/ml-accel/generate.out \
    --dependency=afterok:${baselines_id} \
    ml-accel.slurm $config save-training-data
)

training_id=$(sbatch \
    --parsable \
    --n-tasks=1 \
    --cpus-per-task=8 \
    --time=9:00:00 \
    --mem=16G \
    --array=0-35 \
    -J training \
    -o logs/ml-accel/training-%a.out \
    --dependency=afterok:${generate_id} \
    ml-accel.slurm $config train-network
)
