#!/bin/bash

name=$1
config="config/$name.toml"
cd /home/dsc7746/python-msgwam

mkdir -p data/$name/input
mkdir -p data/$name/strategies
mkdir -p logs/$name

reference_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=128G \
    --cpus-per-task=8 \
    --time=36:00:00 \
    -J reference \
    -o logs/$name/reference.out \
    ml-accel.slurm $config \
        save-descending-jets \
        save-spectrum \
        integrate:reference
)

echo $reference_id