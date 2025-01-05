#!/bin/bash

name="idealized"
config="config/$name.toml"
cd /home/dsc7746/python-msgwam

reference_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --cpus-per-task=8 \
    --time=36:00:00 \
    --mem=128G \
    -J reference \
    -o logs/ml-accel/reference.out \
    ml-accel.slurm $config \
        save-descending-jets \
        save-spectrum \
        integrate:reference \
        save-coarsenings
)
