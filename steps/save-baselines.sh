#!/bin/bash

name=$1
dep_arg=$2

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=3:00:00 \
    -J baselines \
    -o logs/$name/baselines.out \
    $dep_arg \
    submit.slurm config/$name.toml \
        save-strategy:coarse \
        save-strategy:stochastic:1 \
        save-strategy:stochastic:25 \
        save-strategy:instantaneous
)

echo $job_id
