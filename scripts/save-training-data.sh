#!/bin/bash

name=$1
dep_arg=$2

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=12:00:00 \
    -a 0-24 \
    -J generation \
    -o logs/$name/generation-%a.out \
    $dep_arg \
    submit.slurm config/$name.toml \
        save-training-context \
        save-training-data
)

echo $job_id