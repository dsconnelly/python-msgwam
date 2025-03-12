#!/bin/bash

name=$1
dep_arg=$2

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=5:00:00 \
    -a 0-31 \
    -J train-encoding \
    -o logs/$name/train-encoding-%a.out \
    $dep_arg \
    submit.slurm config/$name.toml \
        train-pipeline:encoding:validation
)

echo $job_id