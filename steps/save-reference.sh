#!/bin/bash

name=$1
dep_arg=$2

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=128G \
    --cpus-per-task=8 \
    --time=36:00:00 \
    -J reference \
    -o logs/$name/reference.out \
    $dep_arg \
    submit.slurm config/$name.toml \
        save-mean-state:descending-jets \
        save-spectrum \
        save-integration:reference
)

echo $job_id