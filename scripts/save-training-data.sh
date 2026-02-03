#!/bin/bash

name=$1
dep_arg=$2

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=4:00:00 \
    -a 0-287 \
    -J ml-accel-training-data \
    -o logs/$name/save-training-data-%a.out \
    $dep_arg \
    submit.slurm config/$name.toml \
        save-training-data
)

dep_arg="--dependency=afterok:$job_id"

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=8G \
    --time=4:30:00 \
    -a 0-71 \
    -J ml-accel-cache-arrays \
    -o logs/$name/cache-arrays-%a.out \
    $dep_arg \
    submit.slurm config/$name.toml \
        cache-arrays:6:compute
)

dep_arg="--dependency=afterok:$job_id"

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=128G \
    --time=2:00:00 \
    -J ml-accel-cache-arrays \
    -o logs/$name/cache-arrays-combine.out \
    $dep_arg \
    submit.slurm config/$name.toml \
        cache-arrays:6:combine
)

echo $job_id
