#!/bin/bash

name=$1
dep_arg=$2

if [[ $name == icon* ]]; then
    arg=""
else
    arg="save-mean-state:descending-jets"
fi

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=128G \
    --cpus-per-task=8 \
    --time=24:00:00 \
    -J reference \
    -o logs/$name/reference.out \
    $dep_arg \
    submit.slurm config/$name.toml \
        $arg \
        save-spectrum \
        save-strategy:reference \
        plot-strategy:reference
)

echo $job_id
