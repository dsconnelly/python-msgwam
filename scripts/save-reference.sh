#!/bin/bash

name=$1
dep_arg=$2

if [[ $name == icon* ]]; then
    arg=""
else
    arg="save-mean-state:gated-oscillation"
fi

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=128G \
    --cpus-per-task=8 \
    --time=18:00:00 \
    -J reference \
    -o logs/$name/reference.out \
    $dep_arg \
    submit.slurm config/$name.toml \
        $arg \
        save-spectrum \
        plot-spectrum \
        plot-mean-state \
        save-strategy:reference \
        plot-strategy:reference
)

echo $job_id
