#!/bin/bash

name=$1
dep_arg=$2

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=64G \
    --time=4:00:00 \
    -J baselines \
    -o logs/$name/baselines.out \
    $dep_arg \
    submit.slurm config/$name.toml \
        save-strategy:ICONlike \
        save-strategy:coarse \
        save-strategy:stochastic:25 \
        save-strategy:instantaneous \
        plot-strategy:ICONlike \
        plot-strategy:coarse \
        plot-strategy:stochastic-25 \
        plot-strategy:instantaneous \
        plot-error-profiles:ICONlike:instantaneous:coarse:stochastic-25
)

echo $job_id
