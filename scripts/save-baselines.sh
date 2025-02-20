#!/bin/bash

name=$1
dep_arg=$2

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=4:00:00 \
    -J baselines \
    -o logs/$name/baselines.out \
    $dep_arg \
    submit.slurm config/$name.toml \
        save-strategy:coarse \
        save-strategy:stochastic:1 \
        save-strategy:stochastic:25 \
        save-strategy:stochastic:100 \
        save-strategy:instantaneous \
        plot-strategy:coarse \
        plot-strategy:stochastic:1 \
        plot-strategy:stochastic:25 \
        plot-strategy:stochastic:100 \
        plot-strategy:instantaneous \
        plot-error-profiles:coarse:stochastic-1:stochastic-25:stochastic-100:instantaneous
)

echo $job_id
