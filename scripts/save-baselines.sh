#!/bin/bash

name=$1
dep_arg=$2

strats=(
    "ICONlike"
    "MiMAlike"
    "instantaneous"
    "stochastic:25"
    "stochastic:64"
    "stochastic:100"
    "coarse:energy"
    "coarse:flux"
    "coarse:cg_r"
)

cmds=""
args=""

for strat in "${strats[@]}"; do
    dashed="${strat//:/-}"
    cmds="${cmds}save-strategy:${strat} plot-strategy:${dashed} "
    args="${args}${dashed}:"
done

cmds="${cmds%?}"
args="${args%?}"

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=64G \
    --time=4:00:00 \
    -J baselines \
    -o logs/$name/baselines.out \
    $dep_arg \
    submit.slurm config/$name.toml $cmds plot-error-profiles:abs::${args}
)

echo $job_id
