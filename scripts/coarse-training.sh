#!/bin/bash

name=$1
config="config/$name.toml"
cd /home/dsc7746/python-msgwam

if [ -n "$2" ]; then
    dep_arg="--dependency=afterok:$2"
else
    dep_arg=""
fi

mkdir -p data/$name/surrogate-coarse

coarse_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=5:00:00 \
    --array=0-191 \
    -J train-surrogate-coarse \
    -o logs/$name/train-surrogate-coarse-%a.out \
    $dep_arg \
    ml-accel.slurm $config train-network:coarse
)

best_coarse_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=5:00:00 \
    -J train-best-surrogate-coarse \
    -o logs/$name/train-best-surrogate-coarse.out \
    --dependency=afterok:$coarse_id \
    ml-accel.slurm $config train-network:coarse:test
)

echo $best_coarse_id
