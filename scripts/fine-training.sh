#!/bin/bash

name=$1
config="config/$name.toml"
cd /home/dsc7746/python-msgwam

if [ -n "$2" ]; then
    dep_arg="--dependency=afterok:$2"
else
    dep_arg=""
fi

mkdir -p data/$name/surrogate-fine

fine_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=5:00:00 \
    --array=0-767 \
    -J train-surrogate-fine \
    -o logs/$name/train-surrogate-fine-%a.out \
    $dep_arg \
    ml-accel.slurm $config train-network:fine
)

best_fine_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=5:00:00 \
    -J train-best-surrogate-fine \
    -o logs/$name/train-best-surrogate-fine.out \
    --dependency=afterok:${fine_id} \
    ml-accel.slurm $config train-network:fine:test
)

echo $best_fine_id
