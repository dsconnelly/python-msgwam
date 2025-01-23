#!/bin/bash

name=$1
config="config/$name.toml"
cd /home/dsc7746/python-msgwam

if [ -n "$2" ]; then
    dep_arg="--dependency=afterok:$2"
else
    dep_arg=""
fi

mkdir -p data/$name/input
mkdir -p data/$name/training

generate_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=16G \
    --time=24:00:00 \
    --array=0-79 \
    -J generate \
    -o logs/$name/generate-%a.out \
    $dep_arg \
    ml-accel.slurm $config \
        save-training-context \
        save-training-data
)

combine_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=128G \
    --time=1:00:00 \
    -J combine \
    -o logs/$name/combine.out \
    --dependency=afterok:$generate_id \
    ml-accel.slurm $config \
        "combine-data:data/${name}/training/u.npy" \
        "combine-data:data/${name}/training/rays.npy" \
        "combine-data:data/${name}/training/flux-coarse.npy" \
        "combine-data:data/${name}/training/flux-fine.npy"
)

echo $combine_id