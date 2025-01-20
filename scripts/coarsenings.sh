#!/bin/bash

name=$1
config="config/$name.toml"
cd /home/dsc7746/python-msgwam

if [ -n "$2" ]; then
    dep_arg="--dependency=afterok:$2"
else
    dep_arg=""
fi

mkdir -p data/$name/coarsenings
mkdir -p plots/$name

coarsenings_id=$(sbatch \
    $dep_arg \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=1:00:00 \
    --array=0-99 \
    -J coarsening \
    -o logs/$name/coarsening-%a.out \
    $dep_arg \
    ml-accel.slurm $config save-coarsenings
)

update_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=1:00:00 \
    -J coarsening-update \
    -o logs/$name/coarsening-update.out \
    --dependency=afterok:$coarsenings_id \
    ml-accel.slurm $config \
        plot-coarse-errors \
        update-config
)

echo $update_id