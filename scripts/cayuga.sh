#!/bin/bash

name="cayuga"
config="config/$name.toml"
cd /home/dsc7746/python-msgwam

mkdir -p data/$name/coarsenings
mkdir -p data/$name/input
mkdir -p data/$name/strategies
mkdir -p data/$name/surrogate-coarse
mkdir -p data/$name/surrogate-fine
mkdir -p data/$name/training

# reference_id=$(sbatch \
#     --parsable \
#     --ntasks=1 \
#     --cpus-per-task=8 \
#     --time=36:00:00 \
#     --mem=128G \
#     -J reference \
#     -o logs/${name}/reference.out \
#     ml-accel.slurm $config \
#         save-descending-jets \
#         save-spectrum \
#         integrate:reference \
#         save-coarsenings \
#         update-config
# )

coarsenings_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --time=1:00:00 \
    --mem=16G \
    -J coarsening \
    --array=0-99 \
    -o logs/${name}/coarsening-%a.out \
    ml-accel.slurm $config save-coarsenings
)

# generate_id=$(sbatch \
#     --parsable \
#     --ntasks=1 \
#     --time=24:00:00 \
#     --mem=16G \
#     -J generate \
#     --array=0-79 \
#     -o logs/${name}/generate-%a.out \
#     --dependency=afterok:${reference_id} \
#     ml-accel.slurm $config \
#         save-training-context \
#         save-training-data
# )

# combine_id=$(sbatch \
#     --parsable \
#     --ntasks=1 \
#     --time=1:00:00 \
#     --mem=128G \
#     -J combine-generate \
#     -o logs/${name}/combine-generate.out \
#     --dependency=afterok:${generate_id} \
#     ml-accel.slurm $config \
#         "combine-data:data/${name}/training/u.npy" \
#         "combine-data:data/${name}/training/rays.npy" \
#         "combine-data:data/${name}/training/flux-coarse.npy" \
#         "combine-data:data/${name}/training/flux-fine.npy"
# )

# fine_id=$(sbatch \
#     --parsable \
#     --ntasks=1 \
#     --cpus-per-task=2 \
#     --time=5:00:00 \
#     --mem=32G \
#     -J train-fine-surrogate \
#     --array=0-1535 \
#     -o logs/${name}/train-surrogate-fine-%a.out \
#     --dependency=afterok:${combine_id} \
#     ml-accel.slurm $config train-network:fine
# )

# coarse_id=$(sbatch \
#     --parsable \
#     --ntasks=1 \
#     --cpus-per-task=2 \
#     --time=5:00:00 \
#     --mem=32G \
#     -J train-coarse-surrogate \
#     --array=0-1535 \
#     -o logs/${name}/train-surrogate-coarse-%a.out \
#     --dependency=afterok:${combine_id} \
#     ml-accel.slurm $config train-network:coarse
# )