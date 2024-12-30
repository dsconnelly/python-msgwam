#!/bin/bash

name="idealized"
config="config/$name.toml"
cd /home/dsc7746/python-msgwam

generate_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --cpus-per-task=2 \
    --time=12:00:00 \
    --mem=16G \
    -J generate \
    --array=0-39 \
    -o logs/ml-accel/generate-%a.out \
    ml-accel.slurm $config save-training-context save-training-data
)

combine_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --time=1:00:00 \
    --mem=64G \
    -J combine-generate \
    --dependency=afterok:${generate_id} \
    -o logs/ml-accel/combine-generate.out \
    ml-accel.slurm $config \
        "combine-data:data/${name}/training/u.npy" \
        "combine-data:data/${name}/training/rays.npy" \
        "combine-data:data/${name}/training/flux-coarse.npy" \
        "combine-data:data/${name}/training/flux-fine.npy"
)

# proxies_id=$(sbatch \
#     --parsable \
#     --ntasks=1 \
#     --cpus-per-task=2 \
#     --time=6:00:00 \
#     --mem=16G \
#     -J save-proxies \
#     --array=0-39 \
#     -o logs/ml-accel/save-proxies-%a.out \
#     ml-accel.slurm $config save-proxies:fine
# )

# combine_id=$(sbatch \
#     --parsable \
#     --ntasks=1 \
#     --mem=64G \
#     --time=1:00:00 \
#     -J combine-basis \
#     --dependency=afterok:${proxies_id} \
#     -o logs/ml-accel/combine-proxies.out \
#     ml-accel.slurm $config \
#         "combine-data:data/${name}/training/proxies-fine-logistic.npy"
# )

# train_id=$(sbatch \
#     --parsable \
#     --ntasks=1 \
#     --cpus-per-task=2 \
#     --time=6:00:00 \
#     --mem=16G \
#     -J train-network \
#     --array=0-71 \
#     --dependency=afterok:${combine_id} \
#     -o logs/ml-accel/train-surrogate-fine-%a.out \
#     ml-accel.slurm $config train-network:fine
# )

# best_id=$(sbatch \
#     --parsable \
#     --ntasks=1 \
#     --cpus-per-task=2 \
#     --time=6:00:00 \
#     --mem=16G \
#     -J train-best-network \
#     --dependency=afterok:${train_id} \
#     -o logs/ml-accel/train-best-surrogate-fine.out \
#     ml-accel.slurm $config train-network:fine:test
# )