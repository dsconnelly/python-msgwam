#!/bin/bash

name="idealized"
config="config/$name.toml"
cd /home/dsc7746/python-msgwam

generate_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --cpus-per-task=8 \
    --time=2:00:00 \
    --mem=16G \
    -J generate \
    --array=0-10 \
    -o logs/ml-accel/generate-%a.out \
    ml-accel.slurm $config save-training-context save-training-data
)

combine_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --time=2:00:00 \
    --mem=64G \
    -J combine-generate \
    --dependency=afterok:${generate_id} \
    -o logs/ml-accel/combine-generate.out \
    ml-accel.slurm $config \
        "combine-data:data/${name}/training/u.npy" \
        "combine-data:data/${name}/training/rays.npy" \
        "combine-data:data/${name}/training/Y-coarse.npy" \
        "combine-data:data/${name}/training/Y-fine.npy"
)

# basis_id=$(sbatch \
#     --parsable \
#     --ntasks=1 \
#     --cpus-per-task=8 \
#     --time=2:00:00 \
#     --mem=16G \
#     -J save-basis \
#     --array=0-19 \
#     -o logs/ml-accel/save-basis-%a.out \
#     --dependency=afterok:${combine_id} \
#     ml-accel.slurm $config \
#         save-basis-coefficients:coarse \
#         save-basis-coefficients:fine
# )

# combine_id=$(sbatch \
#     --parsable \
#     --n-tasks=1 \
#     --mem=128G \
#     --time=2:00:00 \
#     -J combine-basis \
#     --dependency=afterok:${basis_id} \
#     -o logs/ml-accel/combine-basis.out \
#     ml-accel.slurm $config \
#         "combine-data:data/${name}/training/coeffs-fine-logistic.npy" \
#         "combine-data:data/${name}/training/coeffs-coarse-logistic.npy"
# )