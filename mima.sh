#!/bin/bash
set -e

sites=($(ncdump -v site -l 1000 data/mima-scenarios.nc | tail \
    | awk '/site = / {gsub(/[;,"]/, ""); for(i=3; i<=NF; i++) print $i}'))

lats=($(ncdump -v lat -l 1000 data/mima-scenarios.nc \
    | awk '/lat = / {gsub(/[;,]/, ""); for(i=3; i<=NF; i++) print $i}'))

job_ids=()
rnames=()

for i in "${!sites[@]}"; do
    site=${sites[i]}
    lat=${lats[i]}

    cp config/mima-base.toml config/mima-$site.toml
    cp hyperparameters/mima-base.toml hyperparameter/mima-$site.toml
    sed -i "s/^latitude = .*/latitude = $lat/" config/mima-$site.toml

    job_ids+=($(./submit.sh mima-$site save-reference save-coarsenings))
    rnames+=("mima-${site}")
done

dep_list=$(IFS=','; echo "${job_ids[*]}")
rname_arge=$(IFS=':'; echo "${rnames[*]}")

job_id=$(sbatch \
    --parsable \
    --ntasks=1 \
    --mem=32G \
    --time=1:00:00 \
    -J update \
    -o logs/$name/coarsening-update.out \
    --dependency=afterok:${dep_list} \
    submit.slurm config/mima-$site.toml \
        update-config:${rname_args}
)

for i in "${!sites[@]}"; do
    site=${sites[i]}
    ./submit.sh mima-$site save-baselines
done
