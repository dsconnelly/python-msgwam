#!/bin/bash
set -e

sites=($(ncdump -v site -l 1000 data/mima-scenarios.nc | tail \
    | awk '/site = / {gsub(/[;,"]/, ""); for(i=3; i<=NF; i++) print $i}'))

lats=($(ncdump -v lat -l 1000 data/mima-scenarios.nc \
    | awk '/lat = / {gsub(/[;,]/, ""); for(i=3; i<=NF; i++) print $i}'))

is_calibration() {
    case "$1" in
        "copenhagen"   | \
        "new-york"     | \
        "miami"        | \
        "singapore"    | \
        "santiago"     | \
        "amundsen-sea" )
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

job_ids=()
rnames=()

for i in "${!sites[@]}"; do
    site=${sites[i]}
    lat=${lats[i]}

    lat_tropics="25"
    abs_lat=$(echo "if ($lat < 0) -1 * $lat else $lat" | bc)
    if (( $(echo "$abs_lat > $lat_tropics" | bc) )); then
        extr="true"
    else
        extr="false"
    fi

    if ! is_calibration "$site"; then
        continue
    fi

    cp config/mima-base.toml config/mima-$site.toml
    cp hyperparameters/mima-base.toml hyperparameters/mima-$site.toml
    sed -i "s/^latitude = .*/latitude = $lat/" config/mima-$site.toml
    sed -i "s/^extrinsic = .*/extrinsic = $extr/" config/mima-$site.toml

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
    -J group-update \
    -o logs/mima-$site/coarsening-update.out \
    --dependency=afterok:${dep_list} \
    submit.slurm config/mima-$site.toml \
        update-config:${rname_args}
)

for i in "${!sites[@]}"; do
    site=${sites[i]}

    if ! is_calibration "$site"; then
        continue
    fi

    ./submit.sh -d $job_id mima-$site save-baselines
done
