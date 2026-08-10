#!/bin/bash

dep_arg=""
function make_dep() {
    echo "--dependency=afterok:$1"
}

while getopts "d:" opt; do
    case $opt in
        d)
            dep_arg=$(make_dep $OPTARG)
            ;;
        \?)
            echo "Invalid option: -$OPTARG"
            exit 1
            ;;
    esac
done

shift $((OPTIND-1))

if [ $# -lt 1 ]; then
    echo "Error: must specify config name"
    exit 1
fi

name=$1
shift
tasks=("$@")

cd /home/dsc7746/python-msgwam

mkdir -p data/$name/grid-search-coarse
mkdir -p data/$name/grid-search-mima
mkdir -p data/$name/input
mkdir -p data/$name/strategies
mkdir -p data/$name/training

mkdir -p logs/$name
mkdir -p plots/$name

for task in "${tasks[@]}"; do
    job_id=$("scripts/$task.sh" $name $dep_arg)
    dep_arg=$(make_dep $job_id)
done

echo $job_id
