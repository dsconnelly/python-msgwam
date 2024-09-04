#!/bin/bash

fname="logs/testing/case-$1.out"
sbatch -o "$fname" submit.slurm "$@"