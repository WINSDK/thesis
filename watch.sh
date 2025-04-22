#!/bin/bash
set -euo pipefail

rm -f slurm*
sbatch finetune.job
while ! tail -f -slurm* ; do sleep 1 ; done
