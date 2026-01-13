#!/bin/bash

sbatch \
    -A ${JOB_ACC} \
    -n 1 \
    --cpus-per-task=${JOB_CPU:-16} \
    --mem-per-cpu=${JOB_MEM:-2048} \
    --gpus=rtx_4090:1 \
    --time=${JOB_TIME:-"48:00:00"} \
    --job-name=batch_$1 \
    cluster_scripts/batch_$1.sh