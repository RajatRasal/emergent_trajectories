#!/bin/bash

timesteps=(100 200 300 400)
batch_sizes=(32 64 128 256)

for t in "${timesteps[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
    echo "Running with timesteps=$t and batch_size=$batch_size"
    sbatch ./train_job.sh $batch_size $t
  done
done