#!/bin/bash
# #SBATCH -p gpus24
# #SBATCH --gres gpu:1
# #SBATCH --output=./slurm_logs/shapes/slurm.%N.%j.log
# 
# export PATH="/vol/biomedic3/rrr2417/.local/bin:$PATH"
# export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

poetry run python3 -m train \
    --lrate 1e-4 \
    --weight_decay 1e-5 \
    --n_sample 50 \
    --seed $1 \
    --batch_size 64 \
    --n_T 400 \
    --pixel_size 28 \
    --scheduler DDPM \
    --max_steps 20000 \
    --n_feat 64