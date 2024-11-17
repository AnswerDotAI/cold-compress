#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32gb
#SBATCH --partition scavenger
#SBATCH --account scavenger
#SBATCH --qos scavenger
#SBATCH -t 04:00:00

# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export NCCL_IB_DISABLE=1

module add cuda
module add Python3

# ulimit -n 2048
source /fs/nexus-scratch/minghui/cold_compress/.venv/bin/activate


python3 eval.py --cache_config l2 --task gsm
python3 eval.py --cache_config l2sh --task gsm
