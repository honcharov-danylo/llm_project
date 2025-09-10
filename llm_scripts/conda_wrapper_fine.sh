#!/bin/bash
cd "$(dirname "$0")"
# eval "$(/home/$(whoami)/anaconda3/bin/conda shell.bash hook)"
eval "$(/home/dhonchar/anaconda3/bin/conda shell.bash hook)"

source "/home/$(whoami)/.bashrc"
export WANDB_PROJECT=finetune-llm
export CUDA_LAUNCH_BLOCKING=1 # for debugging only, for more accurate stacktracing

conda activate llm_finetuning

$@ 
