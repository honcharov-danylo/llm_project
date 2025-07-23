#!/bin/bash
cd "$(dirname "$0")"
eval "$(/home/$(whoami)/anaconda3/bin/conda shell.bash hook)"

source /home/dhonchar/.bashrc
export WANDB_PROJECT=eval_model

conda activate eval_model

$@
