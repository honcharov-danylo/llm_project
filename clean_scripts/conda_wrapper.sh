#!/bin/bash
cd "$(dirname "$0")"
mkdir -p logs/
eval "$(/home/$(whoami)/anaconda3/bin/conda shell.bash hook)"

source "/home/$(whoami)/.bashrc"

conda activate data_cleaning

$@ 
