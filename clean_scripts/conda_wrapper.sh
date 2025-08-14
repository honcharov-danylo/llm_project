#!/bin/bash
cd "$(dirname "$0")"
eval "$(/home/$(whoami)/anaconda3/bin/conda shell.bash hook)"

source "/home/$(whoami)/.bashrc"

conda activate llm_project

$@ 
