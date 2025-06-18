#!/bin/bash
eval "$(/home/$(whoami)/anaconda3/bin/conda shell.bash hook)" 

source /home/dhonchar/.bashrc

conda activate llm_project

$@ 
