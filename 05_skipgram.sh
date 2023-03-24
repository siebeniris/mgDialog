#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=skipgram
#SBATCH --output=%j.out
#SBATCH --time=30:00:00
#SBATCH --mem=256GB


lang=$1
query=$2


source $HOME/.bashrc
conda activate mg

cd $HOME/mgDialog

python src/topicModeling/skipgram.py --lang "$lang" --query "$query"

