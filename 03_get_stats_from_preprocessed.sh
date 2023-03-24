#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=stats
#SBATCH --output=%j.out
#SBATCH --time=30:00:00
#SBATCH --mem=64GB


lang=$1
query=$2


source $HOME/.bashrc
conda activate mg

cd $HOME/mgDialog

python -m src.analysis.stats_textLen "$lang" "$query"


