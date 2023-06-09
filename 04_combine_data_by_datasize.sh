#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=stats
#SBATCH --output=%j.out
#SBATCH --time=30:00:00
#SBATCH --mem=64GB

pageType=$1
lang=$2
query=$3


source $HOME/.bashrc
conda activate mg

cd $HOME/mgDialog

python -m src.utils.dataframe_utils "$pageType" "$lang" "$query"


