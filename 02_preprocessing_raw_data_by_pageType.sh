#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=preprocess
#SBATCH --output=%j.out
#SBATCH --time=30:00:00
#SBATCH --mem=256GB

pageType=$1
lang=$2
query=$3


source $HOME/.bashrc
conda activate mg

cd $HOME/mgDialog

python -m src.preprocessor.preprocessing "$pageType" "$lang" "$query"


