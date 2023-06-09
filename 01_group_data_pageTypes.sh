#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=preprocess
#SBATCH --output=%j.out
#SBATCH --time=30:00:00
#SBATCH --mem=256GB


lang=$1
query=$2


source $HOME/.bashrc
conda activate mg

cd $HOME/mgDialog

python src/preprocessor/groupbyTypeName.py "$lang" "$query"


