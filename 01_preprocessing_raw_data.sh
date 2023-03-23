#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=preprocess
#SBATCH --output=%j.out
#SBATCH --time=30:00:00
#SBATCH --mem=256GB


query=$1
lang=$2

source $HOME/.bashrc
conda activate mg

cd $HOME/mgDialog

python src/preprocessor/preprocessing "$lang" "$query"


