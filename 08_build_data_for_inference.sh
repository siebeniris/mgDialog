#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=data_infer
#SBATCH --output=%j.out
#SBATCH --time=30:00:00
#SBATCH --mem=256GB


pageType=$1
lang=$2
query=$3


source $HOME/.bashrc
conda activate mg

cd $HOME/mgDialog

#python src/topicModeling/skipgram.py --pageType "$pageType" --lang "$lang" --query "$query"

python -m src.topicModeling.data_build_for_infer --lang "$lang" --query "$query" --pageType "$pageType"