#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=etm_year
#SBATCH --output=%j.out
#SBATCH --time=30:00:00
#SBATCH --mem=256GB


pageType=$1
lang=$2
query=$3
num_topics=$4
year=$5


source $HOME/.bashrc
conda activate mg

cd $HOME/mgDialog

#python src/topicModeling/skipgram.py --pageType "$pageType" --lang "$lang" --query "$query"

python -m src.topicModeling.main --pageType "$pageType" --lang "$lang" --query "$query"  --num_topics "$num_topics" --year "$year" --epochs 200