#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=muse
#SBATCH --output=%j.out
#SBATCH --time=30:00:00
#SBATCH --mem=64GB


lang=$1
SRCEMB=$2
TGTEMB=$3

source $HOME/.bashrc
conda activate muse

cd $HOME/mgDialog/MUSE


python -m unsupervised --src_lang "$lang" --tgt_lang "$lang" --src_emb "$SRCEMB" --tgt_emb "$TGTEMB" --n_refinement 5 --batch_size 128


