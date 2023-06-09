#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --job-name=muse
#SBATCH --output=%j.out
#SBATCH --time=30:00:00
#SBATCH --mem=32GB
#SBATCH --nodelist=nv-ai-01.srv.aau.dk


lang=$1
SRCEMB=$2
TGTEMB=$3
CUDA=$4

source $HOME/.bashrc
conda activate muse

cd $HOME/mgDialog/MUSE


python -m unsupervised --src_lang "$lang" --tgt_lang "$lang" --src_emb "$SRCEMB" --tgt_emb "$TGTEMB" --n_refinement 5 --cuda "$CUDA"


