#!/bin/bash

langs='cs de es hu it nl pl sk sv'

for lang in $langs
do
  echo $lang
  python src/muse_exp/get_muse_vecs.py $lang eu Twitter
done


langs2='cs da de es hu it pl sk sv'

for lang in $langs2
do
  echo $lang
  python src/muse_exp/get_muse_vecs.py $lang un Twitter
done

