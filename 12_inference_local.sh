#!/bin/bash

# cs, de
langs='da es hu it nl pl sk sv'

for lang in $langs
do
  echo $lang
  python -m src.topicModeling.data_build_for_infer --lang $lang --query eu --pageType Twitter
done


langs2='cs da de es hu it pl sk sv'

for lang in $langs2
do
  echo $lang
  python -m src.topicModeling.data_build_for_infer --lang $lang --query un --pageType Twitter
done

