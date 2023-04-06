# Experimenting with MUSE

1. trim MUSE fasttext embeddings to the vocab of we have 
`python src/muse_exp/get_muse_vecs.py #lang #query #pageType`

output are in the folder `data/fasttext/de_eu_Twitter.vec`

2. convert gensim wv format to txt:
`python src/muse_exp/utils.py #lang #query #pageType`

3. run MUSE on the two vecs.
`python -m unsupervised --src_lang da --tgt_lang da --src_emb ../data/tp/eu/da/da_Twitter/embeddings.wordvectors.txt --tgt_emb ../data/fasttext/da_eu_Twitter.vec --n_refinement 5`



# install start
- in LINUX
- python=3.9
- install numpy, scipy
- install pytorch: `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
- install faiss-gpu: `conda install -c pytorch faiss-gpu`

# transfer data to server.
