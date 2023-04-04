import os
import pickle
import argparse

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import savemat

parser = argparse.ArgumentParser(description='Building data for inferring topics')

# data and file related arguments
parser.add_argument('--lang', type=str, default="en", help="The language of the data")
parser.add_argument('--query', type=str, default='eu', help='The name of corpus, either eu or un')
parser.add_argument("--pageType", type=str, default="Twitter", help="PageType of the data data/tp/eu/cs_Twitter")
parser.add_argument("--year", type=str, default=None, help="The year of the data data/tp/eu/cs_Twitter_2014")
parser.add_argument('--month', type=str, default=None, help="The month of language data/tp/eu/cs_Twitter_2014-01/")

args = parser.parse_args()

#############
# get the data directory to save and extract data.
if args.year is not None:
    data_dir = f"data/tp/{args.query}/{args.lang}/{args.lang}_{args.pageType}_{args.year}"
elif args.month is not None:
    data_dir = f"data/tp/{args.query}/{args.lang}/{args.lang}_{args.pageType}_{args.year}-{args.month}"
else:
    data_dir = f"data/tp/{args.query}/{args.lang}/{args.lang}_{args.pageType}"

# file path and data dir
# data_dir = f'output/preprocessed/forTp/'

vocab_filepath = os.path.join(data_dir, 'vocab.pkl')

with open(vocab_filepath, 'rb') as f:
    vocab = pickle.load(f)

word2id = dict([(w, j) for j, w in enumerate(vocab)])
id2word = dict([(j, w) for j, w in enumerate(vocab)])

# to infer file

to_infer_filepath = os.path.join(data_dir, f'{args.lang}_{args.pageType}.csv') # sv_Twitter.csv
df = pd.read_csv(to_infer_filepath, index_col=0)

docs = df.preprocessed_text.tolist()
num_docs = len(docs)

print('len of docs:', num_docs)

data = [[word2id[w] for w in docs[idx_d].split() if w in word2id] for idx_d in range(num_docs)]

print('data size: ', len(data))

df['non_empty_data_tm'] = data

# non-empty but can be duplicated.
df_ = df[df['non_empty_data_tm'].map(lambda d: len(d) > 0)]

print('non empty data tm: ', len(df_))

print(df_['non_empty_data_tm'].tolist()[:2])

# df_preprocessed....
df_.to_csv(os.path.join(data_dir, f'{args.lang}_{args.pageType}_non_empty.csv'))

data = df_['non_empty_data_tm'].tolist()

print('after removing empty data:', len(data))


def create_list_words(in_docs):
    return [x for y in in_docs for x in y]


print('creating lists of words...')
words_data = create_list_words(data)


def create_doc_indices(in_docs):
    aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
    return [int(x) for y in aux for x in y]


doc_indices = create_doc_indices(data)
print(len(np.unique(doc_indices)), len(data))


# compressed
def create_bow(doc_indices, words, n_docs, vocab_size):
    return sparse.coo_matrix(([1] * len(doc_indices), (doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()


print('vocab size:', len(vocab))

num_data = len(data)
print('creating bow representation...')

bow_data = create_bow(doc_indices, words_data, num_data, len(vocab))
print('splitting bow intro token/value pairs and saving to disk...')


def split_bow(bow_in, n_docs):
    indices = np.array([[w for w in bow_in[doc, :].indices] for doc in range(n_docs)], dtype=object)
    counts = np.array([[c for c in bow_in[doc, :].data] for doc in range(n_docs)], dtype=object)
    return indices, counts


bow_tokens, bow_counts = split_bow(bow_data, num_data)

savemat(os.path.join(data_dir, 'data_tokens.mat'), {'tokens': bow_tokens}, do_compression=True)
savemat(os.path.join(data_dir, 'data_counts.mat'), {'counts': bow_counts}, do_compression=True)

print('data ready!')