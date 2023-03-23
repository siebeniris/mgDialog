import gensim
import os
import pandas as pd
import argparse
import multiprocessing

cpu_units = multiprocessing.cpu_count()
parser = argparse.ArgumentParser(description='The Embedded Topic Model')

### data and file related arguments
# de/el/en/es/fi/fr/hu/it/nl/pl/sv
parser.add_argument('--lang', type=str, default="sv")
parser.add_argument('--data_folder', type=str, default='output/preprocessed/forTP/',
                    help='a .txt file containing the corpus')
parser.add_argument('--dim_rho', type=int, default=300, help='dimensionality of the word embeddings')
parser.add_argument('--min_count', type=int, default=2, help='minimum term frequency (to define the vocabulary)')
parser.add_argument('--sg', type=int, default=1, help='whether to use skip-gram')
parser.add_argument('--workers', type=int, default=cpu_units, help='number of CPU cores')
parser.add_argument('--negative_samples', type=int, default=10, help='number of negative samples')
parser.add_argument('--window_size', type=int, default=4, help='window size to determine context')
parser.add_argument('--iters', type=int, default=20, help='number of iterationst')

args = parser.parse_args()

data_file = os.path.join(args.data_folder, args.lang, f'{args.lang}_built.csv')
emb_file = os.path.join(args.data_folder, args.lang, 'embeddings.wordvectors')


# Class for a memory-friendly iterator over the dataset
class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        df = pd.read_csv(self.filename)
        for line in df.preprocessed_text:
            yield line.split()


sentences = MySentences(filename=data_file)

print(list(sentences)[:10])
print('start training skipgram word embeddings....')
print(f"open {data_file} ....")
model = gensim.models.Word2Vec(sentences, min_count=args.min_count, sg=args.sg, vector_size=args.dim_rho,
                               epochs=args.iters, workers=args.workers, negative=args.negative_samples,
                               window=args.window_size)

# Write the embeddings to a file
print(f'writing to {emb_file}')
word_vectors = model.wv
word_vectors.save(emb_file)
