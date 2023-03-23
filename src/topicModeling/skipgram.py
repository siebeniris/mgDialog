import gensim
import os
import pandas as pd
import argparse
import multiprocessing

from yaml import load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

cpu_units = multiprocessing.cpu_count()
parser = argparse.ArgumentParser(description='The Embedded Topic Model')

### data and file related arguments
parser.add_argument('--lang', type=str, default="en")
parser.add_argument('--query', type=str, default="eu")

parser.add_argument('--dim_rho', type=int, default=300, help='dimensionality of the word embeddings')
parser.add_argument('--min_count', type=int, default=2, help='minimum term frequency (to define the vocabulary)')
parser.add_argument('--sg', type=int, default=1, help='whether to use skip-gram')
parser.add_argument('--workers', type=int, default=cpu_units, help='number of CPU cores')
parser.add_argument('--negative_samples', type=int, default=10, help='number of negative samples')
parser.add_argument('--window_size', type=int, default=4, help='window size to determine context')
parser.add_argument('--iters', type=int, default=20, help='number of iterationst')

args = parser.parse_args()


# Class for a memory-friendly iterator over the dataset
class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        df = pd.read_csv(self.filename)
        for line in df.preprocessed_text:
            yield line.split()


def processing_one_file(file, outputfile):
    sentences = MySentences(filename=file)

    print('start training skipgram word embeddings....')
    print(f"open {file} ....")
    model = gensim.models.Word2Vec(sentences, min_count=args.min_count, sg=args.sg, vector_size=args.dim_rho,
                                   epochs=args.iters, workers=args.workers, negative=args.negative_samples,
                                   window=args.window_size)

    print(f"writing to {outputfile}")
    word_vectors = model.wv
    word_vectors.save(outputfile)


def processing_files_by_lang(lang, query):
    data_folder = os.path.join(f"data/tp/")
    lang_folder = os.path.join(data_folder, query, lang)
    print(lang_folder)
    for foldername in os.listdir(lang_folder):
        month_folder = os.path.join(lang_folder, foldername)
        csv_file = os.path.join(month_folder, f"{foldername}.csv")
        outputfile = os.path.join(month_folder, "embeddings.wordvectors")
        if not os.path.exists(outputfile):
            processing_one_file(csv_file, outputfile)
        else:
            print(f"outputfile {outputfile} exists!")


def preprocessing_all_langs(query="eu"):
    with open("data/config.yaml") as f:
        langs = load(f, Loader=Loader)["langs"]

    for lang in langs:
        print("processing lang ", lang)
        processing_files_by_lang(lang, query)


def main(lang="", query="eu"):
    if lang != "":
        processing_files_by_lang(lang, query)
    else:

        preprocessing_all_langs(query)


if __name__ == '__main__':
    main(args.lang, args.query)

