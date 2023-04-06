from glob import glob
from gensim.models.keyedvectors import KeyedVectors
import pickle
import plac

def load_muse_vecs(lang):
    data_folder = "data/fasttext"
    model = KeyedVectors.load_word2vec_format(f"{data_folder}/wiki.multi.{lang}.vec")
    return model


def get_vocab_all(lang, query, pageType="Twitter"):
    # filter the fasttext embeddings with existing vocab

    vocab_all = []
    for file in glob(f"data/tp/{query}/{lang}/{lang}_{pageType}*/vocab.pkl", recursive=True):
        print(file)
        with open(file, "rb") as f:
            vocab = pickle.load(f)
            vocab_all += vocab
    vocab_all = list(set(vocab_all))
    return vocab_all


def get_overlapping_vecs(lang, query, pageType):
    vocab = get_vocab_all(lang, query, pageType)
    model = load_muse_vecs(lang)
    vocab_overlap = len(set(list(model.key_to_index.keys())).intersection(set(vocab)))
    print(f"{lang}->{query}->{pageType}: vocab len {len(vocab)}")
    vec_dim = 300
    counter = 0
    with open(f"data/fasttext/{lang}_{query}_{pageType}.vec", "w") as f:
        f.write(f"{vocab_overlap} {vec_dim}\n")
        for word in vocab:
            try:
                vec = list(model[word])
                f.write(word + ' ')
                vec_str = ['%.9f' % val for val in vec]
                vec_str = " ".join(vec_str)
                f.write(vec_str + '\n')
            except Exception as msg:
                counter += 1
    print(f"OOV : {counter}")


if __name__ == '__main__':
    plac.call(get_overlapping_vecs)
