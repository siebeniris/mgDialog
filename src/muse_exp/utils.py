import glob

from gensim.models.keyedvectors import KeyedVectors


def convert_w2v_txt(lang, query, pageType):
    for file in glob.glob(f"data/tp/{query}/{lang}/{lang}_{pageType}*/embeddings.wordvectors", recursive=True):
        print(file)
        wv = KeyedVectors.load(file)

        new_file = file + ".txt"
        with open(new_file, "w") as f:
            vocab_len = len(wv.key_to_index)

            f.write(f"{vocab_len} 300\n")
            for word in wv.key_to_index:
                vec = list(wv[word])
                f.write(word + ' ')
                vec_str = ['%.9f' % val for val in vec]
                vec_str = " ".join(vec_str)
                f.write(vec_str + '\n')
        print(f"saved to {new_file}")


if __name__ == '__main__':
    import plac

    plac.call(convert_w2v_txt)

