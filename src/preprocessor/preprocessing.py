import os
import string
import unicodedata

import numpy as np
import pandas as pd
from simplemma import text_lemmatizer
import stopwordsiso
from tqdm import tqdm
from pandarallel import pandarallel

from yaml import load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from src.preprocessor.defines import *
from src.utils.load_lists import load_pageTypes

pandarallel.initialize()

SPECIAL_CHARS = ['&nbsp;', '&lt;', '&gt;', '&amp;', '&quot;', '&apos;', '&cent;', '&pound;', '&yen;', '&euro;',
                 '&copy;', '&reg;']


def cleaning_text_for_tp(text, stopwords, lang):
    # field: fullText
    # blogs and forum, YouTube have longer content
    # clean hashtags,

    # clean
    text = Patterns.URL_PATTERN.sub(r'', text)
    text = Patterns.RESERVED_WORDS_PATTERN.sub(r'', text)
    text = Patterns.SMILEYS_PATTERN.sub(r'', text)
    text = Patterns.EMOJIS_PATTERN.sub(r'', text)
    text = Patterns.NUMBERS_PATTERN.sub(r'', text)
    text = Patterns.MENTION_PATTERN.sub(r'', text)

    # remove special chars, starting with &.
    for CHAR in SPECIAL_CHARS:
        text = text.replace(CHAR, '')

    # unicode.
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))

    # lower-cased, tokenized and lemmatized
    text_lemmatized = text_lemmatizer(text.lower(), lang=lang)

    # remove stopwords
    inter = set(text_lemmatized).intersection(stopwords)
    for i in inter:
        text_lemmatized.remove(i)

    if len(text) > 1:
        return " ".join(text_lemmatized).lower()
    else:
        return np.NaN


def create_dir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)


def processing_by_lang(pageType, lang, query="eu"):
    folderdir = os.path.join(f"data/pageTypes/{query}/{lang}/{pageType}")
    outputdir_query = os.path.join("data/preprocessed", query)
    create_dir(outputdir_query)

    outputdir_lang = os.path.join("data/preprocessed", query, lang)
    create_dir(outputdir_lang)

    outputdir_pageType = os.path.join("data/preprocessed", query, lang, pageType)
    create_dir(outputdir_pageType)

    stopwords = stopwordsiso.stopwords(lang)

    for file in tqdm(os.listdir(folderdir)):
        if file.endswith(".csv"):
            filepath = os.path.join(folderdir, file)
            outputfile = os.path.join(outputdir_pageType, file)

            if not os.path.exists(outputfile):

                print(f"processing file {filepath}")
                df = pd.read_csv(filepath, low_memory=False, lineterminator="\n")

                df["preprocessed_text"] = df["fullText"].parallel_apply(cleaning_text_for_tp,
                                                                        args=(stopwords, lang))

                df = df.dropna(subset=["preprocessed_text"])
                df["LEN"] = df["preprocessed_text"].astype(str).str.split(" ").str.len()
                df = df[["resourceId", "date", "preprocessed_text", "countryCode", "pageTypeName", "month", "fullText",
                         "LEN"]]

                print(f"output to {outputfile}")
                df.to_csv(outputfile, index=False)

        else:
            print(
                f"outputfile {outputfile} exists!"
            )


def preprocessing_all_langs(pageType, query="eu"):
    with open("data/config.yaml") as f:
        langs = load(f, Loader=Loader)["langs"]

    for lang in langs:
        print("processing lang ", lang)
        processing_by_lang(pageType, lang, query)


def main(pageType, lang="", query="eu"):
    if lang != "":
        processing_by_lang(pageType, lang, query)
    else:

        preprocessing_all_langs(pageType, query)


if __name__ == '__main__':
    import plac

    plac.call(main)
