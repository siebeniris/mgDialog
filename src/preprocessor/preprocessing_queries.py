import json
import os
import string
import unicodedata

import numpy as np
from simplemma import text_lemmatizer
import stopwordsiso

from yaml import load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from src.preprocessor.defines import *


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

    extra = {"links", "or", "site", "and", "for", "link", "det", "den", "der", "das", "die" }
    texts = list(set(text_lemmatized).difference(extra))
    texts = [x for x in texts if len(x)>1]

    if len(text) > 1:
        return texts
    else:
        return np.NaN


def processing_by_lang(lang, query="eu"):
    inputfile = os.path.join(f"data/queries/{query}/{lang}.txt")

    with open(inputfile) as f:
        words = f.read()

    stopwords = stopwordsiso.stopwords(lang)
    texts = cleaning_text_for_tp(words, stopwords, lang)

    outputdir = f"data/keywords/{query}/"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    outputfile = os.path.join(outputdir, f"{lang}.json")

    with open(outputfile, "w") as f:
        json.dump(texts, f)


def preprocessing_all_langs(query="eu"):
    with open("data/config.yaml") as f:
        langs = load(f, Loader=Loader)["langs"]

    for lang in langs:
        print("processing lang ", lang)
        processing_by_lang(lang, query)


def main(lang="", query="eu"):
    if lang != "":
        processing_by_lang(lang, query)
    else:

        preprocessing_all_langs(query)


if __name__ == '__main__':
    import plac

    plac.call(main)
