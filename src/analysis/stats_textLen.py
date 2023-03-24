import os.path
import os
from collections import defaultdict
import json

import pandas as pd

from src.utils.load_lists import load_pageTypes
from src.utils.read_files import load_languages

langs = load_languages()


def get_stats_by_lang(lang, query):
    pageTypes = load_pageTypes(query)

    for pageType in pageTypes:
        inputfolder = f"data/preprocessed/{query}/{lang}/{pageType}"
        if os.path.exists(inputfolder) and os.path.isdir(inputfolder):
            pagetype_len = []
            for file in os.listdir(inputfolder):
                filepath = os.path.join(inputfolder, file)
                df = pd.read_csv(filepath, low_memory=False, lineterminator="\n")
                lens = df.LEN.tolist()
                pagetype_len += lens
            if len(pagetype_len) > 0:
                outputfolder = f"data/stats/{query}/"
                if not os.path.exists(outputfolder):
                    os.makedirs(outputfolder)
                outputfile = os.path.join(outputfolder, f"{lang}_{pageType}.json")
                print(f"{pageType} -> {outputfile}")
                with open(outputfile, "w") as f:
                    json.dump({pageType: pagetype_len}, f)


def get_stats_all_langs(query):
    for lang in langs:
        get_stats_by_lang(lang, query)


def main(lang="", query="eu"):
    if lang != "":
        get_stats_by_lang(lang, query)
    else:
        get_stats_all_langs(query)


if __name__ == '__main__':
    import plac

    plac.call(main)
