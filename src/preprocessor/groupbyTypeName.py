import pandas as pd
import os
from tqdm import tqdm


from yaml import load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper



pageTypes_eu = ["Blogs", "Forums", "Instagram", "News", "Reddit", "Review", "Tumblr", "Twitter", "YouTube"]
# review only for english

pageTypes_un = ["Blogs", "Facebook", "Forums", "Instagram", "News", "Reddit", "Tumblr", "Twitter", "YouTube"]


# facebook only for en

def create_dir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)


def processing_one_file(file, lang="en", query="eu"):
    # take from raw data
    df = pd.read_csv(file, index_col=0, lineterminator="\n", low_memory=False)
    fields = ["date", "resourceId", "countryCode", "fullText", "pageTypeName"]
    df = df[fields].dropna(subset=["fullText"])
    df["month"] = df["date"].str[:7]

    outputdir = "data/pageTypes"
    create_dir(outputdir)

    filename = os.path.basename(file)
    if query == "eu":
        pageTypes = pageTypes_eu
    else:
        pageTypes = pageTypes_un

    outputdir_query = os.path.join(outputdir, query)
    create_dir(outputdir_query)

    outputdir_lang = os.path.join(outputdir_query, lang)
    create_dir(outputdir_lang)

    for type in pageTypes:
        outputdir_type = os.path.join(outputdir_lang, type)
        create_dir(outputdir_type)
        try:

            df_type = df[df["pageTypeName"]==type]
            if len(df_type) > 0:
                df_type.to_csv(os.path.join(outputdir_type, filename), index=False)
                print(f"{lang} -> {type} ->{filename} -> {len(df_type)}")
        except Exception as msg:
            print(f"exception {msg}")


def processing_by_lang(lang, query="eu"):
    folderdir = f'data/raw/{query}/{lang}'

    for file in tqdm(os.listdir(folderdir)):
        filepath = os.path.join(folderdir, file)
        processing_one_file(filepath, lang, query)


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
