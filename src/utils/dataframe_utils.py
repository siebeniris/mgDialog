import os
from collections import defaultdict

import pandas as pd

from src.utils.read_files import load_datasize_twitter, load_datasize_news


def combine_data_by_size(pageType, lang, query):
    # first deal with twitter, other types needs to be sentence tokenized.
    inputfolder = f"data/preprocessed/{query}/{lang}/{pageType}"
    outputfolder = f"data/tp/{query}/{lang}/"
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)


    # lang_pageType_year
    years = [2014, 2015, 2018, 2019, 2020]

    if pageType == "Twitter":
        datasize = load_datasize_twitter(lang, query)

    elif pageType == "News":
        datasize = load_datasize_news(lang, query)

    else:
        datasize = "low"

    fields = ["resourceId", "date", "countryCode", "preprocessed_text", "month", "LEN"]

    if datasize == "low":

        df_ls = []
        for file in os.listdir(inputfolder):
            if file.endswith(".csv"):
                filepath = os.path.join(inputfolder, file)
                df = pd.read_csv(filepath, low_memory=False, lineterminator="\n")
                df = df[fields]
                df_ls.append(df)

        df_concat = pd.concat(df_ls)
        print(f"{lang} -> {datasize} -> {len(df_concat)}")

        df_concat.to_csv(os.path.join(outputfolder, f"{lang}_{pageType}.csv"), index=False)

    elif datasize == "medium":
        # groupby year

        year_file_dict = defaultdict(list)
        for file in os.listdir(inputfolder):
            if file.endswith(".csv"):
                year, month = file.replace(".csv", "")
                year_file_dict[year].append(os.path.join(inputfolder, file))

        for year, filepaths in year_file_dict.items():
            df_ls_year = []
            for filepath in filepaths:
                df = pd.read_csv(filepath, low_memory=False, lineterminator="\n")
                df = df[fields]
                df_ls_year.append(df)

            df_concat = pd.concat(df_ls_year)
            print(f"{lang} -> {datasize} -> {len(df_concat)}")
            df_concat.to_csv(os.path.join(outputfolder, f"{lang}_{pageType}_{year}.csv"), index=False)

    elif datasize == "high":

        for file in os.listdir(inputfolder):
            if file.endswith(".csv"):
                year, month = file.replace(".csv", "")
                filepath = os.path.join(inputfolder, file)
                df = pd.read_csv(filepath, low_memory=False, lineterminator="\n")
                df = df[fields]
                print(f"{lang} -> {datasize} -> {len(df)}")
                df.to_csv(os.path.join(outputfolder, f"{lang}_{pageType}_{year}_{month}.csv"), index=False)


if __name__ == '__main__':
    import plac
    plac.call(combine_data_by_size)
