import os
import numpy as np
import pandas as pd


def get_largest_file_size(inputfolder="data/preprocessed/"):
    print("inputfolder:", inputfolder)
    file_sizes = []
    for file in os.listdir(inputfolder):
        if file.endswith(".csv"):
            file_size = os.path.getsize(os.path.join(inputfolder, file))
            file_sizes.append(file_size)
    print(np.max(file_sizes))


def data_split(inputfile, outputfile, datasize):
    file_size = os.path.getsize(inputfile)
    if file_size > datasize:
        print(f"processsing {inputfile}")
        chunks = file_size // datasize
        df = pd.read_csv(inputfile, low_memory=False, lineterminator="\n")
        for idx, df_group in df.groupby(np.arange(len(df) // chunks)):

            outputfile_name = outputfile.replace(".csv", f"_{idx}.csv")
            print(f"save to {outputfile_name}")
            df_group.to_csv(outputfile_name, index=False)


def split_data_for_preprocessing(pageType, lang, query):
    datasize = 200000
    inputfolder = os.path.join("data/pageTypes", query, lang, pageType)
    outputfolder = os.path.join("data/preprocessed", query, lang, pageType)

    for file in os.listdir(inputfolder):
        outputfile = os.path.join(outputfolder, file)
        if not os.path.exists(outputfile):
            inputfile = os.path.join(inputfolder, file)
            data_split(inputfile, outputfile, datasize)


if __name__ == '__main__':
    import plac

    plac.call(split_data_for_preprocessing)
