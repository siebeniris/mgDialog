import os
import numpy as np


def get_largest_file_size(inputfolder="data/preprocessed/"):
    print("inputfolder:", inputfolder)
    file_sizes = []
    for file in os.listdir(inputfolder):
        if file.endswith(".csv"):
            file_size = os.path.getsize(os.path.join(inputfolder, file))
            file_sizes.append(file_size)
    print(np.max(file_sizes))


if __name__ == '__main__':
    import plac
    plac.call(get_largest_file_size)