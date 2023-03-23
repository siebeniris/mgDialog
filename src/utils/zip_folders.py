import os
import zipfile
import glob


def zip_folder(inputfolder="data/raw/eu"):
    for folder in os.listdir(inputfolder):
        folderpath = os.path.join(inputfolder, folder)
        print(folderpath)
        if os.path.isdir(folderpath):
            outputfilename = folderpath + ".zip"
            print(f"output {outputfilename}")
            with zipfile.ZipFile(outputfilename, "w") as f:
                for file in glob.glob(os.path.join(inputfolder, folder)+"/**.csv"):
                    f.write(file)

if __name__ == '__main__':
    import plac
    plac.call(zip_folder)