from yaml import load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def load_languages():
    with open("data/config.yaml") as f:
        langs = load(f, Loader=Loader)["langs"]
    return langs


def load_keywords_by_lang(lang):
    with open("data/config.yaml") as f:
        data = load(f, Loader=Loader)

    keywords = data["keywords"]
    keywords_dict = {x: item[x] for item in keywords for x in item}
    return keywords_dict[lang]


def load_datasize_twitter(lang, query):
    with open("data/config.yaml") as f:
        data = load(f, Loader=Loader)

    if query == "eu":
        datasizes = data["datasize_twitter_eu"]
    else:
        datasizes = data["datasize_twitter_un"]
    datasize_dict = {x: item[x] for item in datasizes for x in item}
    return datasize_dict[lang]


def load_datasize_news(lang, query):
    with open("data/config.yaml") as f:
        data = load(f, Loader=Loader)

    if query == "eu":
        datasizes = data["datasize_news_eu"]
    else:
        datasizes = data["datasize_news_un"]
    datasize_dict = {x: item[x] for item in datasizes for x in item}
    return datasize_dict[lang]
