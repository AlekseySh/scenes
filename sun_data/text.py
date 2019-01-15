from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import wordnet as wn

from sun_data.utils import get_names, get_domains

nltk.download('wordnet')


def synsets_from_names(names):
    synsets = []
    for name in names:
        synset_list = wn.synsets(name)
        if synset_list:
            synsets.append(synset_list[0])
        else:
            synsets.append(None)
    return synsets


def hypernyms_from_synsets(synsets):
    hypernyms = []
    for synset in synsets:
        if synset is not None:
            hypernyms_list = synset.hypernyms()
            if hypernyms_list:
                hypernyms.append(hypernyms_list[0])
            else:
                hypernyms.append(None)
        else:
            hypernyms.append(None)
    return hypernyms


def topics_from_synsets(synsets):
    topics = []
    for synset in synsets:
        if synset is not None:
            topic_list = synset.topic_domains()
            if topic_list:
                topics.append(topic_list[0])
            else:
                topics.append(None)
        else:
            topics.append(None)
    return topics


def synsets_to_words(synsets):
    words = []
    for synset in synsets:
        if synset is not None:
            word = synset.name().split('.')[0]
            words.append(word)
        else:
            words.append(' ')
    return words


def save_general_table():
    raw_names = get_names(need_beutify=False)
    names = get_names(need_beutify=True)
    synsets = synsets_from_names(names)
    hypernyms = hypernyms_from_synsets(synsets)
    domains = get_domains()

    data = {
        'raw_names': raw_names,
        'names': names,
        'synsets': synsets_to_words(synsets),
        'hypernyms': synsets_to_words(hypernyms),
        'domains': domains
    }
    table_path = Path(__file__).parent / 'files' / 'general.csv'
    df = pd.DataFrame(data)
    df.to_csv(table_path, index=False)


if __name__ == '__main__':
    save_general_table()
