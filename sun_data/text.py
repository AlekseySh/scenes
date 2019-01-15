from pathlib import Path

import pandas as pd
from nltk.corpus import wordnet as wn
import json
import numpy as np

from sun_data.getters import get_names, get_domains
from sklearn.preprocessing import LabelEncoder

_table_path = Path(__file__).parent / 'files' / 'mappings.csv'


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


def main():
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

    df = pd.DataFrame(data)
    df.to_csv(_table_path, index=False)

    w_exist = df['domains'].values != ''
    domains_exist = df['domains'][w_exist]
    enum_domains = LabelEncoder().fit_transform(domains_exist)
    enum_domains = [int(domain) for domain in enum_domains]

    name_to_enum = dict(zip(df['raw_names'][w_exist], enum_domains))

    mapping_path = Path(__file__).parent / 'files' / 'NameToEnum.json'
    with open(mapping_path, 'w') as j:
        json.dump(fp=j, obj=name_to_enum)


if __name__ == '__main__':
    main()
