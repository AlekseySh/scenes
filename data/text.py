from collections import Counter
from pathlib import Path

import pandas as pd
from nltk.corpus import wordnet as wn

from data.getters import get_sun_names, get_sun_domains

_save_path = Path(__file__).parent / 'files' / 'mappings.csv'


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
    raw_names = get_sun_names(need_beutify=False)
    names = get_sun_names(need_beutify=True)
    synsets = synsets_from_names(names)
    hypernyms = hypernyms_from_synsets(synsets)
    domains = get_sun_domains()

    data = {
        'raw_names': raw_names,
        'names': names,
        'synsets': synsets_to_words(synsets),
        'hypernyms': synsets_to_words(hypernyms),
        'domains': domains
    }

    df = pd.DataFrame(data)
    df.to_csv(_save_path, index=False)

    counter = Counter(df.domains)
    print(sum(counter.values()) - 204)


if __name__ == '__main__':
    main()
