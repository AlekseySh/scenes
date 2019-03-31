from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import wordnet as wn

from sun_data.utils import get_sun_names

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


def synsets_to_words(synsets):
    words = []
    for synset in synsets:
        if synset is not None:
            word = synset.name().split('.')[0]
            words.append(word)
        else:
            words.append(' ')
    return words


def save_auxilary_table() -> None:
    raw_names = get_sun_names(need_beutify=False)
    names = get_sun_names(need_beutify=True)
    synsets = synsets_from_names(names)
    hypernyms = hypernyms_from_synsets(synsets)

    data = {
        'raw_names': raw_names,
        'names': names,
        'synsets': synsets_to_words(synsets),
        'hypernyms': synsets_to_words(hypernyms)
    }
    table_path = Path(__file__).parent / 'files' / 'aux_mapping.csv'
    df = pd.DataFrame(data)
    df.to_csv(table_path, index=False)


if __name__ == '__main__':
    save_auxilary_table()
