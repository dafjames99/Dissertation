import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))


from utils.paths import DATA_DICT
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

jobs = pd.read_csv(DATA_DICT['jobs'])
repos = pd.read_csv(DATA_DICT['github']['repositories'])

def n_grammer(texts, ngram_range = (2, 3)):
    vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words='english', min_df=3)
    X = vectorizer.fit_transform(texts)
    phrase_counts = zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0])
    common_phrases = sorted(phrase_counts, key=lambda x: -x[1])
    return phrase_counts, common_phrases

phrase_counts, common_phrases = n_grammer(repos['description'])

i = 0
for phrase, count in common_phrases[i:i + 100]:
    print(f"{phrase} - {count}")


