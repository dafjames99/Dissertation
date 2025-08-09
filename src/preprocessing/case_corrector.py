import pandas as pd
import re
from collections import Counter, defaultdict
from utils.paths import DATA_DICT

df = pd.read_csv(DATA_DICT['jobs'])
all_text = " ".join(df['description'].tolist())

tags_df = pd.read_csv(DATA_DICT['github']['tags']['categorized'])
tags = tags_df['tag'].tolist()

words = re.findall(r'\b\w[\w/+-]*\b', all_text)

word_counter = Counter(words)

smart_casing_map = {}
lower_to_variants = defaultdict(list)

for word in word_counter:
    lw = word.lower()
    lower_to_variants[lw].append((word, word_counter[word]))

for tag in tags:
    if tag in lower_to_variants:
        variants = sorted(lower_to_variants[tag], key=lambda x: -x[1])
        smart_casing_map[tag] = variants[0][0] 
    else:
        smart_casing_map[tag] = tag

map2 = {}
for tag, map in smart_casing_map.items():
    if tag != map:
        map2[tag] = map

df = pd.DataFrame(data = {'lower': list(map2.keys()), 'proper': list(map2.values())}) 
df.to_csv(DATA_DICT['github']['tags']['case_map'], index = False)