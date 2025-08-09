import sys
from pathlib import Path
import pandas as pd
import numpy as np

src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))


df1 = pd.read_csv('data/ESCO/digitalSkillsCollection_en.csv')
df1['broaderConceptUri_aslist'] = df1['broaderConceptUri'].str.split('|')
df1['broaderConceptPT_aslist'] = df1['broaderConceptPT'].str.split('|')

df1['broaderConceptPT_aslist'] = df1['broaderConceptPT_aslist'].apply(lambda x: [i.strip() for i in x])
df1['broaderConceptUri_aslist'] = df1['broaderConceptUri_aslist'].apply(lambda x: [i.strip() for i in x])

df1['modified_label'] = df1['preferredLabel'].str.replace(' (computer programming)', '')
df1['altLabels'] = df1['altLabels'].fillna('')

concepts = [
    'information and communication technologies not elsewhere classified',
    'principles of artificial intelligence'
]

def concat_lists(list_of_lists, unique = False):
    if unique:
        out = set()
    else:
        out = list()
    for l in list_of_lists:
        for i in l:
            if unique: out.add(i)
            else: out.append(i)
    return list(out)        
    
all_broad_concepts = concat_lists(df1['broaderConceptPT_aslist'].to_list(), unique=True)
df1[df1['broaderConceptPT'].str.contains('programming', case = False)].modified_label.to_list()
df1.iloc[1283].description

df1[df1['preferredLabel'].str.contains('programming')]

df1.iloc[173].altLabels


skills = {
    'label': [],
    'altLabels': []
}

for i, r in df1.iterrows():
    skills['label'].append(r['modified_label'])
    alt = r['altLabels'].split('|')
    if len(alt) != 0:
        alt = [x.strip() for x in alt]
    skills['altLabels'].append(alt)
df = pd.DataFrame(data = skills)

df[df['label'].str.contains('machine learning')].iloc[0].altLabels
all_skills = []
for i, r in df.iterrows():
    # if len(r['altLabels']) != 0:
    #     [all_skills.append(i) for i in r['altLabels']]
    all_skills.append(r['label'])
refined = []
for i in all_skills:
    if len(i) > 0:
        refined.append(i)



pd.DataFrame(data = {'skill': refined}).to_csv('data/all_skills.csv', index = False)