import sys
from pathlib import Path
import json
import pandas as pd
from termcolor import colored
from collections import Counter
from termcolor import colored

src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))
from utils.paths import DATA_DICT

# ------- 1. Peek example chunks -------

def visualize_tokens(tokens, labels):
    for token, label in zip(tokens, labels):
        if label.startswith('B-'):
            print(colored(token, 'green'), end=' ')
        elif label.startswith('I-'):
            print(colored(token, 'cyan'), end=' ')
        else:
            print(token, end=' ')
    print("\n")

def visualise_chunks(start, end = None, count = 1):
    if (end is None) or (end < start):
        end = start + count
    with open(DATA_DICT['ner_train']) as f:
        lines = f.readlines()[start:end]
    for i, l in enumerate(lines):
        data = json.loads(l)
        print(f"\n--- CHUNK {start + i} ---")
        visualize_tokens(data['tokens'], data['labels'])

visualise_chunks(10, 12)

# ------- 2. Assess prevalence of different label categories -------
def label_counts():
    label_counter = Counter()
    with open(DATA_DICT['ner_train']) as f:
        for line in f:
            data = json.loads(line)
            label_counter.update(data["labels"])
    return label_counter

counter = label_counts()
counter.most_common()



# ------- 3. What chunks are not tagged at all? -------
all_chunks, filtered_chunks, non_chunks = [], [], []
matched_tags = Counter()

with open(DATA_DICT['ner_train']) as f:
    for i, chunk in enumerate(f.readlines()):    
        data = json.loads(chunk)
        all_chunks.append(data)
        if any(l != "O" for l in data["labels"]):
            filtered_chunks.append(data)
            labels = data['labels']
            tokens = data['tokens']
            
        else:
            data['chunk_id'] = i
            non_chunks.append(data)
    
print(f"Retained {len(filtered_chunks)} of {len(all_chunks)} chunks with at least one entity.")

visualise_chunks(non_chunks[5]['chunk_id'])

df = pd.read_csv(DATA_DICT['jobs'])

matched_tags = Counter()
chunk_with_tag = 0
total_chunks = 0

with open(DATA_DICT['ner_train']) as f:
    for line in f:
        data = json.loads(line)
        total_chunks += 1
        labels = data['labels']
        tokens = data['tokens']
        chunk_tags = set()
        for token, label in zip(tokens, labels):
            if label.startswith("B-"):
                chunk_tags.add(token.lower())
        if chunk_tags:
            chunk_with_tag += 1
            matched_tags.update(chunk_tags)

print(f"Total chunks: {total_chunks}")
print(f"Chunks with at least one tagged skill: {chunk_with_tag} ({chunk_with_tag/total_chunks:.2%})")
print(f"Unique matched terms: {len(matched_tags)}")
print(matched_tags.most_common(10))


