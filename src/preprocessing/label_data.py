import sys
import re
import json
import pandas as pd
from transformers import AutoTokenizer
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))

from utils.vars import MODEL, MAX_LEN
from utils.paths import DATA_DICT



jobs = pd.read_csv(DATA_DICT['jobs'])
tags_categories = pd.read_csv(DATA_DICT['github']['categorized_tags'])
tag_sets = tags_categories[['tag', 'category']].values.tolist()

tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize_and_label(description, job_id, tag_list):
    chunks = []
    
    encoding = tokenizer(
        description,
        return_offsets_mapping=True,
        return_attention_mask=False,
        add_special_tokens=False
    )
    
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offsets = encoding['offset_mapping']
    
    labels = ['O'] * len(tokens)
    
    for term, category in tag_list:
        for match in re.finditer(re.escape(term.lower()), description):
            start_char, end_char = match.span()
            for idx, (start, end) in enumerate(offsets):
                if start >= start_char and end <= end_char:
                    if labels[idx] == 'O':
                        labels[idx] = 'B-' + category if start == start_char else 'I-' + category

    for chunk_id, i in enumerate(range(0, len(tokens), MAX_LEN)):
        chunk_tokens = tokens[i:i+MAX_LEN]
        chunk_labels = labels[i:i+MAX_LEN]
        chunk_ids = encoding['input_ids'][i:i+MAX_LEN]
        chunk_offsets = offsets[i:i+MAX_LEN]
        
        chunks.append({
            'job_id': job_id,
            'chunk_id': chunk_id,
            'tokens': chunk_tokens,
            'labels': chunk_labels,
            'input_ids': chunk_ids,
            'offsets': chunk_offsets
        })

    return chunks

if __name__ == "__main__":
    all_chunks = []
    for _, row in jobs.iterrows():
        job_id = row['job_id']
        desc = row['description']
        chunks = tokenize_and_label(desc, job_id, tag_sets)
        all_chunks.extend(chunks)

    with open(DATA_DICT['ner_train'], "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")



