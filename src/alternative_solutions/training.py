import json
import spacy
from collections import Counter
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))

from utils.paths import DATA_DICT

tokenizer = AutoTokenizer.from_pretrained("jjzha/jobspanbert-base-cased")
model = AutoModel.from_pretrained("jjzha/jobspanbert-base-cased")

with open(DATA_DICT['ner_train'], 'r') as f:
    data = [json.loads(l) for l in f.readlines()]
    

    
data[0].keys()

len(data)s