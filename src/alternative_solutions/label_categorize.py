from utils.paths import DATA_DICT
import pandas as pd
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")
df = pd.read_csv(DATA_DICT['github']['repositories'])
candidates = ["Framework", "Library", "Model", "Dataset", "Tool"]
texts = df['description'] + df['readme']
texts.iloc[0]
results = []
for description in texts:
    embeddings = model.encode([description] + candidates, convert_to_tensor=True)
    cos_sim = util.cos_sim(embeddings[0], embeddings[1:])
    results.append(cos_sim)
for repo, res in zip(df['repository'], results):
    print(repo)
    for i, label in enumerate(candidates):
        print(f"\t{label}: {res[0][i]:.4f}")
