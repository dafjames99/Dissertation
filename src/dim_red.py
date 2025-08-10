from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import sys
import numpy as np, pandas as pd
from datetime import datetime, timezone as tz
from typing import Literal
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))

from representation import DataLoader, SimilarityCalculator
from utils.paths import DATA_DICT
from utils.plots import n_colors

RANDOM_STATE = 42

data = DataLoader('v2', 'b')
sim_data = SimilarityCalculator(data)

def cosine_sim(a, b):
    return normalize(a) @ normalize(b).T

def recall_at_k(simA, simB, k = 10):
    topA = np.argpartition(-simA,  k-1, axis = 1)[: , :k]
    topB = np.argpartition(-simB,  k-1, axis = 1)[: , :k]
    overlap = [len(set(a).intersection(b)) / k for a, b in zip(topA, topB)]
    return overlap

k = 10
topOrig = np.argpartition(-sim_data.matrix,  k-1, axis = 1)[: , :k]

fig, ax = plt.subplots()

ks = list(range(2, 20, 2))
colors = n_colors(len(ks))

for i, k in enumerate(ks):
    overlaps = []
    ns = np.linspace(1, 80, 80, dtype = int)
    
    for n in ns:
        pca = PCA(n, random_state=RANDOM_STATE)
        pca.fit(data.repo.embedding)
        repo_red = pca.transform(data.repo.embedding)
        job_red = pca.transform(data.job.embedding)
        sim_red = cosine_sim(repo_red, job_red)
        overlaps.append(np.mean(recall_at_k(sim_data.matrix, sim_red, k = k))*100)
        # print(f'n={n} | recall @ {k} <- VS. -> baseline: {np.mean(overlap):.3f}')

    ax.plot(ns, overlaps, color = colors[i], label = k, linewidth = 0.5)
fig.legend(title = 'k', loc = 'outside right')
fig.suptitle('K Nearest Neighbor Recall\nPCA(n) vs. No-PCA')
ax.set_xlabel('n PCA components')
ax.set_ylabel('Recall rate (%)')
plt.show()

i, j = 0, 0
r_sample, j_sample = repo_emb_p[i, :].reshape(1, -1), job_emb_p[j:,].reshape(1, -1)
r_sample @ j_sample.T
cosine_sim(r_sample, j_sample)

def plot_explained_var(pca: PCA, threshold, percentiles_increment = 0.1):
    def critpoint(cumsum, threshold):
        try:
            return np.argwhere(cumsum >= threshold).min()
        except:
            return np.nan
        
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    crit = critpoint(cumsum, threshold)

    percentiles = np.arange(0, 1, percentiles_increment)
    critpoints = [critpoint(cumsum, t) for t in percentiles]
    
    percentile_ser = pd.Series(index = percentiles, data = critpoints)

    fig, axs = plt.subplots(1, 2, figsize = (20, 8))
    if not np.isnan(crit):
        axs[0].text(s = f'({int(crit)}, {cumsum[crit]:.2f})', x = crit + 0.5, y = cumsum[crit] - 0.02)
        axs[0].axvline(crit, linestyle = '-', alpha = 0.5, color = 'red')
        axs[0].axhline(cumsum[crit], linestyle = '-', alpha = 0.5, color = 'red')
    axs[0].axhline(threshold, linestyle = '--', alpha = 0.5, color = 'green')
    axs[0].plot(cumsum)
    axs[1].bar(x = [f'{x * 100: .0f}%' for x in percentile_ser.index],height = percentile_ser)
    for i, v in enumerate(percentile_ser):
        if np.isnan(v):            
            axs[1].text(s = f'> {pca.components_.shape[0]}', x = i, y = 1, ha = 'center')
        else:
            axs[1].text(s = f'{int(v)}', x = i, y = v + 1, ha = 'center')
    plt.show()
    return percentile_ser

ser = plot_explained_var(pca, 0.5, 0.025)

