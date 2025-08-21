import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import sys 
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(src_path))

from utils.paths import DATA_DICT
from models.embedding.compare_embeddings import SimilarityCalculator

OPTIMAL_MODEL = 'v2_b_pca128'
model = SimilarityCalculator.from_model_name(OPTIMAL_MODEL)

sim_df = model.squeeze_by_period('Q')
star_df = model.data.stars.Q
star_df.columns = pd.to_datetime(star_df.columns).to_period('Q')

similarity_values_notfull = sim_df.values
star_values = star_df.values

similarity_values = np.zeros_like(star_values, dtype = np.float64)

for i, col in enumerate(star_df.columns):
    where_col = np.argwhere(sim_df.columns == col)
    if len(where_col) > 0:
        for j in range(star_values.shape[0]):
            similarity_values[j, i] = similarity_values_notfull[j, int(where_col[0][0])]

repos = sim_df.index.tolist()


def compute_repo_metrics(index, star_values, similarity_values, sim_thresh=0.75, pop_thresh=0.75, max_lag=2):
    """
    Compute per-repository metrics relating popularity and semantic similarity.
    
    Parameters:
    -----------
    star_values: n x m ndarray
        Repository popularity (e.g., stars) for n repositories over m periods (quarters).
    similarity_values: n x m ndarray
        Repository-job similarity for n repositories over m periods (same column alignment).
    sim_thresh: float
        Threshold for considering similarity "high".
    pop_thresh: float
        Threshold for considering popularity "high".
    max_lag: int
        Max number of periods to shift for lag analysis (+/- max_lag).
    
    Returns:
    --------
    DataFrame: n x metrics dataframe, with metrics per repository.
    """
    
    n_repos, n_periods = star_values.shape
    metrics_list = []

    for i in range(n_repos):
        pop = star_values[i].copy()
        sim = similarity_values[i].copy()

        # Pad pre-creation periods with zeros if NaN exists at the start
        # Assumes NaNs indicate periods before repo existed
        pop = np.nan_to_num(pop, nan=0.0)
        sim = np.nan_to_num(sim, nan=0.0)

        # Correlations
        if np.any(pop) and np.any(sim):  # Only compute if there is at least some data
            pearson_corr = np.corrcoef(sim, pop)[0,1]
            spearman_corr, _ = spearmanr(sim, pop)
        else:
            pearson_corr = np.nan
            spearman_corr = np.nan

        # Lag analysis
        best_lag = 0
        max_corr = pearson_corr if not np.isnan(pearson_corr) else np.nan
        for lag in range(-max_lag, max_lag+1):
            if lag == 0 or np.isnan(max_corr):
                continue
            if lag < 0:
                corr = np.corrcoef(sim[:lag], pop[-lag:])[0,1]
            else:  # lag > 0
                corr = np.corrcoef(sim[lag:], pop[:-lag])[0,1]
            if not np.isnan(corr) and corr > max_corr:
                max_corr = corr
                best_lag = lag

        # Summary stats
        mean_pop = np.mean(pop)
        mean_sim = np.mean(sim)
        frac_high = np.sum((sim > sim_thresh) & (pop > pop_thresh)) / n_periods

        metrics_list.append({
            'repo_idx': i,
            'pearson_corr': pearson_corr,
            'spearman_corr': spearman_corr,
            'max_corr': max_corr,
            'best_lag': best_lag,
            'mean_popularity': mean_pop,
            'mean_similarity': mean_sim,
            'frac_high_alignment_popularity': frac_high
        })

    return pd.DataFrame(metrics_list, index = index)


metrics_df = compute_repo_metrics(repos, star_values, similarity_values, sim_thresh=0.3, pop_thresh = 0.3, max_lag = 20)

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=metrics_df,
    x='best_lag',
    y='max_corr',
    hue='mean_popularity',  # optional: color by popularity
    size='mean_similarity',  # optional: size by mean similarity
    palette='viridis',
    sizes=(20,200)
)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel("Best Lag (quarters)")
plt.ylabel("Max Correlation")
plt.title("Repository Popularity vs Semantic Alignment")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()


plt.figure(figsize=(7,5))
sns.histplot(metrics_df['pearson_corr'].dropna(), bins=20, kde=True, color='skyblue')
plt.xlabel("Pearson Correlation")
plt.title("Distribution of Repository-Popularity vs Similarity Correlations")
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=metrics_df,
    x='mean_popularity',
    y='frac_high_alignment_popularity',
    hue='max_corr',
    palette='coolwarm',
    size='mean_similarity',
    sizes=(20,200)
)
plt.xlabel("Mean Popularity")
plt.ylabel("Fraction of High Alignment & Popularity")
plt.title("Repos with High Popularity & Alignment")
plt.show()


top_indices = metrics_df.sort_values('max_corr', ascending=False).head(3)['repo_idx']

for idx in top_indices:
    plt.figure(figsize=(10,4))
    plt.plot(star_values[idx], label='Popularity', marker='o')
    plt.plot(similarity_values[idx], label='Similarity', marker='x')
    plt.xlabel("Quarter")
    plt.ylabel("Value")
    plt.title(f"Repository {repos[idx]} Popularity vs Similarity Over Time")
    plt.legend()
    plt.show()
    
    
sns.pairplot(metrics_df[['pearson_corr', 'spearman_corr', 'mean_popularity', 'mean_similarity', 'frac_high_alignment_popularity']])
plt.suptitle("Pairwise Relationships Between Metrics", y=1.02)
plt.show()
