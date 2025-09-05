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
fig_dir = 'latex/figures'
OPTIMAL_MODEL = 'v2_c'
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
print(star_values.shape, similarity_values.shape)

def shift_array(arr, lag):
    """
    Shift an array by `lag` periods (non-circular).
    Positive lag -> shift SIM backward (earlier values align to later pop periods)
    Negative lag -> shift SIM forward (later values align to earlier pop periods)
    Fills empty positions with 0.
    """
    shifted = np.zeros_like(arr)
    if lag > 0:
        # shift backward: sim[i] moves to i-lag
        shifted[:-lag] = arr[lag:]
    elif lag < 0:
        # shift forward: sim[i] moves to i-lag
        shifted[-lag:] = arr[:lag]
    else:
        shifted = arr.copy()
    return shifted

def compute_repo_metrics(index, star_values, similarity_values, sim_thresh=0.75, pop_thresh=0.75, max_lag=2, frac_at_mean = False):
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
        for lag in range(max_lag + 1):
            sim_lagged = shift_array(sim, lag)
            if np.any(pop) and np.any(sim_lagged):
                corr = np.corrcoef(sim_lagged, pop)[0,1]
            else:
                corr = np.nan
            if not np.isnan(corr) and (np.isnan(max_corr) or corr > max_corr):
                max_corr = corr
                best_lag = lag

        # Summary stats# Only consider periods where the repo exists (pop > 0)
        pop_mask = pop > 0
        mean_pop = np.mean(pop[pop_mask]) if np.any(pop_mask) else np.nan
        repo_age = np.sum(pop_mask)
        
        sim_mask = sim > 0
        mean_sim = np.mean(sim[sim_mask]) if np.any(sim_mask) else np.nan

        # Repo age = number of active periods

        # mean_pop = np.mean(pop)
        # mean_sim = np.mean(sim)
        valid_mask = sim != 0
        sim_lagged_best = shift_array(sim, best_lag)
        valid_mask_lagged = sim_lagged_best != 0

        if frac_at_mean:
            frac_high = np.sum((sim > mean_sim) & (pop > mean_pop) & valid_mask) / np.sum(valid_mask) if np.any(valid_mask) else np.nan
            frac_high_lagged = np.sum((sim_lagged_best > mean_sim) & (pop > mean_pop) & valid_mask_lagged) / np.sum(valid_mask_lagged) if np.any(valid_mask_lagged) else np.nan
        else:
            frac_high = np.sum((sim > sim_thresh) & (pop > pop_thresh) & valid_mask) / np.sum(valid_mask) if np.any(valid_mask) else np.nan
            frac_high_lagged = np.sum((sim_lagged_best > sim_thresh) & (pop > pop_thresh) & valid_mask_lagged) / np.sum(valid_mask_lagged) if np.any(valid_mask_lagged) else np.nan

        metrics_list.append({
            'repo_idx': i,
            'pearson_corr': pearson_corr,
            'spearman_corr': spearman_corr,
            'max_corr': max_corr,
            'best_lag': best_lag,
            'mean_popularity': mean_pop,
            'mean_similarity': mean_sim,
            'repo_age': repo_age,
            'frac_high_alignment_popularity': frac_high,
            'frac_high_alignment_popularity_at_best_lag': frac_high_lagged
        })

    return pd.DataFrame(metrics_list, index = index)


metrics_df = compute_repo_metrics(repos, star_values, similarity_values, sim_thresh=0.3, pop_thresh = 0.3, max_lag = 50, frac_at_mean=True)
metrics_df.to_csv(DATA_DICT['models']['star_eval'])

plt.figure(figsize=(10,6))
sns.histplot(metrics_df['repo_age'].dropna(), bins=20, kde=True, color='skyblue')
plt.xlabel("Repository Age (in Quarters)")
# plt.title("Distribution of Repository-Ages")
plt.savefig(fig_dir + '/repo-ages.png')

data_list = []
for i, repo_name in enumerate(repos):
    for sim_val in similarity_values[i]:
        # Only include non-zero similarity values
        if sim_val > 0:
            data_list.append({"repo": repo_name, "similarity": sim_val})

sim_df = pd.DataFrame(data_list)

# Step 2: Compute mean similarity per repo to order
mean_sim_order = sim_df.groupby("repo")["similarity"].mean().sort_values().index
import math

n_repos = len(mean_sim_order)
chunk_size = math.ceil(n_repos / 2)  # split into 2 chunks

for i in range(0, n_repos, chunk_size):
    chunk = mean_sim_order[i:i+chunk_size]
    plt.figure(figsize=(12, 10))  # smaller figure per chunk
    sns.boxplot(
        data=sim_df,
        y="repo",
        x="similarity",
        order=chunk,
        orient="h",
        palette="coolwarm"
    )
    ax = plt.gca()
    for y in range(len(chunk)):
        ax.axhline(y=y, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
    for x in np.arange(0.45, 0.7, 0.025):
        ax.axvline(x=x, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
    
    plt.tick_params(axis='y', labelsize=14)  # smaller font for readability
    plt.xlabel("Similarity")
    plt.ylabel("Repository")
    plt.title(f"Similarity Distributions per Repository (Repos {i+1}–{i+len(chunk)})")
    plt.tight_layout()
    plt.savefig(fig_dir + f'/sim-boxplot_{i//chunk_size + 1}.png', dpi=300)
    plt.close()



plt.figure(figsize=(10,6))
sns.scatterplot(
    data=metrics_df,
    x='mean_popularity',
    y='mean_similarity',
    # hue='max_corr',
    palette='coolwarm',
    # size='mean_similarity',
    # sizes=(20,200)
)
plt.xlabel("Mean Popularity")
plt.ylabel("Mean Similarity")
plt.title("Mean Similarity & Popularity")
plt.savefig(fig_dir + '/mean-sim-pop.png')
# plt.show()


plt.figure(figsize=(8,6))
sns.scatterplot(
    data=metrics_df,
    x='best_lag',
    y='max_corr',
    hue='repo_age',  # optional: color by popularity
    # size='mean_similarity',  # optional: size by mean similarity
    palette='coolwarm',
    # sizes=(20,200)
)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel("Best Lag (quarters)")
plt.ylabel("Max Correlation")
plt.title("Lags with Best Correlation")
plt.legend(title = 'Repository Age \n(in Quarters)',loc = 'lower right')
plt.savefig(fig_dir + '/lag-max_corr.png')
# plt.show()


plt.figure(figsize=(7,5))
sns.histplot(metrics_df['pearson_corr'].dropna(), bins=20, kde=True, color='skyblue')
plt.xlabel("Pearson Correlation")
plt.title("Correlation: Sim vs Pop")
plt.savefig(fig_dir + '/dist-corrs.png')
# plt.show()

plt.figure(figsize=(7,5))
sns.histplot(metrics_df['max_corr'].dropna(), bins=20, kde=True, color='skyblue')
plt.xlabel("Max Correlation")
plt.title("Correlation: Sim vs Pop \n(at best respective best lags)")
plt.savefig(fig_dir + '/dist-max-corr.png')
# plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=metrics_df,
    x='max_corr',
    y='frac_high_alignment_popularity',
    hue='mean_similarity',
    palette='coolwarm',
    # size='mean_similarity',
    # sizes=(20,200)
)
plt.xlabel("Max Correlation")
plt.ylabel("Fraction of High Alignment & Popularity")
plt.title("Repos with High Popularity & Alignment")
plt.savefig(fig_dir + '/frac_align.png')
# plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=metrics_df,
    x='max_corr',
    y='frac_high_alignment_popularity_at_best_lag',
    hue='mean_similarity',
    palette='coolwarm',
    # size='mean_similarity',
    # sizes=(20,200)
)
plt.xlabel("Max Correlation")
plt.ylabel("Fraction of High Alignment & Popularity at best lags")
plt.title("Repos with High Popularity & Alignment at best lags")
plt.savefig(fig_dir + '/frac_align_best_lag.png')
# plt.show()



# indices = metrics_df.sort_values('max_corr', ascending=False).head(3)['repo_idx']
repo_desired = ['scikit-learn/scikit-learn', 'pytorch/pytorch', 'huggingface/transformers', 'opencv/opencv', 'google-research/bert', 'wandb/wandb', 'langchain-ai/langchain']
indices = [repos.index(r) for r in repo_desired] 
for idx in indices:
    fig, ax1 = plt.subplots(figsize=(10, 4))

    # Plot popularity on left y-axis
    ax1.plot(star_values[idx], label='Popularity', marker='o', color='tab:blue')
    ax1.set_xlabel("Quarter")
    ax1.set_ylabel("Popularity (stars)", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a second y-axis for similarity
    ax2 = ax1.twinx()
    ax2.plot(similarity_values[idx], label='Similarity', marker='x', color='tab:red', linestyle = 'none')
    ax2.set_ylabel("Similarity", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.title(f"Repository {repos[idx]} Popularity vs Similarity Over Time")
    plt.savefig(fig_dir + f'/repo_{repos[idx].replace('/', '_')}.png')
    plt.tight_layout()
    # plt.show()
# repo_desired = ['scikit-learn/scikit-learn', 'pytorch/pytorch', 'huggingface/transformers', 'opencv/opencv', 'google-research/bert', 'wandb/wandb', 'langchain-ai/langchain']
# indices = [repos.index(r) for r in repo_desired] 
    
sns.pairplot(metrics_df[['pearson_corr', 'spearman_corr', 'mean_popularity', 'mean_similarity', 'frac_high_alignment_popularity']])
plt.suptitle("Pairwise Relationships Between Metrics", y=1.02)
plt.savefig(fig_dir + '/pairplot.png')
# plt.show()

metrics_list = []

for i, repo_name in enumerate(repos):
    sim_vals = similarity_values[i]
    pop_vals = star_values[i]
    
    # Only consider non-zero similarity values
    sim_active = sim_vals[sim_vals > 0]
    pop_active = pop_vals[pop_vals > 0]
    
    if len(sim_active) > 0:
        sim_range = sim_active.max() - sim_active.min()
        mean_sim = sim_active.mean()
    else:
        sim_range = np.nan
        mean_sim = np.nan
    
    mean_pop = pop_vals[pop_vals > 0].mean() if np.any(pop_vals > 0) else np.nan
    total_pop = pop_vals.sum()
    age = np.sum(pop_vals > 0)
    
    metrics_list.append({
        'repo': repo_name,
        'sim_range': sim_range,
        'mean_similarity': mean_sim,
        'mean_popularity': mean_pop,
        'total_popularity': total_pop,
        'age': age
    })

metrics_df2 = pd.DataFrame(metrics_list)

x = metrics_df2['mean_popularity']
y = metrics_df2['sim_range']
colors = metrics_df2['total_popularity']
sizes = metrics_df2['age']*10  # scale for visibility

plt.figure(figsize=(10,6))
sc = plt.scatter(x, y, c=colors, s=sizes, cmap='viridis', alpha=0.7)
plt.xlabel("Mean Popularity")
plt.ylabel("Similarity Range")
plt.title("Similarity Range vs Mean Popularity (Repo Age & Total Popularity Highlighted)")
plt.colorbar(sc, label='Total Popularity')
import numpy as np
import pandas as pd

n_repos, n_periods = similarity_values.shape
repo_ages = pd.Series(metrics_df['repo_age']).value_counts().sort_index()
quarters = np.arange(n_periods)
creations = dict.fromkeys(quarters, 0)
for age in repo_ages.index:
    creations[n_periods - age] = repo_ages[age]
creations_series = pd.Series(creations)

# Find the quarter of max popularity for each repo
peak_quarters = np.array([np.argmax(star_values[i]) for i in range(n_repos)])

# Count number of repos hitting peak popularity in each quarter
peak_counts = pd.Series(peak_quarters).value_counts().sort_index()

# Make sure all quarters are represented
peak_counts = peak_counts.reindex(quarters, fill_value=0)

mean_sim_per_period = np.nanmean(np.where(similarity_values > 0, similarity_values, np.nan), axis=0)
std_sim_per_period = np.nanstd(np.where(similarity_values > 0, similarity_values, np.nan), axis=0)

fig, ax1 = plt.subplots(figsize=(10,5))

# Mean similarity line
ax1.plot(quarters, mean_sim_per_period, marker='o', color='tab:blue', label='Mean Similarity')
ax1.fill_between(
    quarters,
    mean_sim_per_period - std_sim_per_period,
    mean_sim_per_period + std_sim_per_period,
    color='tab:blue', alpha=0.2
)
ax1.set_xlabel("Quarter")
ax1.set_ylabel("Mean Similarity", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Secondary y-axis for repo creation and peak counts
ax2 = ax1.twinx()

bar_width = 0.4

# Original repo creations
ax2.bar(creations_series.index - bar_width/2, creations_series, width=bar_width, alpha=0.3, color='tab:orange', label='Repos Created')

# Peak popularity
ax2.bar(peak_counts.index + bar_width/2, peak_counts, width=bar_width, alpha=0.3, color='tab:green', label='Repos Peak Popularity')

ax2.set_ylabel("Number of Repos", color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.title("Mean Similarity Trend Over Time with Repo Creation and Peak Popularity")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir + '/sim_over_time_vs_creation_peak.png')
# plt.show()


df = pd.read_csv(DATA_DICT['jobs'], parse_dates=['date'])
# Plot 1: Total job postings per quarter as a column chart
df['quarter'] = df['date'].dt.to_period('Q')

# Generate full range of quarters
all_quarters = pd.period_range(df['quarter'].min(), df['quarter'].max(), freq='Q')

# Count postings per quarter
quarterly_counts = df.groupby('quarter').size().reindex(all_quarters, fill_value=0)

# Plot 1: Column chart of total job postings per quarter
plt.figure(figsize=(12, 5))
quarterly_counts.plot(kind='bar', color='steelblue')
plt.title("Total Job Postings per Quarter")
plt.xlabel("Quarter")
plt.ylabel("Number of Postings")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(fig_dir + '/job_date_variation.png')
# plt.show()

# Plot 2: Horizontal bar chart of job mapped title counts
job_counts = df['mapped_title'].value_counts()

plt.figure(figsize=(10, 6))
job_counts.plot(kind='barh', color='skyblue')
plt.title("Job Mapped Title Counts")
plt.xlabel("Number of Postings")
plt.ylabel("Job Mapped Title")
plt.gca().invert_yaxis()  # largest on top
plt.tight_layout()
plt.savefig(fig_dir + '/job_title_variation.png')
plt.show()
