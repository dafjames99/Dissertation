import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import numpy as np
from scipy.stats import zscore
import pandas as pd
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal
from IPython.display import display, HTML

src_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(src_path))

from utils.plots import n_colors
from utils.paths import DATA_DICT, SENTENCE_MODEL
from models.embedding.compare_embeddings import SimilarityCalculator, UtilityClass, EmbeddingSimilarity

OPTIMAL_MODEL = 'v2_b_pca128'
model = SimilarityCalculator.from_model_name(OPTIMAL_MODEL)


EmbeddingSimilarity.repo_stars_similarity()




# --- 0) Assumptions ---
# You already have df_A (values in [-1, 1]) and df_B (counts per quarter).
# Both indexed by repository name; columns are quarters like "2025Q1".
period, agg_fn = "Q", 'mean'
q_sim = model.squeeze_by_period(period, agg_fn)
print(q_sim)
model.data.stars.Q.columns = pd.to_datetime(model.data.stars.Q.columns).to_period('Q')
df_A = model.df
df_B = model.data.stars.Q
# --- 1) Align rows & quarters (keep origmodel.data.stars.Qinal column labels but sort chronologically) ---

common_rows = df_A.index.intersection(df_B.index)

common_cols = df_A.columns.intersection(df_B.columns)

def _quarter_key(c):
    # works for strings like "2025Q1" or Period objects
    if hasattr(c, "year"):  # Period
        return (c.year, c.quarter)
    s = str(c)
    y, q = s.split("Q")
    return (int(y), int(q))

cols_sorted = sorted(common_cols, key=_quarter_key)

A = df_A.loc[common_rows, cols_sorted].astype(float)
B = df_B.loc[common_rows, cols_sorted].astype(float)

# --- 2) Preprocess ---
# 2a) Row-wise z-score of A using pandas (skip NaNs; avoid scipy's edge cases)
row_mean = A.mean(axis=1, skipna=True)
row_std = A.std(axis=1, ddof=0, skipna=True).replace(0, np.nan)  # avoid divide-by-zero
A_z = A.sub(row_mean, axis=0).div(row_std, axis=0)

# 2b) Choose a "driver" version of B. Try diff of log1p for smoother spikes.
B_log = np.log1p(B)
B_drv = B_log.diff(axis=1)       # change in log-stars QoQ (≈ growth)

# --- 3) Lag correlations: positive lag = "B leads A by lag quarters" ---
def lag_corr_series(a, b, max_lag=4, min_n=3):
    """
    Returns dicts {lag: corr} and {lag: n}.
    Positive lag k: correlate A[k:] with B[:-k]  (B at t influences A at t+k).
    Handles NaNs; requires at least min_n overlapping points.
    """
    a = pd.Series(a, dtype=float)
    b = pd.Series(b, dtype=float)
    corrs, ns = {}, {}
    for k in range(-max_lag, max_lag + 1):
        if k > 0:         # B leads A by k
            x, y = a[k:], b[:-k]
        elif k < 0:       # A leads B by -k
            x, y = a[:k], b[-k:]
        else:             # same quarter
            x, y = a, b
        m = x.notna() & y.notna()
        n = int(m.sum())
        ns[k] = n
        corrs[k] = x[m].corr(y[m]) if n >= min_n else np.nan
    return corrs, ns

corr_dict, n_dict = {}, {}
for repo in A_z.index:
    c, n = lag_corr_series(A_z.loc[repo].values, B_drv.loc[repo].values, max_lag=4, min_n=3)
    corr_dict[repo] = c
    n_dict[repo] = n

lags_corr = pd.DataFrame.from_dict(corr_dict, orient="index").sort_index(axis=1)  # rows=repos, cols=lags
lags_n    = pd.DataFrame.from_dict(n_dict,    orient="index").reindex(columns=lags_corr.columns)

# --- 4) Summaries: where does B appear to lead A? ---
pos_lags = [k for k in lags_corr.columns if k > 0]
same_lag = 0

summary_rows = []
for repo in A_z.index:
    s_pos = lags_corr.loc[repo, pos_lags]
    n_pos = lags_n.loc[repo, pos_lags]
    best_lag = s_pos.idxmax()
    best_corr = s_pos.loc[best_lag]
    best_n = int(n_pos.loc[best_lag])
    same_corr = lags_corr.loc[repo, same_lag]
    summary_rows.append({
        "repo": repo,
        "best_pos_lag": int(best_lag),
        "corr_at_best_pos_lag": float(best_corr),
        "n_at_best_pos_lag": best_n,
        "same_quarter_corr": float(same_corr),
    })

summary = pd.DataFrame(summary_rows).sort_values("corr_at_best_pos_lag", ascending=False)
print("Top repos where B (stars growth) leads A the most:")
print(summary.head(10))

# Quick headline counts
num_pos = (summary["corr_at_best_pos_lag"] > 0).sum()
num_strong = (summary["corr_at_best_pos_lag"] >= 0.4).sum()
print(f"\nRepos with positive best-lag corr: {num_pos}/{len(summary)}")
print(f"Repos with best-lag corr ≥ 0.4: {num_strong}/{len(summary)}")

# --- 5) Visual: overlay A_z and B driver for a top repo ---
if not summary.empty and np.isfinite(summary.iloc[0]["corr_at_best_pos_lag"]):
    top_repo = summary.iloc[0]["repo"]
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(cols_sorted, A_z.loc[top_repo, cols_sorted], marker="o", label="A (z-score)")
    ax1.set_ylabel("A (z-score)")
    ax1.set_xlabel("Quarter")
    ax1.set_title(f"{top_repo}: A vs. Δlog(Stars) — best lag={summary.iloc[0]['best_pos_lag']}, corr={summary.iloc[0]['corr_at_best_pos_lag']:.2f}")
    ax2 = ax1.twinx()
    ax2.bar(cols_sorted, B_drv.loc[top_repo, cols_sorted], alpha=0.3, label="Δlog(Stars)")
    ax2.set_ylabel("Δlog(Stars)")
    ax1.legend(loc="upper left")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
