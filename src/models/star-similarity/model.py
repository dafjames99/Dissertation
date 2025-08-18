
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.api import VAR


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
from models.embedding.compare_embeddings import SimilarityCalculator, UtilityClass

OPTIMAL_MODEL = 'v2_b_pca128'
model = SimilarityCalculator.from_model_name(OPTIMAL_MODEL)


# def align_numerics(model, period, agg_fn):
#     q_sim = model.squeeze_by_period(period, agg_fn)
#     qstar_cols = pd.to_datetime(model.data.stars.Q.columns).to_period('Q')

#     common_cols = q_sim.columns.intersection(qstar_cols)
#     common_cols = [c.strftime('%YQ%q') for c in common_cols]

#     A_aligned = q_sim[common_cols].copy()
#     B_aligned = model.data.stars.Q[common_cols].copy()
#     return A_aligned, B_aligned, common_cols

# def time_aligned_check(model, period, agg_fn, repo: str | int):
    
#     A_aligned, B_aligned, _ = align_numerics(model, period, agg_fn)
    
#     B_diff = B_aligned.diff(axis=1)
#     A_z = A_aligned.apply(lambda row: pd.Series(zscore(row), index = A_aligned.columns.tolist()), axis=1)
    
#     if isinstance(repo, int):
#         repo = A_z.index[repo]
    
#     fig, ax1 = plt.subplots(figsize=(10, 5))
#     ax2 = ax1.twinx()

#     ax1.plot(A_aligned.columns, A_z.loc[repo], color='blue', marker='o', label='Similarity (z-score)')
#     ax2.bar([i for i in range(len(B_diff.columns))], B_diff.loc[repo], alpha=0.3, color='orange', label='ΔStars')

#     ax1.set_xlabel("Quarter")
#     ax1.set_ylabel("Similarity (z-score)", color='blue')
#     ax2.set_ylabel("ΔStars", color='orange')
#     ax1.set_title(f"Repo: {repo}")

#     fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()
    
# def lag_heatmap(model, period, agg_fn, max_lag):
#     A_aligned, B_aligned, _ = align_numerics(model, period, agg_fn)
    
#     B_diff = B_aligned.diff(axis=1)
#     A_z = A_aligned.apply(lambda row: pd.Series(zscore(row), index = A_aligned.columns.tolist()), axis=1)
        
#     def lag_corr(a, b, max_lag=max_lag):
#         """Compute correlation at different lags: b leads a"""
#         corrs = {}
#         for lag in range(-max_lag, max_lag+1):
#             if lag < 0:
#                 corr = np.corrcoef(a[-lag:], b[:lag])[0, 1]
#             elif lag > 0:
#                 corr = np.corrcoef(a[:-lag], b[lag:])[0, 1]
#             else:
#                 corr = np.corrcoef(a, b)[0, 1]
#             corrs[lag] = corr
#         return corrs

#     lags_df = pd.DataFrame({
#         repo: lag_corr(A_z.loc[repo].values, B_diff.loc[repo].values, max_lag=max_lag)
#         for repo in A_z.index
#     }).T

#     plt.figure(figsize=(8, 6))
#     plt.imshow(lags_df, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
#     plt.colorbar(label='Correlation')
#     plt.xticks(range(len(lags_df.columns)), lags_df.columns)
#     plt.yticks(range(len(lags_df.index)), lags_df.index)
#     plt.title("Lag correlation (rows=repos, cols=lags)")
#     plt.tight_layout()
#     plt.show()

# def star_and_sim_to_longformat(model, period, agg_fn):
#     a_aligned, b_aligned, cols = align_numerics(model, period, agg_fn)
#     df = pd.DataFrame(columns = ['quarter', 'repo', 'star', 'similarity'])
#     for i, (rowsim, rowstar) in enumerate(zip(a_aligned.itertuples(index = False, name = None), b_aligned.itertuples(index = False, name = None))):
#         for col, sim_, star_ in zip(cols, rowsim, rowstar):
#             df.loc[len(df)] = [col, a_aligned.index[i], star_, sim_]
#     return df


# # time_aligned_check(model, "Q", 'mean', 1)
# # lag_heatmap(model, 'Q', 'mean', 40)
# df = star_and_sim_to_longformat(model, 'Q', 'mean')
# df = df.set_index(['quarter', 'repo'])
# #----------------------
# # 1. Example: your quarterly data
# # --------------------------
# # df should have quarterly DateTimeIndex and columns:
# # 'repos' = repo starts in that quarter
# # 'similarity' = mean similarity score in that quarter

# # Example mock data
# # df = pd.read_csv("your_data.csv", parse_dates=["quarter"])
# # ?/df = df.set_index("quarter").sort_index()

# # --------------------------
# # 2. Stationarity check helpers
# # --------------------------
# def adf_test(series, signif=0.05, name=''):
#     """ADF test for stationarity"""
#     result = adfuller(series.dropna(), autolag='AIC')
#     p_value = result[1]
#     print(f"ADF Test on {name}: p-value={p_value:.4f}")
#     if p_value <= signif:
#         print(f"  => Stationary (rejects unit root at {signif})")
#     else:
#         print(f"  => Non-stationary (fail to reject unit root)")

# def kpss_test(series, signif=0.05, name=''):
#     """KPSS test for stationarity"""
#     result = kpss(series.dropna(), regression='c', nlags="auto")
#     p_value = result[1]
#     print(f"KPSS Test on {name}: p-value={p_value:.4f}")
#     if p_value <= signif:
#         print(f"  => Non-stationary (rejects stationarity at {signif})")
#     else:
#         print(f"  => Stationary (fail to reject stationarity)")

# # --------------------------
# # 3. Transform & test
# # --------------------------
# df['repos_log'] = np.log1p(df['repos'])  # log(count+1)
# df['similarity'] = df['similarity']  # bounded [0,1], no log transform

# print("Before differencing:")
# for col in ['repos_log', 'similarity']:
#     adf_test(df[col], name=col)
#     kpss_test(df[col], name=col)
#     print()

# # --------------------------
# # 4. Differencing if needed
# # --------------------------
# df_diff = df[['repos_log', 'similarity']].diff().dropna()

# print("After first differencing:")
# for col in df_diff.columns:
#     adf_test(df_diff[col], name=col)
#     kpss_test(df_diff[col], name=col)
#     print()

# # Choose stationary form:
# df_stationary = df_diff  # if differencing made them stationary

# # --------------------------
# # 5. Optimal lag selection (VAR-based)
# # --------------------------
# model = VAR(df_stationary)
# lag_order_results = model.select_order(maxlags=8)  # e.g., up to 8 quarters
# print(lag_order_results.summary())

# optimal_lag = lag_order_results.aic  # could use bic, hqic
# optimal_lag_val = int(lag_order_results.selected_orders['aic'])
# print(f"Optimal lag (AIC): {optimal_lag_val}")

# # --------------------------
# # 6. Granger causality test
# # --------------------------
# # Test: does 'repos_log' Granger-cause 'similarity'?
# grangercausalitytests(
#     df_stationary[['similarity', 'repos_log']], 
#     maxlag=optimal_lag_val
# )

# # Test: does 'similarity' Granger-cause 'repos_log'?
# grangercausalitytests(
#     df_stationary[['repos_log', 'similarity']], 
#     maxlag=optimal_lag_val
# )


matrices, rows, cols, tensor = model.repo_stars_similarity('Q')
print(tensor[0])
print(tensor[1])