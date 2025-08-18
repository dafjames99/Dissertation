import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.api import VAR

# --------------------------
# 1. Example: your quarterly data
# --------------------------
# df should have quarterly DateTimeIndex and columns:
# 'repos' = repo starts in that quarter
# 'similarity' = mean similarity score in that quarter

# Example mock data
# df = pd.read_csv("your_data.csv", parse_dates=["quarter"])
# df = df.set_index("quarter").sort_index()

# --------------------------
# 2. Stationarity check helpers
# --------------------------
def adf_test(series, signif=0.05, name=''):
    """ADF test for stationarity"""
    result = adfuller(series.dropna(), autolag='AIC')
    p_value = result[1]
    print(f"ADF Test on {name}: p-value={p_value:.4f}")
    if p_value <= signif:
        print(f"  => Stationary (rejects unit root at {signif})")
    else:
        print(f"  => Non-stationary (fail to reject unit root)")

def kpss_test(series, signif=0.05, name=''):
    """KPSS test for stationarity"""
    result = kpss(series.dropna(), regression='c', nlags="auto")
    p_value = result[1]
    print(f"KPSS Test on {name}: p-value={p_value:.4f}")
    if p_value <= signif:
        print(f"  => Non-stationary (rejects stationarity at {signif})")
    else:
        print(f"  => Stationary (fail to reject stationarity)")

# --------------------------
# 3. Transform & test
# --------------------------
df['repos_log'] = np.log1p(df['repos'])  # log(count+1)
df['similarity'] = df['similarity']  # bounded [0,1], no log transform

print("Before differencing:")
for col in ['repos_log', 'similarity']:
    adf_test(df[col], name=col)
    kpss_test(df[col], name=col)
    print()

# --------------------------
# 4. Differencing if needed
# --------------------------
df_diff = df[['repos_log', 'similarity']].diff().dropna()

print("After first differencing:")
for col in df_diff.columns:
    adf_test(df_diff[col], name=col)
    kpss_test(df_diff[col], name=col)
    print()

# Choose stationary form:
df_stationary = df_diff  # if differencing made them stationary

# --------------------------
# 5. Optimal lag selection (VAR-based)
# --------------------------
model = VAR(df_stationary)
lag_order_results = model.select_order(maxlags=8)  # e.g., up to 8 quarters
print(lag_order_results.summary())

optimal_lag = lag_order_results.aic  # could use bic, hqic
optimal_lag_val = int(lag_order_results.selected_orders['aic'])
print(f"Optimal lag (AIC): {optimal_lag_val}")

# --------------------------
# 6. Granger causality test
# --------------------------
# Test: does 'repos_log' Granger-cause 'similarity'?
grangercausalitytests(
    df_stationary[['similarity', 'repos_log']], 
    maxlag=optimal_lag_val
)

# Test: does 'similarity' Granger-cause 'repos_log'?
grangercausalitytests(
    df_stationary[['repos_log', 'similarity']], 
    maxlag=optimal_lag_val
)
