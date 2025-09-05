import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))

# Generate data
x = np.linspace(0, 4 * np.pi, 500)
sinx = np.sin(x)
cosx = np.cos(x)

# Compute cross-correlation between sin and cos at different lags
lags = np.arange(-100, 101)
corrs = [np.corrcoef(sinx[max(0, lag):len(sinx) + min(0, lag)],
                     cosx[max(0, -lag):len(cosx) - max(0, lag)])[0, 1] for lag in lags]
lag_vals = 4*np.pi / lags
# Find lag with max and min correlation
max_corr_idx = np.argmax(corrs)
min_corr_idx = np.argmin(corrs)

best_lag = lags[max_corr_idx]
worst_lag = lags[min_corr_idx]

best_lag_val = lag_vals[max_corr_idx]
worst_lag_val = lag_vals[min_corr_idx]

# Shifted signals for alignment
cos_shifted_best = np.roll(cosx, best_lag)
cos_shifted_worst = np.roll(cosx, worst_lag)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# (0,0): sin and cos
axs[0, 0].plot(x, sinx, label="sin(x)", color="red")
axs[0, 0].plot(x, cosx, label="cos(x)", color="blue")
axs[0, 0].set_title("sin(x) & cos(x)")
axs[0, 0].legend(loc = 'lower right')

axs[0, 0].set_xticks([])
axs[0, 0].set_yticks([])
axs[0, 0].set_xticklabels([])
axs[0, 0].set_yticklabels([])

# (0,1): sin and cos shifted worst lag (out of phase)
axs[0, 1].plot(x, sinx, label="sin(x)", color="red")
axs[0, 1].plot(x, cos_shifted_worst, label=f"cos(x) shifted", color="blue")
axs[0, 1].set_title("Out-of-phase alignment")
axs[0, 1].legend(loc = 'lower right')

axs[0, 1].set_xticks([])
axs[0, 1].set_yticks([])
axs[0, 1].set_xticklabels([])
axs[0, 1].set_yticklabels([])

# (1,0): lag-correlation curve with max correlation
axs[1, 0].plot(lags, corrs, label="correlation")
axs[1, 0].axvline(0, color="green", linestyle="--", label=f"Default Lag")
axs[1, 0].scatter(0, 0, color="green", s=50)
axs[1, 0].set_title("Lag-Correlation (Default Alignment)")
axs[1, 0].legend(loc = 'lower right')

axs[1, 0].set_xticks([])
axs[1, 0].set_yticks([])
axs[1, 0].set_xticklabels([])
axs[1, 0].set_yticklabels([])

# (1,1): lag-correlation curve with min correlation
axs[1, 1].plot(lags, corrs, label="correlation")
axs[1, 1].axvline(worst_lag, color="purple", linestyle="--", label=f"Worst lag")
axs[1, 1].scatter(worst_lag, corrs[min_corr_idx], color="purple", s=50)
axs[1, 1].set_title("Lag-Correlation (worst alignment)")
axs[1, 1].legend(loc = 'lower right')

axs[1, 1].set_xticks([])
axs[1, 1].set_yticks([])
axs[1, 1].set_xticklabels([])
axs[1, 1].set_yticklabels([])

plt.tight_layout()
plt.savefig('latex/figures/lag-correlation.png')
plt.show()
