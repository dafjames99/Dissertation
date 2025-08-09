import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.cm as cm

base_palette = 'Spectral'
high_contrast = 'Set1'
cmap = {
    'base': cm.get_cmap(base_palette),
    'high_contrast': cm.get_cmap(high_contrast)
}
n_colors = lambda n, palette = 'base': cmap[palette](np.linspace(0, 1, n))