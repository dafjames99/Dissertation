from typing import Literal
from pathlib import Path
import sys

from datetime import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))

from utils.plots import n_colors, cmap
from utils.paths import DATA_DICT

START_DATE = '2022-01-01'
today = dt.today()

df = pd.read_csv(DATA_DICT['github']['stars'])

df['date'] = pd.to_datetime(df['date']).dt.normalize()

def normalize_group(x):
    min_x, max_x = x.min(), x.max()
    if min_x == max_x:
        return 0  # or return x  # or return x*0 + 1 depending on your intent
    return (x - min_x) / (max_x - min_x)

df['normalized_stars'] = df.groupby('fullname')['stars'].transform(normalize_group)


class RepoStarsPlotter:
    def __init__(self, start_date=None, end_date=None):
        self.df = pd.read_csv(DATA_DICT['github']['stars'])
        
        self.df['date'] = pd.to_datetime(self.df['date']).dt.normalize()
        
        self.start_date = start_date or df['date'].min()
        self.end_date = end_date or today
        
        self.date_range = pd.date_range(self.start_date, self.end_date)
        self.df['normalized_stars'] = self.normalize_stars()


    def normalize_stars(self, metric = 'stars'):
        def normalize_group(x):
            min_x, max_x = x.min(), x.max()
            if min_x == max_x:
                return 0
            return (x - min_x) / (max_x - min_x)
        
        return self.df.groupby('fullname')[metric].transform(normalize_group) 



    def _filter_repo_data(self, repo_name: str | list = None, metric: list | str = None):
        if metric is not None:
            cols = ['fullname', 'date']
            if isinstance(metric, str):
                cols.append(metric)
            elif isinstance(metric, list):
                cols.extend(metric)
        else:
            cols = df.columns.tolist()
        mask = (self.df['date'].isin(self.date_range))
        if repo_name is not None:
            if isinstance(repo_name, list):
                mask = mask & (self.df['fullname'].isin(repo_name))
            elif isinstance(repo_name, str):
                mask = mask & (self.df['fullname'] == repo_name)
        return (
            self.df[mask]
            .sort_values(by='date')[cols]
        )

    def _apply_plot(self, ax, repo_name, metric, color, ma_period):
        data = self._filter_repo_data(repo_name, metric)
        if metric == 'cumulative':
            ax.plot(data['date'], data[metric], color=color, label=repo_name)
        else:
            ax.plot(data['date'], data[metric], color=color, label=f"{repo_name} (raw)", alpha=0.4, linestyle='-')
            # Plot moving average overlay
            if ma_period > 1:
                avg_series = np.convolve(data[metric], np.ones(ma_period)/ma_period, mode='same')
                ax.plot(data['date'], avg_series, color=color, label=f"{repo_name} ({ma_period}d MA)", linewidth=2)


    def plot_single_metric(self, repos, metric='stars', figsize=(15,5), hlines=None, ma_period = 1):
        colors = n_colors(len(repos))
        fig, ax = plt.subplots(figsize=figsize)

        for repo, color in zip(repos, colors):
            self._apply_plot(ax, repo, metric, color, ma_period)

        if hlines is not None:
            for h in hlines:
                ax.axhline(int(h), color='r', linestyle='--', alpha=0.5)
        title = f"{metric.capitalize()} over Time"
        if ma_period != 1:
            title += f"\n Moving Average of {ma_period} Days"

        ax.set_title(title)
        ax.legend()
        plt.show()
    

    def plot_combined(self, repos, figsize=(15,10), hlines_daily=None, hlines_cum=None, ma_period = 1):
        colors = n_colors(len(repos))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Plot daily
        for repo, color in zip(repos, colors):
            self._apply_plot(ax1, repo, 'stars', color, ma_period)
        ax1.set_title("Daily Stars")
        if hlines_daily is not None:
            for h in hlines_daily:
                ax1.axhline(int(h), color='r', linestyle='--', alpha=0.5)

        # Plot cumulative
        for repo, color in zip(repos, colors):
            self._apply_plot(ax2, repo, 'cumulative', color, ma_period)
        ax2.set_title("Cumulative Stars")
        if hlines_cum is not None:
            for h in hlines_cum:
                ax2.axhline(int(h), color='r', linestyle='--', alpha=0.5)

        ax1.legend()
        plt.tight_layout()
        plt.show()

    def set_date_range(self, start=None, end=None):
        if start: self.start_date = start
        if end: self.end_date = end
        self.date_range = pd.date_range(self.start_date, self.end_date)

    def make_rectangular(self, repos, metric, frequency, normalize_cols):
        
        self.df['period'] = self.df['date'].dt.to_period(frequency).dt.to_timestamp()

        new_df = pd.DataFrame(columns = ['date'] + repos)

        period_range = self.df['period'].sort_values().drop_duplicates()
        
        for p in period_range:
            on_period = self.df[self.df['period'] == p]
            values = [p]
            for repo in repos:
                sub_slice = on_period[on_period['fullname'] == repo][metric]
                try:
                    values.append(sub_slice.iloc[0])
                except IndexError:
                    values.append(0)
            new_df.loc[len(new_df)] = values
        if normalize_cols:

            for c in new_df.columns[1:]:
                col_min = new_df[c].min()
                col_max = new_df[c].max()
                if col_min != col_max:
                    new_df[c] = (new_df[c] - col_min) / (col_max - col_min)
                else:
                    new_df[c] = 0
        return new_df
        
    def heatmap(self, repos, frequency: Literal['D','W','M','Q','A'] = 'D', metric = 'stars', normalize_cols = True, ax = None):
        data = self.make_rectangular(repos, metric, frequency, normalize_cols)
        data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
        heatmap_data = data[repos]
        heatmap_data.index = data['date']
        if ax is None:
            plt.figure(figsize=(len(repos)*2, 15))  # adjust as needed
            sns.heatmap(heatmap_data, cmap = cmap)
            plt.ylabel("Date")
            plt.xlabel("Repository")
            plt.title("Repository Stars Over Time")
            plt.tight_layout()
            plt.show()
        else:
            sns.heatmap(heatmap_data, cmap = cmap, ax = ax)
            # ax.ylabel("Date")
            # ax.xlabel("Repository")


all_repos = df['fullname'].drop_duplicates().tolist()

plotter = RepoStarsPlotter()

repos = ['pytorch/pytorch', 'huggingface/transformers', 'google-research/bert','keras-team/keras', 'openai/gym',]

# plotter.plot_single_metric(
#     repos,
#     ma_period=30
# )
freqs = ['D','W','M','Q','A']

fig, axs = plt.subplots(len(freqs), 1, figsize = (len(all_repos) / 2, 15 * len(freqs)))
for freq, ax in zip(freqs, axs):
    plotter.heatmap(all_repos, frequency=freq, ax = ax)
