import time
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal
from IPython.display import display, HTML

src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))

from utils.plots import n_colors
from utils.paths import DATA_DICT, SENTENCE_MODEL


class DotDict(dict):
    """
    Upgrades a dict object so that we can create, access, edit and delete elements with .notation!
    e.g.
    obj = DotDict({'a': 1})

    Instead of
        obj['a'] # Output: 1

    we can write
        obj.a # Output: 1

    Much faster (especially for complex objects)!
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class UtilityClass(DotDict):
    def show_schema(self, obj=None, verbose=True):
        """
        Clean method to understand contents of a UtilityClass object
        EVERYTHING downstream from here extends this class!
        """
        if obj is None:
            obj = self  # start with the DataLoader instance

        schema = {}

        def type_info(value):
            tname = type(value).__name__
            if not verbose:
                return tname
            if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
                return f"{tname} (shape={value.shape})"
            elif isinstance(value, (list, tuple, set)):
                return f"{tname} (len={len(value)})"
            else:
                return tname

        for key, value in (
            obj.items() if isinstance(obj, dict) else obj.__dict__.items()
        ):
            if isinstance(value, (DotDict, dict)):
                schema[key] = self.show_schema(obj=value, verbose=verbose)
            else:
                schema[key] = type_info(value)

        return schema


def _maybe_reduced_path(base_path: Path, dr: str | None, dr_dim: int | None) -> Path:
    if dr == "pca" and dr_dim:
        return base_path.with_name(base_path.stem + f"_pca{dr_dim}" + base_path.suffix)
    return base_path


class DataLoader(UtilityClass):
    def __init__(
        self,
        text_variant: str,
        sentence_model_index: str,
        dr: str | None = None,
        dr_dim: int | None = None,
    ):
        self.job = DotDict()
        self.job.df = pd.read_csv(DATA_DICT["jobs"], index_col="job_id")
        self.job.df["fulltext"] = pd.read_csv(
            DATA_DICT["embeddings"][text_variant]["jobs_texts"]
        )["jobs_texts"]
        self.job.ids = self.job.df.index.to_numpy()
        self.job.dates = pd.to_datetime(self.job.df["date"])
        jobs_embed_path = _maybe_reduced_path(
            Path(
                DATA_DICT["embeddings"][text_variant][
                    f"jobs_embed_{sentence_model_index}"
                ]
            ),
            dr,
            dr_dim,
        )
        self.job.embedding = np.load(jobs_embed_path)
        # Ensure L2 normalization (especially important after dimensionality reduction)
        job_norms = np.linalg.norm(self.job.embedding, axis=1, keepdims=True)
        job_norms[job_norms == 0] = 1.0
        self.job.embedding = self.job.embedding / job_norms

        self.repo = DotDict()
        self.repo.df = pd.read_csv(DATA_DICT["github"]["repositories"]["metadata"])
        self.repo.df["fulltext"] = pd.read_csv(
            DATA_DICT["embeddings"][text_variant]["repositories_texts"]
        )["repositories_texts"]
        self.repo.ids = self.repo.df.repository.to_numpy()
        repos_embed_path = _maybe_reduced_path(
            Path(
                DATA_DICT["embeddings"][text_variant][
                    f"repositories_embed_{sentence_model_index}"
                ]
            ),
            dr,
            dr_dim,
        )
        self.repo.embedding = np.load(repos_embed_path)
        # Ensure L2 normalization (especially important after dimensionality reduction)
        repo_norms = np.linalg.norm(self.repo.embedding, axis=1, keepdims=True)
        repo_norms[repo_norms == 0] = 1.0
        self.repo.embedding = self.repo.embedding / repo_norms

        self.stars = DotDict()
        self.stars.D = pd.read_parquet(DATA_DICT["github"]["star_wide"]["daily"])
        self.stars.M = pd.read_parquet(DATA_DICT["github"]["star_wide"]["monthly"])
        self.stars.Q = pd.read_parquet(DATA_DICT["github"]["star_wide"]["monthly"])


class SimilarityCalculator(UtilityClass):
    def __init__(
        self,
        dataloader: DataLoader = None,
        text_variant=None,
        sentence_model_index=None,
        dr: str | None = None,
        dr_dim: int | None = None,
        **kwargs,
    ):
        if not dataloader:
            try:
                self.data = DataLoader(
                    text_variant, sentence_model_index, dr=dr, dr_dim=dr_dim
                )
            except Exception:
                raise ValueError(
                    "If no data loader, need valid text_variant and sentence_model_index"
                )
        else:
            self.data = dataloader
        self.matrix = self.data.repo.embedding @ self.data.job.embedding.T
        self.df = pd.DataFrame(
            self.matrix, columns=self.data.job.ids, index=self.data.repo.ids
        )

    def describe(self, axis=0, **kwargs):
        """Returns a pd.describe() object of the desired axis (0 = By Repository, 1 = By Job)"""
        if axis == 0:
            return self.df.T.describe(**kwargs)
        return self.df.describe(**kwargs)

    def n_repo_sort(
        self,
        job_id: int,
        n: int = 1,
        ascending=False,
        index_only=False,
        value_only=False,
    ) -> pd.Series:
        return self.df[job_id].sort_values(ascending=ascending).iloc[:n]

    def demonstrate(
        self, job_id: int, repo: Literal["highest", "lowest"] | str | int = None
    ):
        if repo == "highest":
            repo = self.n_repo_sort(job_id, ascending=False).index[0]
        elif repo == "lowest":
            repo = self.n_repo_sort(job_id, ascending=True).index[0]

        if not isinstance(repo, int):
            repo = np.argwhere(self.data.repo.ids == repo)[0][0]

        job_title = self.data.job.df.title.iloc[job_id]
        repo_name = self.data.repo.ids[repo]
        print(repo)
        job_text = self.data.job.df.fulltext.iloc[job_id]
        repo_text = self.data.repo.df.fulltext.iloc[repo]

        display(
            HTML(
                f"""
        <h1> <strong> Score: </strong> {self.matrix[repo, job_id]:.4f} </h1> 
        <h1> <strong> Job Text: {job_title}</strong></h1> 
        <p> {job_text} </p>
        <h1> <strong> Repository Text: {repo_name} </strong></h1> 
        <p> {repo_text} </p>
        """
            )
        )


# sim_v1 = SimilarityCalculator(
#     text_variant='v1',
#     sentence_model_index='b'
# )

# sim_v2 = SimilarityCalculator(
#     text_variant='v2',
#     sentence_model_index='b'
# )
# i = 15000
# sim_v1.demonstrate(i, 'highest')
# sim_v2.demonstrate(i, 'highest')


class EmbeddingSimilarity:
    def __init__(self, sentence_model_index, text_variant):
        self.text_variant = text_variant
        self.job_texts = pd.read_csv(
            DATA_DICT["embeddings"][self.text_variant]["jobs_texts"]
        )["jobs_texts"]
        self.repo_texts = pd.read_csv(
            DATA_DICT["embeddings"][self.text_variant]["repositories_texts"]
        )["repositories_texts"]

        self.df_jobs = pd.read_csv(DATA_DICT["jobs"], index_col="job_id")
        self.df_repos = pd.read_csv(DATA_DICT["github"]["repositories"]["metadata"])

        self.job_date = pd.to_datetime(self.df_jobs["date"])
        self.job_id = self.df_jobs.index.tolist()

        self.repositories = self.df_repos["repository"].tolist()
        self.stars = {
            "D": pd.read_parquet(DATA_DICT["github"]["star_wide"]["daily"]),
            "M": pd.read_parquet(DATA_DICT["github"]["star_wide"]["monthly"]),
            "Q": pd.read_parquet(DATA_DICT["github"]["star_wide"]["quarterly"]),
        }

        self.df_jobs["full_text"] = self.job_texts
        self.df_repos["full_text"] = self.repo_texts

        self.sentence_model = SENTENCE_MODEL[sentence_model_index]

        self.job_embedding = np.load(
            DATA_DICT["embeddings"][text_variant][f"jobs_embed_{sentence_model_index}"]
        )
        self.repo_embedding = np.load(
            DATA_DICT["embeddings"][text_variant][
                f"repositories_embed_{sentence_model_index}"
            ]
        )

        self.matrix = self.repo_embedding @ self.job_embedding.T
        self.df = pd.DataFrame(
            self.matrix, columns=self.job_id, index=self.repositories
        )

    def n_most_similar(self, job_id, n):
        col = self.df[job_id].copy()
        col = col.sort_values(ascending=False).iloc[:n]
        return col

    def plot_top_n(self, job_id, n):
        x = self.n_most_similar(job_id, n)

        fig, ax = plt.subplots(figsize=(10, 5))  # wider for long labels

        ax.bar(x.index, x.values, color="skyblue")

        plt.xticks(rotation=45, ha="right")

        ax.set_ylabel("Similarity Score")
        ax.set_xlabel("Repository")
        ax.set_title(f"Top {n} Repositories Similar to Job ID {0}")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    def inspect_job_similarity(self, job_id: int, n: int = 10, n_descriptions: int = 1):
        top_repos = self.n_most_similar(job_id, n)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(top_repos.index, top_repos.values, color="mediumpurple")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        ax.set_ylabel("Similarity Score")
        ax.set_xlabel("Repository")
        ax.set_title(f"Top {n} Similar Repositories for Job ID {job_id}")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

        print("\n📄 Job Description:\n" + "-" * 80)
        print(self.df_jobs.loc[job_id, "description"])

        print("\n📘 Top Repository Descriptions:\n" + "-" * 80)
        for repo in top_repos.index[:n_descriptions]:
            repo_desc = self.df_repos.loc[
                self.df_repos["repository"] == repo, "description"
            ]
            desc_text = (
                repo_desc.values[0]
                if not repo_desc.empty
                else "No description available"
            )
            print(f"\n🔹 {repo}:\n{desc_text}")

    def candlestick(
        self, axis=0, sort_by_mean=True, long=20, small=10, ax=None, grid=None, **kwargs
    ):
        if ax is None:
            figsize = (
                (small, long)
                if kwargs.get("orientation") == "horizontal"
                else (long, small)
            )
            fig, ax = plt.subplots(figsize=figsize)

        grid = (
            grid
            if grid is not None
            else ("y" if kwargs.get("orientation") == "horizontal" else "x")
        )
        ax.grid(axis=grid, linestyle="--", color="r", alpha=0.5)

        if sort_by_mean:
            mean = self.df.mean(axis=(1 if axis == 0 else 0)).sort_values(
                ascending=False
            )
            indices = mean.index
        else:
            indices = self.repositories if axis == 0 else self.job_id

        values = (
            [self.df.loc[i] for i in indices]
            if axis == 0
            else [self.df[i] for i in indices]
        )

        ax.boxplot(values, tick_labels=indices, **kwargs)

    def similarity_stats(self, axis=0):
        values = (
            [self.df.loc[i] for i in self.repositories]
            if axis == 0
            else [self.df[i] for i in self.job_id]
        )
        means, stds = [v.mean() for v in values], [v.std() for v in values]
        return means, stds

    def plot_stats(self, plot_type="hist", axis=0, axs=None, ax_lab_add=None, **kwargs):
        means, stds = self.similarity_stats(axis=axis)
        if plot_type == "hist":
            if axs is None:
                figsize = (
                    (20, 10) if kwargs.get("figsize") is None else kwargs.get("figsize")
                )
                fig, axs = plt.subplots(1, 2, figsize=figsize)
            axs[0].hist(means, **kwargs)
            ax_lab = f"{'Repositories' if axis == 0 else 'Jobs'} Mean Value"
            ax_lab = ax_lab_add + " " + ax_lab if ax_lab_add is not None else ax_lab
            axs[0].set_xlabel(ax_lab)
            axs[1].hist(stds, **kwargs)
            ax_lab = f"{'Repositories' if axis == 0 else 'Jobs'} Standard Deviation"
            ax_lab = ax_lab_add + " " + ax_lab if ax_lab_add is not None else ax_lab
            axs[1].set_xlabel(ax_lab)
            # fig.suptitle(f'{self.sentence_model} - {'Repositories' if axis == 0 else 'Jobs'} Statistics Histogram')
        elif plot_type == "scatter":
            if axs is None:
                figsize = (
                    (20, 10) if kwargs.get("figsize") is None else kwargs.get("figsize")
                )
                fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(means, stds, **kwargs)
            ax.set_xlabel(f"{'Repositories' if axis == 0 else 'Jobs'} Mean Value")
            ax.set_ylabel(
                f"{'Repositories' if axis == 0 else 'Jobs'} Standard Deviation"
            )
            ax.set_title(
                f"{self.sentence_model} - {'Repositories' if axis == 0 else 'Jobs'} Statistics Histogram"
            )

    @staticmethod
    def combine_scatters(*sims: "EmbeddingSimilarity", **kwargs):
        figsize = (20, 10) if kwargs.get("figsize") is None else kwargs.get("figsize")
        colors = (
            n_colors(len(sims), "high_contrast")
            if kwargs.get("c") is None
            else kwargs.get("c")
        )
        fig, ax = plt.subplots(figsize=figsize)
        for sim, c in zip(sims, colors):
            means, stds = sim.similarity_stats()
            ax.scatter(means, stds, c=c, label=sim.sentence_model, **kwargs)
        ax.legend()

    @staticmethod
    def compare_models(*sims: "EmbeddingSimilarity", **kwargs):
        figsize = (20, 10) if kwargs.get("figsize") is None else kwargs.get("figsize")
        colors = (
            n_colors(len(sims), "high_contrast")
            if kwargs.get("c") is None
            else kwargs.get("c")
        )
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        sim_stats = [sim.similarity_stats() for sim in sims]
        labels = [sim.sentence_model for sim in sims]
        for i, (ax, xlabel) in enumerate(zip(axs, ["Mean", "Standard Deviations"])):
            ax.grid(axis="y", linestyle="--", color="r", alpha=0.5)
            ax.boxplot([sim_stats[j][i] for j in range(len(sims))], tick_labels=labels)
            ax.set_title(f"{xlabel}")
        fig.suptitle(
            "Comparing Summary Statistics of Sentence Models\nSummarized Across Repositories"
        )
        plt.show()

    def similarity_ts(self, repository):
        try:
            row = self.df.loc[repository]
        except:
            print('Use a valid repository name: "owner/repo"')
            return
        v = row.values
        d, m, q = [], [], []
        return pd.DataFrame(
            data={
                "D": self.job_date,
                "M": self.job_date.dt.to_period("M"),
                "Q": self.job_date.dt.to_period("Q"),
                "values": v,
            }
        )

    def group_similarity_periods(self, repository, period, method=np.mean):
        df = self.similarity_ts(repository)
        return df.groupby(period)["values"].apply(lambda x: method(x))

    def repo_stars_similarity(
        self,
        repository: str,
        period: Literal["D", "M", "Q"],
        *methods,
        drop_zero=False,
        add_cumulative=True,
    ) -> pd.DataFrame:
        """
        Merge GitHub star counts and job similarity scores for a repository.

        Parameters
        ----------
        repository : str
            Full repository name in "owner/repo" format.
        period : {"D", "M", "Q"}, default="Q"
            Time period for grouping: daily, monthly, or quarterly.
        method : callable, default=np.mean
            Aggregation function for similarity scores (e.g., np.mean, np.sum).
        drop_zero: bool, default=False
            if True, drops rows where stars == 0 (heuristic for returning only after creation)
            Useful for a simplified view - Might be misleading!
        add_cumulative: bool, default=True
            Choose whether to add a stars_cum (cumulative sum) column
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: [period, stars, <method>_similarity].
        """
        for i, method in enumerate(methods):
            sim_ser = self.group_similarity_periods(repository, period, method).copy()
            if i == 0:
                sim_df = pd.DataFrame({f"{method.__name__}_similarity": sim_ser})
            else:
                sim_df[f"{method.__name__}_similarity"] = sim_ser
        # Get star series

        star_ser = self.stars[period].loc[repository].copy()
        star_ser.index = pd.to_datetime(star_ser.index).to_period(period)
        star_df = pd.DataFrame({"stars": star_ser})

        # Get similarity series

        # Merge
        merged = star_df.merge(
            sim_df,
            how="outer",
            left_on=star_df.index,
            right_on=sim_df.index,
            suffixes=("_stars", "_similarity"),
        )
        merged.rename(columns={"key_0": period}, inplace=True)
        if add_cumulative:
            merged["stars_cum"] = merged["stars"].cumsum()
        merged["stars_norm"] = (merged["stars"] - merged["stars"].min()) / (
            merged["stars"].max() - merged["stars"].min()
        )

        # Fill missing similarities with 0
        for method in methods:
            merged[f"{method.__name__}_similarity"] = merged[
                f"{method.__name__}_similarity"
            ].fillna(0)
        if drop_zero:
            return merged[merged["stars"] != 0]
        return merged


def plot_time_series(df, period_col="Q", star_col="stars", similarity_cols=None):
    if similarity_cols is None:
        similarity_cols = [col for col in df.columns if col.endswith("_similarity")]
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot stars
    ax1.plot(df[period_col].astype(str), df[star_col], color="tab:blue", label="Stars")
    ax1.set_xlabel("Period")
    ax1.set_ylabel("Stars", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Plot similarities on secondary y-axis
    ax2 = ax1.twinx()
    for col in similarity_cols:
        ax2.plot(df[period_col].astype(str), df[col], label=col)
    ax2.set_ylabel("Similarity")
    ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.title("Stars and Similarity Metrics Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_similarity_ribbon(df, period_col="Q", remove_zeros=True):
    # Convert period index to string for x-axis labels if needed
    if remove_zeros:
        df = df[df["count_nonzero_similarity"] != 0].copy()
    x = df[period_col].astype(str) if period_col in df.columns else df.index.astype(str)
    mean = df["mean_similarity"]
    std = df["std_similarity"]
    min_ = df["min_similarity"]
    max_ = df["max_similarity"]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot mean line
    ax.plot(x, mean, label="Mean Similarity", color="blue")

    # Fill between mean - std and mean + std
    ax.fill_between(
        x, mean - std, mean + std, color="blue", alpha=0.2, label="Mean ± Std Dev"
    )

    # Plot min and max as dashed lines
    ax.plot(x, min_, label="Min Similarity", color="green", linestyle="--")
    ax.plot(x, max_, label="Max Similarity", color="red", linestyle="--")

    ax.set_xlabel("Period")
    ax.set_ylabel("Similarity Score")
    ax.set_title("Similarity Statistics Over Time")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# sim = EmbeddingSimilarity("b", "v1")

# df = sim.repo_stars_similarity(
#     "pytorch/pytorch",
#     "Q",
#     np.mean,
#     np.median,
#     np.count_nonzero,
#     np.sum,
#     np.min,
#     np.max,
#     np.std,
#     np.var,
#     drop_zero=True,
# )


# similarity_cols = [col for col in df.columns if col.endswith("_similarity")]

# plot_time_series(
#     df, period_col="Q", star_col="stars_norm", similarity_cols=similarity_cols
# )
# plot_similarity_ribbon(df)

# -----------------------------------------------------
# -------------- COMPARISON of v1 and v2 --------------
# -----------------------------------------------------

# sim_v1 = EmbeddingSimilarity("b", "v1")
# sim_v2 = EmbeddingSimilarity("b", "v2")

# long, small = 20, 10

# fig, axs = plt.subplots(1, 2, figsize=(long, long))

# sim_v1.candlestick(orientation="horizontal", showfliers=False, ax=axs[0], grid="both")
# sim_v2.candlestick(orientation="horizontal", showfliers=False, ax=axs[1], grid="both")

# fig, axs = plt.subplots(2, 2, figsize=(long, small))
# bins = 30

# sim_v1.plot_stats(bins=bins, axs=[axs[0][0], axs[0][1]], ax_lab_add="V1")
# sim_v2.plot_stats(bins=bins, axs=[axs[1][0], axs[1][1]], ax_lab_add="V2")


# -----------------------------------------------------
# -------------- TIME-SERIES Analysis--- --------------
# -----------------------------------------------------
a = SimilarityCalculator(None, "v2", "b", "pca", 64)
a.data.repo.embedding[0]
