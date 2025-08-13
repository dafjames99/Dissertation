from sklearn.decomposition import PCA
from pathlib import Path
import argparse
import numpy as np
from utils.paths import DATA_DICT


def _suffix_path(path: Path, suffix: str) -> Path:
    return path.with_name(path.stem + suffix + path.suffix)


def reduce_with_pca(
    text_variant: str, sentence_model_index: str, n_components: int
) -> tuple[Path, Path]:
    jobs_path = Path(
        DATA_DICT["embeddings"][text_variant][f"jobs_embed_{sentence_model_index}"]
    )
    repos_path = Path(
        DATA_DICT["embeddings"][text_variant][
            f"repositories_embed_{sentence_model_index}"
        ]
    )

    jobs = np.load(jobs_path)
    repos = np.load(repos_path)

    X = np.vstack([jobs, repos])
    pca = PCA(n_components=n_components, random_state=0)
    Xr = pca.fit_transform(X)

    jobs_r = Xr[: len(jobs)]
    repos_r = Xr[len(jobs) :]

    out_suffix = f"_pca{n_components}"
    jobs_out = _suffix_path(jobs_path, out_suffix)
    repos_out = _suffix_path(repos_path, out_suffix)
    jobs_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(jobs_out, jobs_r)
    np.save(repos_out, repos_r)
    print(f"Saved PCA embeddings:\n- {jobs_out}\n- {repos_out}")
    return jobs_out, repos_out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_variant", required=True, choices=["v1", "v2"])
    ap.add_argument("--sentence_model_index", required=True, choices=["a", "b", "c"])
    ap.add_argument("--dim", required=True, type=int)
    args = ap.parse_args()
    reduce_with_pca(args.text_variant, args.sentence_model_index, args.dim)
