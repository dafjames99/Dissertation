import sys
from pathlib import Path
from datetime import datetime, timezone as tz

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, ndcg_score
from tqdm import tqdm

src_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(src_path))

from utils.paths import DATA_DICT, EMBEDDING_DIR, EVAL_DIR  # noqa: E402
from representation import SimilarityCalculator  # noqa: E402
from models.weak.model import IntersectionFeatures  # noqa: E402


# ---------------------------- Labels (built once) ----------------------------
kws = pd.read_csv(DATA_DICT["models"]["weak"]["kws"])
kws["Category"] = kws["Category"].astype(str).str.strip()
intersect = IntersectionFeatures()

repos = intersect.df.index.tolist()  # R repos (ids)
jobs = intersect.df.columns.astype(int).tolist()  # J jobs (ids)
categories = sorted(kws["Category"].dropna().unique().tolist())  # K categories

R, J, K = len(repos), len(jobs), len(categories)
repo_idx = {r: i for i, r in enumerate(repos)}
job_idx = {j: i for i, j in enumerate(jobs)}
cat_idx = {c: i for i, c in enumerate(categories)}


def _build_matches(intersect):
    matches = {}
    for i, row in enumerate(intersect.df.itertuples(index=False, name=None)):
        for j, cell in enumerate(row):
            cell_ = []
            for match, match_type in cell:
                entry = {"match": match, "match_type": match_type}
                if match_type == "term":
                    entry["category"] = intersect._term_to_cats.get(match, [])
                else:
                    entry["category"] = intersect._acronym_to_cats.get(match, [])
                cell_.append(entry)
            matches[(i, j)] = cell_
    return matches


matches = _build_matches(intersect)

B = np.zeros((R, J, K), dtype=np.int8)
C = np.zeros_like(B, dtype=np.int32)  # optional counts

for (r, j), match_list in matches.items():
    i, jj = repo_idx[r], job_idx[j]
    seen_per_cat = set()
    for m in match_list:
        ks = [cat_idx.get(c) for c in (m.get("category") or [])]
        ks = [k for k in ks if k is not None]
        if not ks:
            continue
        for k in ks:
            C[i, jj, k] += 1
            seen_per_cat.add(k)
    for k in seen_per_cat:
        B[i, jj, k] = 1


# ---------------------------- Evaluation helpers ----------------------------
def eval_category(k, S, B, topk=(1, 5, 10)):
    APs, ndcgs = [], []
    p_at_k = {k_: [] for k_ in topk}
    prevalence_jobs = 0
    for jj in range(S.shape[1]):
        y_true = B[:, jj, k]
        if y_true.sum() == 0:
            continue
        y_scores = S[:, jj]
        APs.append(average_precision_score(y_true, y_scores))
        ndcgs.append(ndcg_score([y_true], [y_scores], k=10))
        order = np.argsort(-y_scores)
        for kk in topk:
            topk_indices = order[:kk]
            p_at_k[kk].append(y_true[topk_indices].sum() / kk)
        prevalence_jobs += 1

    return {
        "AP_mean": float(np.mean(APs)) if APs else np.nan,
        "NDCG_mean": float(np.mean(ndcgs)) if ndcgs else np.nan,
        "P@k_mean": {
            kk: (float(np.mean(v)) if v else np.nan) for kk, v in p_at_k.items()
        },
        "jobs_evaluated": int(prevalence_jobs),
    }


def evaluate_run(S, B):
    # Per-category
    category_results = {kname: eval_category(k, S, B) for kname, k in cat_idx.items()}

    # Weights by inverse prevalence (exclude zero prevalence)
    prevalences = np.array([B[:, :, k].sum() for k in range(K)])
    nonzero_mask = prevalences > 0
    if nonzero_mask.any():
        w = np.zeros_like(prevalences, dtype=float)
        w_nz = 1.0 / prevalences[nonzero_mask]
        w[nonzero_mask] = w_nz / w_nz.sum()
    else:
        w = np.ones_like(prevalences, dtype=float) / max(K, 1)

    aps = np.array(
        [
            category_results[c]["AP_mean"]
            if not np.isnan(category_results[c]["AP_mean"])
            else 0.0
            for c in categories
        ]
    )
    aggregate_ap_weighted = float((w * aps).sum())

    supported_mask = np.array(
        [category_results[c]["jobs_evaluated"] > 0 for c in categories], dtype=bool
    )
    macro_ap_equal = (
        np.nan if not supported_mask.any() else float(np.nanmean(aps[supported_mask]))
    )

    # MSE between weighted labels and normalized similarity
    S01 = (S - S.min()) / (S.max() - S.min() + 1e-12)
    B_weighted = np.tensordot(B, w, axes=([2], [0]))
    mse = float(np.mean((B_weighted - S01) ** 2))

    # Micro metrics (any-category positive)
    topk_micro = (1, 5, 10)
    micro_APs, micro_ndcgs = [], []
    micro_p_at_k = {k_: [] for k_ in topk_micro}
    jobs_evaluated_micro = 0
    for jj in range(S.shape[1]):
        y_true_any = (B[:, jj, :].sum(axis=1) > 0).astype(int)
        if y_true_any.sum() == 0:
            continue
        y_scores = S[:, jj]
        micro_APs.append(average_precision_score(y_true_any, y_scores))
        micro_ndcgs.append(ndcg_score([y_true_any], [y_scores], k=10))
        order = np.argsort(-y_scores)
        for kk in topk_micro:
            topk_indices = order[:kk]
            micro_p_at_k[kk].append(y_true_any[topk_indices].sum() / kk)
        jobs_evaluated_micro += 1

    micro = {
        "micro_jobs_evaluated": int(jobs_evaluated_micro),
        "micro_ap_mean": (
            float(np.mean(micro_APs)) if jobs_evaluated_micro > 0 else np.nan
        ),
        "micro_ndcg10_mean": (
            float(np.mean(micro_ndcgs)) if jobs_evaluated_micro > 0 else np.nan
        ),
        "micro_p_at_1": (
            float(np.mean(micro_p_at_k[1])) if jobs_evaluated_micro > 0 else np.nan
        ),
        "micro_p_at_5": (
            float(np.mean(micro_p_at_k[5])) if jobs_evaluated_micro > 0 else np.nan
        ),
        "micro_p_at_10": (
            float(np.mean(micro_p_at_k[10])) if jobs_evaluated_micro > 0 else np.nan
        ),
    }

    return {
        "aggregate_ap_weighted": aggregate_ap_weighted,
        "macro_ap_equal": macro_ap_equal,
        "mse": mse,
        "category_results": category_results,
        "prevalences": prevalences,
        "weights": w,
        **micro,
    }


# ---------------------------- Config discovery ----------------------------
def _embeddings_root() -> Path:
    # derive embeddings root from existing paths
    return EMBEDDING_DIR


def discover_configs() -> list[tuple[str, str, str | None, int | None]]:
    root = _embeddings_root()
    variants = [d.name for d in root.iterdir() if d.is_dir()]
    model_idxs = ["a", "b", "c"]
    configs = []

    for v in variants:
        vdir = root / v
        for m in model_idxs:
            # base (no PCA)
            repo_f = vdir / f"repositories_embeddings_{m}.npy"
            job_f = vdir / f"jobs_embeddings_{m}.npy"
            if repo_f.is_file() and job_f.is_file():
                configs.append((v, m, None, None))

            # PCA variants
            for repo_fp in vdir.glob(f"repositories_embeddings_{m}_pca*.npy"):
                dim_str = repo_fp.stem.split("_pca")[-1]
                if not dim_str.isdigit():
                    continue
                dim = int(dim_str)
                job_fp = vdir / f"jobs_embeddings_{m}_pca{dim}.npy"
                if job_fp.is_file():
                    configs.append((v, m, "pca", dim))

    # de-duplicate and keep stable order
    seen = set()
    uniq = []
    for cfg in configs:
        if cfg not in seen:
            uniq.append(cfg)
            seen.add(cfg)
    return uniq


# ---------------------------- Main loop ----------------------------
def main():
    # destination directory (agnostic to keyed mapping)
    eval_dir = EVAL_DIR
    eval_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    per_cat_rows = []

    configs = discover_configs()
    print(f"Discovered {len(configs)} embedding configuration(s) under {EMBEDDING_DIR}")
    print(f"Dataset sizes -> R: {R}, J: {J}, K: {K}")

    for text_variant, sent_model_idx, dr, dr_dim in tqdm(
        configs, desc="Evaluating configs", unit="cfg"
    ):
        # compute similarity
        sim = SimilarityCalculator(
            text_variant=text_variant,
            sentence_model_index=sent_model_idx,
            dr=dr,
            dr_dim=dr_dim,
        )
        S = sim.matrix  # (R x J)

        # evaluate
        res = evaluate_run(S, B)

        # run-level row
        run_row = {
            "run_id": f"{text_variant}_{sent_model_idx}"
            + (f"_pca{dr_dim}" if dr == "pca" and dr_dim else ""),
            "version": text_variant,
            "variant": sent_model_idx,
            "pca_dim": int(dr_dim) if (dr == "pca" and dr_dim) else pd.NA,
            "timestamp": datetime.now(tz=tz.utc).isoformat() + "Z",
            "R": int(R),
            "J": int(J),
            "K": int(K),
            "B_shape": str(tuple(B.shape)),
            "S_shape": str(tuple(S.shape)),
            "aggregate_ap_weighted": res["aggregate_ap_weighted"],
            "macro_ap": res["macro_ap_equal"]
            if not np.isnan(res["macro_ap_equal"])
            else np.nan,
            "mse_weighted_vs_similarity": res["mse"],
            "micro_jobs_evaluated": res["micro_jobs_evaluated"],
            "micro_ap_mean": res["micro_ap_mean"],
            "micro_ndcg10_mean": res["micro_ndcg10_mean"],
            "micro_p_at_1": res["micro_p_at_1"],
            "micro_p_at_5": res["micro_p_at_5"],
            "micro_p_at_10": res["micro_p_at_10"],
        }
        run_rows.append(run_row)

        # per-category rows
        for idx, cat_name in enumerate(categories):
            cres = res["category_results"].get(cat_name, {})
            per_cat_rows.append(
                {
                    "run_id": run_row["run_id"],
                    "version": text_variant,
                    "variant": sent_model_idx,
                    "pca_dim": run_row["pca_dim"],
                    "timestamp": run_row["timestamp"],
                    "category": cat_name,
                    "prevalence": int(res["prevalences"][idx]),
                    "weight": float(res["weights"][idx]),
                    "AP_mean": float(cres.get("AP_mean", np.nan))
                    if not np.isnan(cres.get("AP_mean", np.nan))
                    else np.nan,
                    "NDCG_mean": float(cres.get("NDCG_mean", np.nan))
                    if not np.isnan(cres.get("NDCG_mean", np.nan))
                    else np.nan,
                    "jobs_evaluated": int(cres.get("jobs_evaluated", 0)),
                    "P_at_1": float(cres.get("P@k_mean", {}).get(1, np.nan))
                    if not np.isnan(cres.get("P@k_mean", {}).get(1, np.nan))
                    else np.nan,
                    "P_at_5": float(cres.get("P@k_mean", {}).get(5, np.nan))
                    if not np.isnan(cres.get("P@k_mean", {}).get(5, np.nan))
                    else np.nan,
                    "P_at_10": float(cres.get("P@k_mean", {}).get(10, np.nan))
                    if not np.isnan(cres.get("P@k_mean", {}).get(10, np.nan))
                    else np.nan,
                }
            )

    # Write compiled CSVs once
    pd.DataFrame(run_rows).to_csv(eval_dir / "compiled_eval_runs.csv", index=False)
    pd.DataFrame(per_cat_rows).to_csv(
        eval_dir / "compiled_eval_per_category.csv", index=False
    )
    print(
        f"Saved: {eval_dir / 'compiled_eval_runs.csv'} and {eval_dir / 'compiled_eval_per_category.csv'}"
    )


if __name__ == "__main__":
    main()
