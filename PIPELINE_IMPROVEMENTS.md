# Pipeline Improvement Suggestions

## Executive Summary
- Current state: Partially functional. The core building blocks exist, but several blocking issues prevent a clean end-to-end run.
- Primary fixes: unify embedding I/O and naming, correct CLI argument validation, standardize metadata usage, anchor all paths to project root, secure secrets, and guard heavy code behind main blocks.

## Blocking Issues (to fix first)
- CLI validation bugs (cause immediate failure):
  - `models/embed_create.py` and `preprocessing/text_embed_prepare.py` use `x not in x` instead of membership checks against the allowed lists.
  - Fix to: `if text_variant not in available_variants` and `if sentence_model_index not in available_sentence_models`.
- Embedding I/O mismatch:
  - Embeddings are saved under variant-specific keys, while `representation.py` expects canonical files under `embeddings/value_dir` with different names.
  - Decide one convention and apply consistently for both writing and reading (see “I/O Contract” below).
- Metadata inconsistency:
  - Aggregated repo metadata is produced as `meta_data2`, but some readers still target `meta_data` and/or a `tags` column instead of `topics`.
- Wrong key for stars output:
  - Writer uses a non-existent `star_count` key; the registry provides `stars_dir`.
- Secrets committed to source control:
  - `GITHUB_TOKEN` is hard-coded in `utils/vars.py`.
- Top-level execution on import:
  - `bertopic_pipeline.py` and `representation.py` perform heavy work at module import. Should be gated under `if __name__ == "__main__":` and/or exposed as CLIs.
- CWD-sensitive paths:
  - `utils/paths.py` builds relative paths from the current working directory, not the project root.

## Unification of Embedding I/O Contract
Pick one pattern and make all producers/consumers conform.

Recommended:
- Directory structure: `data/embeddings/values/{text_variant}/`
- Filenames: `{channel}__{model_key}.npy` where:
  - `channel ∈ {jobs, repositories}`
  - `model_key ∈ SENTENCE_MODEL.keys()` (e.g., `a`, `b`, `c`)
- Text files: `data/embeddings/{text_variant}/{channel}.csv` with a single column `{channel}_texts`.

Implications:
- `CreateEmbedding.generate_embeddings()` writes to:
  - `values/{text_variant}/{channel}__{model_key}.npy`
- `representation.py` reads from the same path and should accept CLI args:
  - `--text_variant` and `--sentence_model_index`

Alternative (less preferred): keep flat `value_dir`, but include `{text_variant}` in the filename, e.g., `{channel}__{text_variant}__{model_key}.npy`.

## Path Management (anchor to repo root)
- In `utils/paths.py`, define `ROOT = Path(__file__).resolve().parents[2]` (from `src/utils/paths.py`) and construct all directories from `ROOT`.
- Replace `Path('data')` with `ROOT / 'src' / 'data'` or (preferably) `ROOT / 'src' / 'data'` only if all data lives under `src`. If data should be outside `src`, use `ROOT / 'data'` and adjust references accordingly.
- This removes CWD dependence and makes all scripts runnable from anywhere.

## Metadata Standardization
- Use `meta_data2` as the single canonical repo metadata file across the codebase (or rename to `meta_data` and update writers and readers uniformly).
- Ensure downstream code reads `topics` (string of comma-separated topics) and not `tags` unless you intentionally transform it.

## CLI Validation and UX
- Fix membership checks:
  - `if text_variant not in available_variants: raise ValueError(...)`
  - `if sentence_model_index not in available_sentence_models: raise ValueError(...)`
- Add `choices=` in `argparse` (or Typer) to enforce valid inputs automatically.
- Provide clear `--help` descriptions and echo effective config at runtime.

## Security and Config
- Remove `GITHUB_TOKEN` from `utils/vars.py`.
- Load from environment: `os.environ['GITHUB_TOKEN']` (optionally via `.env` + `python-dotenv`).
- Add any credentials file to `.gitignore` and keep secure.

## Execution Guards and Module Structure
- Wrap all heavy code under `if __name__ == "__main__":`.
- Expose reusable functions for library use; keep CLIs thin wrappers.
- Consider a unified CLI (e.g., Typer) with subcommands:
  - `collect` (GitHub data)
  - `prepare-text` (variants)
  - `embed` (channels/models)
  - `similarity` (compute/plot)
  - `topics` (BERTopic)

## Consistency in Normalization and Similarity
- You currently encode with `normalize_embeddings=True` and use dot-product for similarity (cosine). Keep this consistent across all embedding creation, and document it.
- If you have legacy files without normalization, regenerate or normalize on load to avoid silent drift.

## Reproducibility and Logging
- Add light logging (start/end, shapes, paths, counts, model names) to all CLIs.
- Record run metadata with outputs (model key, text variant, date, commit hash if available).
- Seed any stochastic components when applicable (e.g., topic modeling, UMAP, KMeans).

## Dependency and Resource Notes
- Ensure `en_core_web_sm` and NLTK `stopwords` are installed in `requirements.txt` (or auto-download with a clear message).
- Validate GPU/CPU usage and batch sizes for `SentenceTransformer.encode`.

## Minimal “Happy Path” (after fixes)
1) Data acquisition (populate metadata and optional stars):
   - `python src/data-acquisition/github_data.py --include_stars False`
2) Text preparation (select variant):
   - `python src/preprocessing/text_embed_prepare.py --text_variant texts2`
3) Embedding generation:
   - `python src/models/embed_create.py --text_variant texts2 --sentence_model_index b`
4) Similarity analysis (use same variant and model key):
   - `python src/representation.py --text_variant texts2 --sentence_model_index b`

## Action Checklist
- [ ] Fix CLI membership checks in `embed_create.py` and `text_embed_prepare.py`.
- [ ] Unify embedding I/O (directory + filename convention) and update `representation.py` to match.
- [ ] Standardize on `meta_data2` (or rename to `meta_data`) and update all readers; use `topics` not `tags`.
- [ ] Correct stars output key to `stars_dir`.
- [ ] Anchor paths in `utils/paths.py` to the project root.
- [ ] Move `GITHUB_TOKEN` to environment; remove from `utils/vars.py`.
- [ ] Add `if __name__ == "__main__":` guards and convert heavy modules to CLIs.
- [ ] Add logging and basic run metadata recording.
- [ ] Confirm `en_core_web_sm` and NLTK stopwords are installed or auto-downloaded.

## Optional Enhancements
- Use Parquet for large intermediate tables (metadata, text variants) for speed.
- Add small validation scripts that assert required inputs exist before each stage.
- Add a Makefile or simple task runner targets (`make collect`, `make prepare`, `make embed`, `make sim`). 