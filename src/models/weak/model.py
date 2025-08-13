from __future__ import annotations
from pathlib import Path
import re
from typing import Iterable, List, Optional, Tuple
import sys
import pandas as pd
import ast
from IPython.display import display, HTML
import argparse

try:
    from tqdm.auto import tqdm  # type: ignore

    _HAS_TQDM = True
except Exception:
    tqdm = None  # type: ignore
    _HAS_TQDM = False

from rapidfuzz import fuzz

try:
    import spacy

    _HAS_SPACY = True
except ImportError:
    spacy = None
    _HAS_SPACY = False

src_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(src_path))
from utils.paths import DATA_DICT  # noqa: E402


def _compile_phrase_patterns(phrases: Iterable[str]) -> List[Tuple[str, re.Pattern]]:
    compiled = []
    for phrase in phrases:
        if not phrase:
            continue
        stripped = phrase.strip()
        if not stripped:
            continue
        pattern_text = rf"(?<!\w){re.escape(stripped)}(?!\w)"
        compiled.append((stripped, re.compile(pattern_text, flags=re.IGNORECASE)))
    return compiled


class WeakKeywordModel:
    def __init__(
        self,
        glossary_path: Optional[Path] = None,
        use_lemmatization: bool = False,
        use_fuzzy: bool = False,
        fuzzy_threshold: int = 90,
    ):
        self.use_lemmatization = use_lemmatization
        self.use_fuzzy = use_fuzzy
        self.fuzzy_threshold = fuzzy_threshold
        if glossary_path is None:
            glossary_path = DATA_DICT["models"]["weak"]["kws"]
        self.glossary_path = glossary_path
        glossary_df = pd.read_csv(self.glossary_path)
        self._term_entries = _compile_phrase_patterns(
            glossary_df["Term"]
            .dropna()
            .astype(str)
            .map(str.strip)
            .loc[lambda s: s.ne("")]
            .unique()
        )
        self._acronym_entries = _compile_phrase_patterns(
            glossary_df["Acronym"]
            .dropna()
            .astype(str)
            .map(str.strip)
            .loc[lambda s: s.ne("")]
            .unique()
        )
        if self.use_lemmatization and _HAS_SPACY:
            if not hasattr(WeakKeywordModel, "_nlp"):
                WeakKeywordModel._nlp = spacy.load("en_core_web_sm")
        else:
            WeakKeywordModel._nlp = None

    def _find_matches(self, text: str) -> List[Tuple[str, str]]:
        matches = []
        if not isinstance(text, str) or not text:
            return matches
        processed_text = self._preprocess_text(text)
        tokens = processed_text.split()

        def ngrams(tokens, n):
            return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

        if self.use_fuzzy:
            for phrase, _ in self._term_entries:
                processed_phrase = self._preprocess_text(phrase)
                n = len(processed_phrase.split())
                candidates = ngrams(tokens, n)
                if any(
                    fuzz.ratio(processed_phrase, candidate) >= self.fuzzy_threshold
                    for candidate in candidates
                ):
                    matches.append((phrase, "term"))
            for phrase, _ in self._acronym_entries:
                processed_phrase = self._preprocess_text(phrase)
                n = len(processed_phrase.split())
                candidates = ngrams(tokens, n)
                if any(
                    fuzz.ratio(processed_phrase, candidate) >= self.fuzzy_threshold
                    for candidate in candidates
                ):
                    matches.append((phrase, "acronym"))
        else:
            for phrase, pat in self._term_entries:
                if pat.search(text):
                    matches.append((phrase, "term"))
            for phrase, pat in self._acronym_entries:
                if pat.search(text):
                    matches.append((phrase, "acronym"))
        return matches

    def transform(
        self,
        data: pd.DataFrame | pd.Series,
        text_col: Optional[str] = None,
        progress: bool = False,
        use_lemmatization: Optional[bool] = None,
        use_fuzzy: Optional[bool] = None,
        fuzzy_threshold: Optional[int] = None,
    ) -> pd.DataFrame:
        if isinstance(data, pd.Series):
            df = data.to_frame(name=text_col or "text").copy()
            text_col_name = df.columns[0]
        else:
            if not text_col or text_col not in data.columns:
                raise ValueError("text_col must be provided and exist in the DataFrame")
            df = data.copy()
            text_col_name = text_col

        # Allow runtime override of matching flags
        orig_lemmatization = self.use_lemmatization
        orig_fuzzy = self.use_fuzzy
        orig_threshold = self.fuzzy_threshold
        if use_lemmatization is not None:
            self.use_lemmatization = use_lemmatization
        if use_fuzzy is not None:
            self.use_fuzzy = use_fuzzy
        if fuzzy_threshold is not None:
            self.fuzzy_threshold = fuzzy_threshold

        iterator = df[text_col_name]
        if progress and _HAS_TQDM:
            iterator = tqdm(iterator, total=len(df), desc="Matching")

        records = []
        for row_id, text in zip(df.index, iterator):
            for match, match_type in self._find_matches(text):
                records.append(
                    {
                        "row_id": row_id,
                        "match": match,
                        "match_type": match_type,
                    }
                )

        # Restore original flags
        self.use_lemmatization = orig_lemmatization
        self.use_fuzzy = orig_fuzzy
        self.fuzzy_threshold = orig_threshold
        return pd.DataFrame(records, columns=["row_id", "match", "match_type"])

    @staticmethod
    def intersection_matrix(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        row_ids1,
        row_ids2,
        progress: bool = False,
    ) -> pd.DataFrame:
        """
        For each pair (id1, id2), compute the intersection of (match, match_type) between df1 and df2.
        Returns a DataFrame with row_ids2 as rows and row_ids1 as columns.
        """

        # Build lookup: id -> set of (match, match_type)
        def build_lookup(df):
            return {
                row_id: set(zip(grp["match"], grp["match_type"]))
                for row_id, grp in df.groupby("row_id")
            }

        lookup1 = build_lookup(df1)
        lookup2 = build_lookup(df2)

        out = {}
        col_iter = row_ids1
        if progress and _HAS_TQDM:
            col_iter = tqdm(row_ids1, desc="Columns")
        for c in col_iter:
            set1 = lookup1.get(c, set())
            out_col = {}
            row_iter = row_ids2
            if progress and _HAS_TQDM:
                row_iter = tqdm(row_ids2, desc="Rows", leave=False)
            for r in row_iter:
                set2 = lookup2.get(r, set())
                intersection = list([list(x) for x in (set1 & set2)])
                out_col[r] = intersection
            out[c] = out_col
        return pd.DataFrame(out)

    def display_highlighted_text(
        self, row_id, text_csv_path, matches_csv_path, text_col=None
    ):
        """
        Display the text for a given row_id with all matched terms highlighted.
        - text_csv_path: path to CSV with the original texts (must have row_id as index or column)
        - matches_csv_path: path to CSV with matches (long-form: row_id, match, match_type)
        - text_col: name of the text column (if not the first column)
        """
        texts = pd.read_csv(text_csv_path)
        matches = pd.read_csv(matches_csv_path)
        if text_col is None:
            text_col = (
                texts.columns[1] if texts.columns[0] == "row_id" else texts.columns[0]
            )
        # Get the text for this row_id
        text = texts.iloc[row_id][text_col]
        # Get all matches for this row_id
        row_matches = (
            matches.loc[matches["row_id"] == row_id, "match"].dropna().unique()
        )
        # Sort matches by length descending to avoid partial overlaps
        sorted_matches = sorted(row_matches, key=len, reverse=True)
        # Highlight all matches in the text
        highlighted = text
        for match in sorted_matches:
            highlighted = re.sub(
                rf"(?<!\\w)({re.escape(match)})(?!\\w)",
                r"<mark>\1</mark>",
                highlighted,
                flags=re.IGNORECASE,
            )
        display(HTML(highlighted))

    def _preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        if (
            self.use_lemmatization
            and _HAS_SPACY
            and getattr(WeakKeywordModel, "_nlp", None)
        ):
            doc = WeakKeywordModel._nlp(text)
            text = " ".join([token.lemma_ for token in doc])
        # Optionally, strip punctuation (can be extended)
        text = re.sub(r"[\W_]+", " ", text)
        return text.strip()


class IntersectionFeatures:
    def __init__(self, path: Optional[Path] = None):
        if path is None:
            path = DATA_DICT["models"]["weak"]["intersection"]
        # Load the intersection matrix, parsing list-like strings
        self.df = pd.read_csv(
            path,
            converters={
                col: ast.literal_eval for col in pd.read_csv(path, nrows=1).columns
            },
        )
        # Load glossary and build lookup maps from term/acronym to categories
        try:
            glossary = pd.read_csv(
                DATA_DICT["models"]["weak"]["kws"]
            )  # Term, Acronym, Category
        except Exception:
            glossary = pd.DataFrame(columns=["Term", "Acronym", "Category"])  # fallback

        # Normalize whitespace on key columns to prevent duplicate categories due to stray spaces
        for col in ["Term", "Acronym", "Category"]:
            if col in glossary.columns:
                glossary[col] = glossary[col].astype(str).str.strip()

        # Build maps allowing a term/acronym to belong to multiple categories
        if {"Term", "Category"}.issubset(glossary.columns):
            term_grp = glossary.dropna(subset=["Term", "Category"]).astype(
                {"Term": str, "Category": str}
            )
            self._term_to_cats = (
                term_grp.groupby("Term")["Category"]
                .apply(lambda s: sorted(set(s)))
                .to_dict()
            )
        else:
            self._term_to_cats = {}

        if {"Acronym", "Category"}.issubset(glossary.columns):
            acr_grp = glossary.dropna(subset=["Acronym", "Category"]).astype(
                {"Acronym": str, "Category": str}
            )
            self._acronym_to_cats = (
                acr_grp.groupby("Acronym")["Category"]
                .apply(lambda s: sorted(set(s)))
                .to_dict()
            )
        else:
            self._acronym_to_cats = {}

    def compute_size_matrix(self) -> pd.DataFrame:
        """Return a DataFrame of the same shape, with each cell as len(cell)."""
        return self.df.applymap(lambda x: len(x) if isinstance(x, list) else 0)

    def compute_binary_matrix(self) -> pd.DataFrame:
        """Return a DataFrame: 1 if cell is non-empty list, else 0."""
        return self.df.applymap(lambda x: int(bool(x)) if isinstance(x, list) else 0)

    def compute_custom_matrix(self, func) -> pd.DataFrame:
        """Apply a custom function to each cell."""
        return self.df.applymap(func)

    def _categories_for(self, match: str, match_type: str) -> list[str]:
        """Lookup categories for a (match, match_type) using the glossary maps."""
        if not isinstance(match, str):
            return []
        if match_type == "term":
            return self._term_to_cats.get(match, [])
        if match_type == "acronym":
            return self._acronym_to_cats.get(match, [])
        # Fallback if match_type is unexpected
        return self._term_to_cats.get(match, []) or self._acronym_to_cats.get(match, [])

    def group_cell_by_category(self, cell, agg: str = "count"):
        """
        Group a single cell's list of [match, match_type] pairs by glossary category.

        agg:
          - "count": counts per category
          - "list": list of (match, match_type) per category
          - "set": set of (match, match_type) per category
        """
        if not isinstance(cell, list):
            return {}
        if agg not in {"count", "list", "set"}:
            raise ValueError('agg must be one of {"count","list","set"}')

        out = {}
        for item in cell:
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                continue
            match, match_type = item
            cats = self._categories_for(match, match_type)
            if not cats:
                continue
            for cat in cats:
                if agg == "count":
                    out[cat] = out.get(cat, 0) + 1
                elif agg == "list":
                    out.setdefault(cat, []).append((match, match_type))
                else:  # "set"
                    out.setdefault(cat, set()).add((match, match_type))
        return out

    def group_by_category(self, agg: str = "count") -> pd.DataFrame:
        """
        Return a DataFrame of same shape where each cell is a dict mapping
        category -> aggregated value per 'agg'.
        """
        return self.df.applymap(lambda cell: self.group_cell_by_category(cell, agg))


__all__ = ["WeakKeywordModel", "IntersectionFeatures"]


def get_kws_key(name, use_lemmatization: bool, use_fuzzy: bool) -> str:
    key = name
    if use_lemmatization and use_fuzzy:
        key += "_lemmatize_fuzzy"
    elif use_lemmatization:
        key += "_lemmatize"
    elif use_fuzzy:
        key += "_fuzzy"
    return key


def ensure_keyword_matches(
    name, text_col, use_lemmatization=False, use_fuzzy=False, fuzzy_threshold=90
):
    kws_key = get_kws_key(name, use_lemmatization, use_fuzzy)
    out_path = DATA_DICT["models"]["weak"][kws_key]
    if not out_path.exists():
        inp = pd.read_csv(DATA_DICT["embeddings"]["v1"][f"{name}_texts"])
        model = WeakKeywordModel(
            use_lemmatization=use_lemmatization,
            use_fuzzy=use_fuzzy,
            fuzzy_threshold=fuzzy_threshold,
        )
        df = model.transform(inp, text_col, progress=True)
        df.to_csv(out_path, index=False)
        print(f"Computed and saved keyword matches for {name} with {kws_key}")
    else:
        print(f"Keyword matches for {name} with {kws_key} already exist.")
    return pd.read_csv(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weak Keyword Model Matching Modes")
    parser.add_argument(
        "--use_lemmatization",
        action="store_true",
        help="Enable lemmatization for matching.",
    )
    parser.add_argument(
        "--use_fuzzy", action="store_true", help="Enable fuzzy matching."
    )
    parser.add_argument(
        "--fuzzy_threshold",
        type=int,
        default=90,
        help="Fuzzy match threshold (default: 90)",
    )
    args = parser.parse_args()

    jobs = pd.read_csv(DATA_DICT["jobs"])
    job_ids = jobs["job_id"]

    repos = pd.read_csv(DATA_DICT["github"]["repositories"]["metadata"])
    repo_ids = [i for i in range(len(repos))]

    print(
        f"Example: Matching with lemmatization={args.use_lemmatization}, fuzzy={args.use_fuzzy}, threshold={args.fuzzy_threshold}"
    )
    model = WeakKeywordModel(
        use_lemmatization=args.use_lemmatization,
        use_fuzzy=args.use_fuzzy,
        fuzzy_threshold=args.fuzzy_threshold,
    )
    sample_text = "Developers working on neural networks and deep learnings algorithms."
    print("Matches:", model._find_matches(sample_text))

    kws = {}
    kws["jobs"] = ensure_keyword_matches(
        "jobs",
        "jobs_texts",
        use_lemmatization=args.use_lemmatization,
        use_fuzzy=args.use_fuzzy,
        fuzzy_threshold=args.fuzzy_threshold,
    )
    kws["repositories"] = ensure_keyword_matches(
        "repositories",
        "repositories_texts",
        use_lemmatization=args.use_lemmatization,
        use_fuzzy=args.use_fuzzy,
        fuzzy_threshold=args.fuzzy_threshold,
    )

    intersection_key = "intersection"
    if args.use_lemmatization and args.use_fuzzy:
        intersection_key = "intersection_lemmatize_fuzzy"
    elif args.use_lemmatization:
        intersection_key = "intersection_lemmatize"
    elif args.use_fuzzy:
        intersection_key = "intersection_fuzzy"

    print("Generating intersection ...")
    df_intersection = WeakKeywordModel.intersection_matrix(
        kws["jobs"], kws["repositories"], job_ids, repo_ids
    )
    df_intersection.to_csv(DATA_DICT["models"]["weak"][intersection_key], index=False)
