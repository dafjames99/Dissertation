import sys
import re
import json
import os
from functools import reduce
from pathlib import Path
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))

import utils.job_title_taxonomy as jtt
from utils.paths import DATA_DICT, RAW_DIR, INTERMEDIATE_DIR


# ----- PREPROCESS & COMPILE JOB DATA -> 1 Master csv file -----

# --- INSTRUCTIONS ---
# Run the script: Run ```python preprocessing/preprocess_jobdata.py``` from the `src` directory
# --- This will take some time ---


class PreprocessJobsData:
    def __init__(
        self,
        source: str,
        config: dict,
        intermediate_dir: Path = INTERMEDIATE_DIR,
        raw_dir: Path = RAW_DIR,
    ):
        self.raw_dir = raw_dir
        self.intermediate_dir = intermediate_dir

        self.config = config
        self.source = source
        self.source_config = DATA_DICT["job_posting_sources"][source]

        self.handle = self.source_config["handle"]
        self.out_file = self.source_config["out"]
        self.files = self.source_config["files"]

        if len(self.files) > 1:
            self.in_file = self.files
        else:
            self.in_file = self.files[0]

        if self.config["is_csv"]:
            if self.config.get("merge") is not None:
                self.df = self.merge()
            else:
                self.df = self.load_csv()
        else:
            self.df = self.json_to_df()

        if self.config.get("rename") is not None:
            self.rename()

        self.convert_dates()

        if self.config.get("composite") is not None:
            self.form_composite()

        if self.config.get("filter") is not None:
            self.filter()

    def merge(self):
        merge_config = self.config.get("merge")
        dfs = [pd.read_csv(self.raw_dir / self.source / f) for f in self.in_file]
        return reduce(
            lambda left, right: pd.merge(
                left, right, how=merge_config["how"], on=merge_config["on"]
            ),
            dfs,
        )

    def load_csv(self):
        return pd.read_csv(self.raw_dir / self.source / self.in_file)

    def json_to_df(self):
        data = []
        with open(self.raw_dir / self.source / self.in_file, "r") as f:
            for line in f:
                record = json.loads(line)
                trimmed = {
                    k: record[k] for k in self.config.get("desired_keys") if k in record
                }
                data.append(trimmed)
        return pd.json_normalize(data)

    def rename(self):
        map = self.config.get("rename")
        self.df.rename(columns=map, inplace=True)

    def convert_dates(self):
        fn = self.config.get("date_conversion")
        if fn is not None:
            self.df["date"] = self.df["date"].apply(fn)
        self.df["date"] = pd.to_datetime(
            self.df["date"], format="%Y-%m-%d"
        ).dt.to_period("D")

    def form_composite(self):
        for comp, sub_cols in self.config.get("composite").items():
            self.df[comp] = self.df[sub_cols[0]].fillna("")
            for c in sub_cols[1:]:
                self.df[comp] = self.df[comp] + "\nSkills: " + self.df[c].fillna("")

    def filter(self):
        fn = self.config.get("filter")
        self.df = fn(self.df)

    def save_intermediate(self):
        self.df.to_csv(self.intermediate_dir / self.out_file, index=False)

    @staticmethod
    def create_final(select_columns: list[str], from_dir: Path, out_file: Path) -> None:
        dfs = [
            df.assign(source_file=file.stem)
            for file in from_dir.glob("*.csv")
            if (df := pd.read_csv(file)).get("description") is not None
        ]

        master_df = pd.concat(dfs, ignore_index=True).dropna(subset=["description"])

        master_df["date"] = pd.to_datetime(master_df["date"])
        master_df["month"] = master_df["date"].dt.to_period("M")

        mask = master_df["title"].str.contains(
            jtt.title_match_pattern, regex=True, na=False
        )
        master_df = master_df[mask].copy()

        master_df["clean_title"] = master_df["title"].apply(jtt.clean_title)
        master_df["mapped_title"] = master_df["clean_title"].apply(
            jtt.map_title_to_category
        )

        master_df.reset_index(inplace=True, drop=True)
        master_df["job_id"] = master_df.index

        master_df["description"] = master_df["description"].apply(
            lambda x: re.sub(r"\s+", " ", x).strip() if isinstance(x, str) else ""
        )

        master_df[select_columns].to_csv(out_file, index=False)


PREPROCESSING_CONFIG = {
    "source_1": {
        "is_csv": False,
        "desired_keys": ["text", "name", "dateCreated"],
        "rename": {
            "text": "description",
            "name": "title",
            "dateCreated.$date.$numberLong": "date",
        },
        "date_conversion": lambda x: pd.to_datetime(x, unit="ms", origin="unix"),
    },
    "source_2": {
        "is_csv": False,
        "desired_keys": ["text", "name", "dateCreated"],
        "rename": {
            "text": "description",
            "name": "title",
            "dateCreated.$date": "date",
        },
        "date_conversion": lambda x: re.search(r"^(\d{4}-\d{2}-\d{2})", x).group(),
    },
    "source_3": {
        "is_csv": True,
        "rename": {"job_title": "title", "first_seen": "date"},
        "composite": {"description": ["job_summary", "job_skills"]},
    },
    "source_4": {
        "is_csv": True,
        "rename": {
            "job_title": "title",
            "job_posted_date": "date",
            "job_description_text": "description",
        },
    },
    "source_5": {
        "is_csv": True,
        "rename": {
            "listed_time": "date",
        },
        "date_conversion": lambda x: pd.to_datetime(x, unit="ms", origin="unix"),
    },
    "source_6": {
        "merge": {"how": "inner", "on": "job_link"},
        "is_csv": True,
        "rename": {"job_title": "title", "first_seen": "date"},
        "composite": {"description": ["job_summary", "job_skills"]},
        "filter": lambda df: df[~df["company"].str.contains("university", case=False)],
    },
}

if __name__ == "__main__":
    out_columns = ["job_id", "date", "month", "title", "mapped_title", "description"]
    for source, config in PREPROCESSING_CONFIG.items():
        obj = PreprocessJobsData(
            source, config, intermediate_dir=INTERMEDIATE_DIR, raw_dir=RAW_DIR
        )
        obj.save_intermediate()

    PreprocessJobsData.create_final(
        select_columns=out_columns,
        from_dir=INTERMEDIATE_DIR,
        out_file=DATA_DICT["jobs2"],
    )
