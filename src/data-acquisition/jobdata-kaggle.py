import os
import sys
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))

from utils.paths import DATA_DICT, RAW_DIR, INTERMEDIATE_DIR

# ---- FULL JOB DATA DOWNLOAD ----

# --- INSTRUCTIONS ---

# 1. Create a Kaggle Account
# 2. In kaggle > Account > Settings > API: Click "Create New Token"
# 3. Move the downloaded API key - Run following in terminal: ```mv path/to/downloaded/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json (this second command to remove warnings in python)
# 4. Run the script: Run ```python data-aquisition/jobdata-kaggle.py``` from the `src` directory
# --- This will take some time ---


class KaggleDownloader:
    def __init__(self):
        self.api = KaggleApi()
        self.api.authenticate()

    def download_to_dir(
        self, handle: str, dir: Path, exist_ok: bool = True, unzip: bool = True
    ) -> None:
        print(f"Downloading: {handle}")
        os.makedirs(dir, exist_ok=exist_ok)
        self.api.dataset_download_files(handle, path=dir, unzip=unzip)


if __name__ == "__main__":
    kdl = KaggleDownloader()
    for source, source_config in DATA_DICT["job_posting_sources"].items():
        kdl.download_to_dir(source_config["handle"], dir=RAW_DIR / source)
