# Walkthrough
## Setup
### Python Env
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### GitHub API Token
[Fine-grained GitHub Authentication token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#fine-grained-personal-access-tokens)

### Kaggle API Token
1. Create a Kaggle Account
2. In kaggle > Account > Settings > API: Click "Create New Token"
3. Move the downloaded API key (Run following): 

```bash
mv path/to/downloaded/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Pipeline
Assumes we have:
* Correct directory structure (as detailed in `src/utils/paths.py` - serves as central control of paths)
* Existent `src/data/github/repositories/all_repository_names.csv` (Names of repositories)

### Step 0 - Run from src/ directory
```bash
cd src
```
### Step 1 - Data Collection
```bash
python3 data-acquisition/github_data.py --include_stars=True #GitHub Repositories
python3 data-acquisition/jobdata-kaggle.py #Job Postings
python3 models/weak/generate_tags.py #Produce dictionary for baseline
```
### Step 2 - Preprocessing
```bash
python3 preprocessing/preprocess_jobsdata.py
```

### Step 3 - Model Generation
Get the:
* Weak model intersections
* Embeddings

for the pairs of documents (repo/job pairs).

```bash
python3 models/embedding/model.py --text_variant=v2 --sentence_model_index=c
python3 models/weak/model.py --text_variant=v2 --use_lemmatization --use_fuzzy
```

### Step 4 - Produce Evaluations

Here we compute the similarity matrix (dot product of job & repo embeddings) & evaluate against the baseline intersections.

```bash
python3 models/evaluate.py
```
### Step 5 - Correlation analysis

Specifying a run_id, we find correlation-values against the stars per repository.

```bash
python3 models/star-similarity/correlation_analysis.py
```