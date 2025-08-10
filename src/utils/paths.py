from pathlib import Path
from typing import TypedDict

# --- I have some experience with TypeScript, and I love the static programming it embodies ---

# --- I made this TypedDict structure ---
# --- It helps with Typing the DATA_DICT contents ---
# --- Pylance understands the contents of the DATA_DICT now ---
# --- Autocomplete suggestions in other files when I import DATA_DICT ! ---

# --- This whole file serves as a central & explicit location for me to keep track of what data is where, and an easy method of accessing ---
# --- Side benefit: Easy management of anomyzied data access in other files

class RepoNamesDict(TypedDict):
    source_1: Path
    source_2: Path
    source_3: Path
    all: Path
    
class StarWideDict(TypedDict):
    daily: Path
    monthly: Path
    quarterly: Path
    
class TagsDict(TypedDict):
    categorized: Path
    uncategorized: Path
    case_map: Path
    
class RepositoriesDict(TypedDict):
    meta_data: Path
    names: RepoNamesDict
    
class QueriesDict(TypedDict):
    rate_limit: Path
    description: Path
    tags: Path
    readme: Path
    stars: Path
    
class GQLDict(TypedDict):
    schema: Path
    queries: QueriesDict
    
class GitHubDict(TypedDict):
    metadata_dir: Path
    gql: GQLDict
    star_wide: StarWideDict
    stars: Path
    stars_dir: Path
    repositories: RepositoriesDict
    tags: TagsDict
    ossinsight_collections: Path
    scraped_data: Path
    
class PostingFilesDict(TypedDict):
    source_1: str
    source_2: str
    source_3: str
    source_4: str
    source_5: str
    source_6: dict

class EmbedTextsDict(TypedDict):
    jobs: Path
    repositories: Path

class EmbedDict(TypedDict):
    value_dir: Path
    texts: EmbedTextsDict
    texts2: EmbedTextsDict
    
class KeywordDict(TypedDict):
    stackoverflow: Path

class ModelDict(TypedDict):
    bertopic: Path
    
class DataDict(TypedDict):
    models: ModelDict
    keywords: KeywordDict
    gql: GQLDict
    github: GitHubDict
    jobs: Path
    posting_files: PostingFilesDict
    ner_train: Path
    embeddings: EmbedDict


GQL_DIR = Path('graphql')
DATA_PATH = Path('data')

JOBS_DATA = DATA_PATH / 'jobs'
GITHUB_DATA = DATA_PATH / 'github'

RAW_DIR = JOBS_DATA / 'raw'
INTERMEDIATE_DIR = JOBS_DATA / 'intermediate'

PROCESSED_DIR = JOBS_DATA / 'processed'

DATA_2024 = RAW_DIR / '2024_data'

STAR_COUNT_DIR = GITHUB_DATA / 'star_counts'
STAR_WIDE_DIR = GITHUB_DATA / 'star_wide'
REPOSITORY_DIR = GITHUB_DATA / 'repositories'
TAGS_DIR = GITHUB_DATA / 'tags'

EMBEDDING_DIR = DATA_PATH / 'embeddings'

KEYWORD_DIR = DATA_PATH / 'keywords'
MODEL_DIR = Path('models')
DATA_DICT: DataDict = {
    'models': {
        'bertopic': MODEL_DIR / 'bertopic'
    },
    'keywords': {
        'stackoverflow': KEYWORD_DIR / 'stackoverflow-tag-occurence.csv'
    },
    'gql': {
        'schema': GQL_DIR / "schema.docs.graphql",
        'queries': {
            'rate_limit': GQL_DIR / 'rate_limit.gql',
            'description': GQL_DIR / 'description.gql',
            'topics': GQL_DIR / 'paginated_topictags.gql',
            'readme': GQL_DIR / 'readme.gql',
            'stars': GQL_DIR / 'paginated_stargazers.gql'
        }
    },
    'github': {
        'star_wide': {
            'daily': STAR_WIDE_DIR / 'daily.parquet',
            'monthly': STAR_WIDE_DIR / 'monthly.parquet',
            'quarterly': STAR_WIDE_DIR / 'quarterly.parquet'
        },
        'star_long': GITHUB_DATA / 'stars_long.csv',
        'stars_dir': STAR_COUNT_DIR,
        'metadata_dir': GITHUB_DATA / 'meta_data',
        'repositories': {
            'metadata': REPOSITORY_DIR / 'full_repository_info.csv',
            'names': {
                'source_1':REPOSITORY_DIR / 'ossinsights_collections.csv',
                'source_2': REPOSITORY_DIR / 'top36_trending_gh.csv',
                'source_3': REPOSITORY_DIR / 'adhoc_repositories.csv',
                'all': REPOSITORY_DIR / 'all_repository_names.csv',
            },
        },
        'tags': {
            'categorized': TAGS_DIR / 'tags_categorized.csv',
            'uncategorized': TAGS_DIR / 'uncategorized_tags.csv',
            'case_map': TAGS_DIR / 'tag_case_map.csv'
        },
        'ossinsight_collections': GITHUB_DATA / 'ossinsights_collection_names.csv',
        'scraped_data': REPOSITORY_DIR / 'gh_scraped_data.csv',
    },
    'jobs': PROCESSED_DIR / 'master.csv',
    'job_posting_sources': {
        'source_1': {
            'handle': "techmap/job-postings-ireland-october-2022",
            'files': ["techmap-jobs-export-2022-10_ie.json"],
            'out': 'techmap-jobs-export-2022-10_ie.csv'},
        'source_2': {
            'handle': "techmap/us-job-postings-from-2023-05-05",
            'files': ["techmap-jobs_us_2023-05-05.json"],
            'out': 'techmap-jobs_us_2023-05-05.csv'},
        'source_3': {
            'handle': "asaniczka/linkedin-data-engineer-job-postings",
            'files': ['postings.csv'],
            'out': 'postings.csv'},
        'source_4': {
            'handle': "ivankmk/thousand-ml-jobs-in-usa",
            'files': ['1000_ml_jobs_us.csv'],
            'out': '1000_ml_jobs_us.csv'},
        'source_5': {
            'handle': "arshkon/linkedin-job-postings",
            'files': ['postings.csv'],
            'out': 'source_5_postings.csv'},
        'source_6': {
            'handle': "asaniczka/data-science-job-postings-and-skills",
            'files': ['job_postings.csv', 'job_skills.csv', 'job_summary.csv'],
            'out': 'source_6_postings.csv'}
    },
    'ner_train': PROCESSED_DIR / 'ner_training_data.jsonl',
    'embeddings': {
        'value_dir': EMBEDDING_DIR / 'values',
        'v1': { 
            'jobs_texts': EMBEDDING_DIR / 'v1/jobs_text.csv',
            'repositories_texts': EMBEDDING_DIR / 'v1/repositories_text.csv',
            'jobs_embed_b': EMBEDDING_DIR / 'v1/jobs_embeddings_b.npy',
            'repositories_embed_b': EMBEDDING_DIR / 'v1/repositories_embeddings_b.npy'
        },
        'v2': {
            'jobs_texts': EMBEDDING_DIR / 'v2/jobs_text.csv',
            'repositories_texts': EMBEDDING_DIR / 'v2/repositories_text.csv',
            'jobs_embed_b': EMBEDDING_DIR / 'v2/jobs_embeddings_b.npy',
            'repositories_embed_b': EMBEDDING_DIR / 'v2/repositories_embeddings_b.npy'
        }
    },
    'config': {
        'text_embed_variants': MODEL_DIR / 'text-embed-variants.txt'
    }
}

class SentenceModelDict(TypedDict):
    a: str
    b: str
    c: str
    
SENTENCE_MODEL: SentenceModelDict = {
    'a': 'all-MiniLM-L6-v2',
    'b': 'all-mpnet-base-v2',
    'c': "BAAI/bge-base-en-v1.5"
}

TEXT_VARIANTS = [
    'v1',
    'v2'
]

MODEL = "jjzha/jobspanbert-base-cased"


POS_TAGS = ["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"]
HEURISTIC_KWS = [ "skills", "skill", "experience", "proficient", "knowledge", "familiarity", "competency", "expertise", "background", "ability", "qualifications", "qualification"]
TOPIC_KWS = ['deep learning', 'machine learning', 'natural language processing', 'computer vision']

NOISY_SECTION_PATTERN = r'''
(?ix)
^(
    (install(ation|ing)?|setup|dev\s+install|install\s+(with|locally)) |
    (.*contribut(e|ing|ion)?s?.*|contributors?) |
    (license|licence) |
    (support|contact(s)?|.*community.*|communication) |
    (changelog|history|star\s+history) |
    (citation|citations?|resources?) |
    (resources) |
    (.*patch.*) |
    (courses) |
    (.*build.*) |
    (what\s+is\s+.+|what\s+can\s+.+\s+do\??) |
    (papers)
)$
'''
    