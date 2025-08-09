from functools import wraps
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from markdown import markdown
import re
from difflib import SequenceMatcher
from bs4 import BeautifulSoup
from typing import Literal
from tqdm import tqdm
import sys, pandas as pd, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import spacy
import nltk
from IPython.display import display, HTML
from nltk.corpus import stopwords

src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))

from utils.paths import DATA_DICT, SENTENCE_MODEL, TEXT_VARIANTS, POS_TAGS, NOISY_SECTION_PATTERN, HEURISTIC_KWS, TOPIC_KWS

# ---------------------------------------------------------------------------------
# --------------------------- (Shared) Create Embedding ---------------------------
# ---------------------------------------------------------------------------------    

def skip_if_exists(path: str | Path):
    if not isinstance(path, Path):
        path = Path(path)
    if path.exists():
        print(f"EXIT (Already exists): {path}")
        return True
    return False

class CreateEmbedding:
    def __init__(self, sentence_model_index: str, text_variant: Literal['text', 'text2']):
        self.sentence_model_index = sentence_model_index
        try:
            self.sentence_model = SentenceTransformer(SENTENCE_MODEL[sentence_model_index])
        except:
            raise f"Use a Proper sentence model index - one of [{[k for k in SENTENCE_MODEL.keys()]}]"
        self.text_content_variant = text_variant    
    
    def load_text(self, data_channel: Literal['jobs', 'repositories']) -> list[str]:
        text = pd.read_csv(DATA_DICT['embeddings'][self.text_content_variant][f'{data_channel}_texts'])[f"{data_channel}_texts"].tolist()
        return text

    def generate_embeddings(self, data_channel: Literal['jobs', 'repositories'], path = None, save_to_file: bool = True) -> np.array:
        """ 
        For the specified data-channel (jobs / repositories), generates the correpsonding sentence-transformer embeddings 
        if save_to_file is True, saved to the directory in embeddings/{text_variant}
        """
        text_list = self.load_text(data_channel)

        if path is None:
            try:
                path = DATA_DICT['embeddings'][self.text_content_variant][f"{data_channel}_embed_{self.sentence_model_index}"]
            except:
                print(f'Create an entry for "{data_channel}_embed_{self.sentence_model_index}" in the DATA_DICT')
        if skip_if_exists(path):
            print("LOADING embeddings ...")
            embeddings = np.load(path)
        else:
            print(f"GENERATING: Embeddings for {data_channel} with {self.text_content_variant} texts ...")
            embeddings = self.sentence_model.encode(
                text_list,normalize_embeddings=True,show_progress_bar=True
            )
            if save_to_file:
                np.save(path, embeddings)
        return embeddings


class TextVariant(ABC):
    def __init__(self, df_jobs, df_repos, variant, **kwargs):
        self.kwargs = kwargs
        self.df_jobs = df_jobs
        self.df_repos = df_repos
        self.variant = variant
        self.out_paths = {
            'jobs': DATA_DICT['embeddings'][variant]['jobs_texts'],
            'repositories': DATA_DICT['embeddings'][variant]['repositories_texts']
        }

    def process_and_save(self, kind, build_fn, colname):
        """
        kind: 'jobs' or 'repositories'
        build_fn: function that builds and returns list of texts
        colname: name of CSV column
        """
        path = self.out_paths[kind]
        if skip_if_exists(path):
            return
        data = build_fn()
        self.save_texts(data, path, colname)

    def save_texts(self, data, path, colname):
        pd.DataFrame(data, columns=[colname]).to_csv(path, index=False)

    @abstractmethod
    def build_jobs_texts(self):
        pass

    @abstractmethod
    def build_repos_texts(self):
        pass

    def generate(self):
        self.process_and_save("jobs", self.build_jobs_texts, "jobs_texts")
        self.process_and_save("repositories", self.build_repos_texts, "repositories_texts")

class TextV1(TextVariant):
    def build_jobs_texts(self):
        return (
            self.df_jobs['title'] + ' ' +
            self.df_jobs['description']
        ).tolist()
    
    def build_repos_texts(self):
        return  (
            self.df_repos['owner'] + ' ' +
            self.df_repos['repo'] + ' ' +
            self.df_repos['topics'].str.replace(',', ' ') + ' ' +
            self.df_repos['description'] + ' ' +
            self.df_repos['readme']
        ).tolist()

class TextV2(TextVariant):
    def build_jobs_texts(self):
        job_processor = JobTextProcessor(
            heuristic_keywords=self.kwargs.get('heuristic_kws'),
            topic_keywords=self.kwargs.get('topic_kws'),
            lemmatize = self.kwargs.get('lemmatize')
        )
        job_texts = (
            self.df_jobs['title'] + ' ' +
            self.df_jobs['description']
        ).tolist()
        return job_processor.batch_transform(
            job_texts,
            'BY_contains_topic_chunk',
            'BY_contains_pos',
            pos='PROPN'
        )
    
    def build_repos_texts(self):
        repo_processor = RepositoryTextProcessor(self.kwargs.get('noise_patterns'), lemmatize = self.kwargs.get('lemmatize'))
        repo_texts = (
            self.df_repos['topics'].str.replace(',', ' ') + ' ' +
            self.df_repos['owner'] + ' ' +
            self.df_repos['repo'] + ' ' +
            self.df_repos['description'].str.replace(r'[^\x00-\x7F]+', '', regex=True) + ' ' +
            self.df_repos['readme'].apply(lambda x: repo_processor.readme_cleaner.transform(x))
        ).str.strip().tolist()
        return repo_processor.batch_transform(repo_texts)

VARIANT_REGISTRY = {
    'v1': TextV1,
    'v2': TextV2,
}

class TextProcessor:
    def __init__(
        self,
        lemmatize = True
        ):
        try:
            _ = stopwords.words("english")
        except LookupError:
            nltk.download("stopwords")
        self.nlp = spacy.load("en_core_web_sm")
        case_map = pd.read_csv(DATA_DICT['github']['tags']['case_map'])
        self.case_map_dict = dict(zip(case_map['lower'], case_map['proper']))
        self.stopwords = set(stopwords.words("english"))
        self.lemmatize = lemmatize
    
    def case_correct_word(self, word: str):
        return self.case_map_dict.get(word.lower(), word)
    
    def filter_stopwords(self, doc: spacy.tokens.Doc, contain):
        for i, token in enumerate(doc):
            if token.is_alpha:
                if self.lemmatize: 
                    obj = token.lemma_
                else: 
                    obj = token.text
                if obj.lower() in self.stopwords:
                    contain[i] = False
        return contain
    
    @abstractmethod
    def submethod(self, doc, contain, *args, **kwargs):
        pass
    
    def transform(self, doc: str | spacy.tokens.Doc, *args, visualise = False, **kwargs):
        """
        Returns a tuple: output-text, HTML
        
        The output-text is the transformed-text; HTML displays highlighted text-block, demonstrating which tokens were kept/removed
        """
        if isinstance(doc, str):
            doc = self.nlp(doc)
    
        contain = [True]*len(doc)
        contain = self.submethod(doc, contain, *args, **kwargs)
        contain = self.filter_stopwords(doc, contain)
        
        out, html = [], []
        for i in range(len(doc)):
            obj =  (doc[i].lemma_ if self.lemmatize else doc[i].text)
            if self.case_correct:
                obj =  self.case_correct_word(obj)
            obj += doc[i].whitespace_
            if contain[i]:
                out.append(obj)
                html.append(f'<span style="color: green; font-weight: bold;">{obj}</span>')
            else:
                html.append(f'<span style="color: #999;">{obj}</span>')
        if visualise:
            return ''.join(out), display(HTML(''.join(html)))
        else:
            return ''.join(out), None
        
    def batch_transform(self, texts, *args, **kwargs):
        out = []
        docs = self.nlp.pipe(texts, batch_size=32, n_process=4)
        with tqdm(desc = "Batch Processing Texts", total = len(texts), unit = 'Texts') as pbar:
            for doc in docs:
                out.append(self.transform(
                    doc,
                    *args,
                    visualise=False,
                    **kwargs)[0])
                pbar.update(1)
        return out
           
class JobTextProcessor(TextProcessor):
    def __init__(
        self, 
        heuristic_keywords,
        topic_keywords,
        **kwargs
        ):
        
        super().__init__(**kwargs)
        self.heuristic_keywords = heuristic_keywords
        self.topic_keywords = topic_keywords

    def filter_sentence(self, doc: spacy.tokens.Doc, method, **kwargs):
        return getattr(self, method)(doc = doc, **kwargs)

    def BY_contains_kw(
        self,
        doc: spacy.tokens.Doc,
        **kwargs,
        ) -> bool:
        """ Return True if the sentence contains any of the keywords defined in heuristic_keywords """
        words = [token.text.lower() for token in doc]
        return any(word.lower() in words for word in self.heuristic_keywords)
    
    def BY_contains_pos(
        self,
        doc: spacy.tokens.Doc,
        pos: Literal ["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"],
        remove_entities: list = None,
        **kwargs
        ) -> bool:
        """
        Returns True if a word in the sentence has a word tagged as pos (Part of Speech)
        spaCy NLP tags each token with a POS tag - e.g. "VERB", "NOUN", etc.
        
        remove_entities: write a list of recognised entity types that are discounted as being relevant.
        e.g. if pos = "PROPN", remove_entities = "PERSON" returns False for sentences where the PROPN terms are only PERSON
        """
        if isinstance(pos, str): pos = [pos]
        for token in doc:
            if any([token.pos_ == p for p in pos]): 
                if remove_entities is not None:
                    if token.ent_type_ not in remove_entities:
                        return True
                else:
                    return True
        return False
    
    def BY_contains_topic_chunk(self, doc: spacy.tokens.Doc, **kwargs):
        """
        Returns True if the sentence contains a noun phrase of at least `min_len` tokens.
        """
        for chunk in doc.noun_chunks:
            if any(term in chunk.text.lower() for term in self.topic_keywords):
                return True
        return False
    
    def submethod(self, contain, doc, *methods, logical_operation = any, **kwargs):
        for sentence in doc.sents:
            sentence_doc = self.nlp(sentence.text)
            truth_values = [self.filter_sentence(sentence_doc, m, **kwargs) for m in methods]
            if not logical_operation(truth_values):
                for i in range(sentence.start, sentence.end):
                    contain[i] = False
        return contain
    

class ReadmeCleaner:
    def __init__(
        self,
        noise_patterns,
    ):  
        self.noise_patterns = noise_patterns
        
        
    def is_noisy_title(self, title: str) -> bool:
        return re.match(self.noise_patterns, title.strip(), re.VERBOSE | re.IGNORECASE) is not None

    def strip_inline_code_tags(self, soup: BeautifulSoup) -> BeautifulSoup:
        for tag in soup.find_all(['code', 'pre']):
            tag.decompose()
        return soup
    
    def create_soup(self, text: str) -> BeautifulSoup: 
        return BeautifulSoup(markdown(text), 'html.parser')

    def remove_nonsemantic_lists(self, soup: BeautifulSoup) -> BeautifulSoup:
        
        def is_link_only_li(li):
            """Check if an <li> is a link or a non-semantic heading."""
            allowed = {'a', 'strong', 'em', 'b', 'i', 'code'}
            return all(
                (child.name in allowed or isinstance(child, str))
                for child in li.contents
                if not isinstance(child, str) or child.strip()
            )

        def is_removable_list(ul_or_ol):
            for li in ul_or_ol.find_all('li', recursive=False):
                nested_lists = li.find_all(['ul', 'ol'], recursive=False)
                if nested_lists:
                    if not all(is_removable_list(nl) for nl in nested_lists):
                        return False
                else:
                    if not is_link_only_li(li):
                        return False
            return True
        
        for ul_or_ol in soup.find_all(['ul', 'ol']):
            if is_removable_list(ul_or_ol):
                ul_or_ol.decompose()
        return soup

    def remove_link_only_lists(self, soup: BeautifulSoup) -> BeautifulSoup:
        for list_tag in soup.find_all(['ul', 'ol']):
            lis = list_tag.find_all('li', recursive=False)
            if lis and all(
                len(li.contents) == 1 and li.find('a') and li.find('a') == li.contents[0]
                for li in lis
            ):
                list_tag.decompose()
        return soup


    def remove_unwanted_sections(self, soup: BeautifulSoup) -> BeautifulSoup:
        headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        checked_headers = []
        for header in headers:
            if header in checked_headers:
                continue
            else:
                title = header.get_text(strip=True)
               
                try:
                    level = int(header.name[1])
                except TypeError:
                    continue
                
                siblings = []
               
                for sibling in header.find_next_siblings():
                    if (sibling.name is not None) and (sibling.name.startswith('h')):
                        try: 
                            next_level = int(sibling.name[1])
                            if next_level <= level:
                                break
                        except ValueError:
                            pass
                        except TypeError:
                            continue
                    siblings.append(sibling)
                if len(siblings) == 0:
                    header.decompose()
                else:
                    if self.is_noisy_title(title):
                        header.decompose()
                        for sibling in siblings:
                            if sibling in headers:
                                checked_headers.append(sibling)
                                headers.remove(sibling)
                            sibling.decompose()
        return soup
    
    def full_soup_cleaner(self, soup: BeautifulSoup) -> BeautifulSoup: 
        soup = self.strip_inline_code_tags(soup)
        soup = self.remove_nonsemantic_lists(soup)
        soup = self.remove_unwanted_sections(soup)
        return soup


    def remove_markdown_tables(self, text: str) -> str:
        lines = text.splitlines()
        clean_lines = []
        in_table = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$", stripped):
                if i > 0 and lines[i - 1].strip().startswith("|") and lines[i - 1].count("|") >= 2:
                    in_table = True
                    continue
            if in_table:
                if stripped.startswith("|") and stripped.count("|") >= 2:
                    continue
                else:
                    in_table = False
            if not in_table and stripped.startswith("|") and stripped.count("|") >= 2:
                continue
            clean_lines.append(line)
        return "\n".join(clean_lines)

    
    def strip_rst_badges(self, text: str) -> str:
        text = re.sub(r"(?m)^\|(?:[^\n|]+\|)+\s*$", "", text)
        text = re.sub(r"(?m)^\.\.\s*\|[^|]+\|\s+(image|replace)::.*(?:\n\s*:[^:]+:[^\n]*)*", "", text)
        text = re.sub(r"(?m)^\.\.\s*image::.*(?:\n\s*:[^:]+:[^\n]*)*", "", text)
        return text.strip()

    def strip_urls_keep_text(self, text: str) -> str:
        soup = BeautifulSoup(markdown(text), 'html.parser')
        for a in soup.find_all('a'):
            a.replace_with(a.get_text())
        return soup.text

    def remove_code_blocks(self, text: str) -> str:
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`\n]+`', '', text)
        return text
    
    def remove_excessive_punctuation(self, text: str) -> str:
        return re.sub(r'(?<=\w)[.,;:!?]+(?=\s|$)', '', text) #Trailing Punctuation
    
    def normalize_whitespace(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip() #Excess whitespace
    
    def remove_non_english_or_symbols(self, text: str) -> str:
        return re.sub(r'[^\x00-\x7F]+', '', text) #Emojis
    
    def full_text_cleaner(self, text: str) -> str:
        text = self.strip_rst_badges(text)
        text = self.remove_code_blocks(text)
        text = self.remove_markdown_tables(text)
        text = self.strip_urls_keep_text(text)
        text = self.remove_excessive_punctuation(text)
        text = self.remove_non_english_or_symbols(text)
        return self.normalize_whitespace(text)

    def highlight_diff_html(self, original: str, filtered: str) -> str:
        sm = SequenceMatcher(None, original, filtered)
        result = []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == 'equal':
                result.append(original[i1:i2])
            elif op == 'insert':
                result.append(f"<span style='color: #8dfc7e'>{filtered[j1:j2]}</span>")
            elif op == 'delete':
                result.append(f"<span style='color: #fc7e7e'>{original[i1:i2]}</span>")
            elif op == 'replace':
                result.append(f"<span style='color: #fc7e7e'>{original[i1:i2]}</span>")
                result.append(f"<span style='color: #8dfc7e'>{filtered[j1:j2]}</span>")
        return ''.join(result)

    def strip_url_manually(self, text: str) -> str:
        url_pattern = re.compile(
            r"(https?://\S+|www\.\S+|\[.*?\]\((https?://|www\.)\S+\)|(?<!\w)([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(\/\S*)?)"
        )
        return url_pattern.sub("", text)

    def transform(self, text: str, visualise: bool = False) -> str | tuple[str, HTML]:
        soup = self.create_soup(text)
        soup = self.full_soup_cleaner(soup)
        intermediate_text = soup.get_text('\n')
        intermediate_text = self.full_text_cleaner(intermediate_text)
        out_text = self.strip_url_manually(intermediate_text)
        if visualise:
            visualised_html = self.highlight_diff_html(text, out_text)
            return out_text, HTML(visualised_html)
        else:
            return out_text
        
class RepositoryTextProcessor(TextProcessor):
    def __init__(self, readme_noise_section_pattern, **kwargs):
        super().__init__(**kwargs)
        self.readme_cleaner = ReadmeCleaner(noise_patterns=readme_noise_section_pattern)
        
    def submethod(self, contain, doc):
        return contain


if __name__ == "__main__":
    
    available_sentence_models = SENTENCE_MODEL.keys()

    parser = ArgumentParser()
    
    parser.add_argument(
        '--text_variant',
        default = None
    )
    parser.add_argument(
        '--sentence_model_index',
        default = None
    )
    
    args = parser.parse_args()
    
    for arg, valid in zip([args.text_variant, args.sentence_model_index], [TEXT_VARIANTS, available_sentence_models]):
        if arg is None:
            raise ValueError(f'Must enter a value for text_variant - choose one from: {valid}')
        elif arg not in valid:
            raise ValueError(f'{arg} is not a valid value for text_variant - choose one from: {valid}')

    df_jobs = pd.read_csv(DATA_DICT['jobs'])
    df_repos = pd.read_csv(DATA_DICT['github']['repositories']['metadata'])
    
    for df in [df_repos, df_jobs]:
        for c in df.columns:
            df[c] = df[c].fillna('')

    print(f'GENERATING: {args.text_variant}')
    
    cls = VARIANT_REGISTRY[args.text_variant](
        df_jobs, 
        df_repos,
        args.text_variant,
        noise_patterns = NOISY_SECTION_PATTERN, 
        heuristic_kws = HEURISTIC_KWS, 
        topic_kws = TOPIC_KWS,
        lemmatize = True
    )
    cls.generate()
    
    embedder = CreateEmbedding(sentence_model_index=args.sentence_model_index, text_variant=args.text_variant)
    
    embedder.generate_embeddings('jobs')
    embedder.generate_embeddings('repositories')