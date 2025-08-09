import re
from spellchecker import SpellChecker

exact_titles = [
    "machine learning", "artificial intelligence", "ai",
    "software engineer", "ml", "ml/ai",
    "mlops", "devops", "nlp", "deep learning", "computer vision", 'big data', 'computational scientist'
]

flexible_titles = [
    r"data .*engineer",r"data .*engineering",  r"data.*analyst", r"data .*analytics",
    r"data .*sci.*", r"data .*science", r'computational .*scientist', r"software .*engineer",  r"software .*engineering"
]

PROTECTED_TERMS = {'ml', 'ai', 'ml/ai', 'nlp', 'mlops', 'devops'}

title_match_pattern = r"(?i)\b(" + "|".join([re.escape(term) for term in exact_titles] + flexible_titles) + r")\b"

title_map = {
    r".*data.*analyst.*": "Data Analyst",
    r".*data .*scien.*": "Data Scientist",
    r".*data .*architect.*": "Data Architect",
    r".*data .*engineer.*": "Data Engineer",
    r".*data .*analytics.*": "Data Analytics",
    r"(.*machine learning.*|.*ml engineer.*)": "Machine Learning Engineer",
    r".*mlops.*": "MLOps Engineer",
    r".*software .*engineer.*": "Software Engineer",
    r".*deep learning resaercher.*": "Deep Learning Researcher",
    r".*deep learning.*": "Deep Learning Engineer",
    r".*nlp .*": "NLP Scientist",
    r".*computer vision.*": "Computer Vision Engineer",
    r".*research scientist.*": "Research Scientist",
    r".*computational.*scientist.*": "Computational Scientist",
    r".*ai.*scientist.*": "AI Scientist",
    r"(.*ai.*engineer.*|.*engineer.*ai|.*artificial intelligence engineer.* | .* ai.*developer.*')": "AI Engineer",
    r"(.*ai.*researcher.*|.*artificial intelligence.*researcher.*|.*researcher.*(ai|ml).*)": "AI Researcher",
    r".*devops.*": "DevOps Engineer",
    r".*annotation analyst.*": "AI Annotation Analyst",
    r".*cloud .*architect.*": "Cloud Architect",
    r'(.*big data.*developer.*|.*big data sdet.*|.*engineer.*big data.*)': "Big Data Developer",
    r'.*fullstack.*': 'Fullstack Developer',
    r'.*ai.* architect': 'AI Architect',
    
}

def map_title_to_category(cleaned_title):
    for pattern, label in title_map.items():
        if re.search(pattern, cleaned_title):
            return label
    return "Other"

def clean_title(title):
    title = title.lower()
    title = re.sub(r'^(associate|senior|sr|sr\.|lead|junior|jr|principal|staff|contract|remote|onsite)\s+', '', title)
    title = re.sub(r'^(hris|bi|erp|crm|sap|aws|azure|sql|python|r|java|salesforce)\s+', '', title)
    title = re.sub(r'\s+(team|group|division|department|office|location|unit)$', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title
