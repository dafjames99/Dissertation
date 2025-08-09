import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))


import base64
import re
import pandas as pd
from bs4 import BeautifulSoup
import requests
from collections import Counter
from utils.paths import DATA_DICT
import time

github_url = "https://github.com/"
topics_url = github_url + 'topics'
ai_collection_id = 10010 #Artificial Intelligence

NEW_SEED_SIZE = 30
TAG_MENTION_THRESHOLD = 10
    

with open('utils/github_credentials.txt', 'r') as f:
    GITHUB_TOKEN = f.readline()
    
AUTH_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}"
}

HTML_HEADERS = {
    "Accept": 'application/json'
}

def check_rate_limit():
    """ Adhere to Authorized Rate Limits for GitHub API"""
    r = requests.get("https://api.github.com/rate_limit", headers=AUTH_HEADERS)
    if r.status_code == 200:
        data = r.json()
        remaining = data["rate"]["remaining"]
        reset_time = data["rate"]["reset"]  # UNIX timestamp
        return remaining, reset_time
    return 0, time.time() + 60

def wait_if_necessary(threshold=10):
    """Function to pause requests until rate limit has refreshed """
    remaining, reset = check_rate_limit()
    if remaining < threshold:
        wait_time = reset - time.time()
        if wait_time > 0:
            print(f"Rate limit near exhaustion. Sleeping for {int(wait_time)} seconds...")
            time.sleep(wait_time + 1)  # sleep until reset

def urlify(term, url):
    return url + "/" + '-'.join(term.split(' '))

def get_page(url):
    response = requests.get(url)
    if response.status_code != 200:
        print('Some error')
        return response
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup
    
def get_topics(url):
    soup = get_page(url)
    repos = soup.find_all('article')
    names, tags = [], []
    for repo in repos:
        repo_name = repo.find('h3', {'class':'f3 color-fg-muted text-normal lh-condensed'}).text
        repo_name = re.sub(r'\n', '', repo_name)
        topic_tags = [a.text for a in repo.find_all('a', {'class': 'topic-tag topic-tag-link Link f6 mb-2'})]
        names.append(repo_name)
        tags.append(topic_tags)
    return names, tags

def scrape_multiple_topics(extensions, base_url):
    urls = [urlify(e, base_url) for e in extensions]
    repo_names, repo_tags = [], []
    for url in urls:
        names, tags = get_topics(url)
        repo_names.extend(names)
        repo_tags.extend(tags)
    return pd.DataFrame(data = {'repository': repo_names, 'tags': repo_tags})

def count_tags(list_of_lists):
    all = []
    [all.extend(t) for t in list_of_lists]
    return Counter(all)

def fetch_trending_repos(collection_id):
    """
    Makes an API Request to ossinsight.io for trending repositories within a collection (topic)
    """
    url = f"https://api.ossinsight.io/v1/collections/{collection_id}/ranking_by_stars/"
    obj = requests.request("GET", url, headers={"Accept": 'application/json'}).json()
    return [item['repo_name'] for item in obj['data']['rows']]

def fetch_repo_info(owner: str, repo: str):
    wait_if_necessary()
    base_url = f"https://api.github.com/repos/{owner}/{repo}"
    readme, desc = "", ""
    
    r1 = requests.get(base_url, headers=AUTH_HEADERS)
    if r1.status_code == 200:
        desc = r1.json().get('description', '')
    r2 = requests.get(f"{base_url}/readme", headers=AUTH_HEADERS)
    if r2.status_code == 200: 
        encoded = r2.json().get("content", "")
        try:
            readme = base64.b64decode(encoded).decode("utf-8", errors="ignore")
        except Exception:
            readme = ""
    return desc, readme



# ------ SCRIPT ENTRY POINT ------ 
if __name__ == "__main__":

    # ------ Web Scraping GitHub for topic-relevant Repostories -------

    # 1. Pull Top repositories from seed Topics
    seeds = [
        "machine learning",
        "deep learning",
        "natural language processing",
        "computer vision"
    ]
    
    df = scrape_multiple_topics(seeds, topics_url)

    # 2. Top 30 mentioned topics are NEW seeds
    new_seeds = [a for (a,_) in count_tags(df['tags']).most_common()[:NEW_SEED_SIZE]]

    # 3. Pull Top Repositories for NEW seeds
    df = scrape_multiple_topics(new_seeds, topics_url)

    # 4. Save Scraped Data
    df['repository'] = df['repository'].apply(lambda x: re.sub(r'\s+', ' ', x.strip()).replace(' / ', '/'))
    df['tags'] = df['tags'].apply(lambda x: ','.join(x))
    df[['repository_owner','repository_name']] = df['repository'].str.split('/', expand = True)
    df.drop_duplicates()
    df.to_csv(DATA_DICT['github']['scraped_data'], index = False)

    # ------ Comprehensive List of Relevant Topic Tags -------

    # 1. Compile Tag List
    counter = count_tags(df['tags'])
    top_tags = [a for (a, b) in counter.most_common() if b >= TAG_MENTION_THRESHOLD]

    # 2. MANUAL removal of irrelevant topics
    purge_list = [
        'hacktoberfest',
        'deep-learning-tutorial',
        'book',
        'notebook',
        'chinese',
        'awesome',
        'awesome-list',
        'distributed',
        'tutorial',
        'education',
        'interview',
        'ultralytics',
        'audio',
        'kaggle'
    ]

    for item in purge_list:
        if item in top_tags:
            top_tags.remove(item)


    # ------ Fetch Names & Information of Curated AI-relevant Repositories ------

    # 1. Find Names & Tags of relevant repositories
    tag_list = []
    repos = fetch_trending_repos(ai_collection_id)

    for r in repos:
        url = github_url + r
        soup = BeautifulSoup(requests.get(url).text, 'html.parser')
        tag_list.append([tag.text for tag in soup.find_all('a', {'class': 'topic-tag topic-tag-link'})])
        time.sleep(1)

    df_trending = pd.DataFrame(data = {'repository': repos, 'tags': tag_list})
    df_trending['tags'] = df_trending['tags'].apply(lambda x: [re.sub(r'\s+', '', i) for i in x])

    # 2. GetDescription & README content for the Repositories

    df_trending[['owner', 'repo']] = df_trending['repository'].str.split('/', expand=True)
    results = df_trending.apply(lambda row: fetch_repo_info(row['owner'], row['repo']), axis=1)
    df_trending[['description', 'readme']] = pd.DataFrame(results.tolist(), index=df_trending.index)
    
    # 3. Save Trending GitHub Repository dataset
    df_trending.to_csv(DATA_DICT['github']['repositories'], index=False)

    # ------ Collate Top / Relevant Tags into one dataset ------

    trending_tags = []
    [trending_tags.extend(t) for t in df_trending['tags'].drop_duplicates().tolist()]
    [trending_tags.remove(t) for t in purge_list if t in trending_tags]
    [trending_tags.remove(t) for t in trending_tags if re.match(r'.*tutorial.*', t)]

    final_tag_list = pd.Series(top_tags + trending_tags)
    final_tag_list.drop_duplicates(inplace = True)

    pd.DataFrame(
        data = {'tag': final_tag_list.str.replace('-', ' ')}
    ).to_csv(DATA_DICT['github']['uncategorized_tags'], index = False)
