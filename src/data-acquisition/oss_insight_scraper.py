import time
import pandas as pd
import sys, os
from pathlib import Path
import json
import requests

src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))

from utils.paths import DATA_DICT

desired_collections = pd.read_csv(DATA_DICT['github']['ossinsight_collections'])['collection_name'].to_list()

def get_all_collections():
    url = f"https://api.ossinsight.io/v1/collections"
    res = requests.request("GET", url, headers={"Accept": 'application/json'})
    time.sleep(1)
    return res 

def id_from_name(name):
    collections = get_all_collections().json()
    for c in collections['data']['rows']:
        if c['name'] == name:
            return c['id']
    return 'Not a valid collection name'

def get_collection_data(id):
    url = f"https://api.ossinsight.io/v1/collections/{id}/ranking_by_stars/"
    obj = requests.request("GET", url, headers={"Accept": 'application/json'}).json()
    time.sleep(1)
    return obj

def get_repositories(collection_data):
    return [item['repo_name'] for item in collection_data['data']['rows']]

if __name__ == '__main__':
    names, repos = [], []
    for name in desired_collections:
        collection_repos = get_repositories(get_collection_data(id_from_name(name)))
        repos.extend(collection_repos)
        names.extend([name for _ in range(len(collection_repos))])
    df = pd.DataFrame(data = {'collection': names, 'repository': repos})
    for repo in repos:
        if len(df[df['repository'] == repo]) > 1:
            indices = df[df['repository'] == repo].index
            collections = ' | '.join(df.iloc[indices]['collection'].tolist())
            df.drop(index = indices, inplace = True)
            pd.concat([df, pd.DataFrame(data = {'repository': [repo], 'collection': [collections]})])
    df.to_csv(DATA_DICT['github']['repo_name'], index = False)
    
    