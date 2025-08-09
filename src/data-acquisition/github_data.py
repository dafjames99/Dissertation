import argparse
import sys, time
import numpy as np, pandas as pd
from datetime import datetime, timezone as tz
from typing import Literal
from tqdm import tqdm
from pathlib import Path
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from graphql import build_schema

src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))

from utils.paths import DATA_DICT
from utils.vars import GITHUB_TOKEN

class GitHubGraphQL: 
    def __init__(self, repo_name, github_token, schema_filepath = DATA_DICT['gql']['schema'], **kwargs):
        self.repo_name = repo_name
        try:
            self.owner, self.name = repo_name.split('/')
        except:
            print(f'{repo_name} not a valid repository name; must be of form "owner/name"')
    
        self.client_kwargs = kwargs
        with open(schema_filepath, 'r') as f:
            schema_str = f.read()
            
        self.schema = build_schema(schema_str)
        
        self.transport = AIOHTTPTransport(
            url='https://api.github.com/graphql',
            headers={'Authorization': f'Bearer {github_token}'},
            ssl = True
        )        
        
    def _make_client(self):
        client = Client(
            transport=self.transport,
            schema=self.schema,
            execute_timeout=15,
            **self.client_kwargs
        )
        return client
        
        
    def run_query(self, query: Path | str, client = None, variables: dict = None, rate_wait = True):
        if client is None:
            client = self._make_client()
        if isinstance(query, Path):
            with open(query, 'r') as f:
                query = gql(f.read())
        else:
            query = gql(query)
        
        if rate_wait:
            with open(DATA_DICT['gql']['queries']['rate_limit'], 'r') as f:
                q = gql(f.read())
            rate_result = client.execute(q)
            remaining, reset_at = rate_result['rateLimit']['remaining'], datetime.strptime(rate_result['rateLimit']['resetAt'], '%Y-%m-%dT%H:%M:%SZ')
            if remaining == 0:
                time.sleep((reset_at - datetime.now(tz.utc)).total_seconds())
        
        return client.execute(query, variable_values = variables or {})
    
    def fetch_paginated(self,
                              query: Path | str,
                              result_access: str | list,
                              entry: Literal['nodes', 'edges'],
                              variables: dict = None,
                              tqdm_bar = None):
        after = None
        results = []
        page = 0
        client = self._make_client()
        while True:
            paginated_vars = dict(variables)
            
            if after:
                paginated_vars['after'] = after
            
            data = self.run_query(client=client, query=query, variables=paginated_vars)
            
            try:
                if isinstance(result_access, list):
                    for k in result_access:
                        data = data[k]
                else:
                    data = data[result_access]
                
                page_items = data[entry]
                if tqdm_bar is not None:
                    tqdm_bar.update(1)

                results.extend(page_items)
                
                pageinfo = data.get('pageInfo', {})
                if not pageinfo['hasNextPage']:
                    break
                else:
                    after = pageinfo['endCursor']
                    page += 1
            
            except KeyError:
                print('Error')
                break
            
        return results

    def fetch_property(self, property: Literal['stars', 'description', 'topics', 'readme']):
        variables={"owner": self.owner, "name": self.name}
        if (property == 'stars') or (property == 'topics'):
            vars = {
                'stars': {'result_access': ['repository', 'stargazers'], 'entry': 'edges'},
                'topics': {'result_access': ['repository', 'repositoryTopics'], 'entry': 'nodes'}
            }
            
            with tqdm(desc=f"{property} - {self.owner}/{self.name}", unit="page") as pbar:
                results = self.fetch_paginated(
                    DATA_DICT['gql']['queries'][property],
                    result_access = vars[property]['result_access'],
                    variables=variables,
                    entry =vars[property]['entry'],
                    tqdm_bar=pbar
                )
            if property == 'topics':
                results = ','.join([t['topic']['name'] for t in results])
            elif property == 'stars':
                results = [e['starredAt'] for e in results] 
        elif property == 'description':
            results = self.run_query(
                DATA_DICT['gql']['queries'][property],
                variables=variables
            )
            results = results['repository']['description']

        
        elif property == 'readme':
            
            results = self.run_query(
                DATA_DICT['gql']['queries']['readme'],
                variables=variables
            )
            readme_variants = ["readme_docs_md", "readme_MD", "readme_md", "readme_rst", "readme_txt", "readme_cap"]
            readme_text = None
            for key in readme_variants:
                try:
                    readme_text = results['repository'][key]["text"]
                    if readme_text:
                        break
                except (KeyError, TypeError):
                    continue
            results = readme_text
            # results = results['repository']['readme']['text']
        return results
    
def get_star_df(repository, stars):
    df = pd.DataFrame(stars, columns=["starred_at"])
    
    df["date"] = pd.to_datetime(df["starred_at"]).dt.date
    
    df = df.groupby("date").size().reset_index(name="stars")
    
    df["date"] = pd.to_datetime(df["date"])
    
    df = df.set_index("date").asfreq("D", fill_value=0)
    
    df["cumulative"] = df["stars"].cumsum()
    
    owner, name = repository.split('/')
    df['owner'] = owner
    df['repository'] = name
    df['fullname'] = repository
    
    return df
    
def collate_all_repositories():
    sources = [
        DATA_DICT['github']['repositories']['names']['source_1'],
        DATA_DICT['github']['repositories']['names']['source_2'],
        DATA_DICT['github']['repositories']['names']['source_3']
    ]
    df = pd.concat([pd.read_csv(s)['repository'] for s in sources])
    df = df.drop_duplicates()
    df.to_csv(DATA_DICT['github']['repositories']['names']['all'], index = False)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--include_stars',
        default='False'
    )
    args = parser.parse_args()
    include_stars = args.include_stars
    
    collate_all_repositories()
    repos = pd.read_csv(DATA_DICT['github']['repositories']['names']['all'])['repository'].tolist()
    # properties = ['description', 'topics', 'readme']
    # if include_stars == 'True': properties.append('stars')
    
    complete_repos = [
        f.name
        .replace('.csv', '')
        .replace('_', '/', 1) 
        for f in DATA_DICT['github']['metadata_dir'].glob('*.csv')]
    complete_stars = [
        f.name
        .replace('.csv', '')
        .replace('_', '/', 1) 
        for f in DATA_DICT['github']['stars_dir'].glob('*.csv')]
    
    for repo in repos:
        if repo in complete_repos and repo in complete_stars:
            print(f"{repo} metadata & stars already retrieved")
            continue
        else:
            fetcher = GitHubGraphQL(repo, GITHUB_TOKEN)
            properties = []
            if repo not in complete_repos:
                properties.append('description')
                properties.append('readme')
                properties.append('topics')
            else:
                print(f"{repo} metadata already retrieved")
            if repo not in complete_stars:
                properties.append('stars')
            else:
                print(f'{repo} Stars already Retrieved')
            results = dict()
            results['repository'] = repo
            results['owner'] = fetcher.owner
            results['repo'] = fetcher.name
            for property in properties:
                results[property] = fetcher.fetch_property(property)
            # print(results)
            if include_stars == 'True':
                get_star_df(repo, results['stars']).to_csv(DATA_DICT['github']['star_count'] / f'{fetcher.owner}_{fetcher.name}.csv', index = False)
                results.pop('stars')
            # results['topics'] = ','.join(results['topics'])
            print(results)
            pd.DataFrame(data = {k: [v] for k,v in results.items()}, index = None).to_csv(DATA_DICT['github']['metadata_dir'] / f'{fetcher.owner}_{fetcher.name}.csv', index = False)
    
    # ---- AGGREGATING ALL REPOSITORIES ----
    
    # --- METADATA ---
    df_metadata = pd.concat([pd.read_csv(f) for f in DATA_DICT['github']['metadata_dir'].glob('*.csv')])
    df_metadata.to_csv(DATA_DICT['github']['repositories']['metadata'], index = False)
    
    if include_stars == 'True':
        # --- STARS (LONG FORMAT) ---
        dfs = []
        for f in DATA_DICT['github']['stars_dir'].glob('*.csv'):
            fullname = f.name.replace('.csv', '').replace('_', '/', 1) 
            df_star = pd.read_csv(f)
            df_star['fullname'] = fullname
            dfs.append(df_star)
        df_stars = pd.concat(dfs)
        # --- STARS (WIDE FORMAT; In DAILY, MONTHLY, QUARTERLY, ) ---
        # print(df_stars.columns)
        repos = df_stars['fullname'].drop_duplicates().tolist()

        df_stars['date'] = pd.to_datetime(df_stars['date'])
        date_range = min_date, max_date = df_stars['date'].dt.date.min(), df_stars['date'].dt.date.max()

        full_date_range = pd.date_range(min_date, max_date)

        for (p, name) in [('D', 'daily'), ('M', 'monthly'), ('Q', 'quarterly')]:
            df_stars[p] = df_stars['date'].dt.to_period(p)
            
            group = df_stars.groupby(['fullname', p])['stars'].sum()
            full_p_range = full_date_range.to_period(p).drop_duplicates()
            
            if p == 'D':
                full_p_labels = full_p_range.to_timestamp()
            else:
                full_p_labels = full_p_range.to_timestamp(how='start')

            arr = np.stack([
                np.array([group.get((repo_name, v), 0) for v in full_p_range], dtype = np.int32)
                for repo_name in repos
            ])
            df = pd.DataFrame(
                arr,
                index = repos,
                columns = full_p_labels
            )
            df.to_parquet(DATA_DICT['github']['star_wide'][name])
        
        df_stars.to_csv(DATA_DICT['github']['star_long'], index = False)