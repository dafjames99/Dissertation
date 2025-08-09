import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))


import pandas as pd
from pathlib import Path
from utils.job_title_taxonomy import title_match_pattern
from utils.paths import DATA_DICT
import matplotlib.pyplot as plt
import json
FILE = DATA_DICT['jobs']

def concatenate(array: list[list]) -> list:
    return_arr = []
    for child in array:
        if not isinstance(child, list):
            raise TypeError('Children must be list elements')
        [return_arr.append(item) for item in child]
    return return_arr

# ------------ Loading Data (Type Checking) ------------
df = pd.read_csv(FILE)
df['date'] = pd.to_datetime(df['date'])
# ------------ Monthly Overview of Data ------------
month_range = [str(d.year) + '-' + ('0' if d.month < 10 else '')  +  str(d.month) for d in [df['date'].min(), df['date'].max()]]
count_records_monthly = df.groupby('month').count()['title'].rename('count_month')

all_months = concatenate([[y + '-' +  m for m in [('0' if i < 10 else '')  + str(i) for i in range(1, 13)]] for y in ['2022', '2023', '2024', '2025']])[9:-7]

counts = []
for m in all_months:
    if m in count_records_monthly.index:
        counts.append(
            int(count_records_monthly.loc[
                count_records_monthly.where(
                    count_records_monthly.index == m
                    ).notna()
                ].values[0])
            )
    else:
        counts.append(0)
        
fig, ax = plt.subplots(figsize = (30, 5))
ax.bar(all_months, counts)
ax.axhline(y = 0, color = 'r')
plt.xticks(rotation = 90)
plt.show()

# ------------ Quaterly Overview of Data ------------

def quarterly_map(month_year):
    y, m = month_year.split('-')
    m = int(m)
    if m <= 3:
        q = 'Q1'
    elif m <= 6:
        q = 'Q2'
    elif m <= 9:
        q = 'Q3'
    else:
        q = 'Q4'
    return y + '-' + q


df['year-quarter'] = df['month'].apply(quarterly_map)
count_records_quarterly = df.groupby('year-quarter').count()['title'].rename('count_quarterly')
all_quarters = concatenate([[str(y) + '-' + q for q in ['Q' + str(i) for i in range(1, 5)]] for y in range(2022, 2026)])[3:-2]

counts_quarterly = []
for q in all_quarters:
    if q in count_records_quarterly.index:
        counts_quarterly.append(
            int(count_records_quarterly.loc[
                count_records_quarterly.where(
                    count_records_quarterly.index == q
                    ).notna()
                ].values[0])
            )
    else:
        counts_quarterly.append(0)
      
      
fig, ax = plt.subplots(figsize = (30, 5))
ax.bar(all_quarters, counts_quarterly)
ax.axhline(y = 0, color = 'r')
plt.xticks(rotation = 90)
plt.show()
      
# ------------ Quaterly Overview of Data, with a cap on samples ------------
cap = 435
capped_counts = [c if c < cap else cap for c in counts_quarterly]
fig, ax = plt.subplots(figsize = (30, 5))
ax.bar(all_quarters, capped_counts)
ax.axhline(y = 0, color = 'r')
ax.axhline(y = cap, color = 'g', linewidth = 5)
plt.xticks(rotation = 90)
plt.show()

repo_df = pd.read_csv(DATA_DICT['github']['repositories'])
repo_df['readme'].iloc[16]
