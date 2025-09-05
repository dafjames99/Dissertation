# Cross-domain Sentence Embedding for time-series representation

This project, my MSc dissertation, is an investigation wherein the text-content of Github repositories and job postings, embedded in a shared semantic space, are represented by a similarity model that equates to measuring GitHub repo similarity to Job Postings (i.e.Labour demand) as a function of time (by the Job Postings' date).

Using Github repository star data, we can examine their relationship with the similarity time-series. 

This project employs techniques that leverage existing pre-trained sentence transformers, and experiments on a pipeline configuration that best reflects a baseline representation (in the form of keyword matching) of job/repo similarity. These are then used in conjunction with stars for time-lagged correlation analysis.

# Setup
## Installation 
First install Python (version 3.13.4 is used).
It is recommended that one uses a virtual environment, to reproduce the exact environment and keep its dependencies isolated:

```
python3 -m venv venv
```

then to activate the venv on Linux/MacOS:
```
source venv/bin/activate
```
or with Windows (Powershell),
```
.\venv\Scripts\Activate
```

Now, install the dependencies within `requirements.txt` via
```
pip install -r requirements.txt
```

## Github Token
For the accessing of GitHub's graphQL api, a github token is required. It is recommended that this be stored in PATH env variables on the local machine. More information on acquiring a GitHub access token can be found [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).


