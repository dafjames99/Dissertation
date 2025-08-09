# Dissertation
This is the dissertation project Repository.

**Title**: *Forecasting AI Skill Demand using Innovation Signals from Open-Source Software
Material*

To-Do:

- [ ] Fine-Tune the JobSpanBERT: Do one run with the labels at hand - **LATER experiment** with changes to training data labelling
- [ ] Pipeline for GitHub Stars Representation: Graphs of Trends, Identification of High-Points
- [ ] Define numerical metric for resemblance of textual passage to a GitHub Framework
- [ ] Map terms - "machine learning" etc. - with prevalence over time, normalized (according to job title count) - how often for "data scientist" is the phrase "Natural language processing" or "generative AI" mentioned, and how does this change in time?


**THREE Data Pipelines:** Job Posting data; GitHub Repository Data; Keyword / Terminology Data

**Job Postings**: Found, Downloaded, Pre-processed - one resulting file with relevant job postings across n time-periods

**GitHub Repository**: Given an *owner/name* string, acquire (via GitHub API fetches) the resulting README, description, Topic-Tags and stargazer history

**Keyword / Terminology Data**: With a trival set of keywords (e.g. "Machine Learning"), find connecting topic-tags on GitHub to expand set of terminologies.

1. Get Keywords & GitHub repositories - this can positively feedback into itself; i.e. start with some trivial keywords, find relevant repo's; run again with expanded keyword set.

2. Get Job Posting Data

3. Using keywords, CREATE label dataset from job postings

4. Fine-tune JobSpanBERT on labels

5. Run inference on temporal samples

6. Identify parallel trends in the Job-data skills / frameworks / terms as they relate to the github repositories.

