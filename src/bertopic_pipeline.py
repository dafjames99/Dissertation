import re
import nltk
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from bertopic import BERTopic
from umap import UMAP
import pandas as pd
from utils.paths import DATA_DICT, SENTENCE_MODEL

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)      # Remove punctuation/numbers
    text = text.strip()
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

def preprocess_all(repo_texts):
    print("Preprocessing texts...")
    return [clean_text(text) for text in tqdm(repo_texts)]

def embed_texts(texts, model_index = 'b'):
    model_name = SENTENCE_MODEL[model_index]
    print(f"Loading sentence-transformer model: {model_name}")
    model = SentenceTransformer(model_name)
    print("Embedding texts...")
    return model.encode(texts, show_progress_bar=True)

def build_topic_model(embeddings, texts, **kwargs):
    print("Building topic model...")
    vectorizer_config = kwargs.get('vectorizer_config')
    topic_config = kwargs.get('topic_config')
    vectorizer_model = CountVectorizer(
        stop_words="english",
        **vectorizer_config
    )

    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=True,
        **topic_config
    )

    topics, probs = topic_model.fit_transform(texts, embeddings)
    return topic_model, topics, probs

def run_pipeline(repo_texts, model_index, **kwargs):
    
    cleaned_texts = preprocess_all(repo_texts)
    embeddings = embed_texts(cleaned_texts, model_index = model_index)
    topic_model, topics, probs = build_topic_model(embeddings, cleaned_texts, **kwargs)

    print("\nTop 10 Topics:")
    print(topic_model.get_topic_info().head(10))

    return embeddings, topic_model, topics, probs


repo_texts = pd.read_csv(DATA_DICT["embeddings"]["texts"]['repositories'])['repositories_text']

vectorizer_config = {
    'ngram_range': (1, 3),
    'min_df': 2, 
    'max_df': 0.8
}


# hdbscan_model = KMeans(
#     n_clusters=5
# )
import hdbscan

hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean', prediction_data=True)


topic_config = {
    'hdbscan_model': hdbscan_model,
    
}

sentence_model_index = 'b'

embeddings, topic_model, topics, probs = run_pipeline(
    repo_texts,
    model_index = sentence_model_index,
    topic_config = topic_config,
    vectorizer_config = vectorizer_config
)

# topic_model.save(DATA_DICT['models']['bertopic'], serialization="safetensors", save_embedding_model=SENTENCE_MODEL[sentence_model_index])

from umap import UMAP

umap_model = UMAP(n_neighbors=15, n_components=15, metric='cosine')
reduced_embeddings = umap_model.fit_transform(embeddings)

kmeans = KMeans(n_clusters=4)
labels = kmeans.fit_predict(reduced_embeddings)

sil_score = silhouette_score(reduced_embeddings, labels)
print(f"Reduced Silhouette Score: {sil_score:.3f}")



from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
labels = topic_model.topics_
score = silhouette_score(embeddings, labels)
print(f"Silhouette Score: {score:.3f}")


from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

range_n_clusters = range(2, 15)
scores = []

for n in range_n_clusters:
    model = KMeans(n_clusters=n, random_state=42)
    model.fit(embeddings)
    labels = model.labels_
    score = silhouette_score(embeddings, labels)
    scores.append(score)

plt.plot(range_n_clusters, scores)
plt.xlabel("n_clusters")
plt.ylabel("Silhouette Score")
plt.title("Optimal number of clusters")
plt.show()






# if __name__ == "__main__":
#     # Sample placeholder list; replace this with your actual repo_texts
#     repo_texts = [
#         "A PyTorch-based transformer library for training NLP models.",
#         "An API wrapper for scraping financial data from Yahoo Finance.",
#         "Lightweight command-line utility for JSON processing.",
#         "Dockerized microservice for image processing using OpenCV.",
#         "FastAI-compatible tools for audio signal classification."
#     ]

#     topic_model, topics, probs = run_pipeline(repo_texts)

#     # Uncomment to view visualizations
#     visualize_topics(topic_model)
#     visualize_barchart(topic_model)
