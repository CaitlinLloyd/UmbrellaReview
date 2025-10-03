from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from sentence_transformers import SentenceTransformer
from umap import UMAP
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, PartOfSpeech
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation._base import BaseRepresentation
from typing import Mapping, List, Tuple
import anthropic
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from bertopic import BERTopic

# this is the csv with abstracts included
dataset="abstracts.csv"
# Extract abstracts to train on and corresponding titles
abstracts = dataset["Abstract"]
titles = dataset["Title"]

def bertopic_to_gensim_format(topic_model, documents):
    #"""Convert BERTopic topics to format compatible with gensim coherence"""
  topics = topic_model.get_topics()

  if -1 in topics:
        topics.pop(-1)  # Remove outlier topic

    # Get top words for each topic
  topic_words = []
  for topic_id in topics:
    words = [word for word, _ in topic_model.get_topic(topic_id)[:12]]
    topic_words.append(words)

  return topic_words

def calculate_topic_diversity(model, topk=10):
    """Calculate topic diversity for BERTopic"""
    topics = model.get_topics()

    # Remove outlier topic if present
    if -1 in topics:
        topics.pop(-1)

    if not topics:  # No topics found
        return 0

    # Get unique words across all topics
    all_words = set()
    for topic_id in topics:
        words = [word for word, _ in model.get_topic(topic_id)[:topk]]
        all_words.update(words)

    # Calculate diversity
    total_words = len(topics) * topk
    unique_words = len(all_words)

    return unique_words / total_words if total_words > 0 else 0


umap_model = UMAP(n_neighbors=15, n_components=4, min_dist=0.0, metric='cosine', random_state=42)

vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))


## test different models




top_mods = {}
# All representation models
representation_model = {
    "KeyBERT": keybert_model}

mods = ["pritamdeka/S-PubMedBert-MS-MARCO","all-MiniLM-L6-v2"]
EVAL=pd.DataFrame(columns=['col1', 'col2', 'col3', 'col4'])
for em_model in mods:
# Pre-calculate embeddings - we can use thre
  embedding_model = SentenceTransformer(em_model)
  embeddings = embedding_model.encode(abstracts, show_progress_bar=True)
  for k in [7,10,13,16]:
    cluster_model = KMeans(n_clusters=k)
    topic_model = BERTopic(
# Pipeline models
    embedding_model=embedding_model,
    umap_model=umap_model,  #umap_model=empty_dimensionality_model,
    hdbscan_model=cluster_model,
    #hdbscan_model=HDBSCAN(min_cluster_size=4, metric='euclidean', cluster_selection_method='eom', prediction_data=True),
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    calculate_probabilities=True,

  # Hyperparameters
    top_n_words=20,
    verbose=True
    )
    topics, probs = topic_model.fit_transform(abstracts, embeddings)
    name=f'm_{em_model}_{k}'
    top_mods[name]=topic_model
    topic_words = bertopic_to_gensim_format(topic_model, abstracts)
    # Preprocess documents
    processed_docs = [doc.lower().split() for doc in abstracts]
    dictionary = Dictionary(processed_docs)


# Calculate coherence
    coherence_model_m = CoherenceModel(
    topics=topic_words,
    texts=processed_docs,
    dictionary=dictionary,
    coherence='u_mass'
    )


    coherence_score_m = coherence_model_m.get_coherence()
    diversity_score = calculate_topic_diversity(topic_model)



    x=[coherence_score_m,diversity_score,em_model,k]
    EVAL.loc[len(EVAL)] = x
    
    ## get information for the best model
    topic_model.get_topic_info()

    # Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
topic_model.save("model_out", serialization="safetensors")