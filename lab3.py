# ==========================================================
# Topic Modeling using LDA and NMF
# ==========================================================

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------
# STEP 1: LOAD DATASET
# ----------------------------------------------------------
df = pd.read_csv("news_articles.csv")

# Use text column and remove missing values
documents = df['Content'].dropna().sample(300, random_state=42).tolist()
print("Total documents used:", len(documents))

# ----------------------------------------------------------
# STEP 2: TEXT VECTORIZATION
# ----------------------------------------------------------
# TF-IDF for NMF
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.8,
    min_df=5
)
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Count Vectorizer for LDA
count_vectorizer = CountVectorizer(
    stop_words='english',
    max_df=0.8,
    min_df=5
)
count_matrix = count_vectorizer.fit_transform(documents)

# ----------------------------------------------------------
# STEP 3: FIT TOPIC MODELS
# ----------------------------------------------------------
num_topics = 5

nmf_model = NMF(n_components=num_topics, random_state=42)
nmf_model.fit(tfidf_matrix)

lda_model = LatentDirichletAllocation(
    n_components=num_topics,
    random_state=42,
    learning_method='batch'
)
lda_model.fit(count_matrix)

# ----------------------------------------------------------
# DISPLAY TOPICS FUNCTION
# ----------------------------------------------------------
def display_topics(model, feature_names, top_n=10):
    topics = []
    for idx, topic in enumerate(model.components_):
        words = [feature_names[i] for i in topic.argsort()[:-top_n-1:-1]]
        topics.append(words)
    return topics

nmf_topics = display_topics(nmf_model, tfidf_vectorizer.get_feature_names_out())
lda_topics = display_topics(lda_model, count_vectorizer.get_feature_names_out())

print("\n====== NMF Topics ======")
for i, words in enumerate(nmf_topics, 1):
    print(f"Topic {i}: {' | '.join(words)}")

print("\n====== LDA Topics ======")
for i, words in enumerate(lda_topics, 1):
    print(f"Topic {i}: {' | '.join(words)}")

# ----------------------------------------------------------
# STEP 4: USER INPUT TOPIC ANALYSIS
# ----------------------------------------------------------
user_text = input("\nEnter a news article text: ").strip()

if user_text:
    user_tfidf = tfidf_vectorizer.transform([user_text])
    user_count = count_vectorizer.transform([user_text])

    nmf_dist = nmf_model.transform(user_tfidf)[0]
    lda_dist = lda_model.transform(user_count)[0]

    print("\nTop NMF Topics:")
    for t in nmf_dist.argsort()[-2:][::-1]:
        print(f"Topic {t+1} (Score: {nmf_dist[t]:.3f})")

    print("\nTop LDA Topics:")
    for t in lda_dist.argsort()[-2:][::-1]:
        print(f"Topic {t+1} (Score: {lda_dist[t]:.3f})")

    similarity = cosine_similarity(user_tfidf, tfidf_matrix)[0]
    top_docs = np.argsort(similarity)[-3:][::-1]

    print("\nMost Similar Documents:")
    for idx in top_docs:
        print(f"Similarity: {similarity[idx]:.3f}")
        print(documents[idx][:200])
        print("-" * 60)
else:
    print("No input provided.")
