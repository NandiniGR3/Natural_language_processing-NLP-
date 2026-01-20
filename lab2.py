# ==========================================================
# TEXT CLUSTERING & DOCUMENT SIMILARITY
# TF-IDF + Cosine Similarity
# K-Means & Hierarchical Clustering
# ==========================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage

# ----------------------------------------------------------
# STEP 1: LOAD DATA (CSV / FOLDER / SINGLE TEXT)
# ----------------------------------------------------------

def load_documents(csv_path=None, folder_path=None):
    documents = []

    if csv_path:
        df = pd.read_csv(csv_path)
        documents = df['Content'].dropna().tolist()

    elif folder_path:
        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                with open(os.path.join(folder_path, file), encoding="utf-8") as f:
                    documents.append(f.read())

    return documents


# === Choose input method ===
documents = load_documents(csv_path="news_articles.csv")
documents = documents[:300]   # limit for speed

print("Total documents:", len(documents))

# ----------------------------------------------------------
# STEP 2: TF-IDF VECTORIZATION
# ----------------------------------------------------------
tfidf = TfidfVectorizer(
    stop_words='english',
    max_df=0.8,
    min_df=5
)

tfidf_matrix = tfidf.fit_transform(documents)
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

# ----------------------------------------------------------
# STEP 3: COSINE SIMILARITY
# ----------------------------------------------------------
cosine_sim = cosine_similarity(tfidf_matrix)

print("\nSample Cosine Similarity (first 5 docs):")
print(np.round(cosine_sim[:5, :5], 2))

# ----------------------------------------------------------
# STEP 4: K-MEANS CLUSTERING
# ----------------------------------------------------------
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(tfidf_matrix)

print("\nK-Means Cluster Distribution:")
print(pd.Series(labels).value_counts())

# ----------------------------------------------------------
# STEP 5: HIERARCHICAL CLUSTERING
# ----------------------------------------------------------
linkage_matrix = linkage(tfidf_matrix.toarray(), method='ward')

plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Documents")
plt.ylabel("Distance")
plt.show()

# ----------------------------------------------------------
# STEP 6: USER INPUT DOCUMENT SIMILARITY
# ----------------------------------------------------------
print("\n======= USER TEXT SIMILARITY =======")
user_text = input("Enter a document/text: ").strip()

if user_text:
    user_vec = tfidf.transform([user_text])
    scores = cosine_similarity(user_vec, tfidf_matrix)[0]

    top_docs = np.argsort(scores)[-3:][::-1]

    print("\nMost Similar Documents:\n")
    for idx in top_docs:
        print(f"Similarity Score: {scores[idx]:.3f}")
        print(documents[idx][:250])
        print("-" * 60)
