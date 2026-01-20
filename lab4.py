# ==========================================================
# WORD EMBEDDINGS & SEMANTIC DOCUMENT SIMILARITY
# Word2Vec & GloVe (Pre-trained)
# ==========================================================

import gensim.downloader as api
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------
# STEP 1: LOAD PRE-TRAINED EMBEDDINGS
# ----------------------------------------------------------
print("Loading Word2Vec (Google News 300d)...")
word2vec_model = api.load("word2vec-google-news-300")

print("Loading GloVe (Wikipedia 100d)...")
glove_model = api.load("glove-wiki-gigaword-100")

print("Embeddings loaded successfully!\n")

# ----------------------------------------------------------
# STEP 2: USER INPUT DOCUMENTS
# ----------------------------------------------------------
num_docs = int(input("Enter number of documents to analyze: "))
documents = []

for i in range(num_docs):
    text = input(f"Enter Document {i+1}:\n")
    documents.append(text.strip())

# ----------------------------------------------------------
# STEP 3: DOCUMENT VECTOR FUNCTION
# ----------------------------------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()

def document_vector(doc, model):
    words = preprocess(doc)
    vectors = [model[word] for word in words if word in model]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# ----------------------------------------------------------
# STEP 4: COMPUTE DOCUMENT VECTORS
# ----------------------------------------------------------
word2vec_vectors = np.array([document_vector(doc, word2vec_model) for doc in documents])
glove_vectors = np.array([document_vector(doc, glove_model) for doc in documents])

# ----------------------------------------------------------
# STEP 5: COSINE SIMILARITY
# ----------------------------------------------------------
word2vec_similarity = cosine_similarity(word2vec_vectors)
glove_similarity = cosine_similarity(glove_vectors)

print("\n===== Word2Vec Similarity Matrix =====")
print(np.round(word2vec_similarity, 2))

print("\n===== GloVe Similarity Matrix =====")
print(np.round(glove_similarity, 2))

# ----------------------------------------------------------
# STEP 6: TOP SIMILAR DOCUMENTS
# ----------------------------------------------------------
top_n = int(input("\nEnter number of top similar documents: "))

for i, doc in enumerate(documents):
    print(f"\nDocument {i+1}: {doc[:150]}...")

    # Word2Vec
    sim_scores = word2vec_similarity[i]
    top_idx = np.argsort(sim_scores)[::-1]
    top_idx = [j for j in top_idx if j != i][:top_n]

    print("\nWord2Vec Similar Documents:")
    for j in top_idx:
        print(f"Doc {j+1} | Similarity: {sim_scores[j]:.3f}")

    # GloVe
    sim_scores_glove = glove_similarity[i]
    top_idx_g = np.argsort(sim_scores_glove)[::-1]
    top_idx_g = [j for j in top_idx_g if j != i][:top_n]

    print("\nGloVe Similar Documents:")
    for j in top_idx_g:
        print(f"Doc {j+1} | Similarity: {sim_scores_glove[j]:.3f}")
