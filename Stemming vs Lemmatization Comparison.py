# ================================
# Stemming vs Lemmatization Comparison
# ================================

import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import Counter

# ===== Download Required Resources (run once) =====
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

# ===== Load Dataset =====
csv_path = "C:/Somalingaiah/nandinir/nlp-3rdsem/nlp-lab/part-a/Language Detection.csv"
df = pd.read_csv(csv_path, encoding='utf-8')

print("Dataset Loaded Successfully")
print(df.head())

# ===== Initialize NLP Tools =====
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# ===== POS Tag Mapping =====
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# ===== Comparison Function =====
def compare_stemming_lemmatization(sentence):
    words = word_tokenize(str(sentence))
    
    stemmed_words = [stemmer.stem(w) for w in words]
    
    pos_tags = pos_tag(words)
    lemmatized_words = [
        lemmatizer.lemmatize(w, get_wordnet_pos(tag))
        for w, tag in pos_tags
    ]
    
    return words, stemmed_words, lemmatized_words

# ===== Select English Sentences =====
english_df = df[df['Language'] == 'English'].sample(5, random_state=42)

all_original = []
all_stemmed = []
all_lemmatized = []

print("\n--- Stemming vs Lemmatization Comparison ---\n")

for i, text in enumerate(english_df['Text'], start=1):
    words, stems, lemmas = compare_stemming_lemmatization(text)
    
    all_original.extend(words)
    all_stemmed.extend(stems)
    all_lemmatized.extend(lemmas)
    
    print(f"Sentence {i}: {text}")
    print("Original     :", words)
    print("Stemmed      :", stems)
    print("Lemmatized   :", lemmas)
    print("-" * 70)

# ===== Quantitative Analysis =====
original_unique = len(set(all_original))
stemmed_unique = len(set(all_stemmed))
lemmatized_unique = len(set(all_lemmatized))

# ===== Plot 1: Vocabulary Size Comparison =====
plt.figure()
plt.bar(
    ["Original", "Stemmed", "Lemmatized"],
    [original_unique, stemmed_unique, lemmatized_unique]
)
plt.title("Vocabulary Size After Preprocessing")
plt.xlabel("Text Representation")
plt.ylabel("Number of Unique Words")
plt.show()

# ===== Plot 2: Word Reduction Impact =====
original_count = len(all_original)
stemmed_count = len(all_stemmed)
lemmatized_count = len(all_lemmatized)

plt.figure()
plt.plot(
    ["Original", "Stemmed", "Lemmatized"],
    [original_count, stemmed_count, lemmatized_count],
    marker='o'
)
plt.title("Word Count Before and After Normalization")
plt.xlabel("Processing Stage")
plt.ylabel("Total Words")
plt.show()
