# Word Frequency Analysis and Stop Word Removal
# Counting word frequencies using English and custom stop words

import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Download required NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset (first 10 rows)
csv_path = "C:/Somalingaiah/nandinir/nlp-3rdsem/nlp-lab/part-a/wiki_sentences_v2.csv"
df = pd.read_csv(csv_path, nrows=10)

# Combine all sentences into a single text string
text_data = " ".join(df['sentence'].astype(str).tolist())

print("\n=== ORIGINAL TEXT SAMPLE ===")
print(text_data[:300])

# ---------------- PREPROCESSING ----------------

# 1. Convert text to lowercase
text_data = text_data.lower()

# 2. Remove punctuation and numbers
text_data = re.sub(r'[^a-z\s]', '', text_data)

# 3. Normalize whitespace
text_data = re.sub(r'\s+', ' ', text_data).strip()

# ---------------- TOKENIZATION ----------------

# Tokenize text into words
word_tokens = word_tokenize(text_data)

print("\n========== ORIGINAL WORD TOKENS ==========")
print(word_tokens[:50])

# ---------------- STOP WORD REMOVAL ----------------

# English stop words
english_stops = set(stopwords.words('english'))

filtered_words = [
    word for word in word_tokens
    if word not in english_stops
]

print("\n========== AFTER ENGLISH STOP WORD REMOVAL ==========")
print(filtered_words[:50])

# Custom stop words
custom_stops = {"said", "later", "connie"}

final_words = [
    word for word in filtered_words
    if word not in custom_stops
]

print("\n========== AFTER CUSTOM STOP WORD REMOVAL ==========")
print(final_words[:50])

# ---------------- WORD FREQUENCY ANALYSIS ----------------

word_freq = Counter(final_words)

print("\n========== WORD FREQUENCY DISTRIBUTION ==========")
for word, freq in word_freq.most_common(20):
    print(f"{word}: {freq}")
