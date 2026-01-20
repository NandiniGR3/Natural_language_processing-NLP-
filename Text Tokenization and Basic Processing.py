# Text Tokenization and Basic Processing
# Sentence-level, Word-level, and Character-level Tokenization
# Numbers are preserved as tokens

import pandas as pd
import nltk
import re
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
csv_path = "C:/Somalingaiah/nandinir/nlp-3rdsem/nlp-lab/part-a/wiki_sentences_v2.csv"
df = pd.read_csv(csv_path, nrows=15)

# Combine all sentences into one text
text_data = " ".join(df['sentence'].astype(str).tolist())

print("\n=== ORIGINAL TEXT SAMPLE ===")
print(text_data[:300])

# ---------------- PREPROCESSING STEPS ----------------

# 1. Convert to lowercase (numbers remain unchanged)
text_data = text_data.lower()

# 2. Remove punctuation ONLY (numbers are preserved)
text_data = text_data.translate(
    str.maketrans('', '', string.punctuation)
)

# 3. Remove extra whitespaces
text_data = re.sub(r'\s+', ' ', text_data).strip()

# ---------------- TOKENIZATION ----------------

# Sentence-level tokenization
sentence_tokens = sent_tokenize(text_data)

# Word-level tokenization (numbers preserved)
word_tokens = word_tokenize(text_data)

# 4. Stopword removal (optional but recommended)
stop_words = set(stopwords.words('english'))
filtered_words = [w for w in word_tokens if w not in stop_words]

# Character-level tokenization
character_tokens = list(text_data)

# ---------------- OUTPUT ----------------

print("\n==== SENTENCE TOKENS ====")
for i, sent in enumerate(sentence_tokens, 1):
    print(f"{i}. {sent}")

print("\n==== WORD TOKENS (Numbers Preserved) ====")
print(word_tokens)

print("\n==== WORD TOKENS (After Stopword Removal) ====")
print(filtered_words)

print("\n==== CHARACTER TOKENS ====")
print(character_tokens)
