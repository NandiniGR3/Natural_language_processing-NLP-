# Multi-language Tokenization
# Sentence-level, Word-level, and Character-level tokenization
# Handling Unicode and special characters

import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download tokenizer models (run once)
nltk.download('punkt')

# Load multilingual dataset
csv_path = "C:/Somalingaiah/nandinir/nlp-3rdsem/nlp-lab/part-a/Language Detection.csv"
df = pd.read_csv(csv_path)

print("=== Dataset Loaded Successfully ===")
print(df.head())

# Select random multilingual samples
sample_data = df.sample(10, random_state=42)
texts = sample_data['Text'].astype(str).tolist()
languages = sample_data['Language'].tolist()

# --------- MULTI-LANGUAGE TOKENIZATION ---------
for i, (text, lang) in enumerate(zip(texts, languages), start=1):
    print("\n==============================================")
    print(f"Sample {i} | Language: {lang}")
    print("Original Text:", text)

    # Ensure Unicode-safe text
    text = text.encode('utf-8', errors='ignore').decode('utf-8')

    # Sentence Tokenization (fallback for unsupported languages)
    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = [text]

    # Word Tokenization (Unicode-aware)
    try:
        words = word_tokenize(text)
    except Exception:
        words = text.split()

    # Character-level Tokenization (Unicode-safe)
    characters = list(text)

    # --------- DISPLAY RESULTS ---------
    print("\nSentence Tokens:")
    for j, sent in enumerate(sentences, 1):
        print(f"{j}. {sent}")

    print("\nWord Tokens:")
    print(words)

    print("\nCharacter Tokens:")
    print(characters)
