# Antonym-Based Text Contrast Analysis using WordNet

import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

# ======= Download Required NLTK Resources (run once) =======
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# ======= Function to Get Antonyms =======
def get_antonyms(word):
    antonyms = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            for ant in lemma.antonyms():
                antonyms.add(ant.name().replace("_", " "))
    return list(antonyms)

# ======= Antonym-Based Contrast Function =======
def antonym_contrast(sentence):
    words = word_tokenize(str(sentence))
    contrast_pairs = {}

    for word in words:
        # Consider only alphabetic words
        if word.isalpha():
            antonyms = get_antonyms(word.lower())
            if antonyms:
                contrast_pairs[word] = antonyms
    return contrast_pairs

# ======= Load Dataset =======
csv_path = "C:/Somalingaiah/nandinir/nlp-3rdsem/nlp-lab/part-a/Language Detection.csv"
df = pd.read_csv(csv_path, encoding='utf-8')

print("Dataset Loaded Successfully!")

# ======= Filter Only English Sentences =======
english_df = df[df['Language'] == 'English'].sample(5, random_state=42)

# ======= Perform Antonym-Based Contrast Analysis =======
print("\n=== Antonym-Based Text Contrast Analysis from Dataset ===\n")

for i, text in enumerate(english_df['Text'], start=1):
    print(f"Sentence {i}: {text}")

    contrast = antonym_contrast(text)

    if contrast:
        print("→ Words with antonyms found:")
        for word, ants in contrast.items():
            print(f"   {word} ↔ {', '.join(ants[:5])}")  # limit for readability
    else:
        print("→ No antonym relationships found.")

    print("-" * 60)
