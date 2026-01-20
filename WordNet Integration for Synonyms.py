# Synonym Expansion using WordNet
# Extracting synonyms for English text using WordNet lexical database

import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

# ======= Download Required NLTK Resources (run once) =======
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# ======= Load Dataset =======
csv_path = "C:/Somalingaiah/nandinir/nlp-3rdsem/nlp-lab/part-a/Language Detection.csv"
df = pd.read_csv(csv_path, encoding='utf-8')

print("Dataset Loaded Successfully!")

# ======= Function to Get Synonyms Using WordNet =======
def get_synonyms(word):
    synonyms = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
    return list(synonyms)

# ======= Synonym Expansion Function =======
def synonym_expansion(sentence):
    words = word_tokenize(str(sentence))
    synonym_pairs = {}
    
    for word in words:
        # Consider only alphabetic words
        if word.isalpha():
            synonyms = get_synonyms(word.lower())
            if synonyms:
                synonym_pairs[word] = synonyms
    return synonym_pairs

# ======= Filter Only English Sentences =======
english_df = df[df['Language'] == 'English'].sample(5, random_state=42)

# ======= Perform Synonym Expansion =======
print("\n=== WordNet Synonym Expansion from Dataset ===\n")

for i, text in enumerate(english_df['Text'], start=1):
    print(f"Sentence {i}: {text}")
    
    expansion = synonym_expansion(text)
    
    if expansion:
        print("→ Words with synonyms found:")
        for word, syns in expansion.items():
            print(f"   {word} ↔ {', '.join(syns[:5])}")  # limit to 5 synonyms
    else:
        print("→ No synonyms found.")
    
    print("-" * 60)
