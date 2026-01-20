# Lemmatization Using WordNet
# Context-aware lemmatization using POS tagging

import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# ======= Download Required NLTK Resources (run once) =======
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# ======= Load Dataset =======
csv_path = "C:/Somalingaiah/nandinir/nlp-3rdsem/nlp-lab/part-a/Language Detection.csv"
df = pd.read_csv(csv_path, encoding='utf-8')

print("Dataset Loaded Successfully!")
print(df.head())

# ======= Initialize Lemmatizer =======
lemmatizer = WordNetLemmatizer()

# ======= Helper Function: POS Tag Mapping =======
def get_wordnet_pos(treebank_tag):
    """
    Map Treebank POS tags to WordNet POS tags
    for accurate lemmatization
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# ======= Context-Aware Lemmatization Function =======
def context_aware_lemmatize(sentence):
    words = word_tokenize(str(sentence))
    pos_tags = pos_tag(words)
    lemmas = []

    for word, tag in pos_tags:
        if word.isalpha():
            wordnet_pos = get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(word.lower(), wordnet_pos)
            lemmas.append(lemma)

    return lemmas

# ======= Sample English Sentences =======
english_df = df[df['Language'] == 'English'].sample(5, random_state=42)

print("\n=== Context-Aware Lemmatization using WordNet ===\n")

for i, text in enumerate(english_df['Text'], start=1):
    print(f"Sentence {i}: {text}")
    lemmas = context_aware_lemmatize(text)
    print("→ Lemmatized Words:", lemmas)
    print("-" * 60)
