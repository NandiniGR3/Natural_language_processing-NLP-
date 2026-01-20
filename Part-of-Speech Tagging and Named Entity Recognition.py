# Part-of-Speech Tagging and Named Entity Recognition (NER)

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import UnigramTagger, DefaultTagger
from nltk.corpus import treebank

# ===== Download Required NLTK Data =====
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('treebank')

# ===== Load Dataset =====
csv_path = "C:/Somalingaiah/nandinir/nlp-3rdsem/nlp-lab/part-a/Language Detection.csv"
df = pd.read_csv(csv_path, encoding='utf-8')

print("Dataset Loaded Successfully!\n")

# ===== Select Sample English Sentences =====
english_df = df[df['Language'] == 'English'].sample(5, random_state=42)

print("Selected English Sentences:\n")
for i, text in enumerate(english_df['Text'], 1):
    print(f"{i}. {text}")
print("\n")

# ===== POS Tagging =====
print("PART-OF-SPEECH TAGGING USING MULTIPLE TAGGERS:\n")

# Default Tagger (assigns NN to unknown words)
default_tagger = DefaultTagger('NN')

# Unigram Tagger trained on Treebank corpus
train_sents = treebank.tagged_sents()
unigram_tagger = UnigramTagger(train=train_sents, backoff=default_tagger)

# Built-in Averaged Perceptron Tagger
def perceptron_pos_tag(tokens):
    return nltk.pos_tag(tokens)

# ===== Perform POS Tagging =====
for i, text in enumerate(english_df['Text'], start=1):
    print("=" * 80)
    print(f"Sentence {i}: {text}\n")
    tokens = word_tokenize(str(text))

    print("Default Tagger:")
    print(default_tagger.tag(tokens), "\n")

    print("Unigram Tagger:")
    print(unigram_tagger.tag(tokens), "\n")

    print("Averaged Perceptron Tagger:")
    print(perceptron_pos_tag(tokens), "\n")

# ===== Named Entity Recognition (NER) =====
print("\nNAMED ENTITY RECOGNITION (NER):\n")

for i, text in enumerate(english_df['Text'], start=1):
    tokens = word_tokenize(str(text))
    pos_tags = nltk.pos_tag(tokens)
    ne_tree = nltk.ne_chunk(pos_tags)

    print("=" * 80)
    print(f"Sentence {i}: {text}")
    print("\nDetected Named Entities:")

    for subtree in ne_tree.subtrees():
        if subtree.label() in ['PERSON', 'ORGANIZATION', 'GPE']:
            entity = " ".join(word for word, pos in subtree.leaves())
            print(f"{entity}  -->  {subtree.label()}")

    print("=" * 80, "\n")
