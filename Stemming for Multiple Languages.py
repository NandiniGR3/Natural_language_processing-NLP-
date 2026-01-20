# Stemming using Porter and Snowball Stemmers
# Multi-language stemming comparison

import pandas as pd
import nltk
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize

# ======= Download Required NLTK Data (run once) =======
nltk.download('punkt')

# ======= Load Dataset =======
csv_path = "C:/Somalingaiah/nandinir/nlp-3rdsem/nlp-lab/part-a/Language Detection.csv"
df = pd.read_csv(csv_path, encoding='utf-8')

print("Dataset Loaded Successfully!")
print(df.head())

# ======= Define Stemmers =======
porter = PorterStemmer()
snowball_languages = SnowballStemmer.languages

print(f"\nSnowball Supported Languages: {snowball_languages}\n")

# ======= Function to Perform Stemming =======
def stem_sentence(sentence, language):
    words = word_tokenize(str(sentence))

    # English: Compare Porter and Snowball
    if language.lower() == 'english':
        porter_stems = [porter.stem(w.lower()) for w in words if w.isalpha()]
        snowball_stems = [
            SnowballStemmer("english").stem(w.lower())
            for w in words if w.isalpha()
        ]
        return porter_stems, snowball_stems

    # Non-English languages supported by Snowball
    elif language.lower() in snowball_languages:
        try:
            snow_stemmer = SnowballStemmer(language.lower())
            snow_stems = [snow_stemmer.stem(w.lower()) for w in words if w.isalpha()]
            return None, snow_stems
        except:
            return None, []

    else:
        return None, []

# ======= Sample Sentences from Different Languages =======
sample_df = df.sample(5, random_state=42)

print("\n=== Multi-language Stemming Comparison ===\n")

for i, row in enumerate(sample_df.itertuples(), start=1):
    text, lang = row.Text, row.Language
    print(f"Sentence {i}: {text}")
    print(f"Language: {lang}")

    porter_stems, snowball_stems = stem_sentence(text, lang)

    if lang.lower() == 'english':
        print("→ Porter Stems:   ", porter_stems)
        print("→ Snowball Stems: ", snowball_stems)
    elif snowball_stems:
        print("→ Snowball Stems: ", snowball_stems)
    else:
        print("→ No stemming available for this language.")

    print("-" * 60)
