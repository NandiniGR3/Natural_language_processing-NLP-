# ==========================================================
# Multilingual NLP for Indian Languages
# Hindi Processing + Transliteration + English Translation
# ==========================================================

from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import re

# Translation (safe import)
try:
    from deep_translator import GoogleTranslator
    TRANSLATION_AVAILABLE = True
except:
    TRANSLATION_AVAILABLE = False

# ----------------------------------------------------------
# Step 1: Hindi Input
# ----------------------------------------------------------
text = input("Enter Hindi text:\n").strip()

# ----------------------------------------------------------
# Step 2: Normalization
# ----------------------------------------------------------
factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("hi")
normalized_text = normalizer.normalize(text)

print("\nNormalized Text:")
print(normalized_text)

# ----------------------------------------------------------
# Step 3: Tokenization
# ----------------------------------------------------------
tokens = indic_tokenize.trivial_tokenize(normalized_text)
print("\nTokens:")
print(tokens)

# ----------------------------------------------------------
# Step 4: Stopword Removal
# ----------------------------------------------------------
hindi_stopwords = {"और", "की", "है", "को", "में", "से", "पर", "।", ","}

filtered_tokens = [
    t for t in tokens
    if t not in hindi_stopwords and re.fullmatch(r"[\u0900-\u097F]+", t)
]

print("\nTokens after Stopword Removal:")
print(filtered_tokens)

# ----------------------------------------------------------
# Step 5: Transliteration
# ----------------------------------------------------------
iast_text = transliterate(normalized_text, sanscript.DEVANAGARI, sanscript.IAST)
itrans_text = transliterate(normalized_text, sanscript.DEVANAGARI, sanscript.ITRANS)

print("\nIAST Transliteration:")
print(iast_text)

print("\nITRANS Transliteration:")
print(itrans_text)

# ----------------------------------------------------------
# Step 6: English Translation (SAFE MODE)
# ----------------------------------------------------------
print("\nEnglish Translation:")

if TRANSLATION_AVAILABLE:
    try:
        translator = GoogleTranslator(source="hi", target="en")
        translation = translator.translate(normalized_text)
        print(translation)
    except Exception as e:
        print("Translation failed due to network restriction.")
        print("Reason:", e)
else:
    print("Translation library not available.")
    print("Install using: pip install deep-translator")
