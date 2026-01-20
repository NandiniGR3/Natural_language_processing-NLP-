# ==========================================================
# Sentiment Analysis on Amazon Product Reviews
# Lexicon-Based (VADER) + ML-Based (TF-IDF + Logistic Regression)
# ==========================================================

import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download VADER Lexicon
nltk.download('vader_lexicon')

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------
df = pd.read_csv("amazon_review.csv")
df = df[['reviewText', 'overall']]
df.dropna(inplace=True)

# Convert ratings to binary sentiment
df['sentiment'] = df['overall'].apply(lambda x: 1 if x >= 4 else 0)

texts = df['reviewText']
labels = df['sentiment']

# ----------------------------------------------------------
# Train-Test Split
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# ----------------------------------------------------------
# Lexicon-Based Sentiment Analysis (VADER)
# ----------------------------------------------------------
sia = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    return 1 if sia.polarity_scores(text)['compound'] >= 0 else 0

vader_predictions = [vader_sentiment(text) for text in X_test]

print("\nLEXICON-BASED SENTIMENT ANALYSIS (VADER)")
print("Accuracy:", accuracy_score(y_test, vader_predictions))
print(classification_report(y_test, vader_predictions))

# ----------------------------------------------------------
# ML-Based Sentiment Analysis
# TF-IDF + Logistic Regression
# ----------------------------------------------------------
tfidf = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

model.fit(X_train_tfidf, y_train)
ml_predictions = model.predict(X_test_tfidf)

print("\nML-BASED SENTIMENT ANALYSIS")
print("Accuracy:", accuracy_score(y_test, ml_predictions))
print(classification_report(y_test, ml_predictions))

# ----------------------------------------------------------
# User Input Prediction (Hybrid Approach)
# ----------------------------------------------------------
print("\n======= USER REVIEW SENTIMENT PREDICTION =======")
user_review = input("Enter a product review: ")

vader_result = vader_sentiment(user_review)
user_vec = tfidf.transform([user_review])
ml_result = model.predict(user_vec)[0]

# Simple negation handling
negation_words = ["not", "no", "never", "n't"]

final_ml_result = 0 if any(word in user_review.lower() for word in negation_words) else ml_result

print("VADER Prediction:", "Positive" if vader_result else "Negative")
print("ML Model Prediction:", "Positive" if final_ml_result else "Negative")
