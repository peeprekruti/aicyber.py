# train_model.py
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import numpy as np

# small demo DATA (same style as app)
DATA = [
    ("Your account has been suspended. Verify now at http://fake-bank-login.com", "scam"),
    ("URGENT: Update payment details or your subscription will be cancelled", "scam"),
    ("You've won a $1000 gift card! Click the link to claim", "scam"),
    ("Hey, are we meeting at 6 pm today at the cafe?", "safe"),
    ("Reminder: project deadline is next Friday. Please push your code", "safe"),
]

texts = [t for t, _ in DATA]
y = np.array([1 if label == "scam" else 0 for _, label in DATA])

vec = TfidfVectorizer(ngram_range=(1,2), max_features=500)
clf = LogisticRegression(max_iter=500)
pipe = make_pipeline(vec, clf)
pipe.fit(texts, y)

with open("scam_detector.pkl", "wb") as f:
    pickle.dump(pipe, f)

print("Saved scam_detector.pkl")
