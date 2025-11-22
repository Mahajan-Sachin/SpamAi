import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import nltk
import re
import string

from merge import merge_datasets
 
nltk.download("stopwords")
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop = set(stopwords.words("english"))
lemm = WordNetLemmatizer()


def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [w for w in text.split() if w not in stop]
    words = [lemm.lemmatize(w) for w in words]
    return " ".join(words)

df = merge_datasets()
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
df["clean_text"] = df["message"].apply(preprocess)

X = df["clean_text"]
y = df["label_num"]

# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 
tfidf = TfidfVectorizer(max_features=5000)
tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)

model = LinearSVC()
model.fit(tfidf_train, y_train)

 
preds = model.predict(tfidf_test)
print("Accuracy:", accuracy_score(y_test, preds))
 
with open("../models/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("../models/svm_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Updated model saved successfully!")
