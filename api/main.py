from flask import Flask, request, jsonify
import pickle
import sys
import pandas as pd

# Add AI folder
sys.path.append("../ai")
from llm_check import compare_and_save, ai_check

app = Flask(__name__)

# Load SVM + TF-IDF
with open("../models/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("../models/svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)


# 1️⃣ SVM Prediction Only
@app.post("/predict-svm")
def predict_svm():
    data = request.json
    message = data.get("message", "")

    vec = tfidf.transform([message])
    pred = svm_model.predict(vec)[0]

    return jsonify({
        "svm_label": "spam" if pred == 1 else "ham",
        "svm_int": int(pred)
    })


# 2️⃣ AI (LLM) Only
@app.post("/advanced-check")
def advanced_check():
    data = request.json
    message = data.get("message", "")

    ai_result = ai_check(message)
    return jsonify(ai_result)


# 3️⃣ Combined (optional)
@app.post("/predict")
def full_predict():
    data = request.json
    message = data.get("message", "")

    vec = tfidf.transform([message])
    svm_pred = svm_model.predict(vec)[0]

    ai_result = compare_and_save(message, svm_pred)

    return jsonify({
        "svm_prediction": "spam" if svm_pred == 1 else "ham",
        "svm_int": int(svm_pred),
        "ai_result": ai_result
    })


@app.get("/")
def home():
    return jsonify({"message": "Spam Detection API is running!"})


if __name__ == "__main__":
    print("🚀 Starting Flask API on http://127.0.0.1:5000 ...")
    app.run(debug=False, host="127.0.0.1", port=5000, use_reloader=False)
