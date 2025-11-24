import streamlit as st
import joblib
import pandas as pd
import os

from ai.llm_check import llm_check


st.set_page_config(page_title="AI Spam Detector", layout="wide")

# ----------------------
# Load ML Model
# ----------------------
MODEL_PATH = "model/svm_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

svm_model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_svm(message):
    X = vectorizer.transform([message])
    pred = svm_model.predict(X)[0]
    return int(pred)  # 1 = spam, 0 = ham


def save_disagreement(msg, ai_label):
    """Save disagreement rows for future retraining."""
    path = "data/new_training_samples.csv"

    df = pd.DataFrame([{
        "message": msg,
        "label": ai_label
    }])

    header = not os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=header)


# ----------------------
# UI Section
# ----------------------

st.title("📡 AI-Augmented Spam Detector")
st.write("Hybrid ML + LLM Spam Detection System")

message = st.text_area("Enter SMS message:", height=120)

if st.button("🔍 Predict (SVM Only)"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        pred = predict_svm(message)
        label = "spam" if pred == 1 else "ham"

        st.subheader("📌 Results")
        st.write(f"**SVM Prediction:** `{label}`")

        st.session_state["svm_result"] = pred
        st.session_state["sms"] = message


# ----------------------
# Advanced Check (AI)
# ----------------------

if "svm_result" in st.session_state:

    st.divider()
    st.subheader("🤖 Advanced Check (LLM AI)")

    if st.button("🧠 Run AI Check"):
        msg = st.session_state["sms"]
        svm_pred = st.session_state["svm_result"]

        with st.spinner("Analyzing with Qwen LLM..."):
            ai = llm_check(msg)

        st.write("### 🧠 AI (LLM) Prediction")
        st.json(ai)

        ai_label = ai.get("label", "error").lower()
        ai_num = 1 if ai_label == "spam" else 0

        # disagreement?
        if ai_label == "error":
            st.error("❌ AI returned invalid JSON.")
        else:
            if ai_num != svm_pred:
                st.warning("⚠ ML and AI disagree — saved for retraining.")
                save_disagreement(msg, ai_label)
            else:
                st.success("✔ ML and AI agree.")
