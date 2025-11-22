import streamlit as st
import requests
import time

API_URL = "http://127.0.0.1:5000"

st.set_page_config(
    page_title="AI Spam Classifier",
    page_icon="📩",
    layout="centered"
)

st.markdown("""
# 📩 AI-Augmented Spam Detector  
### Hybrid ML + Optional AI Semantic Check  
---
""")

# -----------------------------------------
# INPUT
# -----------------------------------------
message = st.text_area("✉️ Enter SMS message:", height=120)

# Store results using session_state
if "svm_result" not in st.session_state:
    st.session_state.svm_result = None

if "ai_result" not in st.session_state:
    st.session_state.ai_result = None


# -----------------------------------------
# BUTTON 1 → SVM ONLY
# -----------------------------------------
if st.button("🔍 Predict (SVM Only)", use_container_width=True):
    if not message.strip():
        st.warning("⚠ Please type a message first.")
        st.stop()

    with st.spinner("Running SVM model..."):
        try:
            svm_res = requests.post(f"{API_URL}/predict-svm",
                                    json={"message": message}).json()
            st.session_state.svm_result = svm_res
        except Exception as e:
            st.error(f"❌ API Error (SVM): {e}")

    st.session_state.ai_result = None  # Reset AI result


# Display SVM result if available
if st.session_state.svm_result:
    svm_res = st.session_state.svm_result
    svm_label = svm_res["svm_label"]
    color = "🟢" if svm_label == "ham" else "🔴"

    st.markdown("## 📌 Results")
    st.write("---")
    st.markdown(f"### {color} SVM Prediction")
    st.write(f"**Label:** {svm_label}")
    st.write("---")

# -----------------------------------------
# BUTTON 2 → OPTIONAL AI CHECK (LLM)
# -----------------------------------------
if st.session_state.svm_result:  # Show only after SVM result
    if st.button("🤖 Advanced Check (AI)", use_container_width=True):
        with st.spinner("Calling AI model..."):
            try:
                ai_res = requests.post(f"{API_URL}/advanced-check",
                                       json={"message": message}).json()
                st.session_state.ai_result = ai_res
            except Exception as e:
                st.error(f"❌ API Error (AI): {e}")


# Display AI results if available
if st.session_state.ai_result:
    ai_res = st.session_state.ai_result
    ai_label = ai_res.get("label", "error")

    st.markdown("## 🧠 AI (LLM) Prediction")
    st.write(f"**Label:** {ai_label}")
    st.write(f"**Confidence:** {ai_res.get('confidence', '?')}")
    st.info(ai_res.get("explanation", "No explanation available."))
