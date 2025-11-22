import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json, re, os
import pandas as pd

MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"

print("[INFO] Loading Qwen 0.5B Chat model...")

device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map={"": device},
    torch_dtype=torch.float32
)


# -----------------------------------------------------------
#  SAFE CLEANER FOR LLM OUTPUT
# -----------------------------------------------------------
def clean_text(s: str) -> str:
    """Remove dangerous characters, control bytes and clean LLM output."""
    if not s:
        return ""

    # Remove control chars and invalid bytes
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    # Remove non-UTF8 safely
    s = s.encode("utf-8", "ignore").decode("utf-8")

    # Remove leftover weird chars
    s = re.sub(r"[\x00-\x1F\x7F]", " ", s)

    return s.strip()


# -----------------------------------------------------------
#  STRICT JSON EXTRACTOR
# -----------------------------------------------------------
def extract_json(raw: str):
    """Extract valid JSON object from LLM response."""
    raw = clean_text(raw)

    # Find JSON { ... }
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        return {
            "label": "error",
            "confidence": 0,
            "explanation": "No JSON returned"
        }

    json_text = clean_text(match.group())

    # Attempt to parse JSON safely
    try:
        return json.loads(json_text)
    except Exception as e:
        return {
            "label": "error",
            "confidence": 0,
            "explanation": f"Invalid JSON: {e}"
        }


# -----------------------------------------------------------
#  GENERATION CALL
# -----------------------------------------------------------
def generate_response(prompt):
    """Use Qwen Chat template + model.generate"""

    messages = [
        {"role": "system",
         "content": "You are a strict AI that must respond with ONLY a JSON object. No explanations, no markdown, no extra text."},
        {"role": "user", "content": prompt}
    ]

    # Build chat input template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(device)

    # Generate
    output_ids = model.generate(
        inputs.input_ids,
        max_new_tokens=200
    )

    generated = output_ids[0][inputs.input_ids.shape[1]:]

    raw = tokenizer.decode(generated, skip_special_tokens=True)
    return clean_text(raw)


# -----------------------------------------------------------
#  MAIN CLASSIFIER
# -----------------------------------------------------------
def ai_check(text):
    """Return structured JSON using LLM."""

    prompt = f"""
Classify the following SMS as **spam** or **ham**.

Respond with STRICT JSON ONLY :

{{
  "label": "",
  "confidence": 0.0,
  "explanation": ""
}}

Message: "{clean_text(text)}"
"""

    raw = generate_response(prompt)
    return extract_json(raw)


# -----------------------------------------------------------
#  SVM + LLM COMPARISON
# -----------------------------------------------------------
def compare_and_save(message, svm_pred):
    """Compare SVM & AI, save disagreements."""

    ai = ai_check(message)
    ai_label = ai.get("label", "").lower()
    ai_num = 1 if ai_label == "spam" else 0

    if ai_num != svm_pred:
        print("⚠ DISAGREEMENT FOUND — saving sample")

        save = "../new_training_samples.csv"
        row = {"message": message, "label": ai_label}

        df = pd.DataFrame([row])
        header = not os.path.exists(save)

        df.to_csv(save, mode="a", index=False, header=header)

    else:
        print("✔ SVM and AI predictions agree")

    return ai
