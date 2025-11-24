import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json, re

MODEL = "Qwen/Qwen1.5-0.5B-Chat"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="cpu",
    torch_dtype=torch.float32
)


def call_llm(prompt):
    """Generate text using Qwen chat template"""
    messages = [
        {"role": "system", "content": "You are an assistant that MUST return valid JSON only."},
        {"role": "user", "content": prompt}
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([chat_text], return_tensors="pt")

    output = model.generate(
        inputs.input_ids,
        max_new_tokens=150
    )

    generated = output[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def clean_json(text):
    """Extract and decode clean JSON."""
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {"label": "error", "confidence": 0, "explanation": "No JSON found"}

    try:
        return json.loads(match.group())
    except Exception as e:
        return {"label": "error", "confidence": 0, "explanation": f"Invalid JSON: {e}"}


def llm_check(message):
    """Run AI classification"""
    prompt = f"""
Classify this SMS as spam or ham.

Return JSON ONLY in this format:

{{
  "label": "spam/ham",
  "confidence": 0.0,
  "explanation": "brief reason"
}}

SMS: "{message}"
"""

    out = call_llm(prompt)
    return clean_json(out)
