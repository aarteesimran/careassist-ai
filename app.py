import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="CareAssist AI", layout="centered")

st.title("CareAssist AI")
st.caption("Educational demo. Not medical advice. Verify outputs with official clinician instructions.")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

tokenizer, model = load_model()

text = st.text_area("Paste medical notes or care instructions", height=180)

col1, col2 = st.columns(2)

def require_text(user_text: str) -> bool:
    if not user_text or not user_text.strip():
        st.warning("Please enter text first.")
        return False
    return True

def run_prompt(prompt: str, max_len: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_len,
        do_sample=False,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

with col1:
    if st.button("Simplify Note", use_container_width=True):
        if require_text(text):
            prompt = (
                "Rewrite the caregiver instructions below in simple, clear language for a caregiver with no medical training.\n"
                "Keep all important details.\n"
                "Do not shorten to one sentence.\n"
                "Do not add new facts.\n\n"
                f"Instructions:\n{text}"
            )
            result = run_prompt(prompt, max_len=220)
            st.subheader("Simplified Version")
            st.write(result)

with col2:
    if st.button("Generate Daily Checklist", use_container_width=True):
        if require_text(text):
            prompt = (
                "You are assisting a caregiver.\n"
                "Create a clear daily checklist from the instructions below.\n"
                "Return exactly 8 checklist items.\n"
                "Each item must be on a new line starting with '- '.\n"
                "Include all important actions mentioned.\n"
                "Do not summarize into one sentence.\n"
                "Do not add new facts.\n\n"
                f"Instructions:\n{text}"
            )
            result = run_prompt(prompt, max_len=260)
            st.subheader("Daily Checklist")
            st.write(result)

st.divider()
st.write("For real-world deployment: add privacy controls, logging, and human review.")
