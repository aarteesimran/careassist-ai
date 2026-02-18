import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="CareAssist AI", layout="centered")

st.title("CareAssist AI")
st.caption("Educational demo. Not medical advice. Verify outputs with official clinician instructions.")

@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-small")

model = load_model()

text = st.text_area("Paste medical notes or care instructions", height=180)

col1, col2 = st.columns(2)

def require_text(user_text: str) -> bool:
    if not user_text or not user_text.strip():
        st.warning("Please enter text first.")
        return False
    return True

def run_prompt(prompt: str, max_len: int) -> str:
    out = model(prompt, max_length=max_len, do_sample=False)
    return out[0]["generated_text"].strip()

with col1:
    if st.button("Simplify Note", use_container_width=True):
        if require_text(text):
            prompt = (
                "Simplify the following medical note for a caregiver with no medical background. "
                "Use plain language. Do not add any new facts. If information is missing, say what is missing.\n\n"
                f"Text:\n{text}"
            )
            result = run_prompt(prompt, max_len=240)
            st.subheader("Simplified version")
            st.write(result)

with col2:
    if st.button("Generate Daily Checklist", use_container_width=True):
        if require_text(text):
            prompt = (
                "Create a daily caregiver checklist from the text below. "
                "Return 6 to 10 short checklist items. Do not invent facts. "
                "If times are not provided, label items as Morning, Afternoon, Evening.\n\n"
                f"Text:\n{text}"
            )
            result = run_prompt(prompt, max_len=260)
            st.subheader("Daily checklist")
            st.write(result)

st.divider()
st.write("If this were a real product: add privacy controls, human review, and a safety policy for medical content.")
