import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import fitz  # PyMuPDF for PDF text extraction

# Load fine-tuned T5 model and tokenizer
MODEL_PATH = "C:/Users/User/Desktop/CG_SUM+BOT/Codes"  # Replace with actual path
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

# Streamlit UI
st.set_page_config(page_title="AI Policy Wizard", page_icon="📜", layout="wide")

st.markdown("<h1 class='title'>📜 AI Policy Wizard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Smart Policy Summarization & Generation</p>", unsafe_allow_html=True)

# Home Button
if st.button("⬅️ Back to Home"):
    st.experimental_rerun()

# Tabs for Navigation
tab1, tab2 = st.tabs(["📄 Policy Summarizer", "🏛️ Policy Generator"])

# Function to extract text from PDFs
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join([page.get_text("text") for page in doc])

# Summarization function
def summarize_text(text):
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Policy generation function
def generate_policy(scenario):
    input_text = "generate policy: " + scenario
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    policy_ids = model.generate(inputs, max_length=300, min_length=100, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(policy_ids[0], skip_special_tokens=True)

# Policy Summarizer Section
with tab1:
    st.header("📄 Upload a Policy Document for Summarization")
    uploaded_file = st.file_uploader("Upload a policy document (TXT, PDF)", type=["txt", "pdf"])
    text_content = ""
    if uploaded_file:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".pdf"):
            text_content = extract_text_from_pdf(uploaded_file)
        elif file_name.endswith(".txt"):
            text_content = uploaded_file.read().decode("utf-8")
        
        if text_content:
            st.text_area("Extracted Policy Content", text_content, height=200)
            if st.button("Summarize Policy"):
                with st.spinner("Generating summary..."):
                    summary = summarize_text(text_content)
                    st.success("Summary Generated!")
                    st.write(summary)
                    st.download_button("📥 Download Summary", summary, "summary.txt", "text/plain")

# Policy Generator Section
with tab2:
    st.header("🏛️ Generate a Policy Based on a Scenario")
    scenario = st.text_area("Describe your scenario", height=150)
    if st.button("Generate Policy"):
        with st.spinner("Generating policy..."):
            policy = generate_policy(scenario)
            st.success("Policy Generated!")
            st.write(policy)
            st.download_button("📥 Download Policy", policy, "policy.txt", "text/plain")
