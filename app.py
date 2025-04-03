import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF

# Configure API
GOOGLE_API_KEY = "AIzaSyBa8iVj1__2W2uStQ-G7iPPnEnAoPTet5o"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Streamlit UI
st.set_page_config(page_title="AI Policy Wizard", page_icon="📜", layout="wide")

# Custom CSS for Centered Title & Beautiful UI
st.markdown(
    """
    <style>
        /* Center the title */
        .title {text-align: center; font-size: 40px; font-weight: bold; color: #2E3B55;}
        .subtitle {text-align: center; font-size: 18px; color: #555; margin-bottom: 30px;}
        
        /* Button Styling */
        .stButton>button {
            width: 100%;
            height: 50px;
            font-size: 18px;
            border-radius: 10px;
            background-color: #2E3B55;
            color: white;
            border: none;
        }
        .stButton>button:hover {
            background-color: #1E2A3A;
        }

        /* Centering Containers */
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        /* Stylish Text Areas */
        .stTextArea textarea {
            font-size: 16px;
            border-radius: 10px;
            border: 1px solid #CCC;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

def navigate(page):
    st.session_state.page = page
    st.rerun()  # Use st.rerun() instead of st.experimental_rerun()

# Home Button
if st.session_state.page != "home":
    if st.button("⬅️ Back to Home"):
        navigate("home")

# 🎯 Home Page: Choose Summarizer or Generator
if st.session_state.page == "home":
    st.markdown('<h1 class="title">📜 AI Policy Wizard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Smart Policy Summarization & Generation</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="container">
        <p style="font-size: 18px; text-align: center;">
        Welcome to AI Policy Wizard! 🚀  
        Choose an option to get started:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Button Layout
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Policy Summarizer", key="summarizer"):
            navigate("summarizer")
    with col2:
        if st.button("🏛️ Policy Generator", key="generator"):
            navigate("generator")

elif st.session_state.page == "summarizer":
    st.header("📄 Upload a Policy Document for Summarization")
    uploaded_file = st.file_uploader("Upload a document (TXT, PDF)", type=["txt", "pdf"])

    content = ""
    if uploaded_file:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".pdf"):
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                content = "\n".join([page.get_text() for page in doc])
        elif file_name.endswith(".txt"):
            content = uploaded_file.read().decode("utf-8")

        if content:
            st.text_area("Extracted Policy Content", content, height=200)
            if st.button("Summarize Policy"):
                with st.spinner("Please wait... Generating summary..."):
                    response = model.generate_content(f"Summarize the following policy document: {content}")
                    summary = response.text
                    st.success("Summary Generated!")
                    st.write(summary)

                    # Download button
                    st.download_button("📥 Download Summary", summary, "summary.txt", "text/plain")

elif st.session_state.page == "generator":
    st.header("🏛️ Generate a Policy Based on a Scenario")
    scenario = st.text_area("Describe your scenario", height=150)

    if st.button("Generate Policy"):
        with st.spinner("Please wait... Generating policy..."):
            response = model.generate_content(f"Generate a policy based on the following scenario: {scenario}")
            policy = response.text
            st.success("Policy Generated!")
            st.write(policy)

            # Download button
            st.download_button("📥 Download Policy", policy, "policy.txt", "text/plain")
