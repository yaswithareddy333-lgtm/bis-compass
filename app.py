import streamlit as st
import faiss
import pickle
import numpy as np
import time
import json
import re
import fitz
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="BIS Compass", layout="wide")

# -------------------- STYLING --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg: #070d18;
    --surface: #0c1524;
    --border: #1e2f4a;
    --blue: #3b82f6;
    --text: #e8edf5;
    --text-muted: #9fb3c8;
}

/* GLOBAL */
html, body, .stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

/* HEADER */
.app-header {
    text-align: center;
    margin-bottom: 40px;
    animation: fadeIn 0.8s ease-in-out;
}

.app-title {
    font-size: 36px;
    font-weight: 600;
}

.app-title span {
    color: var(--blue);
}

.app-desc {
    font-size: 15px;
    color: var(--text-muted);
    margin-top: 8px;
    line-height: 1.6;
}

/* LABEL FIX */
label {
    color: #cfe3ff !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* INPUT */
.stTextInput input, .stTextArea textarea {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 10px;
    padding: 12px;
    font-size: 14px;
    transition: 0.2s;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border: 1px solid var(--blue);
    box-shadow: 0 0 10px rgba(59,130,246,0.3);
}

.stTextArea textarea {
    height: 120px;
}

/* REDUCE COLUMN GAP */
div[data-testid="stHorizontalBlock"] {
    gap: 6px !important;
}

/* FEATURE BUTTONS (CARDS) */
.stButton > button {
    height: 80px;
    font-size: 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--blue);
    border-radius: 12px;
    font-weight: 500;
    transition: all 0.2s ease;
    width: 100%;
}

/* HOVER ANIMATION */
.stButton > button:hover {
    border: 1px solid var(--blue);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(59,130,246,0.25);
}

/* MAIN BUTTON */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    color: white;
    font-weight: 600;
    font-size: 15px;
    height: 48px;
    transition: 0.2s;
}

div.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(59,130,246,0.4);
}

/* RESULT */
.result-box {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    white-space: pre-wrap;
    font-size: 14px;
    line-height: 1.7;
    animation: fadeIn 0.5s ease-in-out;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: var(--surface);
    border: 1px dashed var(--border);
    border-radius: 10px;
    padding: 10px;
}

/* FOOTER */
.footer {
    text-align: center;
    margin-top: 50px;
    color: var(--text-muted);
    font-size: 12px;
}

/* FADE ANIMATION */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD --------------------
@st.cache_resource
def load_rag():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("index.faiss")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return model, index, chunks

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# -------------------- HEADER --------------------
st.markdown("""
<div class="app-header">
    <div class="app-title">BIS <span>Compass</span></div>
    <div class="app-desc">
        AI Compliance Assistant for MSMEs<br>
        Check BIS standards, detect risks, and get actionable fixes instantly
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------- FEATURE SELECTION --------------------
col1, col2, col3 = st.columns(3, gap="small")

if "mode" not in st.session_state:
    st.session_state.mode = "find"

with col1:
    if st.button("Find Standards"):
        st.session_state.mode = "find"

with col2:
    if st.button("Compliance Check"):
        st.session_state.mode = "compliance"

with col3:
    if st.button("PDF Analyzer"):
        st.session_state.mode = "pdf"

mode = st.session_state.mode
st.markdown("<br>", unsafe_allow_html=True)

# -------------------- FIND --------------------
if mode == "find":
    product = st.text_input("Product Description")

    if st.button("Analyze Compliance", type="primary"):
        if product:
            with st.spinner("Analyzing..."):
                model, index, chunks = load_rag()
                query_vec = model.encode([product]).astype("float32")
                _, indices = index.search(query_vec, 8)

                context = "\n\n".join([chunks[i]["text"] for i in indices[0]])

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "Return top 5 BIS standards"},
                        {"role": "user", "content": f"{product}\n{context}"}
                    ]
                )

                result = response.choices[0].message.content

            st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

# -------------------- COMPLIANCE --------------------
elif mode == "compliance":
    product2 = st.text_input("Product Description")
    declared = st.text_area("Standards Followed")

    if st.button("Analyze Compliance", type="primary"):
        if product2 and declared:
            with st.spinner("Analyzing..."):
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "Give compliance result, score, issues, fixes"},
                        {"role": "user", "content": f"{product2}\n{declared}"}
                    ]
                )

                result = response.choices[0].message.content

            st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

# -------------------- PDF --------------------
elif mode == "pdf":
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file and st.button("Analyze Compliance", type="primary"):
        with st.spinner("Analyzing PDF..."):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()

            text = text[:3000]

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Analyze BIS compliance"},
                    {"role": "user", "content": text}
                ]
            )

            result = response.choices[0].message.content

        st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown("""
<div class="footer">
BIS Compass · IIT Tirupati Hackathon 2026
</div>
""", unsafe_allow_html=True)
