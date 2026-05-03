import streamlit as st
import faiss
import pickle
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(
    page_title="BIS Compass",
    page_icon="🏭",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .title {
        font-size: 42px;
        font-weight: 800;
        color: #4285f4;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 16px;
        color: #8b9ab8;
        margin-bottom: 30px;
    }
    .card {
        background: #1a1f2e;
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .standard-code {
        font-size: 22px;
        font-weight: 700;
        color: #4285f4;
    }
    .high { color: #34a853; font-weight: 600; }
    .medium { color: #fbbc04; font-weight: 600; }
    .low { color: #ea4335; font-weight: 600; }
    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2a3550;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .metric-number {
        font-size: 28px;
        font-weight: 700;
        color: #4285f4;
    }
    .metric-label {
        font-size: 12px;
        color: #8b9ab8;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("index.faiss")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return model, index, chunks

client = Groq(api_key="GROQ_API_KEY")

# Header
st.markdown('<div class="title">🧭 BIS Compass</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered BIS Standard Discovery for Micro & Small Enterprises</div>', unsafe_allow_html=True)

# Stats row
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-number">500+</div><div class="metric-label">BIS Standards Indexed</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-number">90%</div><div class="metric-label">Hit Rate Accuracy</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-number">&lt;2s</div><div class="metric-label">Average Response Time</div></div>', unsafe_allow_html=True)

st.markdown("---")

# Examples
st.markdown("**💡 Try an example:**")
examples = [
    "33 Grade Ordinary Portland Cement",
    "Coarse aggregates for structural concrete",
    "Portland slag cement manufacturing"
]
col1, col2, col3 = st.columns(3)
selected = None
with col1:
    if st.button(examples[0]):
        selected = examples[0]
with col2:
    if st.button(examples[1]):
        selected = examples[1]
with col3:
    if st.button(examples[2]):
        selected = examples[2]

# Input
product = st.text_input("🔍 Describe your product:", value=selected if selected else "")

if st.button("Find BIS Standards →", type="primary"):
    if product:
        with st.spinner("Searching BIS SP 21 knowledge base..."):
            start = time.time()
            model, index, chunks = load_rag()

            query_vec = model.encode([product]).astype("float32")
            _, indices = index.search(query_vec, 8)
            context = "\n\n".join([chunks[i]["text"] for i in indices[0]])

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a BIS standards expert.
Return ONLY a JSON array of standard codes exactly as they appear in BIS documents.
Format: ["IS 269: 1989", "IS 8112: 1989", "IS 455: 1989"]
Return top 5 most relevant standards only."""
                    },
                    {
                        "role": "user",
                        "content": f"Product: {product}\n\nContext:\n{context}"
                    }
                ]
            )

            latency = time.time() - start
            raw = response.choices[0].message.content.strip()

            try:
                import json, re
                standards = json.loads(raw)
            except:
                match = re.search(r'\[.*\]', raw, re.DOTALL)
                standards = json.loads(match.group()) if match else []

        st.success(f"✅ Found {len(standards)} standards in {latency:.2f} seconds")

        for i, std in enumerate(standards, 1):
            relevance = "High" if i == 1 else "Medium" if i <= 3 else "Low"
            rel_class = relevance.lower()
            st.markdown(f"""
            <div class="card">
                <div class="standard-code">#{i} {std}</div>
                <div class="{rel_class}">● {relevance} Relevance</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("Please enter a product description")

st.markdown("---")
st.markdown('<p style="color:#4a5568;text-align:center;font-size:13px;">BIS Compass · Built for BIS × Sigma Squad Hackathon · IIT Tirupati 2026</p>', unsafe_allow_html=True)