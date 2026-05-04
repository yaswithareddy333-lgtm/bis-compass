import streamlit as st
import faiss
import pickle
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load RAG resources
@st.cache_resource
def load_rag():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("index.faiss")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return model, index, chunks

# Configure Groq
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# App
st.title("🏭 BIS Compliance Checker")
st.caption("AI-powered BIS standard discovery for building materials")

product = st.text_input("Enter your product description:")

if st.button("Check Compliance"):
    if product:
        with st.spinner("Searching BIS standards..."):
            start = time.time()

            # Load RAG
            model, index, chunks = load_rag()

            # Search
            query_vec = model.encode([product]).astype("float32")
            _, indices = index.search(query_vec, 8)
            context = "\n\n".join([chunks[i]["text"] for i in indices[0]])

            # Ask Groq
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a BIS standards expert.
Use ONLY the provided context to recommend standards.
Respond in this exact format:

STANDARDS:
1. [IS code] - [Standard name] - Relevance: High/Medium/Low
2. [IS code] - [Standard name] - Relevance: High/Medium/Low
3. [IS code] - [Standard name] - Relevance: High/Medium/Low

REASON:
[One line explanation]

COMPLIANCE TIPS:
- [Tip 1]
- [Tip 2]"""
                    },
                    {
                        "role": "user",
                        "content": f"Product: {product}\n\nContext from BIS SP21:\n{context}"
                    }
                ]
            )

            latency = time.time() - start
            result = response.choices[0].message.content

            st.success(f"✅ Found in {latency:.2f} seconds")
            st.markdown(result)
    else:
        st.warning("Please enter a product description")
