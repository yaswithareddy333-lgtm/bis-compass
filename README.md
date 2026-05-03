# 🧭 BIS Compass - BIS Standards Recommendation Engine

AI-powered BIS standard discovery for Micro & Small Enterprises.

## What it does
- User enters a product description
- System searches BIS SP 21 document using RAG
- Returns top 5 relevant BIS standards instantly

## How to run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Build the index (run once)
python ingest.py

### 3. Run the app
streamlit run app.py

### 4. Run judge evaluation
python inference.py --input public_test_set.json --output my_results.json

## Evaluation Results
- Hit Rate @3: 90%
- MRR @5: 0.80
- Avg Latency: 1.10 seconds

## Tech Stack
- Streamlit (UI)
- FAISS (vector search)
- Sentence Transformers (embeddings)
- Groq LLaMA 3.3 (LLM)
- PyMuPDF (PDF parsing)

## Team
- Yaswitha