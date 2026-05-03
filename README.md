# BIS Compass
AI-powered application to check BIS (Bureau of Indian Standards) compliance using LLM + RAG.
## Features
* Find relevant BIS standards from product description
* Compliance analysis with score, issues, and fixes
* PDF document compliance checking
## Tech Stack
* Streamlit
* FAISS
* Sentence Transformers
* Groq API
* PyMuPDF
## Live Demo
[Click here to use the app](https://bis-compass-s7rkw4kr6ortagvrefvzps.streamlit.app/)
## Run Locally
bash
pip install -r requirements.txt
streamlit run app.py
## Environment Variable
Create a `.env` file:
GROK_API_KEY=your_api_key_here
## Project Structure
app.py
requirements.txt
index.faiss
chunks.pkl
README.md
## Deployment
1. Upload project to GitHub
2. Deploy on Streamlit Cloud
3. Add secret:
4. GROK_API_KEY = your_api_key_here
