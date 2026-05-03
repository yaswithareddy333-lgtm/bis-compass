import argparse
import json
import time
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load RAG
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("index.faiss")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

client = Groq(api_key="GROQ_API_KEY")

def get_standards(query):
    start = time.time()
    
    # Search
    query_vec = model.encode([query]).astype("float32")
    _, indices = index.search(query_vec, 8)
    context = "\n\n".join([chunks[i]["text"] for i in indices[0]])
    
    # Ask Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a BIS standards expert.
Return ONLY a JSON array of standard codes exactly as they appear in BIS documents.
Format each code EXACTLY like this: "IS 269: 1989" (IS + space + number + colon + space + year)
Example: ["IS 269: 1989", "IS 8112: 1989", "IS 455: 1989"]"""
            },
            {
                "role": "user",
                "content": f"Product: {query}\n\nContext:\n{context}"
            }
        ]
    )
    
    latency = time.time() - start
    raw = response.choices[0].message.content.strip()
    
    # Parse JSON safely
    try:
        standards = json.loads(raw)
        standards = [s.strip() for s in standards]
    except:
        import re
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        standards = json.loads(match.group()) if match else []
    
    return standards, round(latency, 3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    with open(args.input, "r") as f:
        queries = json.load(f)
    
    results = []
    for item in queries:
        print(f"Processing: {item['id']}")
        standards, latency = get_standards(item["query"])
        results.append({
    "id": item["id"],
    "query": item["query"],
    "expected_standards": item.get("expected_standards", []),
    "retrieved_standards": standards,
    "latency_seconds": latency
})
        print(f"  Done: {standards}")
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()