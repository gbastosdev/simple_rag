import json
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer, util
from query_request import QueryRequest
import requests, os, uvicorn

app = FastAPI()

documents = [
    {
        "id": 1,
        "text": "Quantum computing leverages the principles of superposition and entanglement to perform computations at speeds unattainable by classical computers. Unlike classical bits, quantum bits or qubits can exist in multiple states simultaneously, allowing for parallel computation. Researchers are exploring quantum algorithms such as Shor's algorithm for integer factorization and Grover's algorithm for search problems."
    },
    {
        "id": 2,
        "text": "The field of Artificial Intelligence (AI) has evolved from symbolic reasoning in the 1950s to modern machine learning techniques. Early AI focused on rule-based systems, but limitations led to the rise of statistical learning and neural networks. Today, AI applications range from natural language processing to autonomous systems, with ongoing research in explainability and ethical AI."
    },
    {
        "id": 3,
        "text": "Blockchain is a decentralized ledger technology that ensures transparency and security through cryptographic hashing and consensus mechanisms. Each block contains a cryptographic hash of the previous block, linking them together to form an immutable chain. Applications of blockchain extend beyond cryptocurrencies, including supply chain management and secure digital identity."
    }
]

model = SentenceTransformer("all-MiniLM-L6-v2")
embedded_docs = {
    doc["id"]: model.encode(doc['text'],convert_to_tensor=True) for doc in documents
}

@app.post("/query")
def query(req: QueryRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="Request without entries!")

    embedded_query = model.encode(req.query,convert_to_tensor=True)
    best_document = {}
    best_score = float("-inf")

    for doc in documents:
        score = util.cos_sim(embedded_query, embedded_docs[doc["id"]])
        if score > best_score:
            best_score = score
            best_document = doc

    prompt = f"You are an AI assistant. Your answer will be exact like this document {best_document['text']}. User: {req.query}. Assistant:"

    try:
        url = 'http://localhost:11432/api/chat'
        data = {
                "model": "tinyllama",
                "messages": [{
                    "role": "user",
                    "content": prompt
                }],
                "stream": False
            }
        data_to_send = json.dumps(data).encode('utf-8')
        response = requests.post(url, data=data_to_send)
        res = response.json()
        return {"response": res.get("message").get("content")}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("fastrag:app", host="0.0.0.0", port=8000, reload=True)