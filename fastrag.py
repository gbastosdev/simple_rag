from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer, util
from query_request import QueryRequest
import requests, os

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
    embedded_query = model.encode(req.query,convert_to_tensor=True)
    best_document = {}
    best_score = float("-inf")

    for doc in documents:
        score = util.cos_sim(embedded_query, embedded_docs[doc["id"]])
        if score > best_score:
            best_score = score
            best_document = doc

    prompt = f"You are an AI assistante. Base your answer on this document {best_document['text']}\n\nUser: {req.query}\n Assistant:"

    try:
        api_key = os.getenv("OA_KEY")
        url = 'https://api.openapi.com/v1/chat/completions'
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "Qwen/QwQ-32B-Preview",
            "messages": [{"role": "system", "content": prompt}]
        }
        response = requests.post(url, headers=headers, json=data)
        res = response.json()
        return {"response": res.get("choices")[0].get("message").get("content")}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
