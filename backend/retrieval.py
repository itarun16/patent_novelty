import faiss
import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load PatentSBERTa
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(
    "AI-Growth-Lab/PatentSBERTa"
)

model = AutoModel.from_pretrained(
    "AI-Growth-Lab/PatentSBERTa"
).to(DEVICE)

model.eval()


def embed(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        emb = model(**inputs).last_hidden_state.mean(dim=1)

    emb = emb.cpu().numpy().astype("float32")
    faiss.normalize_L2(emb)

    return emb


# -----------------------------
# Load Index
# -----------------------------



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_PATH = os.path.join(BASE_DIR, "patent.index")

index = faiss.read_index(INDEX_PATH)

META_PATH = os.path.join(BASE_DIR, "patent_metadata.pkl")

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

def search_patents(query, k=5):

    emb = embed(query)

    D, I = index.search(emb, k)

    results = []

    for idx, score in zip(I[0], D[0]):

        patent = metadata[idx]

        results.append({
            "id": patent["id"],
            "claims": patent["claims"],
            "score": float(score)
        })

    return results