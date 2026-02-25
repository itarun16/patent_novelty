import faiss
import pickle
import numpy as np
import os
from google import genai

# ------------------------------------
# GLOBALS
# ------------------------------------
index = None
metadata = None

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

EMBED_MODEL = "gemini-embedding-001"


# ------------------------------------
# INIT RETRIEVAL
# ------------------------------------
def init_retrieval():

    global index, metadata

    print("Initializing retrieval...")

    BASE_DIR = os.path.dirname(
        os.path.abspath(__file__)
    )

    INDEX_PATH = os.path.join(
        BASE_DIR,
        "patent.index"
    )

    META_PATH = os.path.join(
        BASE_DIR,
        "patent_metadata.pkl"
    )

    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    print("✅ Retrieval ready")


# ------------------------------------
# LAZY LOAD (Railway-safe)
# ------------------------------------
def ensure_loaded():
    global index
    if index is None:
        init_retrieval()


# ------------------------------------
# EMBEDDING
# ------------------------------------
def embed(text):

    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text
    )

    emb = np.array(
        result.embeddings[0].values,
        dtype="float32"
    ).reshape(1, -1)

    faiss.normalize_L2(emb)

    return emb


# ------------------------------------
# SEARCH
# ------------------------------------
def search_patents(query, k=5):

    ensure_loaded()

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