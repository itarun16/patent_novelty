import os
import json
import pickle
import faiss
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai

# =====================================
# INIT
# =====================================
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

EMBED_MODEL = "gemini-embedding-001"

print("✅ Gemini configured")


# =====================================
# EMBED FUNCTION
# =====================================
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


# =====================================
# LOAD PATENTS
# =====================================
patents = []

with open("real_patents.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        patents.append(json.loads(line))

print("Patents loaded:", len(patents))


# =====================================
# BUILD EMBEDDINGS
# =====================================
embeddings = []
metadata = []

for patent in tqdm(patents):

    text = patent.get("claims", "")

    if not text:
        continue

    try:
        emb = embed(text)
        embeddings.append(emb)
        metadata.append(patent)

    except Exception as e:
        print("Embedding failed:", e)


if len(embeddings) == 0:
    raise RuntimeError("No embeddings generated!")

embeddings = np.vstack(embeddings)

print("Embedding shape:", embeddings.shape)


# =====================================
# BUILD FAISS
# =====================================
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print("FAISS index size:", index.ntotal)


# =====================================
# SAVE
# =====================================
faiss.write_index(index, "patent.index")

with open("patent_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✅ INDEX BUILT SUCCESSFULLY")