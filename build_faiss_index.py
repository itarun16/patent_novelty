import json
import pickle
import faiss
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# =====================================
# DEVICE
# =====================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using:", DEVICE)


# =====================================
# LOAD PATENTSBERTA
# =====================================
tokenizer = AutoTokenizer.from_pretrained(
    "AI-Growth-Lab/PatentSBERTa"
)

model = AutoModel.from_pretrained(
    "AI-Growth-Lab/PatentSBERTa"
).to(DEVICE)

model.eval()


# =====================================
# EMBEDDING FUNCTION
# =====================================
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


# =====================================
# LOAD PATENTS
# =====================================
patents = []

with open(
    "real_patents.jsonl",
    "r",
    encoding="utf-8"
) as f:

    for line in f:
        patents.append(json.loads(line))


print("Patents loaded:", len(patents))


# =====================================
# BUILD EMBEDDINGS
# =====================================
embeddings = []
metadata = []

for patent in tqdm(patents):

    text = patent["claims"]

    try:
        emb = embed(text)

        embeddings.append(emb)
        metadata.append(patent)

    except:
        continue


embeddings = np.vstack(embeddings)

print("Embedding shape:", embeddings.shape)


# =====================================
# BUILD FAISS INDEX
# =====================================
index = faiss.IndexFlatIP(
    embeddings.shape[1]
)

index.add(embeddings)

print("FAISS index size:", index.ntotal)


# =====================================
# SAVE
# =====================================
faiss.write_index(
    index,
    "patent.index"
)

with open(
    "patent_metadata.pkl",
    "wb"
) as f:
    pickle.dump(metadata, f)

print("✅ INDEX BUILT SUCCESSFULLY")