#!/usr/bin/env python3
"""
Multimodal Patent Index Builder (Updated)
-----------------------------------------
• PatentSBERTa → FAISS text index
• CLIP → stored multimodal embeddings
• CPU compatible
• Handles relative paths by prepending BASE_IMAGE_PATH
• Skips missing/unreadable images with warnings
• Prints summary of patents with missing images
"""

import os
import json
import pickle
import faiss
import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# =====================================
# CONFIG
# =====================================
DATASET_PATH = "dataset_claim_2images.jsonl"
TEXT_INDEX_PATH = "claims_only.index"
META_PATH = "multimodal_metadata.pkl"

SBERT_MODEL = "AI-Growth-Lab/PatentSBERTa"
DEVICE = "cpu"

# Base path where images are stored
BASE_IMAGE_PATH = r"C:\Users\Tarun\OneDrive\Documents\newdataset"

# =====================================
# LOAD MODELS
# =====================================
print("Loading PatentSBERTa...")
sbert = SentenceTransformer(SBERT_MODEL, device=DEVICE)

print("Loading CLIP...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
clip_model = clip_model.to(DEVICE)
clip_model.eval()

print("✅ Models loaded")

# =====================================
# CLIP HELPERS
# =====================================
def clip_text_embed(text):
    with torch.no_grad():
        tokens = open_clip.tokenize([text])
        features = clip_model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0].astype("float32")

def clip_image_embed(path):
    if not os.path.exists(path):
        print(f"⚠️ Image not found: {path}")
        return None
    try:
        image = clip_preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            features = clip_model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0].astype("float32")
    except Exception as e:
        print(f"⚠️ Failed to embed image {path}: {e}")
        return None

# =====================================
# LOAD DATASET WITH FULL IMAGE PATHS
# =====================================
patents = []
patents_missing_images = []

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        patent = json.loads(line)
        # prepend base path and normalize
        patent["images"] = [
            os.path.normpath(os.path.join(BASE_IMAGE_PATH, img))
            for img in patent.get("images", [])
        ]
        if any(not os.path.exists(img) for img in patent["images"]):
            patents_missing_images.append(patent["patent_id"])
        patents.append(patent)

print("Patents loaded:", len(patents))
if patents_missing_images:
    print(f"⚠️ Patents with missing images: {len(patents_missing_images)}")

# =====================================
# BUILD EMBEDDINGS
# =====================================
text_embeddings = []
metadata = []

for patent in tqdm(patents):

    patent_id = patent.get("patent_id")
    claims = patent.get("claims", "")
    images = patent.get("images", [])

    if not claims:
        continue

    try:
        # --- SBERT embedding (for FAISS recall)
        sbert_emb = sbert.encode(
            claims,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        text_embeddings.append(sbert_emb)

        # --- CLIP embeddings
        clip_text_emb = clip_text_embed(claims)

        clip_image_embs = []
        for img_path in images:
            emb = clip_image_embed(img_path)
            if emb is not None:
                clip_image_embs.append(emb)

        metadata.append({
            "patent_id": patent_id,
            "claims": claims,
            "clip_text_embedding": clip_text_emb,
            "clip_image_embeddings": clip_image_embs
        })

    except Exception as e:
        print(f"Embedding failed for {patent_id}: {e}")

if len(text_embeddings) == 0:
    raise RuntimeError("No embeddings generated!")

text_embeddings = np.vstack(text_embeddings)

print("SBERT embedding shape:", text_embeddings.shape)

# =====================================
# BUILD FAISS (Cosine Similarity)
# =====================================
dim = text_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(text_embeddings)

print("FAISS index size:", index.ntotal)

# =====================================
# SAVE
# =====================================
faiss.write_index(index, TEXT_INDEX_PATH)

with open(META_PATH, "wb") as f:
    pickle.dump(metadata, f)

print("✅ MULTIMODAL INDEX BUILT SUCCESSFULLY")

# =====================================
# SUMMARY
# =====================================
if patents_missing_images:
    print("\nSummary: Patents with missing images:")
    for pid in patents_missing_images:
        print(f" - {pid}")
else:
    print("\nAll patents have images present.")