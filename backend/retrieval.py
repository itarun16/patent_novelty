import faiss
import pickle
import numpy as np
import os
import torch
import open_clip
from PIL import Image
from sentence_transformers import SentenceTransformer

# ------------------------------------
# GLOBALS
# ------------------------------------
index = None
metadata = None
sbert = None
clip_model = None
clip_preprocess = None

SBERT_MODEL = "AI-Growth-Lab/PatentSBERTa"
DEVICE = "cpu"


# ------------------------------------
# INIT RETRIEVAL
# ------------------------------------
def init_retrieval():

    global index, metadata, sbert
    global clip_model, clip_preprocess

    print("Initializing retrieval...")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    INDEX_PATH = os.path.join(BASE_DIR, "claims_only.index")
    META_PATH = os.path.join(BASE_DIR, "multimodal_metadata.pkl")

    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    print("Loading PatentSBERTa...")
    sbert = SentenceTransformer(SBERT_MODEL, device=DEVICE)

    print("Loading CLIP...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai"
    )
    clip_model = clip_model.to(DEVICE)
    clip_model.eval()

    print("✅ Retrieval ready")


# ------------------------------------
# LAZY LOAD
# ------------------------------------
def ensure_loaded():
    global index
    if index is None:
        init_retrieval()


# ------------------------------------
# SBERT EMBEDDING (FAISS)
# ------------------------------------
def sbert_embed(text):

    emb = sbert.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    return emb.reshape(1, -1)


# ------------------------------------
# CLIP HELPERS
# ------------------------------------
def clip_text_embed(text):
    with torch.no_grad():
        tokens = open_clip.tokenize([text])
        features = clip_model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0]


def clip_image_embed(path):
    try:
        image = clip_preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            features = clip_model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0]
    except:
        return None


# ------------------------------------
# SEARCH (Cross-Modal Fusion)
# ------------------------------------
def search_patents(user_claim, user_images=None, k=5):

    ensure_loaded()

    # -------- SBERT Recall --------
    query_emb = sbert_embed(user_claim)
    D, I = index.search(query_emb, 20)  # retrieve more for reranking

    # -------- CLIP Embeddings --------
    user_claim_clip = clip_text_embed(user_claim)

    user_image_embs = []
    if user_images:
        for img_path in user_images:
            emb = clip_image_embed(img_path)
            if emb is not None:
                user_image_embs.append(emb)

    results = []

    for idx, text_score in zip(I[0], D[0]):

        patent = metadata[idx]

        clip_text_score = float(
            np.dot(user_claim_clip, patent["clip_text_embedding"])
        )

        image_claim_score = 0.0
        claim_image_score = 0.0

        # User Image ↔ Candidate Claim
        if user_image_embs:
            sims = [
                np.dot(img_emb, patent["clip_text_embedding"])
                for img_emb in user_image_embs
            ]
            image_claim_score = float(np.mean(sims))

        # User Claim ↔ Candidate Images
        if user_image_embs and patent["clip_image_embeddings"]:
            sims = [
                np.dot(user_claim_clip, img_emb)
                for img_emb in patent["clip_image_embeddings"]
            ]
            claim_image_score = float(np.mean(sims))

        # ---- Final Fusion Score ----
        final_score = (
            0.5 * float(text_score)
            + 0.3 * clip_text_score
            + 0.2 * (image_claim_score + claim_image_score) / 2
        )

        results.append({
            "id": patent["patent_id"],
            "claims": patent["claims"],
            "score": final_score,
            "text_score": float(text_score),
            "clip_text_score": clip_text_score,
            "image_claim_score": image_claim_score,
            "claim_image_score": claim_image_score
        })

    # Sort by fused score
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:k]