from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import pdfplumber

from retrieval import search_patents, init_retrieval
from gemini import rerank, init_gemini


# ------------------------------------------------
# FASTAPI INIT
# ------------------------------------------------
app = FastAPI(title="AI Patent Examiner")


# ------------------------------------------------
# CORS (frontend access)
# ------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------
# STARTUP — LOAD MODELS SAFELY
# ------------------------------------------------
@app.on_event("startup")
async def startup():

    print("\n🚀 Starting AI Patent Examiner...\n")

    # Load heavy systems in background thread
    await asyncio.to_thread(init_retrieval)
    await asyncio.to_thread(init_gemini)

    print("\n✅ SYSTEM READY\n")


# ------------------------------------------------
# HEALTH CHECK (IMPORTANT FOR RENDER)
# ------------------------------------------------
@app.get("/")
def root():
    return {"status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ------------------------------------------------
# PDF CLAIM EXTRACTION
# ------------------------------------------------
def extract_claim_text(file):

    text = ""

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"

    return text.strip()


# ------------------------------------------------
# SEARCH ENDPOINT
# ------------------------------------------------
@app.post("/search")
async def search(file: UploadFile = File(...)):

    # ---------- Extract Claim ----------
    claim_text = extract_claim_text(file.file)

    if not claim_text:
        return {"error": "No text found in PDF"}

    # ---------- FAISS Retrieval ----------
    retrieved = search_patents(claim_text)

    # ---------- Gemini Rerank ----------
    reranked = rerank(claim_text, retrieved)

    return {
        "claim": claim_text,
        "faiss_results": retrieved,
        "gemini_results": reranked
    }