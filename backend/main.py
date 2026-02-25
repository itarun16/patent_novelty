from fastapi import FastAPI, UploadFile, File
import io
import fitz
import pdfplumber

from retrieval import search_patents, init_retrieval
from gemini import rerank, init_gemini, image_relevance_score

app = FastAPI()


# -----------------------------------
# STARTUP
# -----------------------------------
@app.on_event("startup")
async def startup():

    print("🚀 Starting AI Patent Examiner...")

    init_retrieval()
    init_gemini()

    print("✅ SYSTEM READY")


# -----------------------------------
# TEXT EXTRACTION
# -----------------------------------
def extract_claim_text(pdf_file):

    text = ""

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    return text


# -----------------------------------
# IMAGE EXTRACTION
# -----------------------------------
def extract_pdf_images(pdf_bytes):

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    images = []

    for page in doc:

        for img in page.get_images(full=True):

            xref = img[0]
            base_image = doc.extract_image(xref)
            images.append(base_image["image"])

    return images


# -----------------------------------
# SEARCH ENDPOINT
# -----------------------------------
@app.post("/search")
async def search(file: UploadFile = File(...)):

    pdf_bytes = await file.read()

    claim_text = extract_claim_text(
        io.BytesIO(pdf_bytes)
    )

    if not claim_text:
        return {"error": "No text found"}

    images = extract_pdf_images(pdf_bytes)

    retrieved = search_patents(claim_text)

    reranked = rerank(claim_text, retrieved)

    # Image relevance (TOP 3 only)
    for r in reranked[:3]:

        r["image_analysis"] = image_relevance_score(
            claim_text,
            images,
            r
        )

    return {
        "claim": claim_text,
        "faiss_results": retrieved,
        "gemini_results": reranked
    }


@app.get("/health")
def health():
    return {"status": "running"}