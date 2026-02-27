from fastapi import FastAPI, UploadFile, File
import io
import fitz
import pdfplumber
import base64

from retrieval import search_patents, init_retrieval
from gemini import init_gemini, multimodal_examiner

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

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    images = []

    for page in doc:

        img_list = page.get_images(full=True)

        for img in img_list:

            xref = img[0]

            base_image = doc.extract_image(xref)

            images.append({
                "bytes": base_image["image"],
                "mime": base_image["ext"]  # png / jpeg
            })

    return images


# -----------------------------------
# SEARCH ENDPOINT
# -----------------------------------
@app.post("/search")
async def search(file: UploadFile = File(...)):

    # --------------------------------
    # Read uploaded PDF
    # --------------------------------
    pdf_bytes = await file.read()

    claim_text = extract_claim_text(
        io.BytesIO(pdf_bytes)
    )

    if not claim_text:
        return {"error": "No text found"}

    # --------------------------------
    # Extract images from PDF
    # --------------------------------
    images = extract_pdf_images(pdf_bytes)
    # images = [{"bytes":..., "mime":"png"}, ...]

    # --------------------------------
    # FAISS retrieval
    # --------------------------------
    retrieved = search_patents(claim_text)

    # --------------------------------
    # MULTIMODAL GEMINI EXAMINATION
    # (single call per candidate)
    # --------------------------------
    gemini_results = []

    for candidate in retrieved[:3]:   # top-3 only

        result = multimodal_examiner(
            claim_text,
            images,
            candidate
        )

        gemini_results.append(result)

    # --------------------------------
    # Encode images for frontend
    # --------------------------------
    encoded_images = []

    for img in images:
        encoded_images.append(
            base64.b64encode(
                img["bytes"]
            ).decode()
        )
    # --------------------------------
    # Response
    # --------------------------------
    return {
        "claim": claim_text,
        "images": encoded_images,
        "faiss_results": retrieved,
        "gemini_results": gemini_results
    }  


@app.get("/health")
def health():
    return {"status": "running"}