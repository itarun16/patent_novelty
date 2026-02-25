import streamlit as st
import requests
import fitz

# -----------------------------------
# BACKEND URL
# -----------------------------------
API_URL = "https://patentnovelty-production.up.railway.app/search"


# -----------------------------------
# TEXT EXTRACTION
# -----------------------------------
def extract_text(upload):

    doc = fitz.open(
        stream=upload.read(),
        filetype="pdf"
    )

    text = ""

    for page in doc:
        text += page.get_text()

    return text


# -----------------------------------
# UI
# -----------------------------------
st.title("AI Patent Examiner")

uploaded = st.file_uploader(
    "Upload Claim PDF",
    type="pdf"
)

if uploaded:

    claim_text = extract_text(uploaded)

    st.subheader("Extracted Claim")
    st.write(claim_text)

    # IMPORTANT → reset file pointer
    uploaded.seek(0)

    response = requests.post(
        API_URL,
        files={
            "file": (
                uploaded.name,
                uploaded,
                "application/pdf"
            )
        }
    )

    data = response.json()

    # -----------------------------
    # FAISS
    # -----------------------------
    st.subheader("FAISS Retrieval")

    for r in data["faiss_results"]:
        st.write(
            f"{r['id']} | {r['score']:.2f}"
        )

    # -----------------------------
    # GEMINI
    # -----------------------------
    st.subheader("Gemini Examiner")

    for r in data["gemini_results"]:

        st.write(
            f"{r['patent_id']} "
            f"| Score={r['final_score']:.2f}"
        )
        st.write(r["reason"])

        if "image_analysis" in r:
            st.write(
                "🖼 Image Score:",
                r["image_analysis"]["image_score"]
            )
            st.write(
                r["image_analysis"]["reason"]
            )

        st.divider()