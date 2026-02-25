import streamlit as st
import requests
import fitz

API_URL = "https://patentnovelty-production.up.railway.app/search"


def extract_text(upload):

    doc = fitz.open(
        stream=upload.read(),
        filetype="pdf"
    )

    text = ""

    for page in doc:
        text += page.get_text()

    return text


st.title("AI Patent Examiner")

uploaded = st.file_uploader(
    "Upload Claim PDF",
    type="pdf"
)

if uploaded:

    claim_text = extract_text(uploaded)

    st.subheader("Extracted Claim")
    st.write(claim_text)

    # RESET file pointer (IMPORTANT)
    uploaded.seek(0)

    # SEND FILE (NOT JSON)
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

    st.subheader("FAISS Retrieval")

    for r in data["faiss_results"]:
        st.write(
            f"{r['id']} | {r['score']:.2f}"
        )

    st.subheader("Gemini Examiner")

    for r in data["gemini_results"]:
        st.write(
            f"{r['patent_id']} "
            f"| Score={r['final_score']:.2f}"
        )
        st.write(r["reason"])