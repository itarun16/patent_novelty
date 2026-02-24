import streamlit as st
import requests
import fitz

API_URL = "http://127.0.0.1:8000/search"


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

    response = requests.post(
        API_URL,
        json={"claim": claim_text}
    )

    data = response.json()

    st.subheader("FAISS Retrieval")

    for r in data["retrieved"]:
        st.write(
            f"{r['id']} | {r['score']:.2f}"
        )

    st.subheader("Gemini Examiner")

    for r in data["reranked"]:
        st.write(
            f"{r['patent_id']} "
            f"| Score={r['final_score']:.2f}"
        )
        st.write(r["reason"])