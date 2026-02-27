import streamlit as st
import requests
import fitz
import base64

# =====================================
# CONFIG
# =====================================
API_URL = "https://patentnovelty-production.up.railway.app/search"

st.set_page_config(
    page_title="AI Patent Examiner",
    layout="wide"
)

# =====================================
# GLOBAL STYLING (Professional Look)
# =====================================
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

h1 {
    font-weight: 700;
}

[data-testid="stMetricValue"] {
    font-size: 28px;
    font-weight: 600;
}

.card {
    padding: 1rem 1.2rem;
    border: 1px solid rgba(200,200,200,0.15);
    border-radius: 10px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# TEXT EXTRACTION
# =====================================
@st.cache_data
def extract_text(upload):
    doc = fitz.open(stream=upload.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# =====================================
# HEADER
# =====================================
st.title("AI Patent Examiner")
st.caption("Upload a claim PDF to perform similarity analysis using FAISS and Gemini")

uploaded = st.file_uploader(
    "Upload Claim PDF",
    type="pdf"
)

# =====================================
# MAIN FLOW
# =====================================
if uploaded:

    # ---------- PROCESSING STATUS ----------
    with st.status("Processing document...", expanded=True) as status:

        st.write("Extracting claim text...")
        claim_text = extract_text(uploaded)

        st.write("Sending document to backend...")
        uploaded.seek(0)

        with st.spinner("Running similarity analysis..."):
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

        if response.status_code != 200:
            st.error("Backend error. Please try again.")
            st.stop()

        data = response.json()

        status.update(
            label="Analysis completed",
            state="complete"
        )

    st.divider()

    # =====================================
    # CLAIM TEXT
    # =====================================
    with st.expander("Extracted Claim Text", expanded=False):
        st.write(claim_text)

    st.divider()

    # =====================================
    # EXTRACTED IMAGES
    # =====================================
    st.subheader("Extracted Images")

    if data["images"]:
        cols = st.columns(2)
        for i, img_str in enumerate(data["images"]):
            with cols[i % 2]:
                st.image(
                    base64.b64decode(img_str),
                    use_container_width=True
                )
    else:
        st.info("No images detected in the document.")

    st.divider()

    # =====================================
    # FAISS RESULTS (CLEAN CARDS)
    # =====================================
    st.subheader("FAISS Retrieval Results")

    if data["faiss_results"]:

        for r in data["faiss_results"]:

            with st.container(border=True):

                col1, col2 = st.columns([4,1])

                with col1:
                    st.markdown(
                        f"**Patent ID:** {r['id']}"
                    )

                with col2:
                    st.metric(
                        label="Similarity",
                        value=f"{r['score']:.2f}"
                    )

    else:
        st.info("No FAISS results available.")

    st.divider()

    # =====================================
    # GEMINI RESULTS
    # =====================================
    st.subheader("Gemini Examiner Analysis")

    if data["gemini_results"]:

        for r in data["gemini_results"]:

            with st.container(border=True):

                col1, col2 = st.columns([4,1])

                with col1:
                    st.markdown(f"### {r['patent_id']}")
                    st.write(r["reason"])

                with col2:
                    st.metric(
                        "Final Score",
                        f"{r['final_score']:.2f}"
                    )

                if "image_analysis" in r:
                    st.markdown("---")
                    st.write("Image Analysis Score:",
                             r["image_analysis"]["image_score"])
                    st.write(r["image_analysis"]["reason"])

    else:
        st.info("No Gemini analysis available.")