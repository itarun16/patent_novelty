import os
import json
import re
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

model = None


# -----------------------------
# INITIALIZE GEMINI (LAZY LOAD)
# -----------------------------
def init_gemini():

    global model

    print("Initializing Gemini...")

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")

    genai.configure(api_key=GEMINI_API_KEY)

    model = genai.GenerativeModel(
        "gemini-2.5-flash"
    )

    print("✅ Gemini ready")


# -----------------------------
# RERANK
# -----------------------------
def rerank(user_claim, retrieved):

    global model

    if model is None:
        raise RuntimeError("Gemini not initialized")

    prior_text = ""

    for r in retrieved:
        prior_text += f"""
Patent: {r['id']}
Claim: {r['claims']}
Similarity: {r['score']}
------------------
"""

    prompt = f"""
Return ONLY VALID JSON.

Format EXACTLY:

[
 {{"patent_id":"USXXXX","final_score":0.9,"reason":"text"}}
]

USER CLAIM:
{user_claim}

PRIOR ART:
{prior_text}
"""

    try:

        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0}
        )

        raw = response.text or ""

        print("\n===== GEMINI RAW =====")
        print(raw)
        print("======================\n")

        # -------------------------
        # SAFE JSON EXTRACTION
        # -------------------------
        match = re.search(r"\[.*\]", raw, re.S)

        if not match:
            print("⚠ Gemini returned invalid JSON")

            return [
                {
                    "patent_id": r["id"],
                    "final_score": r["score"],
                    "reason": "Fallback: JSON parse failed"
                }
                for r in retrieved
            ]

        return json.loads(match.group())

    except Exception as e:

        print("🔥 Gemini failure:", e)

        return [
            {
                "patent_id": r["id"],
                "final_score": r["score"],
                "reason": "Fallback ranking (LLM error)"
            }
            for r in retrieved
        ]