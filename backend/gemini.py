import os
import json
import re
from google import genai

model = None

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# -----------------------------
# INIT GEMINI
# -----------------------------
def init_gemini():

    global model

    print("Initializing Gemini...")

    model = "gemini-2.5-flash"

    print("✅ Gemini ready")


# -----------------------------
# RERANK
# -----------------------------
def rerank(user_claim, retrieved):

    global model

    if model is None:
        init_gemini()

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

        response = client.models.generate_content(
            model=model,
            contents=prompt
        )

        raw = response.text or ""

        print("\n===== GEMINI RAW =====")
        print(raw)
        print("======================\n")

        match = re.search(r"\[.*\]", raw, re.S)

        if not match:
            return [
                {
                    "patent_id": r["id"],
                    "final_score": r["score"],
                    "reason": "Fallback ranking"
                }
                for r in retrieved
            ]

        return json.loads(match.group())

    except Exception as e:

        print("Gemini error:", e)

        return [
            {
                "patent_id": r["id"],
                "final_score": r["score"],
                "reason": "Fallback ranking"
            }
            for r in retrieved
        ]