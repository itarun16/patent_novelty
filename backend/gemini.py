import os
import json
import re
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = "gemini-2.5-flash"


# -----------------------------------
def init_gemini():
    print("Initializing Gemini...")
    print("✅ Gemini ready")


# -----------------------------------
# TEXT RERANK
# -----------------------------------
def rerank(user_claim, retrieved):

    prior_text = ""

    for r in retrieved:
        prior_text += f"""
Patent: {r['id']}
Claim: {r['claims']}
Similarity: {r['score']}
------------------
"""

    prompt = f"""
Return ONLY JSON list.

Format:
[
{{"patent_id":"ID","final_score":0.9,"reason":"text"}}
]

USER CLAIM:
{user_claim}

PRIOR ART:
{prior_text}
"""

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    raw = response.text or ""

    match = re.search(r"\[.*\]", raw, re.S)

    if not match:
        return [
            {
                "patent_id": r["id"],
                "final_score": r["score"],
                "reason": "Fallback"
            }
            for r in retrieved
        ]

    try:
        return json.loads(match.group())
    except:
        return [
            {
                "patent_id": r["id"],
                "final_score": r["score"],
                "reason": "Fallback"
            }
            for r in retrieved
        ]


# -----------------------------------
# IMAGE ANALYSIS
# -----------------------------------
def image_relevance_score(user_claim, images, candidate):

    prompt = f"""
Return JSON ONLY:

{{"image_score":0.0-1.0,"reason":"short"}}

USER CLAIM:
{user_claim}

CANDIDATE CLAIM:
{candidate["reason"]}
"""

    parts = [prompt]

    for img in images[:2]:
        parts.append({
            "mime_type": "image/png",
            "data": img
        })

    response = client.models.generate_content(
        model=model,
        contents=parts
    )

    raw = response.text or ""

    match = re.search(r"\{.*\}", raw, re.S)

    if not match:
        return {
            "image_score": 0.0,
            "reason": "Fallback image analysis"
        }

    try:
        return json.loads(match.group())
    except:
        return {
            "image_score": 0.0,
            "reason": "Parsing error"
        }