import os
import json
import re

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL = "gemini-2.5-flash"


# -------------------------------
def init_gemini():
    print("Initializing Gemini...")
    print("✅ Gemini ready")


# -------------------------------
# SINGLE MULTIMODAL EXAMINER
# -------------------------------
def multimodal_examiner(user_claim, images, candidate):

    try:

        prompt = f"""
You are acting as a professional patent examiner.

Your task is to evaluate whether the USER invention is disclosed by the CANDIDATE patent.

You must perform THREE independent analyses:

---------------------------------------
1. CLAIM vs CLAIM ANALYSIS
---------------------------------------
Compare the USER CLAIM and the CANDIDATE CLAIM.

Determine:
- Technical feature overlap
- Missing features
- Novel or distinguishing elements
- Whether the CANDIDATE claim would anticipate the USER claim

---------------------------------------
2. USER IMAGES vs CANDIDATE CLAIM
---------------------------------------
The user images represent the USER invention.

Evaluate:
- Do the CANDIDATE claims describe what is shown visually?
- Are core structural or functional elements present?

---------------------------------------
3. CANDIDATE IMAGES vs USER CLAIM
---------------------------------------
If candidate patent images are available, determine:
- Whether those images disclose what is claimed in the USER claim.
- Whether a skilled examiner would consider the invention visually disclosed.

---------------------------------------
SCORING RULES
---------------------------------------
claim_score:
Similarity of textual claims only.

image_score:
Similarity between visual disclosures and claims.

final_score:
Overall likelihood that the USER invention is already disclosed.

Range: 0.0 to 1.0

Guideline:
0.0 = completely different inventions
0.5 = partial overlap
1.0 = same invention / highly anticipatory

---------------------------------------
OUTPUT FORMAT (STRICT JSON ONLY)
---------------------------------------
{{
  "claim_score": 0.0,
  "image_score": 0.0,
  "final_score": 0.0,
  "reason": "Technical examiner-style explanation explaining overlap and differences."
}}

USER CLAIM:
{user_claim}

CANDIDATE CLAIM:
{candidate.get("claims","")}

FAISS SIMILARITY (reference only):
{candidate.get("score",0)}
"""

        contents = [prompt]

        # attach user images
        for img in images[:2]:

            mime = f"image/{img['mime']}"

            contents.append(
                types.Part.from_bytes(
                    data=img["bytes"],
                    mime_type=mime
                )
            )

        response = client.models.generate_content(
            model=MODEL,
            contents=contents
        )

        raw = response.text or ""

        match = re.search(r"\{.*\}", raw, re.S)

        if not match:
            raise ValueError("No JSON returned")

        result = json.loads(match.group())

        result["patent_id"] = candidate["id"]

        return result

    except Exception as e:

        print("MULTIMODAL ERROR:", e)

        # fallback using FAISS score
        return {
            "patent_id": candidate["id"],
            "final_score": candidate["score"],
            "image_score": 0.0,
            "reason": "Fallback (Gemini unavailable)"
        }