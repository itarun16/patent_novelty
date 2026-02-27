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
You are a professional patent examiner.

The USER has provided:
- A textual claim
- Supporting images describing their invention

Your task:
Evaluate whether the CANDIDATE PATENT discloses what is shown in the USER IMAGES.

Compare:
1. Technical disclosure overlap
2. Whether the CANDIDATE CLAIM covers what is shown in the images
3. Training method similarity (if applicable)

Important:
- Images describe the USER invention.
- Determine if the CANDIDATE patent teaches what is visible in the images.
- Do NOT compare images against the USER claim text.
- Compare images strictly against the CANDIDATE claim.

Return JSON ONLY:

{{
"final_score":0.0-1.0,
"image_score":0.0-1.0,
"reason":"short technical explanation"
}}

USER CLAIM (text reference only):
{user_claim}

CANDIDATE CLAIM:
{candidate.get("claims","")}

FAISS SIMILARITY:
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