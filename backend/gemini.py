from dotenv import load_dotenv
load_dotenv()
import os
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set")

genai.configure(api_key=GEMINI_API_KEY)

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
        print("\n================ GEMINI RAW RESPONSE ================\n")

        print("FULL RESPONSE OBJECT:")
        print(response)

        print("\n---------------- TEXT OUTPUT ----------------\n")

        try:
            print(response.text)
        except:
            print("No response.text available")

        print("\n=====================================================\n")

        raw = response.text or ""

        # -------------------------
        # Extract JSON safely
        # -------------------------
        match = re.search(r"\[.*\]", raw, re.S)

        if not match:
            print("⚠ Gemini returned non-JSON:")
            print(raw)

            # fallback
            return [
                {
                    "patent_id": r["id"],
                    "final_score": r["score"],
                    "reason": "Fallback: Gemini parsing failed"
                }
                for r in retrieved
            ]

        return json.loads(match.group())

    except Exception as e:

        print("🔥 Gemini failure:", e)

        # HARD FAILSAFE
        return [
            {
                "patent_id": r["id"],
                "final_score": r["score"],
                "reason": "Fallback ranking (LLM error)"
            }
            for r in retrieved
        ]