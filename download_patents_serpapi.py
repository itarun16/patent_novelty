from serpapi import GoogleSearch
import json
from tqdm import tqdm
import time

# ===================================
# INSERT YOUR SERPAPI KEY
# ===================================
SERPAPI_KEY = "bd82f3f7b9b073ff2016aa0fdb4628904138283bfc27b0438bb2c3bdaf4f7661"

QUERY = "machine learning patent"

OUTPUT = open(
    "real_patents.jsonl",
    "w",
    encoding="utf-8"
)

seen = set()

# -----------------------------------
# paginate results
# -----------------------------------
for start in tqdm(range(0, 100, 10)):

    params = {
        "engine": "google_patents",
        "q": QUERY,
        "api_key": SERPAPI_KEY,
        "start": start
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    patents = results.get(
        "organic_results", []
    )

    for p in patents:

        patent_id = p.get("publication_number")

        if not patent_id or patent_id in seen:
            continue

        seen.add(patent_id)

        claims_text = ""

        # sometimes snippet contains claim info
        claims_text = (
            p.get("snippet", "")
            + " "
            + p.get("title", "")
        )

        record = {
            "id": patent_id,
            "claims": claims_text
        }

        OUTPUT.write(
            json.dumps(record) + "\n"
        )

    time.sleep(2)  # avoid rate limit

OUTPUT.close()

print("✅ Real patents collected")