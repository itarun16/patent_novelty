from serpapi import GoogleSearch
import json
from tqdm import tqdm
import time
SERPAPI_KEY = "bd82f3f7b9b073ff2016aa0fdb4628904138283bfc27b0438bb2c3bdaf4f7661"

QUERIES = [
    "neural network training patent",
    "machine learning optimization patent",
    "deep learning model patent",
    "computer vision neural network patent",
    "transformer architecture patent"
]

OUTPUT = open("real_patents.jsonl", "w", encoding="utf-8")

seen = set()

for query in QUERIES:

    print(f"\nCollecting: {query}")

    for start in tqdm(range(0, 500, 10)):

        params = {
            "engine": "google_patents",
            "q": query,
            "api_key": SERPAPI_KEY,
            "start": start
        }

        results = GoogleSearch(params).get_dict()
        patents = results.get("organic_results", [])

        for p in patents:

            patent_id = p.get("publication_number")

            if not patent_id or patent_id in seen:
                continue

            seen.add(patent_id)

            record = {
                "id": patent_id,
                "claims":
                    p.get("title","") + " " +
                    p.get("snippet","")
            }

            OUTPUT.write(json.dumps(record) + "\n")

        time.sleep(1.5)

OUTPUT.close()

print("✅ Large patent dataset created")