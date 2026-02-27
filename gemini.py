
import os
import pandas as pd
import json
import re
import time
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -------------------------
# CONFIG
# -------------------------
DATASET_PATH = r"C:\Users\Tarun\OneDrive\Documents\project2k26\PatentLMM-main\similarity score\PATENTMATCH\dataset\patentmatch_test_balanced.tsv"   # <-- PLACEHOLDER
MODEL = "gemini-2.5-flash"
NUM_SAMPLES = 100
BATCH_SIZE = 10
SLEEP_BETWEEN_CALLS = 12   # free tier safe

# -------------------------
# INIT
# -------------------------
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# -------------------------
# GEMINI BATCH FUNCTION
# -------------------------
def gemini_batch_predict(samples):

    formatted_pairs = ""

    for i, row in enumerate(samples):
        formatted_pairs += f"""
PAIR {i}
TEXT_A:
{row['text']}

TEXT_B:
{row['text_b']}
"""

    prompt = f"""
You are a patent similarity evaluator.

For each PAIR, determine whether TEXT_A and TEXT_B describe the same invention.

Return JSON ONLY in this format:

{{
  "predictions": [
    {{"pair_id": 0, "prediction": 0 or 1}},
    ...
  ]
}}

{formatted_pairs}
"""

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

    except ClientError as e:
        if "429" in str(e):
            print("Rate limit hit. Sleeping 20 seconds...")
            time.sleep(20)
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )
        else:
            raise e

    raw = response.text or ""
    match = re.search(r"\{.*\}", raw, re.S)

    if not match:
        raise ValueError("No JSON returned")

    return json.loads(match.group())


# -------------------------
# BENCHMARK
# -------------------------
def benchmark():

    df = pd.read_csv(DATASET_PATH, sep="\t")

    df_sample = df.sample(NUM_SAMPLES, random_state=42)
    samples = df_sample.to_dict(orient="records")

    all_true = []
    all_pred = []

    print(f"Evaluating {NUM_SAMPLES} samples in batches of {BATCH_SIZE}...\n")

    for start in range(0, len(samples), BATCH_SIZE):

        batch = samples[start:start+BATCH_SIZE]
        batch_number = start // BATCH_SIZE + 1

        print(f"Processing batch {batch_number}")

        result = gemini_batch_predict(batch)

        predictions = {
            item["pair_id"]: item["prediction"]
            for item in result["predictions"]
        }

        for i, row in enumerate(batch):
            true_label = int(row["label"])
            pred = int(predictions.get(i, 0))

            all_true.append(true_label)
            all_pred.append(pred)

        time.sleep(SLEEP_BETWEEN_CALLS)

    # -------------------------
    # METRICS
    # -------------------------
    acc = accuracy_score(all_true, all_pred)
    prec = precision_score(all_true, all_pred)
    rec = recall_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred)
    cm = confusion_matrix(all_true, all_pred)

    print("\n==============================")
    print(f"Samples tested: {NUM_SAMPLES}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("==============================\n")

    # -------------------------
    # SAVE RESULTS
    # -------------------------
    results_df = pd.DataFrame({
        "true_label": all_true,
        "prediction": all_pred
    })

    results_df.to_csv("gemini_results.csv", index=False)
    print("Saved predictions to gemini_results.csv")


if __name__ == "__main__":
    benchmark()