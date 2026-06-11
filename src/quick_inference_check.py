"""
src/quick_inference_check.py
============================
Run this LOCALLY (on your own laptop) BEFORE building Docker,
to confirm the model loads and predicts correctly from SST-2 data.

Prerequisites (run once):
  pip install transformers torch datasets

Usage:
  HF_TOKEN=<your-token> python src/quick_inference_check.py
"""

import os

from transformers import pipeline

# ── Config — update after Task 5 push ────────────────────────────────────────
HF_MODEL = os.environ.get("HF_MODEL", "rohit-2145/roberta-sst2-v4")
HF_TOKEN = os.environ.get("HF_TOKEN")

# ── Five sample sentences from SST-2 validation set ──────────────────────────
SAMPLES = [
    ("it 's a charming and often affecting journey .", "POSITIVE"),
    ("unflinchingly bleak and desperate", "NEGATIVE"),
    (
        "allows us to hope that nolan is poised to embark a major career as a commercial director",
        "POSITIVE",
    ),
    (
        "the acting , costumes , music , cinematography and sound are all astounding given the production 's austere locales .",
        "POSITIVE",
    ),
    ("it 's predictable , formulaic and , worst of all , boring .", "NEGATIVE"),
]

print("=" * 60)
print(f"Model: {HF_MODEL}")
print("=" * 60)

clf = pipeline(
    task="text-classification",
    model=HF_MODEL,
    tokenizer=HF_MODEL,
    use_auth_token=HF_TOKEN or None,
    truncation=True,
    max_length=128,
)

correct = 0
for text, expected in SAMPLES:
    result = clf(text)[0]
    match = "✓" if result["label"] == expected else "✗"
    correct += int(result["label"] == expected)
    print(f"{match}  [{result['label']:8s} {result['score']:.3f}]  {text[:60]}")

print(
    f"\nAccuracy on 5 samples: {correct}/{len(SAMPLES)} = {correct/len(SAMPLES)*100:.0f}%"
)
