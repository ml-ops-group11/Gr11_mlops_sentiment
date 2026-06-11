"""
Task 2: Data Preparation & Normalisation
Dataset : SST-2 (Stanford Sentiment Treebank, binary sentiment)
Source  : Hugging Face datasets — 'sst2'  (~67 k training samples, well under 50 k limit when sampled)
Why SST-2?
  - Text modality → aligns with DistilBERT (Task 3)
  - Binary labels (0=negative, 1=positive) → simple id2label mapping
  - Freely available, no manual download needed
  - Small enough to train in Kaggle free-tier GPU hours
"""

import json
import os
import re

import pandas as pd
from datasets import load_dataset

# ── 1. Load raw data ─────────────────────────────────────────────────────────
print("Loading SST-2 dataset …")
raw = load_dataset("stanfordnlp/sst2")  # train / validation / test splits

# Keep only train + validation (test labels are hidden in SST-2)
train_raw = raw["train"].to_pandas()
val_raw = raw["validation"].to_pandas()

print(f"Raw train rows : {len(train_raw)}")
print(f"Raw val   rows : {len(val_raw)}")

# ── 2. Inspect ────────────────────────────────────────────────────────────────
print("\n--- Class distribution (train) ---")
print(train_raw["label"].value_counts())

print("\n--- Sample rows ---")
print(train_raw.head(3))

print("\n--- Missing values ---")
print(train_raw.isnull().sum())


# ── 3. Clean ──────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Cleaning decisions (justified):
    1. Lowercase  → reduces vocabulary size; DistilBERT tokeniser handles
                    casing internally, but consistent casing helps.
    2. Strip extra whitespace → artefacts from HTML scraping in SST-2.
    3. Remove non-ASCII control characters → can corrupt tokenisation.
    We deliberately keep punctuation because sentiment often depends on
    exclamation marks, question marks, etc.
    """
    text = text.lower()
    text = re.sub(r"[^\x20-\x7E]", " ", text)  # keep printable ASCII only
    text = re.sub(r"\s+", " ", text).strip()
    return text


train_raw["sentence"] = train_raw["sentence"].astype(str).map(clean_text)
val_raw["sentence"] = val_raw["sentence"].astype(str).map(clean_text)

# Drop any rows that became empty after cleaning
train_raw = train_raw[train_raw["sentence"].str.len() > 0].drop_duplicates(
    subset="sentence"
)
val_raw = val_raw[val_raw["sentence"].str.len() > 0].drop_duplicates(subset="sentence")

# ── 4. Sample to stay within Kaggle free-GPU limits ──────────────────────────
# 10 000 train + 1 000 val gives fast iterations while keeping the task real.
TRAIN_SIZE = 10_000
VAL_SIZE = 1_000

train_df = train_raw.sample(
    n=min(TRAIN_SIZE, len(train_raw)), random_state=42
).reset_index(drop=True)
val_df = val_raw.sample(n=min(VAL_SIZE, len(val_raw)), random_state=42).reset_index(
    drop=True
)

print(f"\nSampled train : {len(train_df)} rows")
print(f"Sampled val   : {len(val_df)} rows")
print("\nLabel distribution after sampling:")
print(train_df["label"].value_counts())

# ── 5. Save cleaned CSVs ─────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
train_df[["sentence", "label"]].to_csv("data/train.csv", index=False)
val_df[["sentence", "label"]].to_csv("data/val.csv", index=False)
print("\nSaved  data/train.csv  and  data/val.csv")

# ── 6. Save id2label mapping ─────────────────────────────────────────────────
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {v: k for k, v in id2label.items()}

with open("data/id2label.json", "w") as f:
    json.dump(id2label, f, indent=2)

print("Saved data/id2label.json →", id2label)
