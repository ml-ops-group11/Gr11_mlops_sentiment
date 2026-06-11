"""
Tasks 3 & 4 — Model Loading + Fine-Tuning on Kaggle with W&B Tracking
=======================================================================
WHAT THIS FILE IS:
  This is the code you will paste into a Kaggle Notebook cell-by-cell.
  Each "# ── CELL N ──" comment marks a separate notebook cell.
  Run them top-to-bottom in order.

DATASET: SST-2 (Stanford Sentiment Treebank, binary sentiment)
  Loaded from Hugging Face datasets library.
  10 000 train / 1 000 validation samples (sampled for Kaggle free-GPU).
"""

# ── CELL 1 ── Install / upgrade packages ─────────────────────────────────────
# Run this once at notebook start-up
# !pip install -q transformers datasets wandb huggingface_hub scikit-learn --upgrade

# ── CELL 2 ── Load secrets (Kaggle Secrets — never hard-code tokens) ──────────
from kaggle_secrets import UserSecretsClient
import os
import wandb
from huggingface_hub import login

secrets = UserSecretsClient()

os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
HF_TOKEN = secrets.get_secret("HF_TOKEN")

login(token=HF_TOKEN)   
wandb.login()           

print("Secrets loaded ✓")

import json
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score


VERSION      = "v1"         
MODEL_NAME   = "distilbert-base-uncased"
NUM_EPOCHS   = 4            
BATCH_SIZE   = 16           
LEARNING_RATE = 2e-5        
TRAIN_SIZE   = 10_000
VAL_SIZE     = 1_000
SEED         = 123

import re

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\x20-\x7E]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

raw = load_dataset("stanfordnlp/sst2")

def prep_split(split_name, size):
    df = raw[split_name].to_pandas()
    df["sentence"] = df["sentence"].map(clean_text)
    df = df[df["sentence"].str.len() > 0].drop_duplicates("sentence")
    df = df.sample(n=min(size, len(df)), random_state=SEED).reset_index(drop=True)
    return Dataset.from_pandas(df[["sentence", "label"]])

train_ds = prep_split("train",      TRAIN_SIZE)
val_ds   = prep_split("validation", VAL_SIZE)

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

with open("id2label.json", "w") as f:
    json.dump(id2label, f, indent=2)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ── CELL 6 ── Tokenise ────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenise(batch):
    return tokenizer(batch["sentence"], truncation=True, max_length=128)

train_tok = train_ds.map(tokenise, batched=True)
val_tok   = val_ds.map(tokenise,   batched=True)

collator = DataCollatorWithPadding(tokenizer=tokenizer)
print("Tokenisation done ✓")

# ── CELL 7 ── Load model ──────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── CELL 8 ── W&B run init ────────────────────────────────────────────────────
wandb.init(
    project="mlops-assignment3",
    name=f"run-{VERSION}",
    config={
        "model":         MODEL_NAME,
        "epochs":        NUM_EPOCHS,
        "batch_size":    BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "version":       VERSION,
        "platform":      "Kaggle",
        "train_size":    TRAIN_SIZE,
        "val_size":      VAL_SIZE,
    },
)
print(f"W&B run '{VERSION}' initialised ✓")

# ── CELL 9 ── Metrics function ────────────────────────────────────────────────
def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted"),
    }

# ── CELL 10 ── Training arguments ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir                  = f"./results-{VERSION}",
    num_train_epochs            = NUM_EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = 32,
    learning_rate               = LEARNING_RATE,
    warmup_ratio                = 0.1,
    weight_decay                = 0.01,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "accuracy",
    report_to                   = "wandb",
    run_name                    = f"run-{VERSION}",
    seed                        = SEED,
    fp16                        = True,           # use Kaggle T4 half-precision
)

# ── CELL 11 ── Train ──────────────────────────────────────────────────────────
trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_tok,
    eval_dataset    = val_tok,
    tokenizer       = tokenizer,
    data_collator   = collator,
    compute_metrics = compute_metrics,
)

trainer.train()
print("Training complete ✓")

# ── CELL 12 ── Evaluate & log final metrics ───────────────────────────────────
results = trainer.evaluate()
print("Final eval results:", results)

wandb.run.summary.update({
    "final_accuracy": results["eval_accuracy"],
    "final_f1":       results["eval_f1"],
    "final_loss":     results["eval_loss"],
})

# ── CELL 13 ── Push model to Hugging Face Hub (Task 5) ────────────────────────
HF_REPO = f"YOUR-HF-USERNAME/distilbert-sst2-{VERSION}"   # ← replace username

model.push_to_hub(HF_REPO)
tokenizer.push_to_hub(HF_REPO)
print(f"Model pushed to: https://huggingface.co/{HF_REPO}")

# Log model URL to W&B run summary
wandb.run.summary["huggingface_model"] = f"https://huggingface.co/{HF_REPO}"

wandb.finish()
print("W&B run finished ✓")
