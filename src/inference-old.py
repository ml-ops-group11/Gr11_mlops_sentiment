import os
import sys

from transformers import pipeline

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "rohit-2145/roberta-sst2-v4"

# ── Read model config once ────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("HF_MODEL", DEFAULT_MODEL).strip()

print(f"Loading model: {MODEL_NAME}")

clf = pipeline(
    task="text-classification",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    token=os.environ.get("HF_TOKEN"),
    truncation=True,
    max_length=128,
)


def run_inference(text: str) -> dict:
    """Run sentiment analysis."""
    return clf(text)[0]


def main():
    input_text = os.environ.get("INPUT_TEXT", "").strip()

    if not input_text:
        print("ERROR: INPUT_TEXT environment variable is empty or not set.")
        sys.exit(1)

    print(f"\n{'=' * 50}")
    print(f"Input text : {input_text}")
    print(f"Model      : {MODEL_NAME}")
    print(f"{'=' * 50}")

    result = run_inference(input_text)

    print(f"\nPrediction : {result['label']}")
    print(f"Confidence : {result['score']:.4f}")
    print(f"{'=' * 50}\n")

    if result["score"] < 0.5:
        print(
            "WARNING: Low confidence score — check model is loaded correctly."
        )


if __name__ == "__main__":
    main()
    