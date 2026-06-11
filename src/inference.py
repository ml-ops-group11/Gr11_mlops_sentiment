import os
import sys

from transformers import pipeline

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "rohit-2145/roberta-sst2-v4"   # ← set after Task 5

def run_inference(text: str, model_name: str) -> dict:
    """Load model from HF Hub and return prediction."""
    print(f"Loading model: {model_name}")
    clf = pipeline(
        task="text-classification",
        model=model_name,
        tokenizer=model_name,
        use_auth_token=os.environ.get("HF_TOKEN"),
        truncation=True,
        max_length=128,
    )
    result = clf(text)[0]
    return result


def main():
    # ── Read inputs from environment ─────────────────────────────────────────
    input_text = os.environ.get("INPUT_TEXT", "").strip()
    model_name = os.environ.get("HF_MODEL", DEFAULT_MODEL).strip()

    if not input_text:
        print("ERROR: INPUT_TEXT environment variable is empty or not set.")
        sys.exit(1)

    # ── Run inference ─────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Input text : {input_text}")
    print(f"Model      : {model_name}")
    print(f"{'='*50}")

    result = run_inference(input_text, model_name)

    print(f"\nPrediction : {result['label']}")
    print(f"Confidence : {result['score']:.4f}")
    print(f"{'='*50}\n")

    # Quick sanity check — if confidence < 0.5 something is likely wrong
    if result["score"] < 0.5:
        print("WARNING: Low confidence score — check model is loaded correctly.")


if __name__ == "__main__":
    main()
