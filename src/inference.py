import os
import numpy as np
from tokenizers import Tokenizer
from onnxruntime import InferenceSession
from huggingface_hub import hf_hub_download

MODEL_REPO = "rohit-2145/roberta-sst2-v4-onnx"

HF_TOKEN = os.getenv("HF_TOKEN")

# download ONNX model + tokenizer from Hugging Face
model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="model.onnx",
    token=HF_TOKEN
)

tokenizer_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="tokenizer.json",
    token=HF_TOKEN
)

# load
tokenizer = Tokenizer.from_file(tokenizer_path)
session = InferenceSession(model_path)

LABELS = ["negative", "positive"]

text = os.getenv("INPUT_TEXT", "This movie was amazing!")

# tokenize
enc = tokenizer.encode(text)

inputs = {
    "input_ids": np.array([enc.ids], dtype=np.int64),
    "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
}

# inference
outputs = session.run(None, inputs)
logits = outputs[0]

pred = int(np.argmax(logits, axis=1)[0])

print({
    "text": text,
    "prediction": LABELS[pred]
})
