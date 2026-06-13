# Gr11_mlops_sentiment
End to End MLOps Pipeline
MLOps Sentiment Pipeline
End-to-end MLOps pipeline fine-tuning DistilBERT on SST-2 sentiment classification.

Team
Name	Role	GitHub
Reetesh - Admin
Sanjay - Mixed Roles
Rohit- Mixed Roles
Aamir- Mixed Roles


Stack
Model: distilbert-base-uncased (Hugging Face)
Dataset: SST-2 (GLUE)
Tracking: Weights & Biases
Container: Docker + FastAPI
CI/CD: GitHub Actions
Branches
main — protected, requires PR review

develop — integration branch


End-to-End MLOps Pipeline — IIT Jodhpur PGD AI
> **Task:** Binary sentiment classification on SST-2 using RoBERTa-large , fine-tuned on Kaggle, tracked with W&B, containerised with Docker, and automated with GitHub Actions.
---
Live Links

Resource	URL

Kaggle Notebook — Initial	'https://www.kaggle.com/code/g25ait2096/g25ait2096-mlops-sentiment-v1'

Kaggle Notebook — Final	'https://www.kaggle.com/code/rohitragh2145/e2e-sentiment'

Hugging Face Model	`https://huggingface.co/rohit-2145/roberta-sst2-v4`

Docker Image	`https://hub.docker.com/r/joyboy2145/roberta-sst2-inference`

W&B Dashboard	`https://wandb.ai/models-prom-iit-rajasthan8268/huggingface?nw=nwuserg25ait2145`
---
Repository Structure
```
.
├── src/
│   ├── prepare_data.py          # Task 2 — data cleaning
│   ├── train_kaggle.py          # Tasks 3 & 4 — model load + fine-tune
│   ├── inference.py             # Tasks 6 & 7 — Docker + Actions inference
│   └── quick_inference_check.py # Local sanity check
├── data/
│   └── id2label.json            # Label mapping (only file committed)
├── .github/
│   └── workflows/
│       ├── ci.yml               # Task 7.1 — lint on push to develop
│       └── inference.yml        # Task 7.2 — manual inference trigger
├── Dockerfile                   # Task 6
├── requirements.txt
└── README.md
```
---
Setup Instructions
1  Clone & install locally
```bash
git clone https://github.com/YOUR-ORG/mlops-a3.git
cd mlops-a3
pip install -r requirements.txt
```
2  Run data preparation locally
```bash
python src/prepare_data.py
# Outputs: data/train.csv, data/val.csv, data/id2label.json
```
3  Run quick inference check
```bash
export HF_TOKEN=hf_xxxxx
export HF_MODEL=YOUR-HF-USERNAME/distilbert-sst2-v1
python src/quick_inference_check.py
```
4  Build & run Docker image
```bash
# Build
docker build \
  --build-arg HF_MODEL_NAME=YOUR-HF-USERNAME/distilbert-sst2-v1 \
  -t mlops-a3-inference:latest .

# Test locally
docker run --rm \
  -e HF_TOKEN=hf_xxxxx \
  -e INPUT_TEXT="This film was absolutely wonderful!" \
  mlops-a3-inference:latest

# Push to Docker Hub
docker tag mlops-a3-inference:latest YOUR-DOCKERHUB/mlops-a3-inference:latest
docker push YOUR-DOCKERHUB/mlops-a3-inference:latest
```
5  Trigger GitHub Actions inference
Go to: GitHub repo → Actions → Inference → Run workflow
Enter any text in the `input_text` box and click Run workflow.
---
Experiment Comparison 

Metric	Version 1	Version 2

Epochs	3	5

Batch size	16	32

Learning rate	3e-5	2e-5

Accuracy	(fill after training)	(fill after training)

F1 (weighted)	(fill after training)	(fill after training)

Val Loss	(fill after training)	(fill after training)

