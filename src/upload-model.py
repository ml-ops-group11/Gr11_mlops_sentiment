from huggingface_hub import HfApi

repo_id = "rohit-2145/roberta-sst2-v4-onnx"

api = HfApi()

api.create_repo(repo_id=repo_id, exist_ok=True)

api.upload_folder(
    folder_path="onnx_model",
    repo_id=repo_id,
    repo_type="model"
)