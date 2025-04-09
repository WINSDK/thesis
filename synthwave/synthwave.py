from pathlib import Path
from . import IS_MACOS, MODEL_DIR

def download_model(repo_id: str, model_path: Path):
    from huggingface_hub import snapshot_download
    download_path = Path(snapshot_download(
        repo_id=repo_id,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.py",
            "tokenizer.model",
            "*.tiktoken",
            "*.txt",
        ],
    ))
    if not download_path.exists():
        raise FileNotFoundError("Model not found in HF repo")
    model_path.parent.mkdir(exist_ok=True, parents=True)
    model_path.symlink_to(download_path, target_is_directory=True)

# parameter repo_id: Either a local path of a model: ./mymodel
#                    or a hugging model name: "bert-base-uncased"
def load_model(repo_id: str):
    model_path = MODEL_DIR / "pretrained" / repo_id.replace("/", "--")
    if not model_path.exists():
        download_model(repo_id, model_path)
    if IS_MACOS:
        from synthwave.mlx import load
        return load(model_path)
    else:
        from unsloth import FastLanguageModel
        return FastLanguageModel.from_pretrained(
            model_name=str(model_path), dtype=None, load_in_4bit=True
        )
