from sys import stderr
from pathlib import Path
import platform
import inspect
import os

MODEL_DIR = Path.cwd() / "models"
IS_MACOS = platform.system() == "Darwin"
try:
    DEBUG = int(os.getenv("DEBUG", "0"))
    if DEBUG < 0 or DEBUG > 4:
        raise ValueError()
except ValueError:
    print("DEBUG variable must be >= 0 && <= 4")
    exit(1)

def error(msg):
    print(msg, file=stderr)
    raise ValueError(msg)

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
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map="mps"
        )
        return model, tokenizer
    else:
        from unsloth import FastModel
        return FastModel.from_pretrained(
            model_name=str(model_path),
            load_in_4bit=True,
            load_in_8bit=False,
        )

def fn_parameters(f):
    assert callable(f), "Not a valid python function"
    return inspect.signature(f).parameters.keys()
