from datasets import Dataset, IterableDataset, DatasetDict
from .helpers import MODEL_DIR, error
from typing import Dict, Union, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import math

from tokenizers import Tokenizer
from unsloth.mlx.models import llama
from unsloth.mlx.trainer.trainer import TrainingArgs, TrainingCallback
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from transformers import AutoTokenizer, EvalPrediction

def quantize_if_necessary(config: Dict, model: nn.Module, weights: Dict):
    quantization = config.get("quantization", None)
    if quantization is None:
        return
    def class_predicate(p, m):
        return isinstance(m, (nn.Linear, nn.Embedding)) and f"{p}.scales" in weights
    nn.quantize(
        model,
        **quantization,
        class_predicate=class_predicate,
    )

def query_model(config: Dict, model_class):
    model_args_class = llama.ModelArgs
    model_args = model_args_class.from_dict(config)
    return model_class(model_args)

def load_weights(config: Dict, model: llama.Model, model_path: Path):
    import glob 
    weights = {}
    for weight_path in glob.glob(str(model_path / "*.safetensors")):
        weights.update(mx.load(weight_path))
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    quantize_if_necessary(config, model, weights)
    model.load_weights(list(weights.items()))
    model.eval()

def load(model_path: Path):
    # Load model manually by first reading HF config.
    with open(model_path / "config.json", "r") as f:
        import json
        config = json.loads(f.read())
    if config["model_type"] not in ["llama", "mistral"]:
        error(f"Model type '{config["model_type"]}' is not supported on mlx")
    model = query_model(config, llama.Model)
    load_weights(config, model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

@dataclass
class FastLanguageModel(nn.Module):
    model: nn.Module
    r: int
    lora_alpha: int
    lora_dropout: int
    random_state: int
    use_rslora: bool

def get_peft_model(**kwargs): return FastLanguageModel(**kwargs)

@dataclass
class Trainer:
    model: FastLanguageModel
    tokenizer: Tokenizer
    args: TrainingArgs
    # data_collator: Optional[DataCollator]
    lr_scheduler_type: str
    learning_rate: float
    warmup_steps: int
    max_steps: int
    weight_decay: float
    optim: str
    train_dataset: Union[Dataset, DatasetDict]
    eval_dataset: Union[Dataset, DatasetDict]
    compute_loss_func: Optional[Callable] = None
    compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None
    callback: Optional[TrainingCallback] = None

def save_adapter_config(num_layers: int, lora_config: Dict):
    from unsloth.mlx.mlx_utils import save_config
    adapter_path = MODEL_DIR / "finetuned"
    adapter_path.mkdir(parents=True, exist_ok=True)
    config = {
        "num_layers" : num_layers,
        "lora_parameters" : lora_config
    }
    save_config(config, adapter_path / "adapter_config.json")

def configure_optimizer(trainer: Trainer):
    from unsloth.mlx.trainer.utils import build_schedule
    if trainer.lr_scheduler_type == "linear":
        arguments = [0.0,trainer.learning_rate,trainer.warmup_steps]
    elif trainer.lr_scheduler_type == "exponential_decay":
        arguments = [trainer.learning_rate,trainer.weight_decay]
    elif trainer.lr_scheduler_type == "step_decay":
        arguments = [trainer.learning_rate,trainer.weight_decay,trainer.warmup_steps]
    elif trainer.lr_scheduler_type == "cosine_decay":
        arguments = [trainer.learning_rate,trainer.max_steps]
    else:
        arguments = [trainer.learning_rate]
    schedule_config = {
        "name": "linear_schedule"
                if trainer.lr_scheduler_type == "linear"
                else trainer.lr_scheduler_type,
        "warmup": trainer.warmup_steps,
        "arguments": arguments,
    }
    lr = build_schedule(schedule_config) if trainer.lr_scheduler_type else trainer.learning_rate
    if trainer.optim.lower().startswith("sgd"):
        opt = optim.SGD(learning_rate=(lr), weight_decay=trainer.weight_decay)
    elif trainer.optim.lower().startswith("rmsprop"):
        opt = optim.RMSprop(learning_rate=(lr))
    elif trainer.optim.lower().startswith("adagrad"):
        opt = optim.Adagrad(learning_rate=(lr))
    elif trainer.optim.lower().startswith("adaDelta"):
        opt = optim.AdaDelta(learning_rate=(lr))
    elif trainer.optim.lower().startswith("adamw"):
        opt = optim.AdamW(learning_rate=(lr),weight_decay=trainer.weight_decay)
    elif trainer.optim.lower().startswith("adam"):
        opt = optim.Adam(learning_rate=(lr))
    elif trainer.optim.lower().startswith("adamax"):
        opt = optim.Adamax(learning_rate=(lr))
    elif trainer.optim.lower().startswith("lion"):
        opt = optim.Lion(learning_rate=(lr), weight_decay=trainer.weight_decay)
    elif trainer.optim.lower().startswith("adafactor"):
        opt = optim.Adafactor(learning_rate=(lr), weight_decay=trainer.weight_decay)
    else:
        raise ValueError("The Optimizer type provided is not supported")
    return opt


def train(trainer: Trainer):
    from unsloth.mlx.trainer.trainer import train
    from unsloth.mlx.trainer.utils import (
        linear_to_lora_layers,
        print_trainable_parameters,
    )
    model, module, args = trainer.model, trainer.model.model, trainer.args
    module.freeze()
    lora_config = {
        "rank": model.r,
        "alpha": model.lora_alpha,
        "dropout": model.lora_dropout,
        "scale": float(model.lora_alpha)/math.sqrt(float(model.r))
                 if model.use_rslora
                 else float(model.lora_alpha)/float(model.r)
    }
    num_layers = min(model.r, len(module.layers)//2)
    optimizer = configure_optimizer(trainer)
    linear_to_lora_layers(module, num_layers, lora_config)
    print_trainable_parameters(module)
    save_adapter_config(num_layers, lora_config)
    mx.random.seed(model.random_state)
    model.train()
    train(
        model=module,
        tokenizer=trainer.tokenizer,
        args=args,
        optimizer=optimizer,
        train_dataset=trainer.train_dataset,
        val_dataset=trainer.eval_dataset,
        training_callback=trainer.callback,
    )
