from .helpers import load_model, IS_MACOS, MODEL_DIR
from .eval import evaluate
from .dsl import UOp, ExternalError
from .typing import infer, Type
from .parser import parse
import os

# For some reason unsloth collects statistics when running
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
del os

__all__ = [
    "load_model",
    "IS_MACOS",
    "MODEL_DIR",
    "ExternalError",
    "UOp",
    "Type",
    "parse",
    "infer",
    "evaluate"
]
