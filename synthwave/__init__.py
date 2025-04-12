from .helpers import load_model, IS_MACOS, MODEL_DIR
from .dsl import ExternalError, UOp, Type
from .dsl import parse, infer, evaluate
import os

# For some reason unsloth collects statistics when running
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

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
