from .helpers import load_model, IS_MACOS, MODEL_DIR
from .eval import evaluate
from .dsl import UOp, ExternalError, pretty_print
from .typing import infer, Type, Scheme
from .parser import parse, parse_poly_type
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
    "Scheme",
    "Type",
    "parse",
    "parse_poly_type",
    "infer",
    "evaluate",
    "pretty_print"
]
