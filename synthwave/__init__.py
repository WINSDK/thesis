from .helpers import load_model, IS_MACOS, MODEL_DIR
import os

# For some reason unsloth collects statistics when running
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

__all__ = ['load_model', 'IS_MACOS', 'MODEL_DIR']
