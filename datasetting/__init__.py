from dataclasses import dataclass
from typing import Dict
from pathlib import Path
from datasets import Dataset
import re
from synthwave import Scheme, parse_poly_type

ROOT = Path(__file__).resolve().parent

def format_prompt(example: str):
    return [
        {"role": "system", "content": ""}, # Empty system prompt for now
        {"role": "user", "content": example}
    ]

@dataclass
class Prompt:
    ty: Scheme
    examples: list[tuple[str, str]]

    @staticmethod
    def of_dataset(prompt: list[Dict[str, str]]) -> "Prompt":
        content = prompt[1]["content"] \
            .removeprefix("Input:\n") \
            .removesuffix("\nOutput:")
        lines = content.splitlines()
        ty = parse_poly_type(lines[0])
        examples = []
        for line in lines[1:]:
            lhs, rhs = line.split('->')
            lhs, rhs = lhs.strip(), rhs.strip()
            examples.append((lhs, rhs))
        return Prompt(ty, examples)

def parse_handcraft(path):
    txt = open(path, "r").read()
    regexp = re.compile(r"Input:(.*?)Output:(.*?)(?=(?:Input:)|\Z)", re.DOTALL)
    examples = []
    for m in regexp.finditer(txt):
        inp = m.group(1).strip()
        answer = m.group(2).strip()
        prompt = format_prompt("Input:\n" + inp + "\nOutput:")
        examples.append({"prompt": prompt, "answer": answer})
    return Dataset.from_list(examples)

handcrafted = parse_handcraft(ROOT / "handcrafted.txt")
