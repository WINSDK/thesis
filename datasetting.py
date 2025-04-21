import re
from datasets import Dataset

def format_prompt(example: str):
    return [
        {"role": "system", "content": ""}, # Empty system prompt for now
        {"role": "user", "content": example}
    ]


def parse_handcraft(path):
    txt = open(path, "r").read()
    regexp = re.compile(r"Input:(.*?)Output:(.*?)(?=(?:Input:)|\Z)", re.DOTALL)
    examples = []
    for m in regexp.finditer(txt):
        inp = m.group(1).strip()
        completion = m.group(2).strip()
        prompt = format_prompt("Input:\n" + inp + "\nOutput:")
        examples.append({"prompt": prompt, "completion": completion})
    return Dataset.from_list(examples)

handcrafted = parse_handcraft("./datasets/handcrafted.txt")
