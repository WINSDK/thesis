from dataclasses import dataclass
from pathlib import Path
from datasets import Dataset
import re
from synthwave import Scheme, parse_poly_type

ROOT = Path(__file__).resolve().parent

REASONING_START = "<start_working_out>"
REASONING_END   = "<end_working_out>"
SOLUTION_START  = "<SOLUTION>"
SOLUTION_END    = "</SOLUTION>"
EOS = "<|endoftext|>"

PROMPT = """
# Constrains

* `String`'s are represented as `List Char`'s.
* Support for partial applition exists.
* You are allowed to provide answers in a point-free form.

# Functions availabe in the language

### Arithmetic primitives

inf :: Float
add :: T -> T -> T
+ :: T -> T -> T
mul :: T -> T -> T
* :: T -> T -> T
sub :: T -> T -> T
- :: T -> T -> T
div :: T -> T -> T
/ :: T -> T -> T
mod :: T -> T -> T
% :: T -> T -> T
pow :: T -> T -> T
** :: T -> T -> T

### Comparisons

if :: T -> A -> B
eq :: T -> T -> Bool
neq :: T -> T -> Bool
gt :: T -> T -> Bool
lt :: T -> T -> Bool
geq :: T -> T -> Bool
leq :: T -> T -> Bool

### Boolean operators

True :: Bool
False :: Bool
not :: Bool -> Bool
and :: Bool -> Bool -> Bool
or :: Bool -> Bool -> Bool

### List utilities

nil :: List T
is_nil :: List T -> Bool
lfold :: List A -> B -> (B -> A -> A) -> B == lfold lst acc (lambda acc x. ...)
rfold :: List A -> (A -> B -> B) -> B -> B == lfold lst (lambda x acc. ...) acc
map :: List A -> (A -> B) -> List B
filter :: List A -> (A -> Bool) -> List A
reduce :: List A -> (A -> Bool) -> List A
zip :: List T -> List T -> List (List T)
length :: List A -> Int
range :: Int -> Int -> List Int
cons :: T -> List T -> List T
head :: List T -> T
tail :: List T -> List T
append :: List T -> T -> List T
reverse :: List T -> List T
sort :: List T -> List T
flatten :: List List T -> List T
min :: List T -> T
max :: List T -> T

### String manipulation

concat :: String -> String -> String
substr :: String -> Int -> Int -> String
split :: String -> String -> List String
join :: List String -> String -> String

### Conversion

show :: T -> String
read :: String -> Int
ord  :: Char -> Int
chr  :: Int -> Char

### Utility/functional

id :: A -> A
compose :: (B -> C) -> (A -> B) -> A -> C

# Task

I will begin by giving you several examples of input-output pairs.
You will then be given a new input, and you must provide the corresponding output.

Input 1:
[1, 2, 3] -> [2, 4, 6]
[] -> []
[9, 9] -> [18, 18]
Output 1:
lambda xs.map xs (* 2)

Input 2:
[1, 2, 3, 6] -> [2, 6]
[9, 8] -> [8]
[1, 2, 3, 4, 5, 6] -> [2, 4, 6]
Output 2:
lambda xs.filter xs (lambda x.eq (mod x 2) 0)

Input 3:
[3, 9, 1, 5] -> [9, 5, 3, 1]
[1, 3] -> [3, 1]
[30, 10, 20] -> [30, 20, 10]
Output 3:
lambda xs.reverse (sort xs)

Input 4:
{}
Output 4:"""

SYSTEM_PROMPT = f"""\
You are given a problem.
Think about the problem and provide your working out.
Place it between <start_working_out> and <end_working_out>.
Then, provide your solution between {SOLUTION_START}{SOLUTION_END}"""

def transform_to_messages(msg):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": PROMPT.format(msg)},
    ]

def transform(ds_name, item):
    return {
        "dataset": ds_name,
        "prompt": transform_to_messages(item["example"]),
        "answer": item["answer"]
    }

@dataclass
class Prompt:
    ty: Scheme
    examples: list[tuple[str, str]]

    @staticmethod
    def of_str(s: str) -> "Prompt":
        lines = s.splitlines()
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
        input = m.group(1).strip()
        answer = m.group(2).strip()
        examples.append({"example": input, "answer": answer})
    return Dataset.from_list(examples)

handcrafted = parse_handcraft(ROOT / "handcrafted.txt")

def transform_metaset3_to_messages(item):
    """Transform metaset3 item to GRPO-compatible format matching PROMPT structure"""
    # Format test cases as input -> output pairs only
    examples = []
    for test in item["tests"]:
        # Extract input values in order of function args
        input_vals = [str(test["input"][arg]) for arg in item["args"].keys()]
        if len(input_vals) == 1:
            input_str = input_vals[0]
        else:
            input_str = f"[{', '.join(input_vals)}]"
        examples.append(f"{input_str} -> {test['output']}")

    example_text = "\n".join(examples)

    return transform_to_messages(example_text)

def flatten_tree_to_lambda(tree):
    """Convert nested list tree structure to lambda expression string"""
    if isinstance(tree, str):
        return tree
    elif isinstance(tree, list):
        if not tree:
            return ""
        if len(tree) == 1:
            return flatten_tree_to_lambda(tree[0])
        # Handle function application
        func = flatten_tree_to_lambda(tree[0])
        args = " ".join(flatten_tree_to_lambda(arg) for arg in tree[1:])
        if args:
            return f"({func} {args})"
        else:
            return func
    else:
        return str(tree)

def load_metaset3_dataset(split="train"):
    """Load metaset3 dataset for GRPO training"""
    import json

    file_path = ROOT / f"metaset3.{split}.jsonl"
    items = []

    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())

            # Format input-output examples properly
            examples = []
            for test in item["tests"]:
                if len(item["args"]) == 1:
                    input_val = str(test["input"][list(item["args"].keys())[0]])
                else:
                    input_val = str(list(test["input"].values()))
                examples.append(f"{input_val} -> {test['output']}")

            transformed = transform("metaset3", {
                "example": "\n".join(examples),
                "answer": flatten_tree_to_lambda(item["short_tree"])
            })
            transformed.update({
                "target_program": str(item["short_tree"]),
                "tests": str(item["tests"]),
                "task_text": " ".join(item["text"]),
                "args_info": str(item["args"]),
                "return_type": item["return_type"],
                "nodes": str(item["nodes"])
            })
            items.append(transformed)

    return Dataset.from_list(items)
