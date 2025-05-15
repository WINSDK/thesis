from dataclasses import dataclass
from pathlib import Path
from datasets import Dataset
import re
from synthwave import Scheme, parse_poly_type

ROOT = Path(__file__).resolve().parent

PROMPT = """
# Constrains

* `String`'s are represented as `List Char`'s.
* Support for partial applition exists.
* You are allowed to provide answers in a point-free form.

# Functions availabe in the language

### Arithmetic primitives

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
{}
Output 3:
lambda \
"""

def transform(ds_name, item):
    messages = [
        {"role": "user", "content": PROMPT.format(item["example"]) },
        {"role": "assistant", "content": item["answer"] }
    ]
    return {
        "dataset": ds_name,
        "messages": messages,
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
