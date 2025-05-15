import math
import synthwave
from lark import Token
from synthwave.parser import term_parser
from synthwave.eval import KNOWN_VARS
from collections import Counter

def parse_score(response):
    for score, f in zip([3.0, 1.5], [synthwave.parse, synthwave.parse_inc]):
        try:
            f(response, known=KNOWN_VARS)
            return score
        except Exception:
            pass
    return 0.0

def single(f, completions, **kwargs):
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = f(response)
        scores.append(score)
    return scores

def valid_syntax(completions, **kwargs):
    return single(parse_score, completions, **kwargs)

def token_entropy(response):
    tree = term_parser.parse(response)
    alphabet = []
    for node in tree.iter_subtrees():
        for child in node.children:
            if not isinstance(child, Token):
                continue
            alphabet.append(child.type)
    freq = Counter(alphabet)
    probs = {s: c / len(alphabet) for s, c in freq.items()}
    return -sum(p * math.log2(p) for p in probs.values())

def diverse_output(completions, **kwargs):
    def f(response):
        # To have a diverse output, you first need a correct output
        if parse_score(response) == 3.0:
            return token_entropy(response)
        else:
            return 0.0
    return single(f, completions, **kwargs)

def infers_type_correctly(completions, **kwargs):
    raise NotImplementedError()
