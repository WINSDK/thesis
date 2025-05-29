import re
import synthwave
from synthwave.eval import KNOWN_VARS
from synthwave.typing import infer, unify, Type, T, fresh_type_var
from datasetting import EOS, REASONING_END, SOLUTION_START, SOLUTION_END

# Add optional EOS token matching
SOLUTION_END_REGEX = SOLUTION_END + r"[\s]{0,}" + "(?:" + re.escape(EOS) + ")?"

MATCH_FORMAT = re.compile(
    rf"{REASONING_END}.*?"\
    rf"{SOLUTION_START}(.+?){SOLUTION_END_REGEX}"\
    rf"$",
    flags = re.MULTILINE | re.DOTALL
)

def single(f, completions, **kwargs):
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = f(response, **kwargs)
        scores.append(score)
    return scores

def match_format_exactly(completions, **kwargs):
    def f(response):
        if MATCH_FORMAT.search(response) is not None:
            return 1.5
        return 0.0
    return single(f, completions, **kwargs)

def match_format_approx(completions, **kwargs):
    def f(response):
        score = 0
        score += 0.5 if response.count(REASONING_END)   == 1 else -1.0
        score += 0.5 if response.count(SOLUTION_START)  == 1 else -1.0
        score += 0.5 if response.count(SOLUTION_END)    == 1 else -1.0
        return score
    return single(f, completions, **kwargs)

def valid_lambda_expr(completions, **kwargs):
    def f(response):
        try:
            match = MATCH_FORMAT.search(response)
            if match is None:
                return 0.0
            solution_code = match.group(1).strip()
            synthwave.parse(solution_code, known=KNOWN_VARS)
            return 3.0
        except Exception:
            return 0.0
    return single(f, completions, **kwargs)

def correct(completions, **kwargs):
    def f(response, **kwargs):
        try:
            match = MATCH_FORMAT.search(response)
            if match is None:
                return 0.0
            solution_code = match.group(1).strip()
            expr = synthwave.parse(solution_code, known=KNOWN_VARS)

            prompt_data = kwargs["prompt_data"]
            tests = eval(prompt_data["tests"])
            args_info = eval(prompt_data["args_info"])

            correct_count = 0
            for test in tests:
                try:
                    params = [test["input"][a] for a in args_info.keys()]
                    if expr(*params) == test["output"]:
                        correct_count += 1
                except Exception:
                    continue

            # Give partial points for getting some correct
            score = 3.0 * (correct_count / len(tests))**2 if tests else 0.0
            return score
        except Exception:
            return 0.0

    return single(f, completions, **kwargs)

def infers_type_correctly(completions, **kwargs):
    def f(response, **kwargs):
        try:
            match = MATCH_FORMAT.search(response)
            if match is None:
                return 0.0
            solution = match.group(1).strip()
            parsed = synthwave.parse(solution, known=KNOWN_VARS)
            ty = infer(parsed)

            prompt_data = kwargs["prompt_data"]
            print(prompt_data["args_info"])
            args_info = eval(prompt_data["args_info"])
            return_type = prompt_data["return_type"]

            def convert_metaset_type(type_str):
                if type_str == "int":
                    return Type(T.Int)
                elif type_str == "float":
                    return Type(T.Float)
                elif type_str == "bool":
                    return Type(T.Bool)
                elif type_str == "char":
                    return Type(T.Char)
                elif type_str == "int[]":
                    return Type(T.List, [Type(T.Int)])
                elif type_str == "float[]":
                    return Type(T.List, [Type(T.Float)])
                elif type_str == "bool[]":
                    return Type(T.List, [Type(T.Bool)])
                elif type_str == "char[]" or type_str == "string":
                    return Type(T.List, [Type(T.Char)])
                else:
                    return fresh_type_var()

            input_types = [convert_metaset_type(type_str) for type_str in args_info.values()]
            output_type = convert_metaset_type(return_type)
            if len(input_types) == 1:
                expected_type = Type(T.Arrow, [input_types[0], output_type])
            else:
                expected_type = Type.arrow(input_types, output_type)
            try:
                unify(ty, expected_type, {})
                return 3.0
            except TypeError:
                # Types don't match
                return 0.0
        except Exception:
            return 0.0
    return single(f, completions, **kwargs)
